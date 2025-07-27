import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
import scipy.optimize as optimize
sys.path.append(r"../../")
from beam_scan import BeamScan
from sar_focus import SAR_Focus
from dot_estimate import DotEstimator
from multiprocessing import Process, Queue
import multiprocessing as mp
import cv2
from tqdm import tqdm
from sinc_interpolation import SincInterpolation


class FScanAzimuth(BeamScan):
    def __init__(self):
        super().__init__()
        self.theta_c = np.deg2rad(0)  # 斜视角
        self.theta_az = np.deg2rad(0.05)
        self.theta_sc = np.deg2rad(0.2)
        self.Br = 320e6
        self.Fr = self.Br*1.3
        self.Tr = self.Tp*10
        self.Nr = int(np.ceil(self.Fr*self.Tr))
        self.Kr = self.Br/self.Tp
        self.alpha = self.Br/(self.theta_sc-self.theta_az)
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+self.theta_sc/2) - np.sin(self.theta_c-self.theta_sc/2))/self.lambda_
        self.Rc = self.R0/np.cos(self.theta_c)
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_
        print("therical res: range:{}, azimuth:{}".format(self.c*self.theta_sc/(2*self.Br*self.theta_az), self.Vr/self.Bd))

        self.PRF = 2000
        print("Bd:{}, B_fov:{}".format(self.Bd, self.B_fov))
        self.Fa = self.PRF ## 初始采样率

        self.points_n = 3
        self.points_r = self.R0 + cp.array([-141, 0, 141])
        self.points_a = cp.array([-99, 0, 99])

        self.Ta = 0.8
        self.Na = int(np.ceil(self.PRF*self.Ta))

        self.R_min = self.R0 - 119
        self.R_max = self.R0 + 119
        self.A_min = -99
        self.A_max = 99

    def dist_gen(self, pre_target):
        
        image_shape = pre_target.shape

        self.points_r = cp.linspace(self.R_min, self.R_max, (image_shape[1]))
        self.points_r = cp.tile(self.points_r[cp.newaxis,:], (image_shape[0], 1)).flatten()
        self.points_a = cp.linspace(self.A_min, self.A_max, (image_shape[0]))
        self.points_a = cp.tile(self.points_a[:,cp.newaxis], (1, image_shape[1])).flatten()
        self.points_n = len(self.points_r)
        print("Distributed shape:",image_shape)

        self.tau_c = 2*self.Rc/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in tqdm(range(self.points_n)):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = np.arctan((self.points_a[i] - self.Vr*mat_eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负

            ftau_l = self.alpha*(-doa+self.theta_c-self.theta_az/2)
            ftau_u = self.alpha*(-doa+self.theta_c+self.theta_az/2)

            ## 接收机频率与时间的关系
            ftau_send = (self.Kr*(mat_tau - 2*R_eta/self.c)) 

            ## 脉宽限制
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            ## 从距离向描述fscan限制
            Wr_fscan = (ftau_send>ftau_l) * (ftau_send<ftau_u)
            # Wr = 1
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            signal_r = phase_r*Wr*Wr_fscan

            Tstrip_tar = self.theta_sc*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + self.eta_c)) < Tstrip_tar/2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += pre_target[i//image_shape[1], i%image_shape[1]]*signal_r*signal_a
        return S_echo


    def echogen(self):
        ##接收机时间窗
        self.tau_c = 2*self.Rc/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = np.arctan((self.points_a[i] - self.Vr*mat_eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负

            ftau_l = self.alpha*(-doa+self.theta_c-self.theta_az/2)
            ftau_u = self.alpha*(-doa+self.theta_c+self.theta_az/2)

            ## 接收机频率与时间的关系
            ftau_send = (self.Kr*(mat_tau - 2*R_eta/self.c)) 

            ## 脉宽限制
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            ## 从距离向描述fscan限制
            Wr_fscan = (ftau_send>ftau_l) * (ftau_send<ftau_u)
            # Wr = 1
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            signal_r = phase_r*Wr*Wr_fscan

            Tstrip_tar = self.theta_sc*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + self.eta_c)) < Tstrip_tar/2

            ## 从方位向描述fscan限制
            # feta_l = 2*self.Vr*cp.sin(-ftau_send/self.alpha +self.theta_c-self.theta_az/2)/self.lambda_
            # feta_u = 2*self.Vr*cp.sin(-ftau_send/self.alpha +self.theta_c+self.theta_az/2)/self.lambda_
            # feta = 2*self.Vr*cp.sin(doa)/self.lambda_
            # Wa_fscan = (feta > feta_l) * (feta < feta_u)
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo

    def azimuth_mosaic(self, echo_ftau_feta):
        [Na, Nr] = echo_ftau_feta.shape
        self.Fa = self.Bd*1.2
        uprate = self.Fa/self.PRF
        if uprate < 1:
            uprate = 1
            self.Fa = self.PRF
        uprate = int(cp.ceil(uprate))
        self.Fa = self.PRF*uprate
        echo_ftau_feta_up = cp.tile(echo_ftau_feta, (uprate, 1))

        Na_up = Na * uprate
        self.Na = Na_up
        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        feta = self.feta_c + cp.arange(-Na_up/2, Na_up/2, 1)*(self.Fa/(Na_up))
        mat_ftau, mat_feta = cp.meshgrid(ftau, feta)
        

        ## PRF 对应的角宽
        theta_width = np.arcsin(self.PRF*self.lambda_/(2*self.Vr))
        ## 频率扫描的多普勒边缘
        feta_l = 2*self.Vr*np.sin(self.theta_c-mat_ftau/self.alpha - theta_width/2)/self.lambda_ - self.feta_c
        feta_u = 2*self.Vr*np.sin(self.theta_c-mat_ftau/self.alpha + theta_width/2)/self.lambda_ - self.feta_c

        # W_fscan = cp.abs(mat_feta-self.feta_c-feta_R0) < self.PRF/2
        W_fscan = (mat_feta - self.feta_c > feta_l)*(mat_feta -self.feta_c < feta_u)


        shift = int(cp.round(self.PRF/(2*self.Fa/Na_up)))
        if int(cp.round(self.theta_sc/self.theta_az)) % 2 == 1:
            shift = 0
        W_fscan = cp.roll(W_fscan, shift, axis=0)
        echo_ftau_feta_up *= W_fscan
        echo_ftau_feta_up = cp.roll(echo_ftau_feta_up, -shift, axis=0)
        return echo_ftau_feta_up
    
    def shift2center(self, echo_tau_feta):
        [Na,Nr] = echo_tau_feta.shape
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.Fa)
        feta = cp.arange(-Na/2, Na/2, 1)*(self.Fa/(Na))

        mat_tau, mat_feta = cp.meshgrid(tau, feta)
        mat_ftau, mat_eta = cp.meshgrid(ftau, eta)
        doa = cp.arcsin(feta*self.lambda_/(2*self.Vr))
        theta_width = np.arcsin(self.PRF*self.lambda_/(2*self.Vr))
        ftau_target = self.alpha*(-doa+self.theta_c)
        ftau_center = 0
        ftau_shift = ftau_center - ftau_target
        mat_ftau_shift = cp.tile(ftau_shift[:, cp.newaxis], (1, Nr))
        Hx_shift = cp.exp(1j*2*cp.pi*mat_ftau_shift*mat_tau)
        echo_tau_feta_shift = echo_tau_feta*Hx_shift

        # echo_ftau_eta = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo_tau_feta_shift)))
        # feta_target = 2*self.Vr*cp.sin(-mat_ftau/self.alpha +self.theta_c-self.theta_az/2)/self.lambda_
        # feta_center = self.feta_c
        # feta_shift = feta_center - feta_target
        # Hy_shift = cp.exp(1j*2*cp.pi*feta_shift*mat_eta)
        # echo_ftau_eta_shift = echo_ftau_eta * Hy_shift
        # echo_tau_feta_shift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(echo_ftau_eta_shift)))

        slope = 2*self.Vr/(self.lambda_*self.alpha)
        ratio = (self.theta_sc)/self.theta_az
        Fr1 = (ratio-1)*self.B_fov*self.lambda_*self.alpha/(2*self.Vr)*self.Tp/self.Tr*2.5
        print(Fr1)
        # Fr1 = self.B_fov*self.lambda_*self.alpha/(2*self.Vr)
        echo_ftau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo_tau_feta_shift, axes=1), axis=1), axes=1)
        echo_ftau_feta=self.tau_down_sample(echo_ftau_feta, Fr1)
        echo_tau_feta_shift = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_ftau_feta, axes=1), axis=1), axes=1)

        [Na, Nr] = echo_ftau_feta.shape
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        mat_tau, mat_feta = cp.meshgrid(tau, feta)
        ftau_center = -Fr1/2
        ftau_shift = ftau_center - ftau_target
        mat_ftau_shift = cp.tile(ftau_shift[:, cp.newaxis], (1, Nr))
        Hxi = cp.exp(-1j*2*cp.pi*mat_ftau_shift*mat_tau)
        echo_tau_feta_shift = echo_tau_feta_shift*Hxi

        echo_ftau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo_tau_feta_shift, axes=1), axis=1), axes=1)
        echo_ftau_feta = self.tau_up_sample(echo_ftau_feta, 250e6*1.5)
        echo_tau_feta_shift = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_ftau_feta, axes=1), axis=1), axes=1)

        return echo_tau_feta_shift
    
    def tau_down_sample(self, image_fft, Fr_down):
        [Na,Nr] = image_fft.shape

        width = Fr_down/(self.Fr/Nr)
        self.Fr = Fr_down
        return image_fft[:, Nr//2-width//2:Nr//2+width//2]
    
    def tau_up_sample(self, image_fft, Fr_up):
        [Na,Nr] = image_fft.shape

        Nr_up = int(cp.ceil(Fr_up/self.Fr*Nr))
        self.Fr = Nr_up*self.Fr/Nr
        echo_fft2_new = cp.zeros((Na, Nr_up), dtype=cp.complex128)
        echo_fft2_new[:, Nr_up//2-Nr/2:Nr_up//2+Nr/2] = image_fft
        return echo_fft2_new
    
        
def distributed_plot(pre_target, extent):
    plt.figure()
    plt.imshow(np.abs(pre_target), aspect='auto', cmap='grey', extent=extent)
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/distributed.png")

def echo_plot(echo, extent):
    plt.figure()
    plt.imshow(np.abs(echo), aspect='auto', cmap='jet', extent=extent)
    plt.xlabel('Range time(us)')
    plt.ylabel('Azimuth time(s)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo.png")

def azimuth_mosaic_plot(echo_tau_feta, extent):
    plt.figure()
    plt.imshow(np.abs(echo_tau_feta), aspect='auto', cmap='jet', extent=extent)
    plt.xlabel('Range Time(us)')
    plt.ylabel('Azimuth Frequency(Hz)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo_range_fft.png")

def echo_fft2_plot(echo_ftau_feta, extent):
    plt.figure()
    plt.imshow(np.abs(echo_ftau_feta), aspect='auto', cmap='jet', extent=extent)
    plt.xlabel('Range Frequency')
    plt.ylabel('Azimuth Frequency')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo_fft2.png")

def image_plot(image, extent, figuresize):
    image_show = np.abs(image)
    image_show = image_show/image_show.max()
    image_show = 20*np.log10(image_show)
    # image_show = geometry_correction(image_show)

    plt.figure(figsize=figuresize)
    plt.imshow(image_show, aspect='auto', cmap='grey', extent=extent, vmin=-30, vmax=0)
    plt.xlabel('Range time(us)')
    plt.ylabel('Azimuth time(s)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/image.png")

def image_fft_plot(image_fft, extent):
    plt.figure()
    plt.imshow(np.abs(image_fft), aspect='auto', cmap='jet', extent=extent)
    plt.xlabel('Range Frequency')
    plt.ylabel('Azimuth Frequency')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/image_fft.png")


def fscan_azimuth_sim(qfunc, qargs):
    # 定义参数
    cp.cuda.Device(1).use()  
    fscan = FScanAzimuth()

    pre_target = cv2.imread("./R-C.jpg", cv2.IMREAD_GRAYSCALE)
    # pre_target = np.ones((60, 60), dtype=np.float64)
    # pre_target[::2, :] = 0
    # pre_target[:, ::2] = 0
    pre_target = cp.array(pre_target)
    divider = 32
    pre_target = pre_target[pre_target.shape[0]//2-pre_target.shape[0]//divider:pre_target.shape[0]//2+pre_target.shape[0]//divider, pre_target.shape[1]//2-pre_target.shape[1]//divider:pre_target.shape[1]//2+pre_target.shape[1]//divider]
    pre_target = (pre_target-pre_target.min())/(pre_target.max()-pre_target.min())
    qfunc.put(distributed_plot)
    qargs.put((pre_target.get(), None))

    echo = fscan.dist_gen(pre_target)
    # echo = fscan.echogen()
    qfunc.put(echo_plot)
    qargs.put((echo.get(), (-fscan.Tr*1e6/2, fscan.Tr*1e6/2, -fscan.Ta/2 + fscan.eta_c.get(), fscan.Ta/2+fscan.eta_c.get())))

    ## 处理多普勒中心，将频域中心移至多普勒中心
    shift = fscan.feta_c / (fscan.Fa/fscan.Na)
    echo_ftau_feta = cp.fft.ifftshift(cp.fft.fft2(cp.fft.ifftshift(echo)))
    echo_ftau_feta = cp.roll(echo_ftau_feta, -int(cp.round(shift)), axis=0)

    # qfunc.put(echo_fft2_plot)
    # qargs.put((echo_ftau_feta.get(), (-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))

    ## 拼接
    echo_ftau_feta = fscan.azimuth_mosaic(echo_ftau_feta)
    # echo_ftau_feta = fscan.spectrum_adjust(echo_ftau_feta)


    qfunc.put(echo_fft2_plot)
    qargs.put((echo_ftau_feta.get(), (-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))
    echo = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(echo_ftau_feta)))

    ## 成像
    focus = SAR_Focus(fscan.Fr, fscan.Tp, fscan.f0, fscan.Fa, fscan.Vr, fscan.Br, fscan.feta_c, fscan.R0, fscan.Kr)
    image = focus.wk_focus(echo, fscan.R0)
    ## 平移

    image_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(image)))
    
    [Na, Nr] = image_fft.shape
    ftau = cp.arange(-Nr/2, Nr/2, 1)*(fscan.Fr/Nr)
    mat_ftau = cp.tile(ftau[cp.newaxis, :], (Na, 1))
    shift = cp.exp(1j*2*cp.pi*fscan.Tr/2*mat_ftau)
    image_fft = image_fft * shift

    image_tau_feta = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(image_fft, axes=1), axis=1), axes=1)
    image_tau_feta = fscan.shift2center(image_tau_feta)

    image_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image_tau_feta, axes=1), axis=1), axes=1)


    image = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_fft)))
    

    # extent = ((fscan.tau_c-fscan.Tr/2)*1e6, (fscan.tau_c+fscan.Tr/2)*1e6, (fscan.eta_c-fscan.Ta/2).get(),(fscan.eta_c+fscan.Ta/2).get())
    extent = None
    x_size = 10
    y_size = x_size*fscan.Na*fscan.Vr/fscan.Fa/(fscan.Nr*fscan.c/(2*fscan.Fr))

    qfunc.put(image_plot)
    qargs.put((image.get(), extent, (x_size, y_size)))

    qfunc.put(image_fft_plot)
    qargs.put((image_fft.get(), (-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))

    # dot_estimate = DotEstimator(fscan.points_n, fscan.c, fscan.Vr, fscan.Fa, fscan.Fr, "../../../fig/fscan_azimuth/")
    # dot_estimate.dot_estimate((image).get(), (30, int(30*cp.round(fscan.Vr/fscan.Fa/(fscan.c/(2*fscan.Fr))).get())), 16)

    print("fscan azimuth simulation done")

def plot_sim(qfunc, qargs):
    print("plot process start")
    process_cnt = 0
    process_list = []
    func_list = []
    while True:
        args = qargs.get()
        func = qfunc.get()
        if args is None or func is None:
            break
        func_process = Process(target=func, args=args)
        func_process.start()
        process_list.append(func_process)
        func_list.append(func)
        process_cnt += 1
    for i in range(process_cnt):
        process_list[i].join()
        print(func_list[i].__name__, " done")

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)  # 使用 spawn 方法启动进程
    qfunc = Queue()
    qargs = Queue()
    plot_process = Process(target=plot_sim, args=(qfunc, qargs))
    plot_process.start()

    simulate_process = Process(target=fscan_azimuth_sim, args=(qfunc, qargs))
    simulate_process.start()
    simulate_process.join()

    ## terminate the plot process
    qfunc.put(None)
    qargs.put(None)
    plot_process.join()
    exit(0)
