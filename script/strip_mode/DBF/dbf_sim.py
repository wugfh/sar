## 

import scipy.io as sci
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
from scipy import signal
sys.path.append(r"../")
from sinc_interpolation import SincInterpolation

data_1 = sci.loadmat("../../../data/English_Bay_ships/data_1.mat")
data_1 = data_1['data_1']
cp.cuda.Device(0).use()

class RangeSIM:
    def __init__(self):
        self.H = 519e3                          #卫星高度
        self.fscan_tau = 0.2806e-6              #频扫通道时延
        self.fscan_N = 10                       #频扫通道数   
        self.fscan_d = 1                        #频扫通道间隔
        self.DBF_N = 10                         #DBF天线数
        self.Re = 6371.39e3                     #地球半径
        self.beta = cp.deg2rad(25)              #天线安装角
        self.d_ra = 1                           #DBF子孔径间距
        self.c = 299792458                      #光速
        self.Fs = 120e6                         #采样率              
        self.Tr = 20e-05                        #脉冲宽度                        
        self.f0 = 30e+09                        #载频                     
        self.PRF = 1950                         #PRF                     
        self.Vr = 7062                          #雷达速度     
        self.B = 30e6                           #信号带宽
        self.fc = -6500                         #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.R0 = self.H/cp.cos(self.beta)
        self.La = 15
        self.Ta = 1
        self.Kr = -self.B/self.Tr
        

    def init_raw_data(self, data1):
        [Na, Nr] = cp.shape(data1)
        data = cp.zeros([int(cp.ceil(Na*1.5)), int(cp.ceil(Nr*1.5))], dtype=cp.complex128)
        [self.Na, self.Nr] = cp.shape(data)
        data[self.Na/2-Na/2:self.Na/2+Na/2, self.Nr/2-Nr/2:self.Nr/2+Nr/2] = data1
        return data
    
    ## 添加高斯白噪声
    def add_noise(self, data, snr):
        data = data + cp.random.randn(self.Na, self.Nr)*cp.sqrt(cp.var(data)/10**(snr/10))
        return data
    
    def init_simparams(self):
        self.Nr = int(cp.ceil(self.Fs*self.Tr))
        self.Na = int(cp.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+cp.linspace(-5000, 5000, self.points_n)
        self.points_a = cp.linspace(-3000, 3000, self.points_n)
        self.Ba = 2*0.886*self.Vr*cp.cos(self.theta_c)/self.La + 2*self.Vr*self.B*cp.sin(self.theta_c)/self.c
        self.Rc = self.R0/cp.cos(self.theta_c)

    def tradition_echogen(self):
        self.init_simparams()
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tr/2
            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            Phase = cp.exp(-4j*cp.pi*R_eta/self.lambda_)*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            S_echo += Wr*Wa*Phase
        return S_echo 

    def dbf_echogen(self):
        self.init_simparams()
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            t0 = R0_tar*2/self.c
            doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            P_sub = 1 ## 子孔径损耗
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
            ## 遍历DBF-SCORE子孔径
            for j in range(self.DBF_N):
                d = (j-(self.DBF_N-1)/2)*self.d_ra
                ti = t0-d*cp.sin(doa-self.beta)/self.c
                Wr = cp.abs(mat_tau-ti)<self.Tr/2
                phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-ti-2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.f0*ti) ## 基带信号
                channel_ri = P_sub*Wr*phase_r
                wi = cp.exp(-2j*cp.pi*d*cp.sin(doa-self.beta)/self.lambda_)
                rwi = channel_ri*wi
                signal_r += rwi

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo
    
    def fscan_echogen(self):
        self.init_simparams()
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)

        f_tau = cp.fft.fftshift(cp.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fs/self.Nr))
        f_eta = self.fc + (cp.linspace(-self.Na/2,self.Na/2-1,self.Na)*(self.PRF/self.Na))
        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = cp.arccos(((self.H+self.Re)**2+R_eta**2-self.Re**2)/(2*(self.H+self.Re)*R_eta))
            signal_t = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)

            deci = self.c/(self.f0+self.Kr*(mat_tau-2*R_eta/self.c))
            pr = cp.sin(self.fscan_N*(cp.pi*(self.fscan_tau*self.c-self.fscan_d*cp.sin(doa-self.beta)))/deci) \
            /(cp.sin(cp.pi*(self.fscan_tau*self.c-self.fscan_d*cp.sin(doa-self.beta))/deci))
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tr/2
            signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            ## 发射机,地面目标接收到的信号为所有通道的汇总
            # for j in range(self.fscan_N):
            #     tj = (j-(self.fscan_N-1)/2)*(self.fscan_tau - self.fscan_d*cp.sin(doa-self.beta)/self.c) 
            #     phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-tj-2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.f0*tj) ## 基带信号
            #     Wr = cp.abs(mat_tau-tj-2*R_eta/self.c)<self.Tr/2
            #     signal_t += phase_r*Wr
            
            # signal_r = signal_t*pr

            # ## 接收机,每个通道的时延不同
            # signal_fftr = cp.fft.fft(signal_t, axis = 1)
            # for j in range(self.fscan_N):
            #     tj = (j-(self.fscan_N-1)/2)*(self.fscan_tau - self.fscan_d*cp.sin(doa-self.beta)/self.c)+2*R_eta/self.c
            #     signal_t_shift = cp.fft.ifft(signal_fftr*cp.exp(-2j*cp.pi*mat_f_tau*tj), axis = 1)
            #     signal_r += signal_t_shift

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo

    def rd_foucus(self, echo):  

        f_tau = cp.fft.fftshift(cp.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fs/self.Nr))
        f_eta = self.fc + (cp.linspace(-self.Na/2,self.Na/2-1,self.Na)*(self.PRF/self.Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fft(echo, self.Nr, axis = 1) 
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hm = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        data_fft_cr = data_fft_r*Hr*Hm
        data_cr = cp.fft.ifft(data_fft_cr, self.Nr, axis = 1)

        ## RCMC
        data_fft_a = cp.fft.fft(data_cr, self.Na, axis=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta = mat_R0/mat_D - mat_R0
        delta = delta*2/(self.c/self.Fs)
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, self.Na, self.Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, self.Na, self.Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag

        ## 方位压缩
        Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifft(data_fft_a_rcmc, self.Na, axis=0)

        data_final = data_ca_rcmc
        # data_final = cp.abs(data_final)/cp.max(cp.max(cp.abs(data_final)))
        # data_final = 20*cp.log10(data_final)
        return data_final

    def get_IRW(self, ehco):
        max_index = cp.argmax(cp.abs(cp.max(cp.abs(ehco), axis=1))) 
        max_value = cp.max(cp.abs(ehco[max_index,:]))
        half_max = max_value/cp.sqrt(2)
        valid = cp.abs(ehco[max_index,:]) > half_max
        irw = cp.sum(valid)
        irw = irw*self.c/(2*self.Fs)
        return irw, max_index
    
    def get_pslr(self, target):
        target_np = cp.asnumpy(target)
        peaks, _ = signal.find_peaks(target_np)

        mainlobe_peak_index = peaks[np.argmax(target_np[peaks])]
        mainlobe_peak_value = target_np[mainlobe_peak_index]

        sidelobe_peaks = np.delete(peaks, np.argmax(target_np[peaks]))
        sidelobe_peak_value = np.max(target_np[sidelobe_peaks])

        pslr = 20 * np.log10(sidelobe_peak_value / mainlobe_peak_value)
        
        return pslr
    
def dbf_simulation():
    dbf_sim = RangeSIM()
    strip_echo = dbf_sim.tradition_echogen()
    strip_image = dbf_sim.rd_foucus(strip_echo)
    dbf_echo = dbf_sim.dbf_echogen()
    dbf_image = dbf_sim.rd_foucus(dbf_echo)

    plt.figure()
    plt.imshow(abs(cp.asnumpy(dbf_echo)), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/dbf_echo.png", dpi=300)

    plt.figure()
    plt.subplot(121)
    plt.contour(abs(cp.asnumpy(strip_image)))
    plt.title("strip mode")
    plt.subplot(122)
    plt.contour(abs(cp.asnumpy(dbf_image)))
    plt.title("dbf mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/sim_dbf.png", dpi=300)

    dbf_res, dbf_index = dbf_sim.get_IRW(dbf_image)
    strip_res, strip_index = dbf_sim.get_IRW(strip_image)
    print("dbf irw: ", dbf_res)
    print("strip irw: ", strip_res)
    print("theoretical irw: ", dbf_sim.c/(2*dbf_sim.B))

    dbf_target = cp.abs(dbf_image[dbf_index, :])
    dbf_target = dbf_target/cp.max(dbf_target)
    strip_target = cp.abs(strip_image[strip_index, :])
    strip_target = strip_target/cp.max(strip_target)
    tau = range(dbf_sim.Nr) 
    plt.figure()
    plt.subplot(121)
    plt.plot(tau, 20*cp.log10(cp.abs(dbf_target)).get())
    plt.title("dbf mode")
    plt.subplot(122)
    plt.plot(tau, 20*cp.log10(cp.abs(strip_target)).get())
    plt.title("strip mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/range.png", dpi=300)
    print("dbf pslr: ", dbf_sim.get_pslr(dbf_target))
    print("strip pslr: ", dbf_sim.get_pslr(strip_target))

def fscan_simulation():
    fscan_sim = RangeSIM()
    echo = fscan_sim.fscan_echogen()
    plt.figure()
    plt.imshow(abs(cp.asnumpy(echo)), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    echo_fft = cp.fft.fft(echo)
    echo_fft = cp.abs(echo_fft)
    echo_fft = 20*cp.log10(echo_fft)
    plt.figure()
    plt.imshow(cp.asnumpy(echo_fft), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


if __name__ == '__main__':
    fscan_simulation()
    # dbf_simulation()





