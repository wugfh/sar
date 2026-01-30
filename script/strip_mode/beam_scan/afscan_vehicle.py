import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
sys.path.append(r"../../")
from beam_scan import BeamScan
from sar_focus import SAR_Focus
from dot_estimate import DotEstimator
from multiprocessing import Process, Queue
import multiprocessing as mp
import cv2
from tqdm import tqdm
from sinc_interpolation import SincInterpolation
from joblib import Parallel, delayed
import scipy.optimize as op


class FScanAzimuth(BeamScan):
    def __init__(self):
        super().__init__()

        
        self.theta_c =  np.deg2rad(14.3)  # 斜视角
        self.theta_az = np.deg2rad(2.9)
        self.theta_lowf = np.deg2rad(17.94) 
        self.theta_upf =  np.deg2rad(10.9)
        self.theta_sc = np.abs(self.theta_upf - self.theta_lowf)  # 扫描角宽
        self.Br = 2e9
        self.Tr = self.Tp*2
        self.Kr = self.Br/self.Tp
        self.alpha = self.Br/(self.theta_upf-self.theta_lowf)
        self.Fr = self.Br*3
        self.Nr = int(np.ceil(self.Fr*self.Tr))
        self.Vr = 8.3
        
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+(self.theta_sc)/2) - np.sin(self.theta_c-self.theta_sc/2))/self.lambda_
        self.R0 = 96
        self.Rc = self.R0/np.cos(self.theta_c)
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_

        beta = 2*self.Vr*self.f0/(self.alpha*self.c)

        self.PRF = 500
        # print("Bd:{}, B_fov:{}".format(self.Bd, self.B_fov))
        self.Fa = self.PRF ## 初始采样率


        self.points_n = 1
        ratio = self.theta_sc/self.theta_az
        Fr1 = self.Br/ratio
        # self.points_r = self.R0 + cp.array([0, -100, 101])
        self.points_r = cp.array([0, -50, 0]) + self.Rc
        # print("points_r:", self.points_r-self.R0)
        self.points_a = cp.array([10, 0, -5])
        # print(self.Br/ratio*(0.989*2/self.c))
        # print(cp.sqrt(2)*(self.c/(2*self.Br/ratio)))

        self.Ta = 5
        self.Na = int(np.ceil(self.PRF*self.Ta))
    
    def set_Br(self, Br):
        self.Br = Br
        self.Fr = self.Br*1.2
        self.Kr = self.Br/self.Tp
        self.alpha = self.Br/(self.theta_sc-self.theta_az)
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+self.theta_sc/2) - np.sin(self.theta_c-self.theta_sc/2))/self.lambda_

    def set_theta(self, theta_sc, theta_az):
        self.theta_sc = np.deg2rad(theta_sc)
        self.theta_az = np.deg2rad(theta_az)
        self.alpha = self.Br/(self.theta_sc-self.theta_az)
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+self.theta_sc/2) - np.sin(self.theta_c-self.theta_sc/2))/self.lambda_


    def dist_gen(self, pre_target):
        
        image_shape = pre_target.shape
        ratio = self.theta_sc/self.theta_az

        self.points_r = cp.arange(-image_shape[1]/2, image_shape[1]/2, 1)*(1.5) + self.R0
        self.points_r = cp.tile(self.points_r[cp.newaxis,:], (image_shape[0], 1)).flatten()
        self.points_a = cp.arange(-image_shape[0]/2, image_shape[0]/2, 1)*(1.5)
        self.points_a = cp.tile(self.points_a[:,cp.newaxis], (1, image_shape[1])).flatten()
        self.points_n = len(self.points_r)
        print("Distributed shape:",image_shape)

        self.tau_c = 2*self.R0/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        def process_point(i):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = cp.arctan((self.points_a[i] - self.Vr*mat_eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负

            ftau_l = self.alpha*(doa-self.theta_c-self.theta_az/2)
            ftau_u = self.alpha*(doa-self.theta_c+self.theta_az/2)
            
            doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
            ftau_mid = self.alpha*(doa1-self.theta_c)
            ## 接收机频率与时间的关系
            ftau_send = (self.Kr*(mat_tau - 2*R_eta/self.c)) 

            ## 脉宽限制
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            ## 从距离向描述fscan限制
            Wr_fscan = (ftau_send>ftau_l) * (ftau_send<ftau_u)
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            signal_r = phase_r*Wr*Wr_fscan

            Tstrip_tar = self.theta_sc*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + self.eta_c)) < Tstrip_tar/2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a

            H0 = cp.exp(-2j*cp.pi*ftau_mid*(mat_tau-2*R_eta/self.c))
            return pre_target[i//image_shape[1], i%image_shape[1]]*signal_r*signal_a

        # Parallel(n_jobs=mp.cpu_count())(delayed(process_point)(i) for i in tqdm(range(self.points_n), desc="Processing points"))
        for i in tqdm(range(self.points_n), desc="Processing points"):
            S_echo += process_point(i)

        return S_echo


    def echogen(self):
        ##接收机时间窗
        self.tau_c = 2*self.R0/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = cp.arctan((self.points_a[i] - self.Vr*mat_eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负
            ftau_u = self.alpha*(doa-self.theta_c-self.theta_az/2)
            ftau_l = self.alpha*(doa-self.theta_c+self.theta_az/2)
            ftau_m = self.alpha*(doa - self.theta_c)

            doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
            ftau_mid = self.alpha*(doa1-self.theta_c)

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

            phase_a = cp.exp(-4j*cp.pi*R_eta*(self.f0)/self.c)
            ## 随着慢时间推移，目标的波束中心频率发生变化，下采样后残余频率偏移
            H0 = cp.exp(-2j*cp.pi*ftau_mid*(mat_tau-2*R_eta/self.c))
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
            return echo_ftau_feta
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
        feta_l = 2*self.Vr*np.sin(self.theta_c+mat_ftau/self.alpha - theta_width/2)/self.lambda_ - self.feta_c
        feta_u = 2*self.Vr*np.sin(self.theta_c+mat_ftau/self.alpha + theta_width/2)/self.lambda_ - self.feta_c

        # W_fscan = cp.abs(mat_feta-self.feta_c-feta_R0) < self.PRF/2
        W_fscan = (mat_feta - self.feta_c > feta_l)*(mat_feta -self.feta_c < feta_u)


        shift = int(cp.round(self.PRF/(2*self.Fa/Na_up)))
        if int(cp.round(self.theta_sc/self.theta_az)) % 2 == 1:
            shift = 0
        W_fscan = cp.roll(W_fscan, shift, axis=0)
        echo_ftau_feta_up *= W_fscan
        echo_ftau_feta_up = cp.roll(echo_ftau_feta_up, -shift, axis=0)
        return echo_ftau_feta_up
    
    def tau_down_sample(self, image_fft, Fr_down, Fr):
        [Na,Nr] = image_fft.shape

        width = int(cp.ceil(Fr_down/(Fr/Nr)))
        return image_fft[:, Nr//2-width/2:Nr//2+width/2]
    
    def tau_up_sample(self, image_fft, Fr_up, Fr):
        [Na,Nr] = image_fft.shape

        Nr_up = int(cp.ceil(Fr_up/Fr*Nr))
        echo_fft2_new = cp.zeros((Na, Nr_up), dtype=cp.complex128)
        echo_fft2_new[:, Nr_up//2-Nr/2:Nr_up//2+Nr/2] = image_fft
        return echo_fft2_new 

    def range_rc(self, echo):
        [Na,Nr] = cp.shape(echo)

        f_tau = ((cp.arange(-Nr/2, Nr/2) * self.Fr / Nr))
        f_eta =  self.feta_c + ((cp.arange(-Na/2, Na/2) * self.Fa / Na))
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2) / self.Fr 
        eta = self.eta_c + cp.arange(-Na/2, Na/2) / self.Fa
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2

        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)


        echo_ftau_feta = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))

 

        H2 = cp.exp(1j*cp.pi*(mat_ftau)**2/self.Kr)
        echo_ftau_feta = echo_ftau_feta * H2
        echo = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(echo_ftau_feta)))
        return echo
    def doppler_shift(self, sig, fa_shift):
        """
        Shift the Doppler frequency of the signal.
        
        Parameters:
        sig (cupy array): Input signal array.
        fa_shift (float): Doppler frequency shift value.
        
        Returns:
        cupy array: Doppler frequency shifted signal.
        """
        Na, Nr = sig.shape
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig))))
        shift = fa_shift * Na / self.PRF
        sig_shift_fft2 = cp.roll(sig_fft2, int(shift), axis=0)
        sig_shift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_shift_fft2)))
        return sig_shift.get()

    def  wk_focus(self, echo, R_ref):
        ## RFM
        [Na,Nr] = cp.shape(echo)


        f_tau = ((cp.arange(-Nr/2, Nr/2) * self.Fr / Nr))
        f_eta = self.feta_c + ((cp.arange(-Na/2, Na/2) * self.Fa / Na))
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2) / self.Fr 
        eta = self.eta_c + cp.arange(-Na/2, Na/2) / self.Fa
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2


        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

    

        mat_ftau_center = (cp.arcsin(mat_feta*self.lambda_/(2*self.Vr))-self.theta_c)*self.alpha

        doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
        ftau_mid = self.alpha*(doa1-self.theta_c)
        Rc = cp.sqrt(self.R0**2 + (self.Vr*mat_eta - 0)**2)
        H0 = cp.exp(-2j*cp.pi*ftau_mid*(mat_tau-2*Rc/self.c))
        # echo = echo*H0
        # # 


        echo_ftau_feta = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))

 

        H2 = cp.exp(1j*cp.pi*(mat_ftau)**2/self.Kr)
        echo_ftau_feta = echo_ftau_feta * H2


        def func(x, y):
            return cp.sqrt((self.f0+x)**2 - self.c**2 * y**2 / (4*self.Vr**2))*cp.cos(self.theta_c) + y*cp.sin(self.theta_c)/(2*self.Vr)




        ftau_new = func(mat_ftau, mat_feta)
        H3 = cp.exp((4j*cp.pi*(R_ref)/self.c)*(ftau_new))
        echo_ftau_feta = echo_ftau_feta * H3
  

        ftau_new1 = cp.sqrt((self.f0+(mat_ftau))**2 - self.c**2 * mat_feta**2 / (4*self.Vr**2))*cp.cos(self.theta_c) + mat_feta*cp.sin(self.theta_c)/(2*self.Vr)

        ## modified stolt mapping
        map_f_tau = ftau_new1-self.f0
        # map_f_tau = cp.sqrt((self.f0+mat_ftau)**2-self.c**2*mat_feta**2/(4*self.vr**2))-self.f0
        delta = (map_f_tau - (mat_ftau))/(self.Fr/Nr) #频率转index
        delta = delta-cp.mean(cp.mean(delta))
        beta = 2
        ## sinc interpolation kernel length, used by stolt mapping
        sinc_N = 8
        focus = SAR_Focus(self.Fr, self.Tp, self.f0, self.Fa, self.Vr, self.Br, self.feta_c, self.R0, self.Kr, self.theta_az)
        echo_ftau_feta_stolt = focus.stolt_interpolation(echo_ftau_feta, delta, Na, Nr, sinc_N)


        # delta = mat_ftau_center/(self.Fr/Nr)
        # sinc_N = 8
        # focus = SAR_Focus(self.Fr, self.Tp, self.f0, self.Fa, self.Vr, self.Br, self.feta_c, self.R0, self.Kr, self.theta_az)
        # echo_ftau_feta_stolt = focus.stolt_interpolation(echo_ftau_feta_stolt, delta, Na, Nr, sinc_N)

        echo_tau_feta_stolt = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_ftau_feta_stolt, axes=1), axis = 1), axes=1)


        
        # echo_tau_feta_stolt = echo_tau_feta_stolt*H4


        echo_stolt = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_tau_feta_stolt, axes = 0), axis = 0), axes=0)
        return echo_stolt

    def  bp_focus(self, echo):
        [Na,Nr] = cp.shape(echo)



        f_tau = ((cp.arange(-Nr/2, Nr/2) * self.Fr / Nr))
        f_eta =  self.feta_c + ((cp.arange(-Na/2, Na/2) * self.Fa / Na))
        tau = self.tau_c + cp.arange(-Nr/2, Nr/2) / self.Fr 
        eta = self.eta_c + cp.arange(-Na/2, Na/2) / self.Fa
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2

        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

    
        doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
        ftau_mid = self.alpha*(doa1-self.theta_c)
        H0 = cp.exp(-2j*cp.pi*ftau_mid*(2*mat_R/self.c))
        # # 

        
        echo_cr = echo
        # echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo_cr, axes=0), axis=0), axes=0)
        # mat_ftau_center = (cp.arcsin(mat_feta*self.lambda_/(2*self.Vr))-self.theta_c)*self.alpha
        # echo_tau_feta = echo_tau_feta * cp.exp(-2j*cp.pi*(mat_ftau_center)*(mat_tau-self.tau_c))
        # echo_cr = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_tau_feta, axes=0), axis=0), axes=0)
        # beta = 2*self.Vr/(self.lambda_*self.alpha)

        output = cp.zeros((Na, Nr), dtype=cp.complex128)
        Tpulze_width = (self.theta_sc+self.theta_az)*mat_R/(self.Vr*cp.cos(self.theta_c)**2)
        sinc_intp = SincInterpolation()
  
        for i in tqdm(range(Na)):
            ## 当前雷达的位置，加上斜视的偏移
            eta_now = (i-Na/2)/self.Fa+self.eta_c
            R_eta = cp.sqrt(mat_R**2 + (self.Vr*(eta_now-mat_eta))**2)
            delta_t = 2*(R_eta-mat_R)/self.c
            delta_n = delta_t/(1/self.Fr)

            # Wa_width =  cp.abs(mat_eta-eta_now+self.eta_c+(mat_tau-self.tau_c)/beta) < Tpulze_width/2
            echo_pulse = cp.tile(echo_cr[i,:], (Na, 1)) 
            doa1 = cp.arccos(mat_R/R_eta)
            ftau_mid = self.alpha*(doa1-self.theta_c)
            # echo_pulse = echo_pulse*cp.exp(-2j*cp.pi*ftau_mid*(mat_tau-self.tau_c))
            intp = sinc_intp.sinc_interpolation(echo_pulse, delta_n, Na, Nr, 8)

            H1 = cp.exp(4j*cp.pi*(R_eta)*ftau_mid/self.c)
            intp = intp*cp.exp(4j*cp.pi*(R_eta-mat_R)*(self.f0)/self.c)   
            output = output + intp
        # feta_shift = self.Fa/2
        # output = output*cp.exp(-1j*2*cp.pi*feta_shift*mat_eta)
        # echo_ftau_eta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(output, axes=1), axis=1), axes=1)
        # echo_ftau_eta = focus.stolt_interpolation(echo_ftau_eta, -delta, Na, Nr, sinc_N)
        # output = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_ftau_eta, axes=1), axis=1), axes=1)
        echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(output, axes = 0), axis=0), axes=0)
        return output, echo_tau_feta

    
    def plot_doppler(self):
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.Fa)
        R0_tar = self.R0
        R_eta = cp.sqrt(R0_tar**2 + (self.Vr*eta - 0)**2)
        doa = np.arctan((0 - self.Vr*eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负
        feta1 = 2*self.Vr*cp.sin(doa)*(self.f0+self.alpha*(doa-self.theta_c))/self.c
        feta2 = 2*self.Vr*cp.sin(doa)*(self.f0)/self.c

        Tstrip_tar1 = self.theta_sc*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
        Wa1 =  cp.abs(eta-(0 + self.eta_c)) < Tstrip_tar1/2
        Tstrip_tar2 = self.theta_az*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
        Wa2 =  cp.abs(eta-(0 + self.eta_c)) < Tstrip_tar2

        feta1 = feta1*Wa1
        feta2 = feta2*Wa2
        plt.figure()
        plt.plot(eta.get(), feta1.get(), label='F-SCAN')
        plt.plot(eta.get(), feta2.get(), label='Stripmap')
        plt.xlabel('Azimuth Time (s)')
        plt.ylabel('Doppler Frequency (Hz)')
        plt.title('Doppler Frequency vs Azimuth Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig("../../../fig/fscan_azimuth/doppler_frequency.png")
    
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

def phase_estimate_plot(echo_tau_feta, Fa):
    abs_val = np.abs(echo_tau_feta)
    max_index = cp.unravel_index(np.argmax(abs_val, axis=None), abs_val.shape)
    max_slice = echo_tau_feta[:, max_index[1]]
    # max_slice = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(max_slice)))

    echo_ftau_feta = np.fft.fftshift(np.fft.fft(np.fft.fftshift(echo_tau_feta, axes=1), axis=1), axes=1)
    fft_slice = echo_ftau_feta[:, echo_ftau_feta.shape[1]//2]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.diff(np.unwrap(np.angle(max_slice)))/(2*np.pi))
    plt.xlabel('Azimuth ')
    plt.ylabel('Azimuth Time')
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(max_slice))
    plt.xlabel('Azimuth time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/phase_estimate.png")

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

def ftau_carrier_plot(ftau_center, extent=None):
    plt.figure()
    plt.imshow((ftau_center), aspect='auto', cmap='jet', extent=extent)
    plt.colorbar()
    plt.xlabel('Range Frequency')
    plt.ylabel('Azimuth Frequency')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/ftau_center.png")


def process_echo(echo, fscan:FScanAzimuth):
    shift = fscan.feta_c / (fscan.Fa/fscan.Na)
    echo_ftau_feta = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))
    echo_ftau_feta = cp.roll(echo_ftau_feta, -int(cp.round(shift)), axis=0)

    ## 拼接
    # echo_ftau_feta = fscan.azimuth_mosaic(echo_ftau_feta)
    # echo = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(echo_ftau_feta)))
    ## 成像
    # image = fscan.range_rc(echo)
    focus = SAR_Focus(fscan.Fr, fscan.Tp, fscan.f0, fscan.Fa, fscan.Vr, fscan.Br, fscan.feta_c, fscan.R0, fscan.Kr, fscan.theta_sc)
    image = focus.wk_focus(echo, fscan.R0)
    estimate = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image, axes=0), axis=0), axes=0)
    return image, estimate

def fscan_azimuth_sim(qfunc, qargs):
    # 定义参数
    cp.cuda.Device(0).use()  
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

    # echo = fscan.dist_gen(pre_target)
    echo = fscan.echogen()
    import scipy.io as sio
    # sio.savemat("./echo.mat", {"echo": echo.get()})
    # mat = sio.loadmat("./echo.mat")
    # echo = cp.array(mat["echo"])
    qfunc.put(echo_plot)
    qargs.put((echo.get(), (-fscan.Tr*1e6/2, fscan.Tr*1e6/2, -fscan.Ta/2 + fscan.eta_c.get(), fscan.Ta/2+fscan.eta_c.get())))
    # # 对echo沿着y轴加kaiser窗
    # kaiser_win = cp.kaiser(echo.shape[0], beta=10)  # beta=8.6常用于旁瓣抑制
    # echo = echo * kaiser_win[:, cp.newaxis]
    echo = fscan.doppler_shift(echo, -fscan.feta_c.get())
    echo_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))
    
    qfunc.put(echo_fft2_plot)
    qargs.put((echo_fft2.get(), (-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))

    ## 处理多普勒中心，将频域中心移至多普勒中心
   
    image, estimate = process_echo(echo, fscan)

    qfunc.put(phase_estimate_plot)
    qargs.put(((estimate).get(), fscan.Fa))
    qfunc.put(ftau_carrier_plot)
    qargs.put((cp.abs(estimate).get(),(-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))

    

    image_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(image)))

    # image_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image, axes=0), axis=0), axes=0)
    # image_tau_feta = fscan.shift2center(image_tau_feta, fscan.alpha*fscan.theta_az)
    # image_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image_tau_feta, axes=1), axis=1), axes=1)

    # ftau = cp.arange(-image_fft.shape[1]/2, image_fft.shape[1]/2, 1)*(fscan.Fr/image_fft.shape[1])
    # mat_ftau = cp.tile(ftau[cp.newaxis, :], (image_fft.shape[0], 1))
    # shift = cp.exp(1j*2*cp.pi*fscan.Tr/2*mat_ftau)
    # image_fft = image_fft * shift

    image = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_fft)))
    image_ftau_eta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image, axes=1), axis=1), axes=1)
    [Na, Nr] = image.shape
    

    # extent = ((fscan.tau_c-fscan.Tr/2)*1e6, (fscan.tau_c+fscan.Tr/2)*1e6, (fscan.eta_c-fscan.Ta/2).get(),(fscan.eta_c+fscan.Ta/2).get())
    extent = None
    x_size = 10
    y_size = x_size*Na*fscan.Vr/fscan.Fa/(Nr*fscan.c/(2*fscan.Fr))
    print("image size:", (x_size, y_size))

    qfunc.put(image_plot)
    qargs.put((image.get(), extent, (x_size, y_size)))

    qfunc.put(image_fft_plot)
    qargs.put((image_fft.get(), (-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2+fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get())))

    dot_estimate = DotEstimator(fscan.points_n, fscan.c, fscan.Vr, fscan.Fa, fscan.Fr, "../../../fig/fscan_azimuth/")
    dot_estimate.dot_estimate((image).get(), (int(2/(fscan.Vr/fscan.Fa)), int(2/(fscan.c/(2*fscan.Fr)))), 16)

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
