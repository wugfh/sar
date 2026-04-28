
import numpy as np
import cupy as cp
import sys

sys.path.append(r"../../")

from sar_focus import SAR_Focus
from sinc_interpolation import SincInterpolation
import scipy.interpolate as intp
from beam_scan import BeamScan
from inverse_conv import recover_dft_phase
import tqdm
from matplotlib import pyplot as plt

class Fscan(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.N*self.d
        self.a = 0.004871409163516
        shift = 2/3.717054305989132
        self.ttd = 2e-9
        lambda_g=self.lambda_/np.sqrt(1-(self.lambda_/(2*self.a))**2)
        self.d = lambda_g/2 +shift* lambda_g

        self.beta = np.deg2rad(45)                  #天线安装角
        self.phi = self.beta + np.deg2rad(14.3)                 #条带中心
        self.B = 2e9                             #信号带宽
        self.Fs = self.B*1.2                            #采样率 
        self.Vr = 260/3.6
        self.PRF = 2500
        self.theta_c = np.deg2rad(3)
        self.theta_width = np.deg2rad(8)
        self.feta_c = 2*self.Vr*np.sin(self.theta_c)/self.lambda_
        self.fc = self.feta_c
        self.Ba = 2*self.Vr*(np.sin(self.theta_width/2+self.theta_c)-np.sin(-self.theta_width/2+self.theta_c))/self.lambda_
        self.Tr = self.Tp*9
        self.La = self.lambda_/self.theta_width
        self.Kr = -np.sign(self.ttd)*self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.d)
        self.H = 3e3
        self.R0 = self.H/np.cos(self.phi)
        self.Rc = self.R0/np.cos(self.theta_c)
        self.re_guard = 0       ##接收窗保护 
        self.theta_az = np.deg2rad(2.5)
        self.set_scanwidth(np.deg2rad(17.9-10.9), np.deg2rad(14.3))
        self.Nr = int(np.ceil(self.Fs*self.Tr))
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.feta_c, self.R0, self.Kr, self.theta_width)
        self.Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*self.R0)
        self.Tswath = (self.H/cp.cos(self.scan_right) - self.H/cp.cos(self.scan_left))/self.c*2
        self.Kfscan = self.B/self.Tswath

        self.Ta = 10
        self.Na = int(np.ceil(self.PRF*self.Ta))
        if self.Na%2==1:
            self.Na += 1
            self.Ta = self.Na/self.PRF

        self.points_n = 12
        # self.points_r = self.R0+np.array([-10,-10,-10,-5,-5,-5,0,0,0,4,4,4,8,8,8])
        self.points_r = self.R0 + np.linspace(-25, 28, self.points_n)
        self.points_y = np.sqrt(self.points_r**2-self.H**2)
        # self.points_a = np.array([-150,0,150,-150,0,150,-150,0,150,-150,0,150,-150,0,150])
        self.points_a = np.linspace(-250, 250, self.points_n)
    def set_Vr(self, Vr):
        self.Vr = Vr
        self.feta_c = 2*self.Vr*np.sin(self.theta_c)/self.lambda_
        self.Ba = 2*self.Vr*(np.sin(self.theta_width/2)-np.sin(-self.theta_width/2))/self.lambda_
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.feta_c, self.R0, self.Kr, self.theta_width)
        self.Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*self.R0)
        self.Na = int(np.ceil(self.PRF*self.Ta))
        if self.Na%2==1:
            self.Na += 1
            self.Ta = self.Na/self.PRF
        print("Ba:", self.Ba)
        print("res:R={},A={}".format(self.c/(2*self.B), self.Vr/(self.Ba)))
        print("Na, Nr:",self.Na, self.Nr)

    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
    
    def set_scanwidth(self, scan_width, scan_center=None):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2 + scan_center
        self.scan_right = self.beta + scan_width/2 + scan_center
    
    def echogen(self, snr_db, forward, down, right):
        ##接收机时间窗
        tau = 2*self.R0/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = (forward)/self.Vr
        # eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        tau = tau[cp.newaxis, :]
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)
        for i in range(self.points_n):
            R0_tar = cp.sqrt((down)**2 + (right-self.points_y[i])**2)

            R_eta = cp.sqrt((down)**2 + (right-self.points_y[i])**2 + (self.Vr*eta - self.points_a[i])**2)
            # doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            doa = cp.arccos(cp.abs(down)/R0_tar)-self.beta
            signal_t = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)

            ## 接收机的频率与时间的对应关系
            f_send = (self.f0+self.Kr*(tau - 2*R_eta/self.c)) 
            f_send = cp.clip(f_send, self.f0-self.B/2, self.f0+self.B/2)

            ## 根据波导缝隙天线阵进行方向图建模
            lambda_now = self.c/f_send
            lambda_g = lambda_now/ cp.sqrt(1-(lambda_now/(2*self.a))**2)

            u1 = 2*cp.pi/lambda_now *self.d* cp.sin(doa) - 2*cp.pi/lambda_g*(self.d-lambda_g/2)
            pr = cp.sin(10*u1/2)/(cp.sin(u1/2)*10)
            # pr = 1
            # pr = (f_send>self.f0-self.B/2)*(f_send<self.f0+self.B/2)*pr
            Wr = cp.abs(tau-(2*R_eta/self.c))<self.Tp/2 
            # Wr = 1
            # signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            phase_r = cp.exp(1j*cp.pi*self.Kr*(tau-2*R_eta/self.c)**2)
            ## 发送机到点目标
            signal_t = Wr*phase_r*pr
            ## 点目标到接收机
            signal_r = signal_t*pr

            Tstrip_tar = self.theta_width*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        # 添加底噪
        noise_power = cp.max(cp.abs(S_echo)) * 10**(-snr_db/20)
        noise = (cp.random.randn(*S_echo.shape) + 1j * cp.random.randn(*S_echo.shape)) * noise_power / cp.sqrt(2)
        S_echo += noise
        return S_echo
    

    def compensate_R(self, sig, dR):
        [Na,Nr] = cp.shape(sig)

        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)
        mat_ftau = cp.tile(ftau[cp.newaxis, :], (Na, 1))
        dtau = 2*dR/self.c
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig))))
        sig_compensated = sig_fft2 * cp.exp(1j*4*cp.pi*(mat_ftau+self.f0)*dR/self.c)


        data_final = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_compensated)))

        return data_final
    
    def compensate_R2(self, sig, error):
        [Na,Nr] = cp.shape(sig)
        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)+self.f0
        feta = cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)+self.feta_c
        error_2d = cp.zeros((Na, Nr), dtype=cp.float32)
        mat_ftau = cp.tile(ftau[cp.newaxis, :], (Na, 1))
        for i in range(Nr):
            error_2d[:,i] = cp.interp(self.f0/ftau[i]*feta, feta, cp.array(error))
        re_value = mat_ftau*error_2d/self.f0

        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig))))
        sig_fft2_compensated = sig_fft2 * cp.exp(-1j*re_value)

        sig = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2_compensated)))

        return sig

    def azimuth_interp(self, sig, forward):
        [Na,Nr] = cp.shape(sig)
        da = (np.arange(Na)-Na//2)*(self.Vr/self.PRF) - self.Rc*np.sin(self.theta_c)
        da = da[:,np.newaxis]
        eta = forward/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*self.feta_c*eta)

        linear_interp = intp.interp1d(cp.squeeze(forward).get(), np.arange(Na), kind='linear', fill_value="extrapolate")
        new_index = cp.array(linear_interp(da))
        delta = new_index - cp.arange(Na)[:, cp.newaxis]
        delta = cp.tile(delta[:, cp.newaxis], (1, Nr))
        sinc_interp = SincInterpolation()
        sig = cp.ascontiguousarray(sig)
        sig_real = sinc_interp.sinc_interpolation(cp.real(sig).T, delta.T, Nr, Na, 8).T
        sig_imag = sinc_interp.sinc_interpolation(cp.imag(sig).T, delta.T, Nr, Na, 8).T
        sig = sig_real + 1j*sig_imag
        eta = cp.array(da)/self.Vr
        sig = sig*cp.exp(2j*cp.pi*self.feta_c*eta)
        return sig
    
    def fscan_dramp(self, sig):
        [Na,Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fs)
        tau = tau[cp.newaxis, :]
        sig = sig*cp.exp(-1j*cp.pi*self.Kfscan*(tau)**2)

        return sig
    
    def fscan_reramp(self, sig):
        [Na,Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fs)
        tau = tau[cp.newaxis, :]
        sig = sig*cp.exp(1j*cp.pi*self.Kfscan*tau**2)

        return sig
    
    def fscan_super_resolution(self, sig):
        [Na,Nr] = cp.shape(sig)
        # window = cp.abs(x)<win_len//2
        f_send = self.f0 + (cp.arange(Nr)-Nr//2)*(self.Fs/Nr)
        Rc = self.H/cp.cos(self.beta+np.deg2rad(14.3))
        tau = 2*Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fs)
        doa = cp.arccos(self.H/(tau*cp.cos(self.theta_c)*self.c/2))-self.beta

        ## 根据波导缝隙天线阵进行方向图建模
        lambda_now = self.c/f_send
        lambda_g = lambda_now/ cp.sqrt(1-(lambda_now/(2*self.a))**2)

        u1 = 2*cp.pi/lambda_now *self.d* cp.sin(doa) - 2*cp.pi/lambda_g*(self.d-lambda_g/2)
        pr = cp.sin(10*u1/2)/(cp.sin(u1/2)*10)
        pr_midx = cp.argmax(pr)
        pr = cp.roll(pr, pr_midx-pr.shape[0]//2)
        win_len = 400
        print("win_len:", win_len)
        x = cp.arange(-Nr/2, Nr/2, 1)
        window =  cp.exp(-0.5 * ((x) / win_len) ** 2)
        # plt.figure()
        # plt.plot(x.get(), pr.get(),label="antenna pattern")
        # plt.plot(x.get(), window.get(), label="window")
        # plt.legend()
        # plt.show()

        win_spectrum = (cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(window))))

        for i in tqdm.tqdm(range(Na), desc="Super-resolution"):
            tmp, info =recover_dft_phase(sig[i, :],win_spectrum, 1e-6,tol=1e-5, max_iter=1000)
            if info != 0:
                print("CG did not converge for column {}".format(i))
            sig[i, :] = cp.array(tmp)
        return sig