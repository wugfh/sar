
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import sys
sys.path.append(r"../../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from matplotlib.font_manager import FontManager
from matplotlib import font_manager

from beam_scan import BeamScan

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


class Fscan(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.N*self.d
        self.a = 0.004871409163516
        shift = 2/3.717054305989132
        self.ttd = 2e-9
        lambda_g=self.lambda_/np.sqrt(1-(self.lambda_/(2*self.a))**2)
        self.d = lambda_g/2 +shift* lambda_g
        self.B = 6e9                             #信号带宽
        self.Fs = self.B*1.2                            #采样率 
        self.Vr = 260/3.6
        self.PRF = 2500
        self.theta_c = 0
        self.theta_width = np.deg2rad(5)
        self.feta_c = 2*self.Vr*np.sin(self.theta_c)/self.lambda_
        self.Ba = 2*self.Vr*(np.sin(self.theta_width/2)-np.sin(-self.theta_width/2))/self.lambda_
        print("Ba:", self.Ba)
        print("res:R={},A={}".format(self.c/(2*self.B), self.Vr/(self.Ba)))
        self.Tr = self.Tp*3
        self.La = self.lambda_/self.theta_width
        self.Kr = -np.sign(self.ttd)*self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.d)
        self.H = 3e3
        self.R0 = self.H/np.cos(self.beta)
        self.Rc = self.R0/np.cos(self.theta_c)
        self.re_guard = 0       ##接收窗保护 
        self.theta_az = np.deg2rad(2.5)
        self.set_scanwidth(np.deg2rad(17.9-10.9))
        self.Nr = int(np.ceil(self.Fs*self.Tr))
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.feta_c, self.R0, self.Kr, self.theta_width)

        self.Ta = 1.2*self.theta_width*self.R0/self.Vr+2
        self.Na = int(np.ceil(self.PRF*self.Ta))
        if self.Na%2==1:
            self.Na += 1
            self.Ta = self.Na/self.PRF
        print(self.Na, self.Nr)
        self.points_n = 15
        self.points_r = self.R0+np.array([-10,-10,-10,-5,-5,-5,0,0,0,4,4,4,8,8,8])
        # self.points_r = np.array([self.R0])
        self.points_a = np.array([-150,0,150,-150,0,150,-150,0,150,-150,0,150,-150,0,150])
        # self.points_a = np.array([0])


    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
    
    def set_scanwidth(self, scan_width):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2
        self.scan_right = self.beta + scan_width/2
    
    def echogen(self, R_error, snr_db, forward):
        ##接收机时间窗
        tau = 2*self.R0/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = (forward)/self.Vr + eta_c
        # eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        [self.Na, self.Nr] = mat_tau.shape
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)

        print("R_error:",cp.abs(R_error).max())
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)+R_error
            # doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            doa = cp.arccos(self.H/self.R0)
            signal_t = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex64)

            ## 接收机的频率与时间的对应关系
            f_send = (self.f0+self.Kr*(mat_tau - 2*R_eta/self.c)) 

            ## 根据波导缝隙天线阵进行方向图建模
            lambda_now = self.c/f_send
            lambda_g = lambda_now/ cp.sqrt(1-(lambda_now/(2*self.a))**2)

            u1 = 2*cp.pi/lambda_now *self.d* cp.sin(doa) - 2*cp.pi/lambda_g*(self.d-lambda_g/2)
            pr = cp.sin(10*u1/2)/(cp.sin(u1/2)*10)
            # pr = 1
            # pr = (f_send>self.f0-self.B/2)*(f_send<self.f0+self.B/2)*pr
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            # Wr = 1
            # signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            ## 发送机到点目标
            signal_t = Wr*phase_r
            ## 点目标到接收机
            signal_r = signal_t

            Tstrip_tar = self.theta_width*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
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
        min_forward = cp.min(cp.array(forward))
        da = min_forward + cp.arange(Na)*(self.Vr/self.PRF)
        for i in range(Nr):
            sig[:,i] = cp.interp(da, cp.array(forward), sig[:,i])
        return sig
