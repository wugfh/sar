
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
        self.B = 4e8                             #信号带宽
        self.Fs = self.B*1.2                            #采样率 
        self.Vr = 260/3.6
        self.PRF = 1200
        self.theta_c = 0
        self.theta_width = np.deg2rad(1)
        self.feta_c = 2*self.Vr*np.sin(self.theta_c)/self.lambda_
        self.Ba = 2*self.Vr*(np.sin(self.theta_width/2)-np.sin(-self.theta_width/2))/self.lambda_
        print("Ba:", self.Ba)
        self.Tr = self.Tp*5
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

        self.Ta = 1.2*self.theta_width*self.R0/self.Vr+1
        self.Na = int(np.ceil(self.PRF*self.Ta))
        print(self.Na, self.Nr)
        self.points_n = 3
        self.points_r = self.R0+np.array([-600,0,600])
        self.points_a = np.zeros(self.points_n)


    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
    
    def set_scanwidth(self, scan_width):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2
        self.scan_right = self.beta + scan_width/2
    
    def echogen(self, R_error, snr_db = 20):
        ##接收机时间窗
        tau = 2*self.R0/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
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
    

    def fscan_rd_ac_focus(self, rcmc):  
        rcmc = cp.array(rcmc)
        [Na, Nr] = cp.shape(rcmc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.feta_c + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        data_fft_a_rcmc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(rcmc, axes=0), axis=0), axes=0)
        mat_R0 = mat_tau*self.c/2
        ## 方位压缩
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*mat_R0)
        # Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        # Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        offset = cp.exp(2j*cp.pi*mat_tau*1/3)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha*offset
        data_ca_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), Na, axis=0), axes=0)

        data_final = data_ca_rcmc
        # data_final = cp.abs(data_final)/cp.max(cp.max(cp.abs(data_final)))
        # data_final = 20*cp.log10(data_final)
        return data_final.get()
