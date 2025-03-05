import scipy.io as sci
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
from scipy import signal
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation

class SAR_Focus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0, Kr):                         
        self.Re = 6371.39e3                     #地球半径
        self.c = 299792458                      #光速
        self.Fs = Fs                                     
        self.Tp = Tp                            #脉冲宽度                        
        self.f0 = f0                            #载频                     
        self.PRF = PRF                          #PRF                     
        self.Vr = Vr                            #雷达速度     
        self.B = B                              #信号带宽
        self.fc = fc                            #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.R0 = R0
        self.Rc = self.R0/cp.cos(self.theta_c)
        self.Kr = Kr
    def rd_focus(self, echo):  
        [Na, Nr] = cp.shape(echo)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Nr/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fft(echo, Nr, axis = 1) 
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hm = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        data_fft_cr = data_fft_r*Hr*Hm
        data_cr = cp.fft.ifft(data_fft_cr, Nr, axis = 1)

        ## RCMC
        data_fft_a = cp.fft.fft(data_cr, Na, axis=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta = mat_R0/mat_D - mat_R0
        delta = delta*2/(self.c/self.Fs)
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag

        ## 方位压缩
        Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifft(data_fft_a_rcmc, Na, axis=0)

        data_final = data_ca_rcmc
        # data_final = cp.abs(data_final)/cp.max(cp.max(cp.abs(data_final)))
        # data_final = 20*cp.log10(data_final)
        return data_final
    
    def stolt_interpolation(self, echo_ftau_feta, delta, Na, Nr, sinc_N):
        echo_ftau_feta = cp.ascontiguousarray(echo_ftau_feta)
        # 初始化数据
        echo_ftau_feta_real = cp.real(echo_ftau_feta).astype(cp.double)
        echo_ftau_feta_imag = cp.imag(echo_ftau_feta).astype(cp.double)

        # 调用核函数
        sinc_intp = SincInterpolation()
        echo_ftau_feta_stolt_real = sinc_intp.sinc_interpolation(
            echo_ftau_feta_real, delta, Na, Nr, sinc_N
        )

        echo_ftau_feta_stolt_imag = sinc_intp.sinc_interpolation(
            echo_ftau_feta_imag, delta, Na, Nr, sinc_N
        )
        echo_ftau_feta_stolt = echo_ftau_feta_stolt_real + 1j * echo_ftau_feta_stolt_imag
        return echo_ftau_feta_stolt

    def wk_focus(self, echo, R_ref):
        ## RFM
        echo_ftau_feta = cp.fft.fft2(echo)

        [Na,Nr] = cp.shape(echo_ftau_feta)

        f_tau = cp.fft.fftshift(((cp.arange(-Nr/2, Nr/2) * self.Fs / Nr)))
        f_eta =  self.fc+((cp.arange(-Na/2, Na/2) * self.PRF / Na))

        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

        H3 = cp.exp((4j*cp.pi*R_ref/self.c)*cp.sqrt((self.f0+mat_ftau)**2 - self.c**2 * mat_feta**2 / (4*self.Vr**2)) + 1j*cp.pi*mat_ftau**2/self.Kr)
        
        echo_ftau_feta = echo_ftau_feta * H3

        map_f_tau = cp.sqrt((self.f0+mat_ftau)**2-self.c**2*mat_feta**2/(4*self.Vr**2))-self.f0
        delta = (map_f_tau - mat_ftau)/(self.Fs/Nr) #频率转index

        ## sinc interpolation kernel length, used by stolt mapping
        sinc_N = 8
        echo_ftau_feta_stolt = self.stolt_interpolation(echo_ftau_feta, delta, Na, Nr, sinc_N)

        echo_stolt = (cp.fft.ifft2((echo_ftau_feta_stolt)))
        return echo_stolt
