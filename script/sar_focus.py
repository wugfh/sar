import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation
from tqdm import tqdm

class SAR_Focus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0, Kr, theta_width):                         
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
        self.La = self.lambda_/theta_width

    def rd_rcmc(self,data_cr):
        [Na, Nr] = cp.shape(data_cr)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)
        f_eta = self.fc + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## RCMC
        data_fft_a = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_cr, axes=0), Na, axis=0), axes=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta =  mat_R0/mat_D - mat_R0 
        delta = delta*2/(self.c/self.Fs)
        print("RCMC delta min,max:", delta.min(), delta.max())
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag
        data_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)
        return data_rcmc
    
    def range_compression(self, echo):
        echo = cp.array(echo)
        [Na, Nr] = cp.shape(echo)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)
        f_eta = self.fc + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hsrc = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        data_fft_cr = data_fft_r*Hr
        data_cr = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(data_fft_cr)))
        return data_cr

    def rd_focus(self, echo):  
        echo = cp.array(echo)
        [Na, Nr] = cp.shape(echo)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)
        f_eta = self.fc + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        data_cr = self.range_compression(echo)

        ## RCMC
        data_rcmc = self.rd_rcmc(data_cr)
        data_fft_a_rcmc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rcmc, axes=0), axis=0), axes=0)
        ## 方位压缩
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*self.R0)
        # Ha = cp.exp(4j*cp.pi*mat_D*self.R0*self.f0/self.c)
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        # offset = cp.exp(2j*cp.pi*self.Fs/3*mat_tau)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), Na, axis=0), axes=0)

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
        [Na,Nr] = cp.shape(echo)

        echo_ftau_feta = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo)))

        f_tau = ((cp.arange(-Nr/2, Nr/2) * self.Fs / Nr))
        f_eta =  self.fc + ((cp.arange(-Na/2, Na/2) * self.PRF / Na))
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2) / self.Fs 

        mat_tau, _ = cp.meshgrid(tau, f_eta)
        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

        ftau_new = cp.sqrt((self.f0+mat_ftau)**2 - self.c**2 * mat_feta**2 / (4*self.Vr**2))*cp.cos(self.theta_c) + mat_feta*cp.sin(self.theta_c)/(2*self.Vr)
        H3 = cp.exp((4j*cp.pi*R_ref/self.c)*ftau_new)

        
        echo_ftau_feta = echo_ftau_feta * H3

        ## modified stolt mapping
        map_f_tau = ftau_new-cp.sqrt((self.f0)**2 - self.c**2 * mat_feta**2 / (4*self.Vr**2))
        # map_f_tau = cp.sqrt((self.f0+mat_ftau)**2-self.c**2*mat_feta**2/(4*self.vr**2))-self.f0
        delta = (map_f_tau - mat_ftau)/(self.Fs/Nr) #频率转index
        delta = delta - cp.mean(cp.mean(delta))

        ## sinc interpolation kernel length, used by stolt mapping
        sinc_N = 8
        echo_ftau_feta_stolt = self.stolt_interpolation(echo_ftau_feta, delta, Na, Nr, sinc_N)
        # echo_ftau_feta_stolt = echo_ftau_feta
        ## focusing
        ## modified stolt mapping, residual azimuth compress
        # mat_R = mat_tau * self.c / 2
        # H4 = cp.exp((4j*cp.pi*(mat_R - R_ref)/self.c)*cp.sqrt((self.f0)**2 - self.c**2 * mat_feta**2 / (4*self.Vr**2)))
        echo_tau_feta_stolt = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_ftau_feta_stolt, axes=1), axis = 1), axes=1)
        # echo_tau_feta_stolt = echo_tau_feta_stolt*H4


        echo_stolt = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(echo_tau_feta_stolt, axes = 0), axis = 0), axes=0)
        return echo_stolt


    def Bp_preprocess(self, S_echo):
        [Na, Nr] = cp.shape(S_echo)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fs/Nr)
        f_eta = self.fc + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)
        mat_f_tau, mat_f_eta = cp.meshgrid(f_tau, f_eta)
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        S_ftau_eta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(S_echo, axes=1), axis=1), axes=1)
        S_ftau_eta = S_ftau_eta*Hr
        echo = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(S_ftau_eta, axes=1), axis=1), axes=1)
        return echo
    

    def Bp_focus(self, echo):
        [Na, Nr] = cp.shape(echo)
        echo_cr = self.Bp_preprocess(echo)
        Rc = self.R0/cp.cos(self.theta_c)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -Rc*cp.sin(self.theta_c)/self.Vr
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2
        output = cp.zeros((Na, Nr), dtype=cp.complex128)
        Tpulze_width = 0.886*self.lambda_*mat_R/(self.La*self.Vr*cp.cos(self.theta_c)**2)

        for i in tqdm(range(Na)):
            ## 当前雷达的位置，加上斜视的偏移
            eta_now = (i-Na/2)/self.PRF+eta_c
            R_eta = cp.sqrt(mat_R**2 + (self.Vr*(mat_eta-eta_now))**2)
            delta_t = 2*(R_eta-mat_R)/self.c
            delta = delta_t/(1/self.Fs)

            ## 加上eta_c是为了对齐波束中心
            Wa_width =  cp.abs(mat_eta-eta_now+eta_c) < Tpulze_width/2
            echo_pulse = cp.ones((Na, 1)) * cp.squeeze(echo_cr[i,:])
            sinc_intp = SincInterpolation()
            intp = sinc_intp.sinc_interpolation(echo_pulse, delta, Na, Nr, 8)
            intp = intp*cp.exp(4j*cp.pi*R_eta/self.lambda_)
            output = output + intp*Wa_width
        
        return output
