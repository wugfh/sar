import numpy as np
import cupy as cp             
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sci
import sys

sys.path.append(r"../")
from sinc_interpolation import SincInterpolation

class BpFocus:

    def __init__(self, ):
        self.c = 299792458                      #光速
        self.Fs = 32317000                      #采样率              
        self.start = 6.5959e-03                 #开窗时间 
        self.Tr = 4.175000000000000e-05         #脉冲宽度                        
        self.f0 = 5.300000000000000e+09         #载频                     
        self.PRF = 1.256980000000000e+03        #PRF                     
        self.Vr = 7062                          #雷达速度     
        self.B = 30.111e+06                     #信号带宽
        self.fc = -6900                         #多普勒中心频率
        self.R0 = self.start*self.c/2
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.La = 15
        self.Ta = 2
        self.Kr = -self.B/self.Tr

    def init_raw_data(self, data1):
        [Na, Nr] = cp.shape(data1)
        data = cp.zeros([int(cp.ceil(Na*1.5)), int(cp.ceil(Nr*1.5))], dtype=cp.complex128)
        [self.Na, self.Nr] = cp.shape(data)
        data[self.Na/2-Na/2:self.Na/2+Na/2, self.Nr/2-Nr/2:self.Nr/2+Nr/2] = data1
        return data

    def echo_generate(self):
        self.Nr = int(cp.ceil(self.Fs*self.Tr))
        self.Na = int(cp.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+cp.linspace(-2000, 2000, self.points_n)
        self.points_a = cp.linspace(-2000, 2000, self.points_n)
        self.Ba = 2*0.886*self.Vr*cp.cos(self.theta_c)/self.La + 2*self.Vr*self.B*cp.sin(self.theta_c)/self.c
        Rc = self.R0/cp.cos(self.theta_c)
        tau = 2*Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -Rc*cp.sin(self.theta_c)/self.Vr
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

    def Bp_preprocess(self, S_echo):
        f_tau = cp.fft.fftshift(cp.arange(-self.Nr/2, self.Nr/2, 1)*(self.Fs/self.Nr))
        f_eta = cp.fft.fftshift(self.fc + cp.arange(-self.Na/2, self.Na/2, 1)*(self.PRF/self.Na))
        mat_f_tau, mat_f_eta = cp.meshgrid(f_tau, f_eta)
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        S_ftau_eta = cp.fft.fft(S_echo, axis=1)
        S_ftau_eta = S_ftau_eta*Hr
        echo = cp.fft.ifft(S_ftau_eta, axis=1)
        return echo
    

    def Bp_foucs(self, echo):
        Rc = self.R0/cp.cos(self.theta_c)
        tau = 2*self.R0/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -Rc*cp.sin(self.theta_c)/self.Vr
        eta = cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2
        output = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        Tpulze_width = 0.886*self.lambda_*mat_R/(self.La*self.Vr*cp.cos(self.theta_c)**2)

        for i in tqdm(range(self.Na)):
            ## 当前雷达的位置，加上斜视的偏移
            eta_now = (i-self.Na/2)/self.PRF+eta_c
            R_eta = cp.sqrt(mat_R**2 + (self.Vr*(mat_eta-eta_now))**2)
            delta_t = 2*(R_eta-mat_R)/self.c
            delta = delta_t/(1/self.Fs)

            ## 加上eta_c是为了对齐波束中心
            Wa_width =  cp.abs(mat_eta-eta_now+eta_c) < Tpulze_width/2
            echo_pulse = cp.ones((self.Na, 1)) * cp.squeeze(echo[i,:])
            sinc_intp = SincInterpolation()
            intp = sinc_intp.sinc_interpolation(echo_pulse, delta, self.Na, self.Nr, 8)
            intp = intp*cp.exp(4j*cp.pi*R_eta/self.lambda_)
            output = output + intp*Wa_width
        
        return output
    
if __name__ == '__main__':
    bp = BpFocus()
    data = sci.loadmat("../../../data/English_Bay_ships/data_1.mat")
    data = data['data_1']
    data = cp.array(data, dtype=cp.complex128)
    # echo = bp.init_raw_data(data)
    echo = bp.echo_generate()
    plt.figure(1)
    plt.imshow(cp.abs(echo).get(), aspect="auto")
    plt.savefig("../../../fig/bp/echo.png", dpi=300)

    echo_pre = bp.Bp_preprocess(echo)
    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(cp.abs(echo_pre).get(), aspect="auto")
    plt.subplot(1,2,2)
    plt.imshow(cp.abs(data).get(), aspect="auto")
    plt.savefig("../../../fig/bp/preprocess.png", dpi=300)

    output = bp.Bp_foucs(echo_pre)
    output = cp.flip(output, axis=0)
    output = cp.abs(output)/cp.max(cp.max(cp.abs(output)))
    output = 20*cp.log10(output+1)
    output = output**0.4
    output = cp.abs(output)/cp.max(cp.max(cp.abs(output)))
    plt.figure(3)
    plt.imshow(cp.abs(output).get(), aspect="auto")
    plt.savefig("../../../fig/bp/bp_dot_result.png", dpi=300)

            


