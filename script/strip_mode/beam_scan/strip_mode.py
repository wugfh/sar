import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
sys.path.append(r"../../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
import logging
import colorlog

from beam_scan import BeamScan

class StripMode(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.d
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.B = self.c / (2*self.dr)  # 信号带宽
        self.Fs = self.B*1.2                            #采样率   
        self.Kr = -self.B/self.Tp 
        # self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)


    def echogen(self):
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tp/2
            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            Phase = cp.exp(-4j*cp.pi*R_eta/self.lambda_)*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            S_echo += Wr*Wa*Phase
        return S_echo.get() 
    
    def nesz(self, doa, Pav):
        Pu = Pav/(self.Tp*self.PRF)

        cons = 128*np.pi**3 * self.K*self.T * self.Ln / (Pu*self.lambda_**2*self.c)
        R0 = self.calculate_R0(doa)

        ### 天线增益
    
        # A_e = self.d*(self.d) * 0.6 ## 天线有效面积
        # unit_gain = 4*np.pi*A_e/(doa_lambda**2)
        # ant_gain = unit_gain*self.N 
        ant_gain = 10**(4)*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2/self.N

        # irw = np.deg2rad(2.5)
        # doa35 = np.linspace(-irw/2, (irw/2), len(doa))+self.beta
        # prm_tmp = self.lambda_
        # prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-self.beta))/prm_tmp)
        # G_r = np.sinc(0.018*np.sin(doa35-self.beta)/prm_tmp)
        # prm = G_r*prm
        # prm = np.abs(prm)/np.max(np.abs(prm))
        # el_loss = irw/np.trapezoid(prm**4, doa35)
        # self.log.info("el pel loss: {}".format(el_loss))

        # irw_az = np.deg2rad(10.9)
        irw_az = 0.886*self.lambda_/(self.La)
        # doa35 = np.linspace(-irw_az/2, (irw_az/2), len(doa))+self.beta
        # G_az = np.sinc(0.04*np.sin(doa35-self.beta)/self.lambda_)
        # G_az = np.abs(G_az)/np.max(np.abs(G_az))
        # az_loss = irw_az/np.trapezoid(G_az**4, doa35)
        # self.log.info("az pel loss: {}".format(az_loss))


        ## 单一单元增益
        # Ar = 0.6*self.Lr*self.La
        # Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)
        # At = 0.6*self.La*self.Lr
        # Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)

        R_eta = R0/np.cos(self.theta_c)
        incident = self.calculate_incident(R_eta)

        var = R0**3 * self.B * np.sin(incident)/(self.Naz*self.Tp*ant_gain**2 *irw_az)
        nesz = cons*var ## el loss and az loss
        return 10*np.log10(nesz)
    
    def rasr(self, doa):
        R0 = self.calculate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        max_R = np.sqrt((self.H+self.Re)**2 - self.Re**2)

        for m in range(-5, 4):
            Rm = R0 + m*self.c*(1/self.PRF)/2
            Rm = Rm*(Rm>self.H)*(Rm<max_R)
            if np.any(Rm > 0):
                start_index = np.argmax(Rm > 0)  # 获取Rm不为0的起始点
                end_index = len(Rm) - np.argmax(Rm[::-1] > 0) - 1  # 获取Rm不为0的终止点
            else:
                continue

            window_Rm = slice(start_index, end_index + 1)  # 创建切片对象

            Rm = Rm[window_Rm]
            doam = self.calculate_doa(Rm)


            ## 单一单元增益
            G_doamr = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2
   
            R_eta = Rm/np.cos(self.theta_c)
            incident = self.calculate_incident(R_eta)
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        return 10*np.log10(rasr)
    
    def resolution(self, doa):
        return np.ones(len(doa))*self.c/(2*self.B)

    def calculate_re_window(self):
        R_min = self.calculate_R0(self.scan_left)
        R_max = self.calculate_R0(self.scan_right)
        # print("R_min: {}, R_max: {}".format(R_min-self.R0, R_max-self.R0))
        return (R_max-R_min)*2/self.c
    
    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
        self.Tr = self.calculate_re_window()
    
    def set_scanwidth(self, scan_width):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2
        self.scan_right = self.beta + scan_width/2
        self.Tr = self.calculate_re_window()