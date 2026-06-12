from matplotlib.pylab import beta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as optimize
from multiprocessing import Process, Queue
import pandas as pd

from matplotlib import font_manager
import os
import scipy.interpolate as interpolate


class FscanDesign:
    def __init__(self):
        self.H = 3e3
        self.c = 299792458 # Speed of light in m/s
        self.EarthMass = 5.972e24 # kg
        self.Re = 6371e3
        self.mode = 0  ## 0: 已知天线 1: 未知天线，用理想天线设计
        self.Gravitational = 6.67430e-11
        self.Ve = 466 # m/s, 地球自转线速度
        self.Vs = 70
        self.da = 0.05 ## 方位向地距分辨率
        self.dg = 0.2  ## 距离向地距分辨率
        self.dr = 0.2 ## 距离向斜距分辨率
        self.f0 = 35e9  ## 载波频率
        self.Tp = 20e-6 ## 脉冲宽度
        self.groud_extent = 2e3
        self.azimuth_extent = 3e3

        plt.figure()
        self.read_ant_pattern("../../../data/250925KaAntenna/1-35-e.xlsx", "../../../data/250925KaAntenna/1-35-a.xlsx")
        plt.plot(np.rad2deg(self.r_angle), self.r_pattern, label="35GHz")
        self.read_ant_pattern("../../../data/250925KaAntenna/1-34-e.xlsx", "../../../data/250925KaAntenna/1-34-a.xlsx")
        plt.plot(np.rad2deg(self.r_angle), self.r_pattern, label="34GHz")
        self.r_pattern34 = self.r_pattern
        self.r_angle34 = self.r_angle
        self.read_ant_pattern("../../../data/250925KaAntenna/1-36-e.xlsx", "../../../data/250925KaAntenna/1-36-a.xlsx")
        plt.plot(np.rad2deg(self.r_angle), self.r_pattern, label="36GHz")
        self.r_pattern36 = self.r_pattern
        self.r_angle36 = self.r_angle
        plt.xlabel("angle/°")
        plt.ylabel("gain/dB")
        plt.grid()
        plt.legend()
        plt.savefig("../../../fig/fscan_design/ant_pattern_r.pdf", dpi=300)

        plt.figure()
        self.read_ant_pattern("../../../data/250925KaAntenna/1-35-e.xlsx", "../../../data/250925KaAntenna/1-35-a.xlsx")
        plt.plot(np.rad2deg(self.a_angle), self.a_pattern, label="35GHz")
        self.read_ant_pattern("../../../data/250925KaAntenna/1-34-e.xlsx", "../../../data/250925KaAntenna/1-34-a.xlsx")
        plt.plot(np.rad2deg(self.a_angle), self.a_pattern, label="34GHz")
        self.read_ant_pattern("../../../data/250925KaAntenna/1-36-e.xlsx", "../../../data/250925KaAntenna/1-36-a.xlsx")
        plt.plot(np.rad2deg(self.a_angle), self.a_pattern, label="36GHz")
        plt.xlabel("angle/°")
        plt.ylabel("gain/dB")
        plt.grid()
        plt.legend()
        plt.savefig("../../../fig/fscan_design/ant_pattern_a.pdf", dpi=300)

        self.read_ant_pattern("../../../data/250925KaAntenna/1-35-e.xlsx", "../../../data/250925KaAntenna/1-35-a.xlsx")


        self.fscan_left = np.deg2rad(10.9066262820108)
        self.fscan_right = np.deg2rad(17.9416367435066)
        self.fscan_center = self.fscan_left + (self.fscan_right - self.fscan_left)/2
        self.fscan_width = self.fscan_right - self.fscan_left
        self.ant_gain = 10**(np.max(self.r_pattern)/10)
        self.lambda_ = self.c / self.f0
        self.beta = np.array([np.deg2rad(61.2)])
        # self.Lr = 0.88*self.lambda_/(self.look_angle_right - self.look_angle_left)  ## 距离向天线长度
        if self.mode == 0:
            self.theta_r = self.calculate_ant_theta_w(self.r_pattern, self.r_angle)  ## 距离向天线波束宽度
            # self.theta_r = np.deg2rad(0.344)
            self.Lr = 0.88*self.lambda_/(self.theta_r) * np.ones_like(self.beta)
        else:
            self.Lr = 1.2 * np.ones_like(self.beta)
            self.theta_r = 0.88*self.lambda_/self.Lr
        print("theta_r:", np.mean(np.rad2deg(self.theta_r)))
        # print("Lr up:", np.min(self.Lr))

        self.Br = 2e9*np.ones_like(self.beta)*self.theta_r/self.fscan_width  ## 距离向调频带宽
        self.Fr = 2.5e9*np.ones_like(self.beta)  ## 距离向采样率
        self.look_angle_left = np.arccos(np.array([self.H/((36.5e-6)*self.c/2)]))

        self.look_angle_right = np.arccos(np.array([self.H/((36.5e-6+2.8e4/self.Fr)*self.c/2)]))
        print("look angle left:", np.rad2deg(self.look_angle_left))
        print("look angle right:", np.rad2deg(self.look_angle_right))
   
        # self.Br = 3.2e9
        self.R0 = self.H/np.cos(self.beta)
        self.PRF = np.array([6000]) ## PRF
            
        self.NB = 1
        if self.mode == 0:
            self.theta_a = self.calculate_ant_theta_w(self.a_pattern, self.a_angle)  ## 方位向天线波束宽度
            # self.theta_a = np.deg2rad(0.344)
            self.La = 0.88*self.lambda_/self.theta_a  ## 方位向天线长度
        else:
            self.La = 1.2 ## 方位向天线长度
            self.theta_a = 0.88*self.lambda_/self.La  ## 方位向天线波束宽度
            self.ant_gain = 4*np.pi/self.lambda_**2 * (self.La)*(self.Lr[0])  ## 天线增益
        print("theta_a:", np.rad2deg(self.theta_a))
        print("ant gain:", 10*np.log10(self.ant_gain))
        ## 斜视角中心
        self.theta_c = np.deg2rad(0)

   
        self.eta = np.arccos(self.H/((self.H/np.cos(self.beta))/np.cos(self.theta_c))) ## 入射角
        # self.Vg = self.Re/(self.Re + self.H)**2 * self.Vs*(self.Re+self.R0*np.cos(self.eta))  ## 地面投影速度
        self.Vr = 70

        self.Bfov = self.Bfov_func(self.theta_a, self.theta_c)
        self.Bd = 0.886*self.Vr/self.da  ## 多普勒带宽
        self.Bd = np.maximum(self.Bd, self.Bfov)
        print("Bfov, Bd:", np.max(self.Bfov), np.max(self.Bd))


        self.K = 1.38e-23                           #玻尔兹曼常数
        self.T = 320                                #温度
        self.Ln = 10**(0.5)                              ## 总体系统损耗        

    def read_ant_pattern(self, file_path_e, file_path_a=None):
        data = pd.read_excel(file_path_e, header=1)
        self.r_angle = np.deg2rad(np.array(data['Elevation (deg)'])) ## rad
        self.r_pattern = np.array(data['Amplitude (dB)']) ## dB
        if file_path_a:
            data_a = pd.read_excel(file_path_a, header=1)
            self.a_angle = np.deg2rad(np.array(data_a['Azimuth (deg)'])) ## rad
            self.a_pattern = np.array(data_a['Amplitude (dB)']) ## dB
        else:
            self.a_pattern = None


    def calculate_ant_theta_w(self, pattern, ant_angle):
        angle_l = 0
        angle_r = 0
        max_val = np.max(pattern)
        for i in range(1,len(pattern)-1):
            if pattern[i] <= -6+max_val and pattern[i+1] >= -6+max_val:
                angle_l = ant_angle[i]
            elif pattern[i] >= -6+max_val and pattern[i+1] <= -6+max_val:
                angle_r = ant_angle[i]
        return (np.abs(angle_r - angle_l))

    def calculate_R0(self, look_angle):
        ## 星载
        if self.H > 100e3:
            tmp_angle = np.arcsin((self.H+self.Re)*np.sin(look_angle)/self.Re)
            tmp_angle = tmp_angle - look_angle
            R0 = self.Re*np.sin(tmp_angle)/np.sin(look_angle)
        ## 机载
        else:
            R0 = self.H/np.cos(look_angle)
        return R0
    
    def calculate_incident(self, R_eta):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
            incident = np.pi - incident
        else:
            incident = np.arccos(self.H/R_eta)
        return incident
    
    
    def calculate_doa(self, R0):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R0**2-(self.H+self.Re)**2)/(2*self.Re*R0)*(R0>self.H))
            doa = np.arcsin(self.Re*np.sin(incident)/(self.H+self.Re))
        else:
            doa = np.arccos(self.H/R0)
        return doa
    
    def calulate_extent(self, psi_start, psi_end, theta_a):
        dstart = self.R0 * np.tan(psi_start)
        dend = self.R0 * np.tan(psi_end)
        dstart_1 = self.R0*(np.tan(psi_start-theta_a/2)) - dstart
        dstart_2 = self.R0*(np.tan(psi_start+theta_a/2)) - dstart
        dend_1 = self.R0*(np.tan(psi_end-theta_a/2)) - dend
        dend_2 = self.R0*(np.tan(psi_end+theta_a/2)) - dend
        if dend_2 < dstart_1 or dend_1 > dstart_2:
            print("Error: The extent of the beam does not overlap.")
            return 0
        return min(dstart_2, dend_2) - max(dstart_1, dend_1)
    
    def calculate_ground_extent(self, beta, theta_w):
        look_angle_left = beta - theta_w/2
        look_angle_right = beta + theta_w/2
        R0_left = self.calculate_R0(look_angle_left)
        R0_right = self.calculate_R0(look_angle_right)
        return (R0_right-R0_left)/np.sin(beta)
    
    def calculate_scanwidth(self, beta, ground_width):
        ground_angle = ground_width/self.Re

        def func(x):
            g_left = np.arcsin((self.H+self.Re)/(self.Re/np.sin(beta-x))) - (beta-x)
            g_right = np.arcsin((self.H+self.Re)/(self.Re/np.sin(beta+x))) - (beta+x)
            return g_right-g_left-ground_angle
        
        equation = lambda x: func(x)

        solu = optimize.fsolve(equation, 0)

        look_angle_left = beta-solu[0]
        look_angle_right = beta+solu[0]

        return look_angle_left, look_angle_right
    
    def Bd_func(self, psi_start, psi_end, theta_a):
            return 2*self.Vr*np.abs(np.sin(psi_start-theta_a/2) - np.sin(psi_end+theta_a/2)) / (self.lambda_)
        
    def Bfov_func(self, theta_a, psi_start):
            return 2*self.Vr*np.abs(np.sin(psi_start+theta_a/2) - np.sin(psi_start-theta_a/2))/self.lambda_
        

    def zebra_diagram(self, prf, tau_rp, prf_low, prf_up):

        frac_min = (tau_rp + self.Tp) * prf  # 发射约束
        frac_max = -tau_rp * prf


        int_min = int(np.min(np.floor(2 * self.H * prf / self.c)))
        int_max = int(np.max(np.ceil(2 * np.sqrt((self.Re+self.H)**2 - (self.Re)**2) * prf / self.c)))

        look_angle_range = np.linspace(np.min(self.beta)-np.deg2rad(5), np.max(self.beta)+np.deg2rad(5), 50000)
        prf_start = np.ceil((prf_low-prf[0])/(prf[1]-prf[0])).astype(int)
        adaptive_pos = np.ones((len(prf), len(look_angle_range)), dtype=bool)
        adaptive_pos[:prf_start, :] = False
        

        plt.figure("PRF selection")

        # 遍历尽可能多的干扰
        for i in range(int_max + 1):
            R1 = (i + frac_min) * self.c / (2 * prf)
            gamma_cos = (R1**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R1 * (self.Re+self.H))
            # gamma_cos = self.H/R1
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R1 * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'b')

            Rn = (i + frac_max) * self.c / (2 * prf)
            gamma_cos = (Rn**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * Rn * (self.Re+self.H))
            # gamma_cos = self.H/Rn
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(Rn * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            end = np.floor((np.deg2rad(gamma1)-look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            end = np.maximum(end, 0)
            end = np.minimum(end, len(look_angle_range)-1)
            start = np.ceil((np.deg2rad(gamma2)-look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            start = np.maximum(start, 0)
            start = np.minimum(start, len(look_angle_range)-1)
            for j in range(len(prf)):
                adaptive_pos[j, start[j]:end[j]] = False

            plt.plot(prf, gamma2, 'b')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.5, color='b')

        # 星下点干扰
        for i in range(30):
            R = (2 * self.H / self.c + i / prf - self.Tp/2) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            # gamma_cos = self.H/R
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'r')

            R = (2 * self.H / self.c + i / prf ) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            # gamma_cos = self.H/R
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)
            
 
            start = np.floor((np.deg2rad(gamma1)-look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            start = np.maximum(start, 0)
            start = np.minimum(start, len(look_angle_range)-1)
            end = np.ceil((np.deg2rad(gamma2)-look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            end = np.maximum(end, 0)
            end = np.minimum(end, len(look_angle_range)-1)
            for j in range(len(prf)):
                adaptive_pos[j, start[j]:end[j]] = False


            plt.plot(prf, gamma2, 'r')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.5, color='r')

        protected_look_angle = np.deg2rad(0.3)
        for i in range(len(self.beta)):
            left = (self.look_angle_left[i])
            right = (self.look_angle_right[i])
            if self.beta[i] < np.deg2rad(30):
                protected_look_angle = np.deg2rad(0.8)
            elif self.beta[i] < np.deg2rad(40):
                protected_look_angle = np.deg2rad(0.5)
            else:
                protected_look_angle = np.deg2rad(0.3)
            left_pos = np.floor((left -protected_look_angle- look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            right_pos = np.ceil((right +protected_look_angle- look_angle_range[0])/(look_angle_range[1]-look_angle_range[0])).astype(int)
            
            select = 0
            while np.all(adaptive_pos[select, left_pos:right_pos] == True) == False and select < len(prf)-1:
                select += 1
            prf_select = np.round(prf[select])
            if(prf_select > prf_up):
                print("Warning: Cannot find suitable PRF for look angle {:.2f}°, discard".format(np.rad2deg(self.beta[i])))
                self.PRF[i] = prf_low
                continue
            else:
                self.PRF[i] = prf_select
            plt.vlines(prf_select, np.rad2deg(left), np.rad2deg(right), colors='k')

        plt.grid()
        plt.xlabel("PRF/Hz")
        plt.ylabel("look angle/°")
        plt.ylim([np.rad2deg(self.beta_below)-1, np.rad2deg(self.beta_up)+1])
        plt.savefig("../../../fig/fscan_design/zebra_diagram.png", dpi=300)

    def aasr(self, prf, Naz, Vr, Bfov):
        len_prf = len(prf)
        Na_band = 1000
        fa_band = np.linspace(-Bfov/2, Bfov/2, Na_band)
        aasr_num = np.zeros(len_prf)
        aasr = np.zeros(len_prf)
        aasr_deno = 0
        center_apeture = (Naz-1)/2
        if Naz%2 == 0:
            center_apeture = Naz/2
        # 计算方位向响应
        for j in range(len_prf):
            alias_fa = (fa_band) - np.floor((fa_band)/prf[j])*prf[j]
            # 计算重构滤波器
            H_matrix = np.zeros((len(fa_band), Naz, Naz), dtype=complex)
            P_matrix = np.zeros((len(fa_band), Naz, Naz), dtype=complex)
            for k in range(Naz):
                for n in range(Naz):
                    H_matrix[:, k, n] = np.exp(- 1j * np.pi * (n * self.La) / Vr * (alias_fa + (k-center_apeture) * prf[j]))


            for k in range(len(fa_band)): 
                P_matrix[k, :, :] = np.linalg.inv(H_matrix[k])

            for i in range(0, 10): 
                ## 天线方向图，这里使用的是矩形天线。如果为其他天线结构，需要更改增益函数
                G_tmp_tx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vs))**2
                G_tmp_rx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vs))**2
                G_tmp = G_tmp_rx * G_tmp_tx

                ## 计算方位向重构系统的响应
                H = np.zeros(Naz, dtype=complex)
                pm = np.zeros((len(fa_band)), dtype=complex)
                for t in range(len(fa_band)):
                    for n in range(Naz):
                        H[n] = np.exp(- 1j * np.pi * (n * self.La) / Vr * (fa_band[t] + i * prf[j]))    
                    Pa = np.squeeze(P_matrix[t, :, :])
                    Ha = np.squeeze(H)
                    tmp = 0
                    if Naz == 1:
                        pm[t] += Ha*Pa
                    else:
                        band_index = int(np.floor((fa_band[t] + prf[j]*center_apeture)/prf[j]))%Naz
                        tmp = Ha@Pa
                        # if i == 1:
                        #     print(band_index, np.abs(tmp)/np.max(np.abs(tmp)))
                        pm[t] += tmp[band_index]
                                

                G_tmp = G_tmp * np.abs(pm)**2
                if i == 0:
                    aasr_deno = np.trapezoid(G_tmp, fa_band)
                else:
                    aasr_num[j] += 2*np.trapezoid(G_tmp, fa_band) ## +-m

            aasr[j] = aasr_num[j] / aasr_deno 
            aasr[j] = 10 * np.log10(aasr[j])
            
        return aasr

    def nesz(self, doa, Pu, beta, Lr, Br, PRF):

        cons = 256*np.pi**3 * self.K*self.T * self.Ln*self.Vs/ (Pu*self.lambda_**3*self.c)
        R0 = self.calculate_R0(doa)      

        ### 天线增益
    
        # A_e = self.d*(self.d) * 0.6 ## 天线有效面积
        # unit_gain = 4*np.pi*A_e/(doa_lambda**2)
        # ant_gain = unit_gain*self.N )
        gain = 1
        # Ae = 0.6*self.La*self.Lr

        # ant_gain = 4*np.pi*Ae/self.lambda_**2
        
        ### 天线方向图,a 为调节系数用于增宽主瓣
        if self.mode == 1:
            a = 1
            u = a*np.pi*Lr/self.lambda_*np.sin(doa-beta)
            ant_gain = (np.sin(u)/u)**2
            ant_gain = ant_gain/np.max(ant_gain)*gain
        else:
            ant_doa = doa-beta+self.fscan_center
            ant_gain_func = interpolate.interp1d(self.r_angle, self.r_pattern, kind='cubic', fill_value="extrapolate")
            ant_gain = np.max(self.r_pattern)*np.ones_like(ant_doa)
            ant_gain[ant_doa<self.fscan_left] = ant_gain_func(ant_doa - (self.fscan_left-self.fscan_center))[ant_doa<self.fscan_left]
            ant_gain[ant_doa>self.fscan_right] = ant_gain_func(ant_doa - (self.fscan_right-self.fscan_center))[ant_doa>self.fscan_right]
            ant_gain = 10**(ant_gain/10)*gain

        R_eta = R0/np.cos(self.theta_c)
        incident = self.calculate_incident(R_eta)

        var = R0**3 * Br * np.sin(incident)/(self.Tp*(self.theta_r/self.fscan_width)*ant_gain**2*PRF)
        nesz = cons*var ## el loss and az loss
        nesz = 10*np.log10(nesz)

        return nesz
    
    def rasr(self, doa, beta, PRF, Lr):
        R0 = self.calculate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        max_R = np.sqrt((self.H+self.Re)**2 - self.Re**2)
        ant_gain_func = interpolate.interp1d(self.r_angle, self.r_pattern, kind='cubic', fill_value="extrapolate")
        for m in range(-5, 4):
            Rm = R0 + m*self.c*(1/PRF)/2
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
            a = 1
            ant_doam = doam-beta+self.fscan_center
            ant_gain = np.max(self.r_pattern)*np.ones_like(ant_doam)
            ant_gain[ant_doam<self.fscan_left] = ant_gain_func(ant_doam - (self.fscan_left-self.fscan_center))[ant_doam<self.fscan_left]
            ant_gain[ant_doam>self.fscan_right] = ant_gain_func(ant_doam - (self.fscan_right-self.fscan_center))[ant_doam>self.fscan_right]
            ant_gain = 10**(ant_gain/10)
            G_doamr = ant_gain
            G_doamt = ant_gain
   
            R_eta = Rm/np.cos(self.theta_c)
            incident = self.calculate_incident(R_eta)
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        rasr = 10*np.log10(rasr)


        return rasr

    def operation_point(self,doa):
        print(np.rad2deg(doa[-1]-doa[0]))
        Bs = self.Br/self.theta_r*self.fscan_width
        Kr = Bs/self.Tp
        theta_p = doa-self.beta+self.fscan_center
        fc = np.array([34e9,35e9,36e9])
        lambda_ = 3e8 / fc  # 波长
        theta = np.array([17.9416367435066, 14.2959686223657, 10.9066262820108])  # 测量角度（度）
        theta = np.deg2rad(theta)  # 转换为弧度
        param = np.polyfit(theta, lambda_, 2)
        lambda_p = param[0]*theta_p**2 + param[1]*theta_p + param[2]
        fp = self.c/lambda_p - self.f0
        fp = np.clip(fp, -Bs/2, Bs/2)
        start = self.H/np.cos(doa)/self.c*2
        end = start + self.Tp*self.theta_r/self.fscan_width
        end_normal = start+self.Tp
        tau_trans = (fp)/Kr+self.Tp/2-self.Tp*self.theta_r/self.fscan_width/2
        tau_start = np.squeeze(tau_trans+start)
        print("tau_trans, tau_start:", tau_trans[0]*1e6, tau_start[0]*1e6)
        tau_end = np.squeeze(tau_trans + end)
        tau_center = np.squeeze(tau_start+self.Tp*self.theta_r/self.fscan_width/2)

        winlen = (np.max(tau_end)-np.min(tau_start))*self.Fr
        print("winlen:", winlen)
        win_start = 36.5e-6
        win_end = win_start+65536/self.Fr
        print("tau start:", np.rad2deg(doa[0]), tau_start[0]*1e6)

        R = np.squeeze(self.H/np.cos(doa))
        Gr = R*np.sin(doa)
        print("R:", np.max(R)-np.min(R))
        print("swath:",np.max(Gr)-np.min(Gr))
        plt.figure()
        plt.plot(tau_start*1e6, R)
        plt.plot(tau_end*1e6, R)
        plt.plot(tau_center*1e6, R, 'r')
        plt.vlines(win_start*1e6, np.min(R), np.max(R), colors='g')
        plt.vlines(win_end*1e6, np.min(R), np.max(R), colors='g')
        plt.fill_betweenx(R, np.squeeze(start)*1e6, np.squeeze(end_normal)*1e6, color='green', alpha=0.2)
        plt.fill_betweenx(R, tau_start*1e6, tau_end*1e6, color='orange', alpha=0.5)

        plt.xlabel("time/μs")
        plt.ylabel("slant range/m")
        plt.savefig("../../../fig/fscan_design/up_chirp.pdf", dpi=2000)

    def fscan_res(self, doa):
            Bs = self.Br/self.theta_r*self.fscan_width
            Kr = Bs/self.Tp
            theta_p = doa-self.beta+self.fscan_center
            fc = np.array([34e9,35e9,36e9])
            lambda_ = 3e8 / fc  # 波长
            theta = np.array([17.9416367435066, 14.2959686223657, 10.9066262820108])  # 测量角度（度）
            theta_bw = np.array([2.9, 2.55, 2.1])  # 波束宽度（度）
            theta = np.deg2rad(theta)  # 转换为弧度
            theta_bw = np.deg2rad(theta_bw)  # 转换为弧度
            param = np.polyfit(theta, lambda_, 2)
            param_bw = np.polyfit(theta, theta_bw, 2)
            lambda_p = param[0]*theta_p**2 + param[1]*theta_p + param[2]
            bw = param_bw[0]*theta_p**2 + param_bw[1]*theta_p + param_bw[2]
            lambda_l = param[0]*(theta_p-bw/2)**2 + param[1]*(theta_p-bw/2) + param[2]
            lambda_u = param[0]*(theta_p+bw/2)**2 + param[1]*(theta_p+bw/2) + param[2]
            fl = self.c/lambda_u - self.f0
            fu = self.c/lambda_l - self.f0
            fbw = np.abs(fu-fl)
            rate = np.abs(np.rad2deg(bw)/(fu-fl))
            print("scan rate, 34:{}, 36:{}".format(rate[-1]*1e9, rate[0]*1e9))
            print("fbw, 34:{}, 36:{}".format(fbw[-1]/1e9, fbw[0]/1e9))

            fbw = np.abs(self.c/lambda_l - self.c/lambda_u)

            gamma = np.array([2.2, 2.5, 2.4])

            Fs = 2.5e9
            fs = np.linspace(-Fs/2, Fs/2, 100000)
            fs = fs[np.newaxis, :]
            H =  np.exp(-0.5 * ((fs) / (fbw/2.4)) ** 2)

            beta = self.beta[0]
            ant_doa = doa-beta+self.fscan_center
            ant_gain_func = interpolate.interp1d(self.r_angle, self.r_pattern, kind='cubic', fill_value="extrapolate")
            ant_gain = ant_gain_func(ant_doa)
 
            lambda_ant = param[0]*ant_doa**2 + param[1]*ant_doa + param[2]
            fant = self.c/lambda_ant-self.f0 - 1e9 + 10e6
            ant_gain_fant = interpolate.interp1d(np.squeeze(fant), np.squeeze(ant_gain), kind='cubic', fill_value="extrapolate")
            ant_H = ant_gain_fant(fs)
            ant_H = 10**(ant_H/10)

            ant_H = ant_H/np.max(ant_H)*np.max(H)

            plt.figure()    
            plt.plot(np.squeeze(fs), np.squeeze(ant_H), label="measured")
            plt.plot(np.squeeze(fs), np.squeeze(H[50,:]), label="approximate")
            plt.legend()
            plt.grid()
            plt.xlabel("frequency/Hz")
            plt.ylabel("normalized amplitude")
            plt.savefig("../../../fig/fscan_design/window_34.pdf", dpi=2000)
            

            H_ifft = np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H), axis=1)))
            max_H_half = np.max(H_ifft, axis=1)/2
            max_H_half = np.tile(max_H_half[:, np.newaxis], (1, H_ifft.shape[1]))
            win_len = np.sum(H_ifft > max_H_half, axis=1)
            res = win_len / Fs *self.c/2

            plt.figure()
            plt.plot(np.rad2deg(doa), res)
            plt.xlabel("look angle/°")
            plt.ylabel("range resolution/m")
            plt.grid()
            plt.savefig("../../../fig/fscan_design/res_vs_look.pdf", dpi=2000)
            plt.figure()
            plt.plot(np.rad2deg(doa), self.c/lambda_p, label="center frequency")
            plt.plot(np.rad2deg(doa), self.c/lambda_l,  label="upper frequency")
            plt.plot(np.rad2deg(doa), self.c/lambda_u, label="lower frequency")
            plt.xlabel("look angle/°")
            plt.ylabel("frequency/Hz")
            plt.legend()
            plt.grid()
            plt.savefig("../../../fig/fscan_design/freq_vs_look.pdf", dpi=2000)

    def fscan_bandwidth(self, theta_in, W):
        def theta_f_fast(f):
            # a = 5.69e-3
            a = 4.78e-3
            beta = 2*np.pi*(np.sqrt(1-(self.c/(2*a*(self.f0+f)))**2)/(self.c/(self.f0+f)))
            theta = np.arcsin(beta*self.c/(2*np.pi*(self.f0+f)))
            return theta

        def theta_f_slow(f):
            a = 5.69e-3
            # a = 7.112e-3
            # f_offset = 6e9
            f_offset = 6.11e9
            # f_offset = 20e9
            # f_offset = 4.5e9
            # a = 4.78e-3
            beta =2*np.pi*(np.sqrt(1-(self.c/(2*a*(self.f0+f)))**2)/(self.c/(self.f0+f)) -  np.sqrt(1-(self.c/(2*a*(self.f0+f_offset)))**2)/(self.c/(self.f0+f_offset)))
            theta = np.arcsin(beta*self.c/(2*np.pi*(self.f0+f)))
            return theta
        
        def center_equation(f):
            # a = 4.78e-3
            # a = 5.69e-3
            a = 7.112e-3
            beta =2*np.pi*(np.sqrt(1-(self.c/(2*a*(self.f0)))**2)/(self.c/(self.f0)) -  np.sqrt(1-(self.c/(2*a*(self.f0+f)))**2)/(self.c/(self.f0+f)))
            theta = np.arcsin(beta*self.c/(2*np.pi*(self.f0)))
            return theta
        
        # equation = lambda f: center_equation(f) - np.deg2rad(-14.3)
        # solu = optimize.fsolve(equation, 2e9)
        # print("center frequency:", solu[0]/1e9, "GHz")

        # derivative of theta_f_slow at f=0 (central difference)
        h = 1e3
        dtheta_df_0 = (theta_f_slow(h) - theta_f_slow(-h)) / (2*h)
        dtheta_df_1 = (theta_f_slow(1e9+h) - theta_f_slow(1e9-h)) / (2*h)
        dtheta_df_2 = (theta_f_slow(-1e9+h) - theta_f_slow(-1e9-h)) / (2*h)
        print("slope derivative 34:{}, 35:{}, 36:{}".format(np.rad2deg(dtheta_df_2)*0.1e9, np.rad2deg(dtheta_df_0)*0.1e9, np.rad2deg(dtheta_df_1)*0.1e9))

        def theta_nl(f):
            return theta_f_slow(f) - theta_f_slow(0) - dtheta_df_0 * f
        
        print("34:{},35:{},36:{}".format(np.rad2deg(theta_f_slow(-1e9)), np.rad2deg(theta_f_slow(0)), np.rad2deg(theta_f_slow(1e9))))
        print("derivative in linear 34:{}, 36{}".format(np.rad2deg(theta_nl(-1e9)), np.rad2deg(theta_nl(1e9))))

        f_bw = self.c/(2*0.16)*0.75
        print("theta_bw:", np.rad2deg(theta_f_slow(f_bw/2)-theta_f_slow(-f_bw/2)))
        theta_bw = theta_f_slow(f_bw/2)-theta_f_slow(-f_bw/2)
        print("res:", self.c/(2*f_bw))
        def func(theta_sc):
            return np.abs(self.H*(np.tan(theta_in-theta_sc/2)-np.tan(theta_in+theta_sc/2)))-W
        
        equation = lambda theta_sc: func(theta_sc)
        solu = optimize.fsolve(equation, 0)
        print("theta_sc:", np.rad2deg(solu[0]-theta_bw))
        # theta_sc = solu[0]-theta_bw
        theta_sc = np.deg2rad(7)
        def func_f(f_sc):
            return np.abs(theta_f_slow(f_sc/2)-theta_f_slow(-f_sc/2))-theta_sc
        equation_f = lambda f_sc: func_f(f_sc)
        solu_f = optimize.fsolve(equation_f, 2e9)
        Br = solu_f[0]
        theta_sc_br = theta_f_slow(Br/2)-theta_f_slow(-Br/2)
        print("fscan scanwidth:", np.rad2deg(theta_sc_br), "°")
        print("fscan bandwidth:", solu_f[0]/1e9, "GHz")
    
    def fscan_design(self):
        # phi_az = self.lambda_/(2*self.da)
        phi_az = np.deg2rad(8)
        PRF = 6000
        fbw = self.c/(2*self.dr)
        # f_bw = 2.3e8
        theta_in = np.deg2rad(60)
        # W = self.groud_extent
        W = 2e3
        def func(theta_sc):
            return np.abs(self.H*(np.tan(theta_in-theta_sc/2)-np.tan(theta_in+theta_sc/2)))-W
        equation = lambda theta_sc: func(theta_sc)
        solu = optimize.fsolve(equation, 0)
        theta_W = solu[0]
        print("theta_W:", np.rad2deg(theta_W))

        G = 10**(30/10)
        eta = 0.5
        print("eta:", eta)
        theta_bw = 4*np.pi*eta/(G*phi_az)
        xi = theta_bw/fbw
        theta_sc = theta_W - theta_bw
        Br = theta_sc/xi

        x = np.linspace(0.7, 2, 100)
        eta_r = 1-np.exp(-2*x)
        win = eta_r > 0.75
        print("eta_r: max, min:", np.max(eta_r), np.min(eta_r))
        eta_op = 2*(1-np.exp(-x))**2/(x)
        eta_op = eta_op[win]
        print("eta_op: max, min:", np.max(eta_op), np.min(eta_op))
        plt.figure()
        plt.plot(x, eta_op)
        plt.show()



        print("parameter: phi_az: {}, theta_sc: {}, f_bw: {}, theta_bw: {}, xi: {}, Br: {}".format(np.rad2deg(phi_az), np.rad2deg(theta_sc),fbw, np.rad2deg(theta_bw), xi*1e9/np.pi*180, Br))
        


if __name__ == "__main__":
    design = FscanDesign()
    # print(design.Vf, design.Ta)
    # print(np.rad2deg(design.omega), np.rad2deg(design.psi_start), np.rad2deg(design.psi_end))
    Pu = 280
    design.fscan_design()
    design.fscan_bandwidth(np.deg2rad(60), 2e3)
    # print(design.Bd, design.Bfov)
    # print(10000/design.Bd)
    # print(design.Vs, design.Vg)
    # print(np.rad2deg(0.886*design.lambda_/design.La), np.rad2deg(0.886*design.lambda_/design.Lr))
    # print(np.rad2deg(design.psi_start), np.rad2deg(design.psi_end), design.Ta)
    # print(design.Br)
    # print(design.Tp/(1/design.PRF))
    # print(design.c/(2*design.Br*np.sin(design.look_angle_left)), design.c/(2*design.Br*np.sin(design.look_angle_right)))
    # print(np.rad2deg(design.look_angle_right-design.look_angle_left), np.rad2deg(design.theta_a))
    
    # prf = np.linspace(5e3, 12e3, 1000)
    # design.zebra_diagram(prf, design.Tp/50, 7e3, 11e3)
    doa = np.linspace(design.look_angle_left[0], design.look_angle_right[0], 100)
    print("look angle range:", np.rad2deg(doa[0]), np.rad2deg(doa[-1]))
    design.operation_point(doa)
    design.fscan_res(doa)

    plt.figure("resolution")
    res = np.array([])
    look_angle = np.array([])
    for i in range(len(design.PRF)):
        print("processing res")
        doa = np.squeeze(np.linspace(design.look_angle_left[i], design.look_angle_right[i], 100))
        res_doa = design.c/(2*design.Br[i]*np.sin(doa))
        plt.plot(np.rad2deg(doa), res_doa, linewidth=1, color='b')
        res = np.concatenate([res, res_doa])
        look_angle = np.concatenate([look_angle, doa])

    plt.xlabel("look angle/°")
    plt.ylabel("resolution/m")
    # plt.ylim([-28, -15])
    plt.grid()
    plt.savefig("../../../fig/fscan_design/res.png", dpi=300)
    print(np.max(res), np.min(res))

    nesz = np.array([])
    look_angle = np.array([])
    plt.figure("NESZ")  
    for i in range(len(design.PRF)):
        doa = np.squeeze(np.linspace(design.look_angle_left[i], design.look_angle_right[i], 100))
        nesz_doa = design.nesz(doa, Pu, design.beta[i], design.Lr[i], design.Br[i], design.PRF[i])
        plt.plot(np.rad2deg(doa), nesz_doa)
        nesz = np.concatenate([nesz, nesz_doa])
        look_angle = np.concatenate([look_angle, doa])

    plt.xlabel("look angle/°")
    plt.ylabel("NESZ/dB")
    # plt.ylim([-28, -15])
    plt.grid()
    plt.savefig("../../../fig/fscan_design/nesz.pdf", dpi=300)
    
    print("NESZ: ", np.max(nesz))
    # doa = np.linspace(design.look_angle_left, design.look_angle_right, 1000)
    # nesz = design.nesz(doa, Pav)
    rasr = np.array([])
    look_angle = np.array([])

    plt.figure("rasr")  
    for i in range(len(design.PRF)):
        doa = np.squeeze(np.linspace(design.look_angle_left[i], design.look_angle_right[i], 100))
        rasr_doa = design.rasr(doa, design.beta[i], design.PRF[i], design.Lr[i])
        plt.plot(np.rad2deg(doa), rasr_doa, linewidth=2)
        rasr = np.concatenate([rasr, rasr_doa])
        look_angle = np.concatenate([look_angle, doa])


    plt.xlabel("look angle/°")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.savefig("../../../fig/fscan_design/rasr.png", dpi=300)
    print("RASR: ", np.max(rasr))

    # aasr_point = np.array([])
    # prf = np.linspace(3e3, 9e3, 1000)
    # for i in range(len(prf)):
    #     aasr_prf = design.aasr(np.array([prf[i]]), 1, design.Vr, design.Bfov)
    #     aasr_point = np.concatenate([aasr_point, aasr_prf])
    # # aasr = design.aasr(prf, 1)
    # aasr_6000 = design.aasr(np.array([6000]), 1, design.Vr, design.Bfov)


    # design.theta_a = np.deg2rad(4.9)
    # design.La = design.lambda_/design.theta_a*0.886
    # design.Bfov = design.Bfov_func(design.theta_a, 0)
    # aasr_point_theoretical = np.array([])
    # for i in range(len(prf)):
    #     aasr_prf = design.aasr(np.array([prf[i]]), 1, design.Vr, design.Bfov)
    #     aasr_point_theoretical = np.concatenate([aasr_point_theoretical, aasr_prf])
    # # aasr = design.aasr(prf, 1)



    # plt.figure("AASR")  
    # # plt.scatter(np.rad2deg(design.beta), aasr_point, label="AASR")
    # # plt.plot(np.rad2deg(design.beta), aasr_point, label="AASR", linewidth=1)
    # plt.plot(prf, aasr_point, label="actual AASR")
    # # plt.plot(prf, aasr_point_theoretical, label="designed AASR")
    # plt.scatter(np.array([6000]), aasr_6000, marker="*", color='r')
    # plt.xlabel("PRF/Hz")
    # plt.ylabel("AASR/dB")
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/aasr.pdf", dpi=300)
    # print("AASR: ", np.max(aasr_6000))

    # angle_width = np.array([])
    # for i in range(len(design.beta)):
    #     left,right = design.calculate_scanwidth(design.beta[i], design.groud_extent)
    #     angle_width = np.concatenate([angle_width, [np.rad2deg(right-left)]])
    # plt.figure("angle_width")
    # plt.plot(np.rad2deg(design.beta), angle_width, label="scan angle width", linewidth=1)
    # plt.xlabel("look angle/°")
    # plt.ylabel("scan angle width/°", fontproperties=my_font)
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/angle_width.png", dpi=300)
    print("脉宽占空比:",np.mean(design.Tp/ (1/design.PRF))*100)
    print("脉宽:",design.Tp)

    # plt.figure("Omega")
    # plt.plot(np.rad2deg(design.beta), np.rad2deg(design.omega), label="Omega", linewidth=1)
    # plt.xlabel("look angle/°")
    # plt.ylabel("Omega/°/s", fontproperties=my_font)
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/omega.png", dpi=300)

    # plt.figure("Theta Width")
    # plt.plot(np.rad2deg(design.beta), np.rad2deg(design.theta_w), label="Squint Angle Width", linewidth=1)
    # plt.xlabel("look angle/°")
    # plt.ylabel("Squint Angle Width/°", fontproperties=my_font)
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/theta_width.png", dpi=300)

    # plt.figure("Ta")
    # plt.plot(np.rad2deg(design.beta), (design.Ta), label="Ta", linewidth=1)
    # plt.xlabel("look angle/°")
    # plt.ylabel("Ta/s", fontproperties=my_font)
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/ta.png", dpi=300)

    # nb = 8
    # baq = 4/8
    # tr = 2*design.groud_extent*np.sin(design.beta)/design.c + design.Tp
    # transmit_speed = nb*design.Fr*tr*design.PRF*baq
    # plt.figure("transmit speed")
    # plt.scatter(np.rad2deg(design.beta), transmit_speed/1e9, label="transmit speed")
    # plt.xlabel("look angle/°")
    # plt.ylabel("transmit speed/Gbps", fontproperties=my_font)
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/transmit_speed.png", dpi=300)


    # # 保存PRF, beta, look_angle_left, look_angle_right到Excel
    # df = pd.DataFrame({
    #     'PRF': design.PRF,
    #     '下视角中心(deg)': np.rad2deg(design.beta),
    #     '下视角左边界(deg)': np.rad2deg(design.look_angle_left),
    #     '下视角右边界(deg)': np.rad2deg(design.look_angle_right)
    # })
    # df.to_excel("../../../fig/fscan_design/PRF_beta_angles.xlsx", index=False)

    # beta = np.deg2rad(np.arange(np.rad2deg(design.beta_below), np.rad2deg(design.beta_up), 0.01))
    # theta_w_35 = np.deg2rad(np.array([0.199,0.279,0.339,0.387,0.4285]))
    # theta_w_40 = np.deg2rad(np.array([0.196,0.275,0.333,0.381,0.421]))
    # plt.figure("theta_w_vs_beta")
    # for i in range(len(theta_w_40)):
    #     gext = design.calculate_ground_extent(beta, theta_w_35[i])/1000
    #     plt.plot(np.rad2deg(beta), gext, label="{}dB beamwidth".format(i+1))
    # plt.xlabel("look angle/°")
    # plt.ylabel("ground extent/km", fontproperties=my_font)
    # plt.title("3.5m*1.5m Antenna")
    # plt.grid()
    # plt.legend()
    # plt.savefig("../../../fig/fscan_design/不同下视角下的幅宽", dpi=300)