
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import sys
sys.path.append(r"../")
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

cp.cuda.Device(2).use()   

class Fscan(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.N*self.d
        self.ttd =  2.00999999999658e-10
        self.B = 2e9                             #信号带宽
        self.Fs = self.B*1.2                            #采样率 
        self.Kr = -np.sign(self.ttd)*self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.d)
        self.Rc = self.R0/np.cos(self.theta_c)
        self.re_guard = self.Tp/2        ##接收窗保护 
        self.set_scanwidth(np.deg2rad(4.5))
        self.Nr = int(np.ceil(self.Fs*self.Tr))
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)

        self.points_n = 5
        self.points_r = self.R0+np.array([-400, 0, 0, 0, 450])
        self.points_a = np.array([0, -40, 0, 40, 0])


    def set_ttd(self, ttd):
        self.ttd = ttd
        self.Kr = -np.sign(self.ttd)*self.B/self.Tp

    def echogen(self):
        ##接收机时间窗
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            signal_t = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)

            ## 接收机频率与时间的关系
            ## 接收机的频率与时间的对应关系
            f_send = (self.f0+self.Kr*(mat_tau - 2*R_eta/self.c)) 
            deci = self.c/f_send
            pr = cp.sin(self.N*(cp.pi*(self.ttd*self.c-self.d*cp.sin(doa-self.beta)))/deci) / (cp.sin(cp.pi*(self.ttd*self.c-self.d*cp.sin(doa-self.beta))/deci))
            # pr = 1
            # pr = (f_send>self.f0-self.B/2)*(f_send<self.f0+self.B/2)*pr
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            # Wr = 1
            # signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            ## 发送机到点目标
            signal_t = pr*Wr*phase_r
            ## 点目标到接收机
            signal_r = signal_t*pr

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo
    
    ## 计算每个目标Doa对应的频率
    def calculate_doaf(self, doa):
        t_inv = self.ttd - self.d*np.sin(doa-self.beta)/self.c
        if np.any(t_inv == 0):
            self.log.warning("some of t_inv are 0")
        f_inv = np.abs(1/(t_inv))
        min_m = np.ceil((self.f0-self.B/2)/(f_inv))
        max_m = np.floor((self.f0+self.B/2)/f_inv)
        if np.any(min_m > max_m):
            self.log.warning("min_m > max_m in some area")
        # self.log.info("min_m: {}, max_m: {}".format(np.mean(min_m), np.mean(max_m)))
        return f_inv*min_m

    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
        self.Tr = self.calculate_re_window()
    
    def set_scanwidth(self, scan_width):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2
        self.scan_right = self.beta + scan_width/2
        self.Tr = self.calculate_re_window() + self.re_guard

    
    def calculate_m(self, doa):
        t_inv = self.ttd - self.d*np.sin(doa-self.beta)/self.c
        f_inv = np.abs(1/(t_inv))
        min_m = np.ceil((self.f0-self.B/2)/(f_inv))
        max_m = np.floor((self.f0+self.B/2)/f_inv)
        if np.any(min_m > max_m):
            self.log.warning("min_m > max_m in some area")
        return min_m*np.sign(self.ttd)

        ## 计算接收窗口大小
    
    def calculate_rx_peak(self, doa):
        doaf = self.calculate_doaf(doa)
        t_peak = (doaf - (self.f0-self.Kr*self.Tp/2))/(self.Kr)
        doaR = self.calculate_R0(doa)
        rx_peak = 2*doaR/self.c + t_peak
        return rx_peak

    def calculate_re_window(self):

        rx_min = sopt.minimize(self.calculate_rx_peak, x0=self.beta, method='Nelder-Mead', bounds=[(self.scan_left, self.scan_right)])

        rx_left = self.calculate_rx_peak(self.scan_left)
        rx_right = self.calculate_rx_peak(self.scan_right)
        f_interval = np.abs(1/(self.N*(self.ttd-self.d*np.sin(self.beta-self.beta)/self.c)))
        interval = np.abs(f_interval/self.Kr)

        return np.max(np.array([rx_left, rx_right]) - self.calculate_rx_peak(rx_min.x)) + interval
    
    def ttd_judge(self, doa):
        t_inv = self.ttd - self.d*np.sin(doa-self.beta)/self.c
        if np.any(t_inv == 0):
            return False
        f_inv = np.abs(1/(t_inv))

        ## none of f_inv is 0 
        min_m = np.ceil((self.f0-self.B/2)/(f_inv))
        max_m = np.floor((self.f0+self.B/2)/f_inv)

        ## if min_m > max_m, then there exist some area that can not be scanned
        if np.any(min_m != max_m):
            return False
        # if np.sum(min_m!=min_m[0]) != 0 or np.sum(max_m!=max_m[0]) != 0:
        #     return False
        
        ## when f are in the range of f0-B/2 and f0+B/2, set m = min_m in order to reduced bandwidth
        m = min_m
        doaf = f_inv*m
        f_interval = np.abs(1/(self.N*(self.ttd-self.d*np.sin(doa-self.beta)/self.c)))
        f_left = doaf - f_interval
        f_right = doaf + f_interval
        band_width = 2*f_interval
        res = self.c/(2*band_width)
        # if  np.sum(m!=m[0]) != 0 or np.min(f_left) < self.f0-self.B/2 or np.max(f_right) > self.f0+self.B/2 or np.max(res) > self.dr:
        #     return False
        if  np.min(f_left) < self.f0-self.B/2 or np.max(f_right) > self.f0+self.B/2 or np.max(res) > self.dr:
            return False
        return True
    
    ## 如果为线性调频，计算每个目标Doa对应的快时间
    def calculate_doaTx(self, doa):
        doaf = self.calculate_doaf(doa)
        t_peak = (doaf - (self.f0-self.Kr*self.Tp/2))/(self.Kr)
        f_interval = np.abs(1/(self.N*(self.ttd-self.d*np.sin(doa-self.beta)/self.c)))
        interval = np.abs(f_interval/self.Kr)
        t_left = t_peak - interval
        t_right = t_peak + interval
        band_width = 2*f_interval

        return t_peak, t_left, t_right, band_width

        
    
    def range_estimate(self):
        self.log.info("target estimate:")
        target_doa = np.arccos(((self.H+self.Re)**2+self.points_r**2-self.Re**2)/(2*(self.H+self.Re)*self.points_r)) ## DoA 信号到达角
        target_peak,target_left,target_right, target_bw= self.calculate_doaTx(target_doa)
        self.log.info("t_peak(us): {}".format(target_peak*1e6))    
        self.log.info("t_left(us): {}".format((target_left)*1e6))
        self.log.info("t_right(us): {}".format((target_right)*1e6))
        self.log.info("target duration(us): {}".format((target_right-target_left)*1e6))
        self.log.info("target bandwidth(Mhz): {}".format((target_bw)/1e6))
        slant = self.calculate_R0(target_doa)
        self.log.info("target slant range(km): {}".format(slant/1e3))
        self.log.info("target doa: {}".format(np.rad2deg(target_doa)))

        doa =  np.linspace(self.scan_left, self.scan_right, self.Nr)
        self.swath_estimate(doa)
        self.swath_estimate(target_doa)


        self.log.info("swath width:{}".format((np.max(target_right)-np.min(target_left))*1e6))
        return target_bw
    
    
    def swath_estimate(self, doa):
        tx_peak, _, _, _ = self.calculate_doaTx(doa)

        doaR = self.calculate_R0(doa)
        rx_peak = 2*doaR/self.c + tx_peak
        t_inv = self.ttd - self.d*np.sin(doa-self.beta)/self.c
        if np.any(t_inv == 0):
            self.log.warning("some of t_inv are 0")
        f_inv = np.abs(1/(t_inv))
        min_m = np.ceil((self.f0-self.B/2)/(f_inv))
        max_m = np.floor((self.f0+self.B/2)/f_inv)
        doaf1 = f_inv*np.min(min_m)
        win1 = (doaf1>self.f0-self.B/2)*(doaf1<self.f0+self.B/2)
        start_index = np.argmax(win1 > 0)  
        end_index = len(win1) - np.argmax(win1[::-1] > 0) - 1 
        win1 = slice(start_index, end_index + 1)  # 创建切片对象
        doaf1 = doaf1[win1]

        doaf2 = np.max(max_m)*f_inv
        win2 = (doaf2>self.f0-self.B/2)*(doaf2<self.f0+self.B/2)
        start_index = np.argmax(win2 > 0)  
        end_index = len(win2) - np.argmax(win2[::-1] > 0) - 1  
        win2 = slice(start_index, end_index + 1)  # 创建切片对象
        doaf2 = doaf2[win2]
  
        t_peak1 = (doaf1 - (self.f0-self.Kr*self.Tp/2))/(self.Kr)
        t_peak2 = (doaf2 - (self.f0-self.Kr*self.Tp/2))/(self.Kr)
        rx_peak1 = 2*doaR[win1]/self.c + t_peak1
        rx_peak2 = 2*doaR[win2]/self.c + t_peak2

        plt.figure()
        plt.subplot(211)
        plt.scatter(np.rad2deg(doa)[win1], rx_peak1*1e6)
        plt.scatter(np.rad2deg(doa)[win2], rx_peak2*1e6)
        plt.grid()
        plt.xlabel("视角/°", fontproperties=my_font)
        plt.ylabel("回波峰值接收时间(us)", fontproperties=my_font)
        plt.title("视角 vs. 回波峰值接收时间", fontproperties=my_font)
        plt.subplot(212)
        plt.scatter(np.rad2deg(doa)[win1], (doaf1-self.f0)/1e6)
        plt.scatter(np.rad2deg(doa)[win2],  (doaf2-self.f0)/1e6)
        plt.grid()
        plt.xlabel("视角/°", fontproperties=my_font)
        plt.ylabel("频率(MHz)", fontproperties=my_font)
        plt.title("视角 vs. 频率", fontproperties=my_font)
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/doa_t_f.png", dpi=300)

    def get_ttd_rasr(self, doa):
        ttd_value = self.ttd
        init_range = 50
        for ttd in np.arange(-3e-9, 3e-9, 1e-12):
            self.set_ttd(ttd)
            if self.ttd_judge(doa) == False:
                continue
            rasr = np.max(self.rasr(doa))
            if rasr < init_range:
                init_range = rasr
                ttd_value = ttd
        if init_range > 20:
            self.log.warning("no ttd is suitable in such condition")
        self.log.info("ttd: {}".format(ttd_value))
        return ttd_value

    def get_ttd_bandwidth(self, doa):
        ttd_value = self.ttd
        init_range = self.B*2
        for ttd in np.arange(-3e-9, 3e-9, 1e-12):
            self.set_ttd(ttd)
            if self.ttd_judge(doa) == False:
                continue
            doaf = self.calculate_doaf(doa)
            bw = np.abs(np.max(doaf)-np.min(doaf))
            if bw < init_range:
                init_range = bw
                ttd_value = ttd
        if init_range > self.B:
            self.log.warning("no ttd is suitable in such condition")
        self.log.info("ttd: {}".format(ttd_value))
        return ttd_value

    def rasr(self, doa):
        R0 = self.calculate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        f = self.calculate_doaf(doa)
        # mat_tau = np.linspace(left, right, 4000)
        prm_tmp = self.c/(f + np.finfo(float).eps)
        max_R = np.sqrt((self.H+self.Re)**2 - self.Re**2)
        # print(max_R/(self.c*(1/self.PRF)/2))
        for m in range(-5,5):
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

            ## 相控阵调制对模糊的抑制
            ## 当波束扫描到doa时，时间为peak，得出此时的模糊角的相控阵增益
            prm_tmp1 = prm_tmp[window_Rm]
            if m == 0:
                prm = self.N
            else:
                prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(doam-self.beta)))/prm_tmp1) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(doam-self.beta))/prm_tmp1)
            # prm = np.zeros(len(doam), dtype=np.complex128)
            # for i in range(self.N):
            #     prm += np.exp(-2j*np.pi*f[window_Rm]*i*(self.ttd-self.d*np.sin(doam-self.beta)/self.c) +1j*np.pi*self.Kr*i**2*(self.ttd-self.d*np.sin(doam-self.beta)/self.c)**2)
            # prm = np.abs(prm)
            # prm = prm**2 ## 功率峰值
            prm = prm**2 ## 双程，发送接收均使用频扫阵列
            # prm = np.trapezoid(prm, mat_tau, axis=0)


            ## 单一单元增益
            G_doamr = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2

            R_eta = Rm/np.cos(self.theta_c)

            incident = self.calculate_incident(R_eta)
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(prm*G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        return 10*np.log10(rasr+1e-10)
    
    
    def resolution(self, doa):
        _, _, _, bw = self.calculate_doaTx(doa)
        res = self.c/(2*bw)
        return res

    def nesz(self, doa, Pu):
        Pav = Pu*(self.Tp*self.PRF)
        doaf = self.calculate_doaf(doa)
        doa_lambda = self.c/doaf
        cons = 128*np.pi**3 * self.K*self.T * self.Ln / (Pu*doa_lambda**2*self.c)
        R0 = self.calculate_R0(doa)
        # peak, left, right, bw = self.calculate_doaTx(doa)
        # tp = np.abs(right-left)
        beam_width = np.linspace(2.5, 2.6, len(doa))
        beam_width = np.deg2rad(beam_width)
        tp = self.Tp*beam_width/self.scan_width
        bw = self.B*beam_width/self.scan_width

        ### 天线增益
    
        # A_e = self.d*(self.d) * 0.6 ## 天线有效面积
        # unit_gain = 4*np.pi*A_e/(doa_lambda**2)
        # ant_gain = unit_gain*self.N 
        ant_gain = 10**(3)

        # irw = np.deg2rad(2.5)
        # doa35 = np.linspace(-irw/2, (irw/2), len(doa))+self.beta
        # prm_tmp = self.lambda_
        # prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-self.beta))/prm_tmp)
        # G_r = np.sinc(0.018*np.sin(doa35-self.beta)/prm_tmp)
        # prm = G_r*prm
        # prm = np.abs(prm)/np.max(np.abs(prm))
        # el_loss = irw/np.trapezoid(prm**4, doa35)
        # self.log.info("el pel loss: {}".format(el_loss))

        # irw_az = np.deg2rad(np.linspace(10.6, 11.2, len(doa)))
        irw_az = 0.886*doa_lambda/(self.La)
        # doa35 = np.linspace(-irw_az/2, (irw_az/2), len(doa))+self.beta
        # G_az = np.sinc(0.04*np.sin(doa35-self.beta)/self.lambda_)
        # G_az = np.abs(G_az)/np.max(np.abs(G_az))
        # az_loss = irw_az/np.trapezoid(G_az**4, doa35)
        # self.log.info("az pel loss: {}".format(az_loss))

        R_eta = R0/np.cos(self.theta_c)
        incident = self.calculate_incident(R_eta)

        var = R0**3 * bw * np.sin(incident)/(tp*ant_gain**2 *irw_az)
        nesz = cons*var ## el loss and az loss
        return 10*np.log10(nesz)
    
    def beam_pattern(self, doa):
        ## 俯仰向相控阵调制
        ### 每个时间，相控阵方向图
        # peak, _, _, _ = self.calculate_doaTx(doa)
        f = self.calculate_doaf(doa)
        f = np.linspace(self.f0-self.B/2, self.f0+self.B/2, 4000)
        ### 波束扫描到peak
        mat_f = f[:, np.newaxis] * np.ones([1, len(doa)])
        # self.log.info(mat_peak)
        doa_ant = np.linspace(np.deg2rad(-100), np.deg2rad(100), len(doa))+self.beta
        mat_doa  = np.tile(doa_ant[np.newaxis, :], (4000, 1))
        # self.log.info(mat_doa)
        prm_tmp = self.c/(mat_f)
        prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(mat_doa-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(mat_doa-self.beta))/prm_tmp)
        # prm = prm**2

        
        Gr = np.sinc(self.d*np.sin(mat_doa-self.beta)/prm_tmp)**2
        prm = prm*Gr
        prm = prm / np.tile(np.max(np.abs(prm), axis=1)[:, np.newaxis], (1, len(doa)))
        # print(Gr*prm)
        # prm = prm*(10**(1.5)/np.max(prm[2000,:]))
        prm = 20*np.log10(np.abs(prm))
        max_index = np.argmax(prm[2000, :]) 
        max_value = prm[2000, max_index]
        half_max = max_value-3
        valid = prm[2000,:] > half_max
        irw = (np.sum(valid)*(doa_ant[1]-doa_ant[0]))
        self.log.info("ant angular width: {}".format(np.rad2deg(irw)))

    
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.rad2deg(doa_ant)-60, f/1e9)
        ax.plot_surface(X, Y, prm, cmap='viridis', edgecolor='none')
        ax.set_xlabel("视角/°", fontproperties=my_font)
        ax.set_ylabel("频率 (GHz)", fontproperties=my_font)
        ax.set_zlabel("归一化增益 (dB)", fontproperties=my_font)
        ax.set_title("Beam Pattern 3D View", fontproperties=my_font)
        ax.set_zlim(-80, 0)
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/fscan_prm_3d.png", dpi=300)


        G_az = np.sinc(0.04*np.sin(doa_ant-self.beta)/self.lambda_)
        plt.figure(figsize=(12,12))
        plt.plot(np.rad2deg(doa_ant)-60, 20*np.log10(np.abs(G_az)), label="f={} GHz".format(35))
        plt.legend()
        plt.grid()
        plt.xlabel("视角/°", fontproperties=my_font)
        plt.ylabel("归一化增益", fontproperties=my_font)
        plt.ylim(-40, 0)
        plt.title("Antenna Orientation Diagram")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/fscan_prm_az.png", dpi=300)

    def fscan_rd_focus(self, echo):  
        echo = cp.array(echo)
        [Na, Nr] = cp.shape(echo)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.fc + cp.fft.fftshift(cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.Rc/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        data_fft_r = cp.fft.fft(echo, Nr, axis = 1) 
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        data_fft_cr = data_fft_r*Hr
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
        return data_final.get()
