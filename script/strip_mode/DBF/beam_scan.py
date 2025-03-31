## 

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
sys.path.append(r"../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus

cp.cuda.Device(0).use()

class BeamScan:
    def __init__(self):
        self.H = 519e3                              #卫星高度
        self.ttd = 0.833e-9                        #频扫通道时延
        self.Lr = 1.5                               #雷达距离向宽度
        self.fscan_N = 60                           #频扫通道数   
        self.fscan_d = self.Lr/self.fscan_N         #频扫通道间隔
        self.DBF_N = 10                             #DBF天线数
        self.Re = 6371.39e3                         #地球半径
        self.beta = np.deg2rad(25)                  #天线安装角
        self.dbf_d = self.Lr/self.DBF_N             #DBF子孔径间距
        self.c = 299792458                          #光速
        self.Fs = 2000e6                            #采样率              
        self.Tp = 20e-6                            #脉冲宽度                        
        self.f0 = 30e+09                            #载频                     
        self.PRF = 1950                             #PRF                     
        self.Vr = 7500                              #雷达速度     
        self.B = 1200e6                             #信号带宽
        self.fc = -1000                             #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = np.arcsin(self.fc*self.lambda_/(2*self.Vr))
        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(self.beta)/self.Re)
        tmp_angle = tmp_angle - self.beta
        self.R0 = self.Re*np.sin(tmp_angle)/np.sin(self.beta)
        self.La = 1.5
        self.Ta = 1
        self.Kr = -self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.fscan_d)
        self.dbf_beam_width = (0.886*self.lambda_/self.dbf_d)
        self.scan_width = min(self.fscan_beam_width, np.deg2rad(19))
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)
        print("fscan beam_width: ", np.rad2deg(self.fscan_beam_width))
        print("scan width: ", np.rad2deg(self.scan_width))
        print("lambda: ", self.lambda_, "d:", self.fscan_d)

    def init_raw_data(self, data1):
        [Na, Nr] = np.shape(data1)
        data = np.zeros([int(np.ceil(Na*1.5)), int(np.ceil(Nr*1.5))], dtype=np.complex128)
        [self.Na, self.Nr] = np.shape(data)
        data[self.Na/2-Na/2:self.Na/2+Na/2, self.Nr/2-Nr/2:self.Nr/2+Nr/2] = data1
        return data
    
    ## 添加高斯白噪声
    def add_noise(self, data, snr):
        data = data + np.random.randn(self.Na, self.Nr)*np.sqrt(np.var(data)/10**(snr/10))
        return data
    
    def calulate_R0(self, look_angle):
        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(look_angle)/self.Re)
        tmp_angle = tmp_angle - look_angle
        R0 = self.Re*np.sin(tmp_angle)/np.sin(look_angle)
        return R0
    
    def calulate_doa(self, R0):
        incident = np.arccos((self.Re**2+R0**2-(self.H+self.Re)**2)/(2*self.Re*R0)*(R0>self.H))
        doa = np.arcsin(self.Re*np.sin(incident)/(self.H+self.Re))
        return doa


    ## 计算接收窗口大小
    def calulate_re_window(self, beam_width):
        R_max = self.calulate_R0(self.beta+beam_width/2)
        R_min = self.calulate_R0(self.beta-beam_width/2)
        return (R_max-R_min)*2/self.c

    def init_simparams(self, mode = "fscan"):
        if(mode == "fscan"):
            self.Tr = self.calulate_re_window(self.scan_width)
            self.Nr = int(np.ceil(self.Fs*self.Tr))
        elif(mode == "dbf"):
            self.Tr = self.calulate_re_window(self.dbf_beam_width)
            self.Nr = int(np.ceil(self.Fs*self.Tr))

        print("receive window length Tr: ", self.Tr*1e6)   
        print("R_width: ", self.Tr*self.c/2) 
        print("Doppler bandwidth: ", 0.886*2*self.Vr*np.cos(self.theta_c)/self.La)
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+np.linspace(-6000,6000, self.points_n)
        self.points_a = np.linspace(-3000, 3000, self.points_n)
        self.Ba = 2*0.886*self.Vr*np.cos(self.theta_c)/self.La + 2*self.Vr*self.B*np.sin(self.theta_c)/self.c
        self.Rc = self.R0/np.cos(self.theta_c)

    def tradition_echogen(self):
        self.init_simparams()
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
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
        return S_echo.get() 

    def dbf_echogen(self):
        self.init_simparams("dbf")
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            t0 = R0_tar*2/self.c
            doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            P_sub = 1 ## 子孔径损耗
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
            ## 遍历DBF-SCORE子孔径
            for j in range(self.DBF_N):
                d = (j-(self.DBF_N-1)/2)*self.dbf_d
                ti = t0-d*cp.sin(doa-self.beta)/self.c
                Wr = cp.abs(mat_tau-ti)<self.Tr/2
                phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-ti-2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.f0*ti) ## 基带信号
                channel_ri = P_sub*Wr*phase_r
                wi = cp.exp(-2j*cp.pi*d*cp.sin(doa-self.beta)/self.lambda_)
                rwi = channel_ri*wi
                signal_r += rwi

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo.get()
    
    def fscan_echogen(self):
        self.init_simparams("fscan")
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)

        f_tau = cp.fft.fftshift(cp.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fs/self.Nr))
        f_eta = self.fc + (cp.linspace(-self.Na/2,self.Na/2-1,self.Na)*(self.PRF/self.Na))
        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)

        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = cp.arccos(((self.H+self.Re)**2+R0_tar**2-self.Re**2)/(2*(self.H+self.Re)*R0_tar)) ## DoA 信号到达角
            signal_t = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
            signal_r = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)

            ## 发送机
            deci = self.c/(self.f0+self.Kr*(mat_tau-2*R_eta/self.c))
            pr = cp.sin(self.fscan_N*(cp.pi*(self.ttd*self.c-self.fscan_d*cp.sin(doa-self.beta)))/deci) / (cp.sin(cp.pi*(self.ttd*self.c-self.fscan_d*cp.sin(doa-self.beta))/deci))
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tr/2
            # signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            # 接收机,地面目标接收到的信号为所有通道的汇总
            for j in range(self.fscan_N):
                tj = (j-(self.fscan_N-1)/2)*(self.ttd - self.fscan_d*cp.sin(doa-self.beta)/self.c) 
                phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-tj-2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.f0*tj) ## 基带信号
                signal_t += phase_r*Wr
            signal_r = signal_t*pr

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo.get()

    ## 计算每个目标Doa对应的频率
    def fscan_calulate_doaf(self, doa):
        t_inv = self.ttd - self.fscan_d*np.sin(doa-self.beta)/self.c
        f_inv = 1/(t_inv)
        m = np.round(self.f0/f_inv)
        if np.all(m != m[0]):
            print("ttd is not appliable")
        return f_inv*m

    ## 如果为线性调频，计算每个目标Doa对应的快时间
    def fscan_calulate_doaTx(self, doa):
        doaf = self.fscan_calulate_doaf(doa)
        t_inv = 1/doaf
        t_peak = (doaf - (self.f0-self.Kr*self.Tp/2))/(self.Kr)
        interval = np.abs(1/(self.fscan_N*self.Kr*t_inv))
        t_left = t_peak - interval
        t_right = t_peak + interval
        band_width = np.abs(interval*self.Kr*2)

        return t_peak, t_left, t_right, band_width

    def fscan_tpeak(self, doa, f0, ttd):
        t_inv = ttd-self.fscan_d*np.sin(doa-self.beta)/self.c
        m = np.round((f0-self.B/2)*t_inv)
        t_peak = m/(self.Kr*t_inv) - (f0-self.B/2)/self.Kr
        return t_peak
    
    def fscan_f0_doa(self):
        doa  = np.linspace(np.deg2rad(10), np.deg2rad(80), 3000)
        # print(mat_doa)
        f = np.linspace(self.f0-self.B/2, self.f0+self.B/2, 3)
        mat_doa  = np.tile(doa[np.newaxis, :], (3, 1))
        mat_f = np.tile(f[:, np.newaxis], (1, len(doa)))
        prm_tmp = self.c/(mat_f)
        prm = np.sin(self.fscan_N*(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(mat_doa-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(mat_doa-self.beta))/prm_tmp)
        prm = prm**2
        plt.figure()
        plt.plot(np.rad2deg(doa), prm[0,:], 'b', label="f0-B/2")
        plt.plot(np.rad2deg(doa), prm[1,:], 'r', label="f0")
        plt.plot(np.rad2deg(doa), prm[2,:], 'g', label="f0+B/2")
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("PRM")
        plt.title("look angle vs. PR")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/fscan_pr_f0.png", dpi=300)
    
    def fscan_range_estimate(self):
        print("target estimate:")
        target_doa = np.arccos(((self.H+self.Re)**2+self.points_r**2-self.Re**2)/(2*(self.H+self.Re)*self.points_r)) ## DoA 信号到达角
        target_peak,target_left,target_right, target_bw= self.fscan_calulate_doaTx(target_doa)
        print("t_peak(us): ",target_peak*1e6)    
        print("t_left(us): ",(target_left)*1e6)
        print("t_right(us): ",(target_right)*1e6)
        print("target duration(us): ", (target_right-target_left)*1e6)
        print("target bandwidth(Mhz): ", (target_bw)/1e6)

        print("\nswath estimate:")
        doa = self.beta + np.linspace(-self.scan_width/2 + self.scan_width/20, self.scan_width/2, self.Nr)
        self.fscan_swath_estimate(doa)

        print("swath width:", (np.max(target_right)-np.min(target_left))*1e6)
        return target_bw

    def fscan_swath_estimate(self, doa):
        tx_peak, _, _, _ = self.fscan_calulate_doaTx(doa)
        doaR = self.calulate_R0(doa)
        rx_peak = 2*doaR/self.c + tx_peak - 2*np.min(doaR)/self.c

        plt.figure()
        plt.subplot(211)
        plt.plot(np.rad2deg(doa), rx_peak*1e6)
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("Rx peak(us)")
        plt.title("look angle vs. Rx time")
        plt.subplot(212)
        plt.plot(np.rad2deg(doa), tx_peak*self.Kr/1e6)
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("Frequency(MHz)")
        plt.title("look angle vs. Frequency")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/doa_t_f.png", dpi=300)

    def fscan_rasr(self, doa):
        R0 = self.calulate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        f = self.fscan_calulate_doaf(doa)
        # mat_tau = np.linspace(left, right, 4000)
        prm_tmp = self.c/(f)
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
            doam = self.calulate_doa(Rm)
            prm_tmp1 = prm_tmp[window_Rm]
            ## 相控阵调制对模糊的抑制
            ## 当波束扫描到doa时，时间为peak，得出此时的模糊角的相控阵增益
            if m == 0:
                prm = self.fscan_N**2
            else:
                prm = np.sin(self.fscan_N*(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(doam-self.beta)))/prm_tmp1) / np.sin(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(doam-self.beta))/prm_tmp1)
                prm = prm**2
            # prm = np.trapezoid(prm, mat_tau, axis=0)


            ## 单一单元增益
            G_doamr = np.sinc(self.fscan_d*np.sin(doam-self.beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.fscan_d*np.sin(doam-self.beta)/self.lambda_)**2
            incident = np.arccos(((self.Re**2+Rm**2-(self.H+self.Re)**2)/(2*self.Re*Rm)))
            gain = np.zeros(len(doa))
            gain[window_Rm] = prm*G_doamr*G_doamt/(Rm**3*np.sin(incident))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        return 10*np.log10(rasr)

    def fsan_aasr(self):
        pass
    def fsan_nesz(self, doa):
        K = 1.38e-23
        T = 300
        Ln = 0.4 ## 总体系统损耗
        Pav = 1e3
        cons = 256*np.pi**3 * K*T * self.Vr * Ln / (Pav*self.lambda_**3*self.c*self.PRF)
        R0 = self.calulate_R0(doa)
        peak, left, right, bw = self.fscan_calulate_doaTx(doa)
        tp = np.abs(right-left)

        ### 每个doa，相控阵的增益
        pr = self.fscan_N**2


        ## 单一单元增益
        Ar = 0.6*self.Lr*self.La
        Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.fscan_d*np.sin(doa-self.beta)/self.lambda_)**2
        At = 0.6*self.La*self.Lr
        Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.fscan_d*np.sin(doa-self.beta)/self.lambda_)**2

        R_eta = R0/np.cos(self.theta_c)
        incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
        incident = np.pi - incident

        var = R0**3 * bw * np.sin(incident)/(Gr*Gt*tp*pr)
        nesz = cons*var
        return 10*np.log10(nesz)

    def fsan_beam_pattern(self, doa):
        ## 俯仰向相控阵调制
        ### 每个时间，相控阵方向图
        # peak, _, _, _ = self.fscan_calulate_doaTx(doa)
        f = self.fscan_calulate_doaf(doa)
        ### 波束扫描到peak
        mat_f = np.linspace(np.min(f), np.max(f), 4000)[:, np.newaxis] * np.ones([1, len(doa)])
        # print(mat_peak)
        mat_doa  = np.tile(np.linspace(np.deg2rad(10), np.deg2rad(80), len(doa))[np.newaxis, :], (4000, 1))
        # print(mat_doa)
        prm_tmp = self.c/(mat_f)
        prm = np.sin(self.fscan_N*(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(mat_doa-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.fscan_d*np.sin(mat_doa-self.beta))/prm_tmp)
        prm = prm**2

        prm = prm / np.tile(np.max(prm, axis=1)[:, np.newaxis], (1, len(doa)))
        Gr = np.sinc(self.fscan_d*np.sin(mat_doa[1,:]-self.beta)/self.lambda_)**2
    
        plt.figure()
        plt.plot(np.rad2deg(mat_doa[2000,:]), prm[2000,:], 'b')
        plt.plot(np.rad2deg(mat_doa[1,:]), prm[1000,:], 'r')
        plt.plot(np.rad2deg(mat_doa[1,:]), prm[3000,:], 'g')
        plt.plot(np.rad2deg(mat_doa[2000,:]), (Gr**2)**2, 'k')
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("PRM")
        plt.title("look angle vs. PRM")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/fscan_prm.png", dpi=300)




    def get_ttd(self, doa):
        ttd_value = 0
        init_range = 0
        for ttd in np.linspace(1e-12, 4e-9, 1000):
            self.ttd = ttd
            rasr = np.max(self.fscan_rasr(doa))
            if rasr < init_range:
                init_range = rasr
                ttd_value = ttd
        print("ttd: ", ttd_value)
        return ttd_value

    def get_range_IRW(self, ehco):
        max_index = np.argmax(np.abs(np.max(np.abs(ehco), axis=1))) 
        max_value = np.max(np.abs(ehco[max_index,:]))
        half_max = max_value/np.sqrt(2)
        valid = np.abs(ehco[max_index,:]) > half_max
        irw = np.sum(valid)
        irw = irw*self.c/(2*self.Fs)
        return irw, max_index
    
    def get_azimuth_IRW(self, ehco):
        max_index = np.argmax(np.abs(np.max(np.abs(ehco), axis=0))) 
        max_value = np.max(np.abs(ehco[:,max_index]))
        half_max = max_value/np.sqrt(2)
        valid = np.abs(ehco[:,max_index]) > half_max
        irw = np.sum(valid)
        irw = irw*self.Vr/(self.PRF)
        return irw, max_index
    
    def get_pslr(self, target):
        target_np = (target)
        peaks, _ = signal.find_peaks(target_np)

        mainlobe_peak_index = peaks[np.argmax(target_np[peaks])]
        mainlobe_peak_value = target_np[mainlobe_peak_index]

        sidelobe_peaks = np.delete(peaks, np.argmax(target_np[peaks]))
        sidelobe_peak_value = np.max(target_np[sidelobe_peaks])

        pslr = 20 * np.log10(sidelobe_peak_value / mainlobe_peak_value)
        
        return pslr
    
def dbf_simulation():
    dbf_sim = BeamScan()
    strip_echo = dbf_sim.tradition_echogen()
    strip_image = dbf_sim.focus.wk_focus(strip_echo, dbf_sim.R0)
    dbf_echo = dbf_sim.dbf_echogen()
    dbf_image = dbf_sim.focus.wk_focus(dbf_echo, dbf_sim.R0)


    plt.figure()
    plt.subplot(121)
    plt.contour(np.abs(strip_image))
    plt.title("strip mode")
    plt.subplot(122)
    plt.contour(np.abs(dbf_image))
    plt.title("dbf mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/sim_dbf.png", dpi=300)

    
    strip_fft = cp.abs((cp.fft.fft2(strip_image))).get()
    strip_fft = strip_fft/cp.max(cp.max(strip_fft))
    strip_fft = 20*np.log10(strip_fft)  
    dbf_fft = cp.abs((cp.fft.fft2(dbf_image))).get()
    dbf_fft = dbf_fft/np.max(np.max(dbf_fft))
    dbf_fft = 20*np.log10(dbf_fft)
    plt.figure()
    plt.subplot(211)
    plt.imshow(((strip_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(((dbf_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/dbf_fft.png", dpi=300)

    dbf_res, dbf_index = dbf_sim.get_range_IRW(dbf_image)
    strip_res, strip_index = dbf_sim.get_range_IRW(strip_image)
    print("dbf irw: ", dbf_res)
    print("strip irw: ", strip_res)
    print("theoretical irw: ", dbf_sim.c/(2*dbf_sim.B))

    dbf_target = np.abs(dbf_image[dbf_index, :])
    dbf_target = dbf_target/np.max(dbf_target)
    strip_target = np.abs(strip_image[strip_index, :])
    strip_target = strip_target/np.max(strip_target)
    tau = range(dbf_sim.Nr) 
    plt.figure()
    plt.subplot(121)
    plt.plot(tau, 20*np.log10(np.abs(dbf_target)))
    plt.title("dbf mode")
    plt.subplot(122)
    plt.plot(tau, 20*np.log10(cp.abs(strip_target)))
    plt.title("strip mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/range.png", dpi=300)
    print("dbf pslr: ", dbf_sim.get_pslr(dbf_target))
    print("strip pslr: ", dbf_sim.get_pslr(strip_target))

def fscan_simulation():
    fscan_sim = BeamScan()
    echo = fscan_sim.fscan_echogen()
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]
    Be = fscan_sim.fscan_range_estimate()

    plt.figure()
    plt.imshow(abs((echo)), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    image = fscan_sim.focus.wk_focus(echo, fscan_sim.R0)
    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    image_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.array(image)), axes=1)).get()
    image_fft = image_fft/np.max(np.max(image_fft))
    image_fft = 20*np.log10(image_fft)  
    plt.figure()
    plt.imshow(((image_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0, 
               extent=[-fscan_sim.Fs/2e6, fscan_sim.Fs/2e6, -fscan_sim.PRF/2, fscan_sim.PRF/2])
    plt.colorbar()

    plt.xlabel("Range Frequency(MHz)")
    plt.ylabel("Azimuth Frequency(Hz)")
    plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    fscan_range_res, fscan_index = fscan_sim.get_range_IRW(np.abs(image))
    fscan_rtarget = np.abs(image[fscan_index, :])
    fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
    x = range(np.shape(fscan_rtarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_rtarget))
    plt.savefig("../../../fig/dbf/fscan_range.png", dpi=300)

    fscan_azimuth_res, fscan_index = fscan_sim.get_azimuth_IRW(np.abs(image))
    fscan_atarget = np.abs(image[:, fscan_index])
    fscan_atarget = fscan_atarget/np.max(fscan_atarget)
    x = range(np.shape(fscan_atarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_atarget))
    plt.savefig("../../../fig/dbf/fscan_azimuth.png", dpi=300)
    
    print("fscan range irw: ", fscan_range_res)
    print("theoretical range irw: ", fscan_sim.c/(2*Be))
    print("fscan azimuth irw: ", fscan_azimuth_res)
    print("theoretical azimuth irw: ", fscan_sim.La/2)
    print("fscan range pslr: ", fscan_sim.get_pslr(fscan_rtarget))
    print("fscan azimuth pslr: ", fscan_sim.get_pslr(fscan_atarget))

def fscan_ka_estimate():
    fscan_sim = BeamScan()
    fscan_sim.init_simparams("fscan")
    doa = np.linspace(-fscan_sim.scan_width/2+np.deg2rad(0) , fscan_sim.scan_width/2-np.deg2rad(0), 3000) + fscan_sim.beta


    t_peak,_,_,_ = fscan_sim.fscan_calulate_doaTx(np.array([fscan_sim.beta]))
    print("center t_peak: ", t_peak)
  
    # fscan_sim.ttd = fscan_sim.get_ttd(doa)
    peak, left, right, bw = fscan_sim.fscan_calulate_doaTx(doa)
    print("beam scan from {} us to {} us".format(np.min(peak)*1e6, np.max(peak)*1e6))
    # fscan_sim.ttd = fscan_sim.get_ttd(doa)
    nesz = fscan_sim.fsan_nesz(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), nesz)
    plt.xlabel("look angle(degree)")
    plt.ylabel("NESZ(dB)")
    plt.grid()
    plt.title("look angle vs. NESZ")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_nesz.png", dpi=300)

    rasr = fscan_sim.fscan_rasr(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), rasr)
    plt.xlabel("look angle(degree)")
    plt.ylabel("RASR(dB)")
    plt.grid()
    plt.title("look angle vs. RASR")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_rasr.png", dpi=300)

    fscan_sim.fsan_beam_pattern(doa)
    fscan_sim.fscan_swath_estimate(doa)





if __name__ == '__main__':
    fscan_ka_estimate()
    # fscan_simulation()
    # dbf_simulation()





