## 

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
import scipy.optimize as optimize
sys.path.append(r"../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
from mpl_toolkits.mplot3d import Axes3D
import logging
import colorlog
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

cp.cuda.Device(3).use()

class BeamScan:
    def __init__(self):
        self.H = 519e3                              #卫星高度  
        self.Re = 6371.39e3                         #地球半径
        self.beta = np.deg2rad(25)                  #天线安装角
        self.c = 299792458                          #光速           
        self.Tp = 40e-6                            #脉冲宽度                        
        self.f0 = 35e+09                            #载频                     
        self.PRF = 1720                             #PRF                         
        self.fc = 0                             #多普勒中心频率
        self.K = 1.38e-23                           #玻尔兹曼常数
        self.T = 300                                #温度
        self.Ln = 0.4                               ## 总体系统损耗
        self.dr = 2                               ## 斜距精度
        self.Gravitational = 6.67e-11;              #万有引力常量
        self.EarthMass = 6e24;                      #地球质量(kg)
        self.Vr = np.sqrt(self.Gravitational*self.EarthMass/(self.Re + self.H))                      
        self.lambda_= self.c/self.f0
        self.theta_c = np.arcsin(self.fc*self.lambda_/(2*self.Vr))
        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(self.beta)/self.Re)
        tmp_angle = tmp_angle - self.beta
        self.R0 = self.Re*np.sin(tmp_angle)/np.sin(self.beta)
        self.La = 10
        self.Ta = 1
        self.log = self.get_logger()
        self.ground_width = 50e3
        self.scan_width = self.calculate_scanwidth(self.ground_width)
        self.Tr = self.calculate_re_window()
        self.d = self.calculate_d(self.scan_width)
        self.d = 0.09
        self.N = 10
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+np.array([-9000, 0, 0, 0, 9000])
        self.points_a = np.array([0, 3000, 0, -3000, 0])
        self.Ba = 2*0.886*self.Vr*np.cos(self.theta_c)/self.La 
        # self.log.info("receive window length Tr: {}".format(self.Tr*1e6))   
        # self.log.info("R_width: {}".format(self.Tr*self.c/2)) 
        self.log.info("Doppler bandwidth: {}".format(0.886*2*self.Vr*np.cos(self.theta_c)/self.La))
        # self.log.info("Doppler center: {}".format(self.fc))
    def set_groundwidth(self, ground_width):
        self.ground_width = ground_width
        self.scan_width = self.calculate_scanwidth(self.ground_width)
        self.Tr = self.calculate_re_window()
    
    def set_scanwidth(self, scan_width):
        self.scan_width = scan_width
        self.scan_left = self.beta - scan_width/2
        self.scan_right = self.beta + scan_width/2
        self.Tr = self.calculate_re_window()

    def get_logger(self, level=logging.INFO):
        # 创建logger对象
        logger = logging.getLogger()
        logger.setLevel(level)
        # 创建控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # 定义颜色输出格式
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        # 将颜色输出格式添加到控制台日志处理器
        console_handler.setFormatter(color_formatter)
        # 移除默认的handler
        for handler in logger.handlers:
            logger.removeHandler(handler)
        # 将控制台日志处理器添加到logger对象
        logger.addHandler(console_handler)
        return logger

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
    
    def calculate_R0(self, look_angle):
        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(look_angle)/self.Re)
        tmp_angle = tmp_angle - look_angle
        R0 = self.Re*np.sin(tmp_angle)/np.sin(look_angle)
        return R0
    
    def slant2ground(self, R):
        ground_angle = np.arccos(((self.H+self.Re)**2+self.Re**2-R**2)/(2*(self.H+self.Re)*self.Re))
        return ground_angle*self.Re

    def calculate_doa(self, R0):
        incident = np.arccos((self.Re**2+R0**2-(self.H+self.Re)**2)/(2*self.Re*R0)*(R0>self.H))
        doa = np.arcsin(self.Re*np.sin(incident)/(self.H+self.Re))
        return doa

    def calculate_scanwidth(self, ground_width):
        ground_angle = ground_width/self.Re

        def func(x):
            g_left = np.arcsin((self.H+self.Re)/(self.Re/np.sin(self.beta-x))) - (self.beta-x)
            g_right = np.arcsin((self.H+self.Re)/(self.Re/np.sin(self.beta+x))) - (self.beta+x)
            return g_right-g_left-ground_angle
        
        equation = lambda x: func(x)

        solu = optimize.fsolve(equation, 0)

        self.scan_left = self.beta-solu[0]
        self.scan_right = self.beta+solu[0]

        self.log.info("scan left: {}    scan right:{}".format(np.rad2deg(self.scan_left), np.rad2deg(self.scan_right)))

        return self.scan_right-self.scan_left
    
        
    def calculate_d(self, scan_width):
        return self.lambda_/(np.sin(scan_width/2) - np.sin(-scan_width/2))


    ## 计算接收窗口大小
    def calculate_re_window(self):
        R_min = self.calculate_R0(self.scan_left)
        R_max = self.calculate_R0(self.scan_right)
        # print("R_min: {}, R_max: {}".format(R_min-self.R0, R_max-self.R0))
        return (R_max-R_min)*2/self.c

    def upsample(self, data, N):
        Na, Nr = cp.shape(data)
        data_fft = cp.fft.fftshift(cp.fft.fft2(data))
        tmp = cp.zeros((N[0]*Na, N[1]*Nr), dtype=complex)
        tmp[N[0]*Na//2-Na/2:N[0]*Na//2+Na/2, N[1]*Nr//2-Nr/2:N[1]*Nr//2+Nr/2] = data_fft
        data_up = cp.fft.ifft2(cp.fft.ifftshift(tmp))
        return data_up.get()

    def get_range_IRW(self, ehco, uprate):
        max_index = np.argmax(np.abs(np.max(np.abs(ehco), axis=1))) 
        max_value = np.max(np.abs(ehco[max_index,:]))
        half_max = max_value/np.sqrt(2)
        valid = np.abs(ehco[max_index,:]) > half_max
        irw = np.sum(valid)
        irw = irw*self.c/(2*uprate*self.Fs)
        return irw, max_index
    
    def get_azimuth_IRW(self, ehco, uprate):
        max_index = np.argmax(np.abs(np.max(np.abs(ehco), axis=0))) 
        max_value = np.max(np.abs(ehco[:,max_index]))
        half_max = max_value/np.sqrt(2)
        valid = np.abs(ehco[:,max_index]) > half_max
        irw = np.sum(valid)
        irw = irw*self.Vr/(self.PRF*uprate)
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
    
    def get_islr(self, target):
        target_np = np.abs(target)**2
        total_power = np.sum(target_np)

        peaks, _ = signal.find_peaks(target_np)
        mainlobe_peak_index = peaks[np.argmax(target_np[peaks])]

        low_peaks,_ = signal.find_peaks(-target_np)
        left_peak = low_peaks[low_peaks < mainlobe_peak_index].max()
        right_peak = low_peaks[low_peaks > mainlobe_peak_index].min()
        mainlobe_power = np.sum(target_np[left_peak:right_peak])
        sidelobe_power = total_power - mainlobe_power
        islr = 10 * np.log10(sidelobe_power / mainlobe_power)
        
        return islr

    def zebra_diagram(self, prf, tau_rp):

        frac_min = -(tau_rp + self.Tp) * prf  # 发射约束
        frac_max = tau_rp * prf

        int_min = int(np.min(np.floor(2 * self.H * prf / self.c)))
        int_max = int(np.max(np.ceil(2 * np.sqrt((self.Re+self.H)**2 - (self.Re)**2) * prf / self.c)))

        plt.figure("PRF selection")

        # 遍历尽可能多的干扰
        for i in range(int_max + 1):
            R1 = (i + frac_min) * self.c / (2 * prf)
            gamma_cos = (R1**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R1 * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R1 * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'b')

            Rn = (i + frac_max) * self.c / (2 * prf)
            gamma_cos = (Rn**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * Rn * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(Rn * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            plt.plot(prf, gamma2, 'b')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.1, color='b')

        # 星下点干扰
        for i in range(6):
            R = (2 * self.H / self.c + i / prf) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'r')

            R = (2 * self.H / self.c + i / prf + self.Tp * 2) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            plt.plot(prf, gamma2, 'r')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.1, color='r')
        
        plt.vlines(x=1720, ymin = np.rad2deg(self.scan_left), ymax = np.rad2deg(self.scan_right), color='g')
        plt.grid()
        plt.xlabel("PRF/Hz")
        plt.ylabel("look angle/°")
        plt.xlim([1.3e3, 3e3])
        plt.ylim([20, 40])
        plt.savefig("../../../fig/dbf/zebra_diagram.png", dpi=300)


    def dot_estimate(self, image, area):
        plt.figure(figsize=(12, 8))
        # Convert numerical values to corresponding letters
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        numerical_values = np.arange(len(alphabet))
        letter_mapping = dict(zip(numerical_values, alphabet))
        for i in range(self.points_n):
            max_index = np.unravel_index(np.argmax(np.abs(image)), image.shape)
            print("Position of the maximum point in the image:", max_index)
            target = image[max_index[0]-area[0]:max_index[0]+area[0], max_index[1]-area[1]:max_index[1]+area[1]]
            uprate = 16
            target_up = self.upsample(cp.array(target), (uprate, uprate))
            image_show = np.abs(target_up)/np.max(np.max(np.abs(target_up)))
            image_show = 20*np.log10(image_show)

            target = target_up
            x = np.array([-area[1]/2, area[1]/2])
            dr = x*self.c/(2*self.Fs)
            y =  np.array([-area[0]/2, area[0]/2])
            da = y*self.Vr/(self.PRF)


            plt.subplot(3, self.points_n, i+1)
            plt.imshow(image_show, aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]], vmin=-60, vmax=0)
            plt.ylabel("Azimuth(m)")
            plt.xlabel("Range(m)")
            colorbar = plt.colorbar()
            colorbar.ax.set_title("dB")
            plt.title("({})".format(letter_mapping[i+1]))


            fscan_range_res, fscan_index = self.get_range_IRW(np.abs(target), uprate)
            fscan_rtarget = np.abs(target[fscan_index, :])
            fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
            x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))

            plt.subplot(3, self.points_n, self.points_n + i+1)
            plt.plot(x_dr, 20*np.log10(fscan_rtarget))
            plt.grid()
            plt.ylim(-60, 0)
            plt.xlabel("Range(m)")
            plt.ylabel("Magnitude(dB)")
            plt.title("({})".format(letter_mapping[self.points_n + i+1]))


            fscan_azimuth_res, fscan_index = self.get_azimuth_IRW(np.abs(target), uprate)
            fscan_atarget = np.abs(target[:, fscan_index])
            fscan_atarget = fscan_atarget/np.max(fscan_atarget)
            x_da = np.linspace(da[0], da[1], len(fscan_atarget))

            plt.subplot(3, self.points_n, 2*self.points_n + i+1)
            plt.plot(x_da, 20*np.log10(fscan_atarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("Azimuth(m)")
            plt.ylabel("Magnitude(dB)")
            plt.title("({})".format(letter_mapping[2*self.points_n + i+1]))

            print("fscan range irw: ", fscan_range_res)
            print("fscan azimuth irw: ", fscan_azimuth_res)
            print("fscan range pslr: ", self.get_pslr(fscan_rtarget))
            print("fscan azimuth pslr: ", self.get_pslr(fscan_atarget))
            print("fscan range islr: ", self.get_islr(fscan_rtarget))
            print("fscan azimuth islr: ", self.get_islr(fscan_atarget))

            image[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2] = 0
        
        plt.tight_layout()

        plt.savefig("../../../fig/dbf/dot_estimate.png", dpi=300)

    
class StripMode(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.d
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.B = self.c / (2*self.dr)  # 信号带宽
        self.Fs = self.B*1.2                            #采样率   
        self.Kr = -self.B/self.Tp 
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)


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
    
    def nesz(self, doa, Pu, N):
        Pav = N*Pu
        cons = 256*np.pi**3 * self.K*self.T * self.Vr * self.Ln / (Pav*self.lambda_**3*self.c*self.PRF)
        R0 = self.calculate_R0(doa)

        ## 单一单元增益
        Ar = 0.6*self.Lr*self.La
        Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2
        At = 0.6*self.La*self.Lr
        Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2

        R_eta = R0/np.cos(self.theta_c)
        incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
        incident = np.pi - incident

        var = R0**3 * self.B * np.sin(incident)/(Gr*Gt*self.Tp)
        nesz = cons*var
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
            incident = np.arccos(((self.Re**2+Rm**2-(self.H+self.Re)**2)/(2*self.Re*Rm)))
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


    
class DBF_SCORE(BeamScan):
    def __init__(self):
        super().__init__()
        self.dbf_beam_width = (0.886*self.lambda_/self.d)
        self.Lr = self.d
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.B = self.c / (2*self.dr)               # 信号带宽
        self.Fs = self.B*1.2                            #采样率   
        self.Kr = -self.B/self.Tp 
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)

        
    def echogen(self):
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
            for j in range(self.N):
                d = (j-(self.N-1)/2)*self.d
                ti = t0-d*cp.sin(doa-self.beta)/self.c
                Wr = cp.abs(mat_tau-ti)<self.Tp/2
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
    
    def rasr(self, doa):
        R0 = self.calculate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        # mat_tau = np.linspace(left, right, 4000)
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
            ## 相控阵调制对模糊的抑制
            ## 当波束扫描到doa时，时间为peak，得出此时的模糊角的相控阵增益
            # prm = np.zeros(len(doam), dtype=np.complex128)
            # for i in range(self.N):
            #     prm += np.exp(2j*np.pi*i*self.d*(np.sin(doam-doa[window_Rm]))/self.lambda_)
            if m == 0:
                prm = self.N
            else:
                prm = np.sin(self.N*np.pi*self.d*np.sin(doam-doa[window_Rm])/self.lambda_) / np.sin(np.pi*self.d*np.sin(doam-doa[window_Rm])/self.lambda_)
            prm = np.abs(prm)**2 ## 功率峰值

            ## 单一单元增益
            G_doamr = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2
            incident = np.arccos(((self.Re**2+Rm**2-(self.H+self.Re)**2)/(2*self.Re*Rm)))
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(prm*G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        return 10*np.log10(rasr)

    
    def nesz(self, doa, Pu):
        ### 每个doa，相控阵的增益
        pr = self.N ## 单程，接收使用相控阵
        Pav = self.N*Pu
        cons = 256*np.pi**3 * self.K*self.T * self.Vr * self.Ln / (Pav*self.lambda_**3*self.c*self.PRF)
        R0 = self.calculate_R0(doa)

        ## 单一单元增益
        Ar = 0.6*self.Lr*self.La
        Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2
        At = 0.6*self.La*self.Lr
        Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2

        R_eta = R0/np.cos(self.theta_c)
        incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
        incident = np.pi - incident

        var = R0**3 * self.B * np.sin(incident)/(Gr*Gt*self.Tp*pr)
        nesz = cons*var
        return 10*np.log10(nesz)
    
    def resolution(self, doa):
        return np.ones(len(doa))*self.c/(2*self.B)

    
class Fscan(BeamScan):
    def __init__(self):
        super().__init__()
        self.Lr = self.N*self.d
        self.ttd =  -2.629e-9
        self.B = 400e6                             #信号带宽
        self.Fs = self.B*1.2                            #采样率 
        self.Kr = -np.sign(self.ttd)*self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.d)
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Nr = int(np.ceil(self.Fs*self.Tr))
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)

    def set_B(self, B):
        self.B = B
        self.Kr = B/self.Tp
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)
        self.Nr = int(np.ceil(self.Fs*self.Tr))

    def set_d(self, d):
        self.Lr = self.N*d
        self.d = d
        self.fscan_beam_width = (0.886*self.lambda_/self.d)
        if(self.fscan_beam_width < self.scan_width):
            self.log.warning("fscan beam width is smaller than scan width!")

    def set_N(self, N):
        self.N = N
        self.Lr = self.N*self.d    

    def set_f0(self, f0):
        self.f0 = f0
        self.lambda_ = self.c/self.f0
        self.fscan_beam_width = (0.886*self.lambda_/self.d) 

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
        S_echo = S_echo.get()
        return S_echo
    
    def fscan_residual_focus(self, image):
        [_, Nr] = np.shape(image)
        doa_target = self.calculate_doa(cp.array(self.points_r))
        doaf = (self.calculate_doaf(doa_target) - self.f0)
        f_interval = cp.abs(1/(self.N*(self.ttd-self.d*cp.sin(doa_target-self.beta)/self.c)))
        f_left_index = cp.clip(cp.floor((doaf - f_interval + self.Fs/2)/(self.Fs/Nr)), 0, Nr-1)
        f_right_index = cp.clip(cp.ceil((doaf + f_interval + self.Fs/2)/(self.Fs/Nr)), 0, Nr-1)
        image_fft = cp.fft.fftshift(cp.fft.fft2(image), axes=1)
        image_feta_new = cp.zeros((self.Na, Nr), dtype=cp.complex128)
        for i in tqdm(range(self.points_n)):
            f_left = int(f_left_index[i])
            f_right = int(f_right_index[i])
            mask = cp.zeros_like(image_fft, dtype=cp.complex128)
            mask[:, f_left:f_right] = 1
            image_feta_new += cp.fft.ifft(cp.fft.ifftshift(image_fft*mask, axes=1), axis=1)
        image_new = cp.fft.ifft(image_feta_new, axis=0)
        return image_new.get()

    
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
    
    def calculate_m(self, doa):
        t_inv = self.ttd - self.d*np.sin(doa-self.beta)/self.c
        f_inv = np.abs(1/(t_inv))
        m = np.ceil((self.f0-self.B/2)/(f_inv))
        return m

    
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
        ground = self.calculate_R0(target_doa)
        ground = self.slant2ground(ground)
        ground = ground - ground[0]
        self.log.info("target ground range(km): {}".format(ground/1e3))
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
        plt.xlabel("look angle/°")
        plt.ylabel("Rx peak(us)")
        plt.title("look angle vs. Rx time")
        plt.subplot(212)
        plt.scatter(np.rad2deg(doa)[win1], (doaf1-self.f0)/1e6)
        plt.scatter(np.rad2deg(doa)[win2],  (doaf2-self.f0)/1e6)
        plt.grid()
        plt.xlabel("look angle/°")
        plt.ylabel("Frequency(MHz)")
        plt.title("look angle vs. Frequency")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/doa_t_f.png", dpi=300)

    def get_ttd_rasr(self, doa):
        ttd_value = self.ttd
        init_range = 0
        for ttd in np.arange(-4e-9, 4e-9, 1e-12):
            self.set_ttd(ttd)
            if self.ttd_judge(doa) == False:
                continue
            rasr = np.max(self.rasr(doa))
            if rasr < init_range:
                init_range = rasr
                ttd_value = ttd
        if init_range == 0:
            self.log.warning("no ttd is suitable in such condition")
        self.log.info("ttd: {}".format(ttd_value))
        return ttd_value

    def get_ttd_bandwidth(self, doa):
        ttd_value = self.ttd
        init_range = self.B*2
        for ttd in np.arange(-10e-9, 10e-9, 1e-12):
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
            prm = prm**2 ## 功率峰值
            prm = prm**2 ## 双程，发送接收均使用频扫相控阵
            # prm = np.trapezoid(prm, mat_tau, axis=0)


            ## 单一单元增益
            G_doamr = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.d*np.sin(doam-self.beta)/self.lambda_)**2
            incident = np.arccos(((self.Re**2+Rm**2-(self.H+self.Re)**2)/(2*self.Re*Rm)))
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(prm*G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        return 10*np.log10(rasr)
    
    
    def aasr(self):
        pass
    
    def resolution(self, doa):
        _, _, _, bw = self.calculate_doaTx(doa)
        res = self.c/(2*bw)
        return res

    def nesz(self, doa, Pu):
        Pav = Pu
        cons = 256*np.pi**3 * self.K*self.T * self.Vr * self.Ln / (Pav*self.lambda_**3*self.c*self.PRF)
        R0 = self.calculate_R0(doa)
        peak, left, right, bw = self.calculate_doaTx(doa)
        tp = np.abs(right-left)

        ### 每个doa，相控阵的增益
        pr = self.N**2  ## 双程，发送接收均使用频扫相控阵

        ## 单一单元增益
        Ar = 0.6*self.Lr*self.La
        Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2
        At = 0.6*self.La*self.Lr
        Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-self.beta)/self.lambda_)**2

        R_eta = R0/np.cos(self.theta_c)
        incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
        incident = np.pi - incident

        var = R0**3 * bw * np.sin(incident)/(Gr*Gt*tp*pr)
        nesz = cons*var
        return 10*np.log10(nesz)
    
    def beam_pattern(self, doa):
        ## 俯仰向相控阵调制
        ### 每个时间，相控阵方向图
        # peak, _, _, _ = self.calculate_doaTx(doa)
        f = self.calculate_doaf(doa)
        f = np.linspace(np.min(f), np.max(f), 4000)
        ### 波束扫描到peak
        mat_f = f[:, np.newaxis] * np.ones([1, len(doa)])
        # self.log.info(mat_peak)
        mat_doa  = np.tile(np.linspace(np.deg2rad(10), np.deg2rad(80), len(doa))[np.newaxis, :], (4000, 1))
        # self.log.info(mat_doa)
        prm_tmp = self.c/(mat_f)
        prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(mat_doa-self.beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(mat_doa-self.beta))/prm_tmp)
        prm = prm**2

        prm = prm / np.tile(np.max(prm, axis=1)[:, np.newaxis], (1, len(doa)))
        Gr = np.sinc(self.d*np.sin(mat_doa[1,:]-self.beta)/self.lambda_)**2
    
        plt.figure()

        plt.plot(np.rad2deg(mat_doa[2000,:]), prm[2000,:], label="f={} GHz".format(f[0]/1e9))
        # plt.plot(np.rad2deg(mat_doa[2000,:]), prm[0,:], label="f={} GHz".format(f[0]/1e9))
        # plt.plot(np.rad2deg(mat_doa[2000,:]), prm[-1,:], label="f={} GHz".format(f[-1]/1e9))
        plt.plot(np.rad2deg(mat_doa[2000,:]), (Gr**2)**2, 'k', label = 'ant unit')
        plt.legend()
        plt.grid()
        plt.xlabel("look angle/°")
        plt.ylabel("Normalized gain")
        # plt.title("Antenna Orientation Diagram")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/fscan_prm.png", dpi=300)


    
def dbf_simulation():
    dbf_sim = DBF_SCORE()
    strip_sim = StripMode()
    strip_echo = strip_sim.echogen()
    strip_image = strip_sim.focus.wk_focus(strip_echo, dbf_sim.R0)
    dbf_echo = dbf_sim.echogen()
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
    dbf_sim.log.info("dbf irw: ", dbf_res)
    strip_sim.log.info("strip irw: ", strip_res)
    dbf_sim.log.info("theoretical irw: ", dbf_sim.c/(2*dbf_sim.B))

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
    dbf_sim.log.info("dbf pslr: ", dbf_sim.get_pslr(dbf_target))
    strip_sim.log.info("strip pslr: ", dbf_sim.get_pslr(strip_target))

def fscan_simulation():
    fscan_sim = Fscan()
    echo = fscan_sim.echogen()
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]
    Be = fscan_sim.range_estimate()

    # plt.figure()
    # plt.imshow(abs((echo)), aspect='auto', cmap='jet')
    # plt.colorbar()
    # plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    image = fscan_sim.focus.wk_focus(echo, fscan_sim.R0)
    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    # image_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.array(image)), axes=1)).get()
    # image_fft = image_fft/np.max(np.max(image_fft))
    # image_fft = 20*np.log10(image_fft)  
    # plt.figure()
    # plt.imshow(((image_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0, 
    #            extent=[-fscan_sim.Fs/2e6, fscan_sim.Fs/2e6, -fscan_sim.PRF/2, fscan_sim.PRF/2])
    # plt.colorbar()

    # plt.xlabel("Range Frequency(MHz)")
    # plt.ylabel("Azimuth Frequency(Hz)")
    # plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    # Find the position of the maximum point in the image
    max_index = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    fscan_sim.log.info("Position of the maximum point in the image:", max_index)
    target = image[max_index[0]-100:max_index[0]+100, max_index[0]-100:max_index[0]+100]
    target_up = fscan_sim.upsample(cp.array(target), (16, 16))
    image_show = np.abs(target)/np.max(np.max(np.abs(target)))
    image_show = 20*np.log10(image_show)
    
    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_upimage.png", dpi=300)

    target = target_up

    fscan_range_res, fscan_index = fscan_sim.get_range_IRW(np.abs(target))
    fscan_rtarget = np.abs(target[fscan_index, :])
    fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
    x = range(np.shape(fscan_rtarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_rtarget))
    plt.savefig("../../../fig/dbf/fscan_range.png", dpi=300)

    fscan_azimuth_res, fscan_index = fscan_sim.get_azimuth_IRW(np.abs(target))
    fscan_atarget = np.abs(target[:, fscan_index])
    fscan_atarget = fscan_atarget/np.max(fscan_atarget)
    x = range(np.shape(fscan_atarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_atarget))
    plt.savefig("../../../fig/dbf/fscan_azimuth.png", dpi=300)
    
    fscan_sim.log.info("fscan range irw: {}".format(fscan_range_res))
    fscan_sim.log.info("theoretical range irw: {}".format(fscan_sim.c/(2*Be)))
    fscan_sim.log.info("fscan azimuth irw: {}".format(fscan_azimuth_res))
    fscan_sim.log.info("theoretical azimuth irw: {}".format(fscan_sim.La/2))
    fscan_sim.log.info("fscan range pslr: {}".format(fscan_sim.get_pslr(fscan_rtarget)))
    fscan_sim.log.info("fscan azimuth pslr: {}".format(fscan_sim.get_pslr(fscan_atarget)))


def fscan_estimate():
    fscan_sim = Fscan()
    # fscan_sim.set_B(4e9)
    # fscan_sim.set_f0(35e9)
    # fscan_sim.set_d(0.030)
    # fscan_sim.set_N(10)
    fscan_sim.log.info("fscan beam width: {}".format(np.rad2deg(fscan_sim.fscan_beam_width)))
    fscan_sim.log.info("vr {}".format(fscan_sim.Vr))
    fscan_sim.log.info("max N {}".format(fscan_sim.dr*4*fscan_sim.f0/fscan_sim.c))
    fscan_sim.log.info("subaperture length {}".format(fscan_sim.d))
    # fscan_sim.dr = 2
    prf = np.linspace(500, 4e3, 3500)
    fscan_sim.zebra_diagram(prf, 1e-6)
    strip_sim = StripMode()
    dbf_sim = DBF_SCORE()
    doa = np.linspace(fscan_sim.scan_left, fscan_sim.scan_right, 3000)
    fscan_sim.set_ttd(fscan_sim.get_ttd_bandwidth(doa))

    t_peak,_,_,_ = fscan_sim.calculate_doaTx(np.array([fscan_sim.beta]))
    fscan_sim.log.info("center t_peak: {}".format(t_peak))
  
    peak, left, right, bw = fscan_sim.calculate_doaTx(doa)
    fscan_sim.log.info("beam scan from {} us to {} us".format(np.min(peak)*1e6, np.max(peak)*1e6))

    pu = 1e2
    nesz_fscan = fscan_sim.nesz(doa, pu)
    nesz_strip = strip_sim.nesz(doa, pu, fscan_sim.N)
    nesz_dbf = dbf_sim.nesz(doa, pu)
    plt.figure()
    plt.plot(np.rad2deg(doa), nesz_fscan, label="F-SCAN")
    plt.plot(np.rad2deg(doa), nesz_strip, label="Stripmap")
    plt.plot(np.rad2deg(doa), nesz_dbf, label="SCORE")
    plt.legend()
    plt.xlabel("look angle/°")
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/nesz.png", dpi=300)

    rasr_fscan = fscan_sim.rasr(doa)
    rasr_strip = strip_sim.rasr(doa)
    rasr_dbf = dbf_sim.rasr(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), rasr_fscan, label="F-SCAN")
    plt.plot(np.rad2deg(doa), rasr_strip, label="stripmap")
    plt.plot(np.rad2deg(doa), rasr_dbf, label="SCORE")
    plt.legend()
    plt.xlabel("look angle/°")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/rasr.png", dpi=300)

    res_fscan = fscan_sim.resolution(doa)
    res_strip = strip_sim.resolution(doa)
    res_dbf = dbf_sim.resolution(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), res_fscan, label="F-SCAN")
    plt.plot(np.rad2deg(doa), res_strip, label="stripmap")
    plt.plot(np.rad2deg(doa), res_dbf, label="SCORE")
    plt.legend()
    plt.xlabel("look angle/°")
    plt.ylabel("Resolution/m")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/resolution.png", dpi=300)
    
    fscan_sim.swath_estimate(doa)
    fscan_sim.beam_pattern(doa)

def fscan_ant(fscan: Fscan):
    doa = np.linspace(fscan.scan_left , fscan.scan_right, 3000)
    N = np.arange(2, 50, 1)
    N_valid = []
    rasr = []
    nesz = []
    res_max = []
    res_min = []
    bw = []
    for n in N:
        fscan.set_N(n)
        fscan.set_ttd(fscan.get_ttd_bandwidth(doa))
        if fscan.ttd_judge(doa) == False:
            continue
        # doa_sin = np.sin(doa-fscan.beta)
        # the_m = 4*fscan.dr/(fscan.N*fscan.lambda_)
        # the_d = np.mean((the_m*fscan.c*fscan.B-fscan.c*(fscan.f0+fscan.B/2))/((np.max(doa_sin)-np.min(doa_sin))*(fscan.f0**2-fscan.B**2/4)))
        # fscan.log.debug("the d {} N {}".format(the_d, fscan.N))
        N_valid.append(n)
        f = fscan.calculate_doaf(doa)
        band_width = np.max(f) - np.min(f)
        bw.append(band_width)
        rasr.append(np.max(fscan.rasr(doa)))
        nesz.append(np.max(fscan.nesz(doa, 1e5)))
        res_max.append(np.max(fscan.resolution(doa)))
        res_min.append(np.min(fscan.resolution(doa)))
    rasr = np.array(rasr)
    nesz = np.array(nesz)
    res_max = np.array(res_max)
    res_min = np.array(res_min)
    N_valid = np.array(N_valid)
    bw = np.array(bw)

    return N_valid, rasr, nesz, res_max, bw


def fscan_ant_d_estimate():
    fscan = Fscan()
    fscan.set_B(2e9)
    fscan.set_f0(34e9)
    fscan.set_scanwidth(np.deg2rad(10))
    fscan.dr= 0.2
    fscan.set_d(0.001)
    N_valid1, rasr1, nesz1, _ , bw1 = fscan_ant(fscan)
    fscan.set_d(0.005)
    N_valid2, rasr2, nesz2, _ , bw2 = fscan_ant(fscan)
    fscan.set_d(0.01)
    N_valid3, rasr3, nesz3, _ , bw3 = fscan_ant(fscan)
    plt.figure()
    plt.plot(N_valid1, rasr1, label="d = {}m".format(0.001), marker='o')
    plt.plot(N_valid2, rasr2, label="d = {}m".format(0.005), marker='^')
    plt.plot(N_valid3, rasr3, label="d = {}m".format(0.01), marker='s')
    plt.xlabel("N")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_drasr.png", dpi=300)
    plt.figure()
    plt.plot(N_valid1, nesz1, label="d = {}m".format(0.001), marker='o')
    plt.plot(N_valid2, nesz2, label="d = {}m".format(0.005), marker='^')
    plt.plot(N_valid3, nesz3, label="d = {}m".format(0.01), marker='s')
    plt.xlabel("N")
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_dnesz.png", dpi=300)
    plt.figure()
    plt.plot(N_valid1, bw1/1e6, label="d = {}m".format(0.001), marker='o')
    plt.plot(N_valid2, bw2/1e6, label="d = {}m".format(0.005), marker='^')
    plt.plot(N_valid3, bw3/1e6, label="d = {}m".format(0.01), marker='s') 
    plt.xlabel("N")
    plt.ylabel("Bandwidth/MHz")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_dbw.png", dpi=300)


def fscan_ant_f_estimate():
    fscan = Fscan()
    fscan.set_B(2e9)
    fscan.set_d(0.01)
    fscan.set_f0(9.8e9)
    fscan.set_scanwidth(np.deg2rad(10))
    fscan.dr = 0.2
    N_valid1, rasr1, nesz1, _ , bw = fscan_ant(fscan)
    fscan.set_f0(32e9)
    N_valid2, rasr2, nesz2, _, bw = fscan_ant(fscan)

    plt.figure()
    plt.plot(N_valid1, rasr1, label="f = 9.8GHz", marker='^')
    plt.plot(N_valid2, rasr2, label="f = 32GHz", marker='o')
    plt.xlabel("N")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_rasr.png", dpi=300)
    plt.figure()
    plt.plot(N_valid1, nesz1, label="f = 9.8GHz", marker='^')
    plt.plot(N_valid2, nesz2, label="f = 32GHz", marker='o')
    plt.xlabel("N")
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_nesz.png", dpi=300)



def fscan_carrier_estimate():
    fscan = Fscan()
    fscan.set_scanwidth(np.deg2rad(10))
    fscan.set_d(0.03)
    fscan.set_B(1e9)
    fscan.set_N(10)
    fscan.dr = 0.2
    fscan.set_ttd(fscan.get_ttd_bandwidth(np.deg2rad(10)))


    carrier = np.arange(5e9, 40e9, 5e8)
    scan = []

    for c in carrier:
        fscan.set_f0(c)
        scan_left = np.arcsin((fscan.ttd*fscan.c-fscan.f0*fscan.ttd*fscan.c/(fscan.f0+fscan.B/2))/fscan.d)
        scan_right = np.arcsin((fscan.ttd*fscan.c-fscan.f0*fscan.ttd*fscan.c/(fscan.f0-fscan.B/2))/fscan.d)
        scan_width = np.abs(scan_right - scan_left)
        scan.append(scan_width)

    scan = np.array(scan)

    plt.figure()
    plt.plot(carrier/1e9, np.rad2deg(scan), marker='o')
    plt.xlabel("Carrier Frequency/GHz")
    plt.ylabel("look angle width/°")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_carrier_scan.png", dpi=300)

def unsuitable():
    fscan = Fscan()
    fscan.set_B(2e9)
    fscan.set_d(0.01)
    fscan.set_groundwidth(50e3)
    doa = np.linspace(fscan.scan_left, fscan.scan_right, 3000)
    fscan.set_ttd(fscan.d*np.sin(fscan.beta)/fscan.c)
    fscan.beam_pattern(doa)
    fscan.log.info("ttd: {}".format(fscan.ttd))

if __name__ == '__main__':
    # fscan_estimate()
    fscan_carrier_estimate()
    # unsuitable()
    
    # fscan_ant_d_estimate()
    # fscan_ant_f_estimate()
    # fscan_simulation()
    # dbf_simulation()





