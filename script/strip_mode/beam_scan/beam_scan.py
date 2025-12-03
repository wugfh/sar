## 

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
import scipy.optimize as optimize
sys.path.append(r"../../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
from mpl_toolkits.mplot3d import Axes3D
import logging
import colorlog
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


class BeamScan:
    def __init__(self):
        self.H = 519e3                              #卫星高度  
        self.Re = 6371.39e3                         #地球半径
        self.beta = np.deg2rad(62.2)                  #天线安装角
        self.c = 299792458                          #光速           
        self.Tp = 2e-6                            #脉冲宽度                        
        self.f0 = 35e+09                            #载频                     
        self.PRF = 200                            #PRF                         
        self.fc = 0                             #多普勒中心频率
        self.K = 1.38e-23                           #玻尔兹曼常数
        self.T = 320                                #温度
        self.Ln = 2.5                              ## 总体系统损耗
        self.dr = 0.2                               ## 斜距精度
        self.Gravitational = 6.67e-11;              #万有引力常量
        self.EarthMass = 6e24;                      #地球质量(kg)
        self.Vr = np.sqrt(self.Gravitational*self.EarthMass/(self.Re + self.H))        
        # self.Vr = 70
        self.Vg = self.Vr*self.Re/(self.Re + self.H)  # 地面速度 
        self.lambda_= self.c/self.f0
        self.theta_c = np.arcsin(self.fc*self.lambda_/(2*self.Vr))
        if self.H > 100e3:
            tmp_angle = np.arcsin((self.H+self.Re)*np.sin(self.beta)/self.Re)
            tmp_angle = tmp_angle - self.beta
            self.R0 = self.Re*np.sin(tmp_angle)/np.sin(self.beta)
        else:
            self.R0 = self.H/np.cos(self.beta)
        self.La = 1
        self.Ta = 2
        self.log = self.get_logger()
        self.d = 0.013
        self.N = 13
        self.Na = int(np.ceil(self.PRF*self.Ta))
        self.Ba = 2*0.886*self.Vg*np.cos(self.theta_c)/self.La 
        self.Naz = np.ceil(self.Ba/self.PRF)
        # self.log.info("Doppler center: {}".format(self.fc))

    def set_B(self, B):
        self.B = B
        self.Kr = B/self.Tp
        # self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)
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
        ## 星载
        if self.H > 100e3:
            tmp_angle = np.arcsin((self.H+self.Re)*np.sin(look_angle)/self.Re)
            tmp_angle = tmp_angle - look_angle
            R0 = self.Re*np.sin(tmp_angle)/np.sin(look_angle)
        ## 机载
        else:
            R0 = self.H/np.cos(look_angle)
        return R0
    
    def slant2ground(self, R):
        ground_angle = np.arccos(((self.H+self.Re)**2+self.Re**2-R**2)/(2*(self.H+self.Re)*self.Re))
        return ground_angle*self.Re

    def calculate_doa(self, R0):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R0**2-(self.H+self.Re)**2)/(2*self.Re*R0)*(R0>self.H))
            doa = np.arcsin(self.Re*np.sin(incident)/(self.H+self.Re))
        else:
            doa = np.arccos(self.H/R0)
        return doa
    
    def calculate_incident(self, R_eta):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
            incident = np.pi - incident
        else:
            incident = np.arccos(self.H/R_eta)
        return incident

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


    def upsample(self, data, N):
        Na, Nr = cp.shape(data)
        data_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(data)))
        tmp = cp.zeros((N[0]*Na, N[1]*Nr), dtype=complex)
        tmp[N[0]*Na/2-Na/2:N[0]*Na/2+Na/2, N[1]*Nr/2-Nr/2:N[1]*Nr/2+Nr/2] = data_fft
        data_up = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(tmp)))
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
        plt.ylabel("视角/°", fontproperties=my_font)
        plt.xlim([1.3e3, 3e3])
        plt.ylim([20, 40])
        plt.savefig("../../../fig/dbf/zebra_diagram.png", dpi=300)

            
    def aasr(self, prf, Naz=1):
        len_prf = len(prf)
        Na_band = 1000
        fa_band = np.linspace(-self.Ba/2, self.Ba/2, Na_band)
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
                    H_matrix[:, k, n] = np.exp(- 1j * np.pi * (n * self.La) / self.Vr * (alias_fa + (k-center_apeture) * prf[j]))


            for k in range(len(fa_band)): 
                P_matrix[k, :, :] = np.linalg.inv(H_matrix[k])

            for i in range(0, 10): 
                G_tmp_tx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vr))**2
                G_tmp_rx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vr))**2
                G_tmp = G_tmp_rx * G_tmp_tx

                ## 计算方位向重构系统的响应
                H = np.zeros(Naz, dtype=complex)
                pm = np.zeros((len(fa_band)), dtype=complex)
                for t in range(len(fa_band)):
                    for n in range(Naz):
                        H[n] = np.exp(- 1j * np.pi * (n * self.La) / self.Vr * (fa_band[t] + i * prf[j]))    
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
            
        # print("aasr: ", aasr)
        return aasr


    def dot_estimate(self, image, area):
        plt.figure(figsize=(12, 8))
        # Convert numerical values to corresponding letters
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        numerical_values = np.arange(len(alphabet))
        letter_mapping = dict(zip(numerical_values, alphabet))
        image_copy = image.copy()
        cnt = 0
        while cnt < self.points_n:
            max_index = np.unravel_index(np.argmax(np.abs(image_copy)), image_copy.shape)
            if max_index[0] < area[0]//2 or max_index[0] > image_copy.shape[0]-area[0]//2 or max_index[1] < area[1]//2 or max_index[1] > image_copy.shape[1]-area[1]//2:
                print("The maximum point is out of the area.")
                continue

            print("Position of the maximum point in the image:", max_index)
            target = image_copy[max_index[0]-area[0]:max_index[0]+area[0], max_index[1]-area[1]:max_index[1]+area[1]]
            uprate = 16
            target_up = self.upsample(cp.array(target), (uprate, uprate))
            image_show = np.abs(target_up)/np.max(np.max(np.abs(target_up)))
            image_show = 20*np.log10(image_show)

            target = target_up
            x = np.array([-area[1]/2, area[1]/2])
            dr = x*self.c/(2*self.Fs)
            y =  np.array([-area[0]/2, area[0]/2])
            da = y*self.Vr/(self.PRF)


            plt.subplot(3, self.points_n, cnt+1)
            plt.imshow(image_show, aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]], vmin=-60, vmax=0)
            plt.ylabel("方位向(m)", fontproperties=my_font)
            plt.xlabel("距离向(m)", fontproperties=my_font)
            colorbar = plt.colorbar()
            colorbar.ax.set_title("dB")
            plt.title("({})".format(letter_mapping[cnt+1]))


            fscan_range_res, fscan_index = self.get_range_IRW(np.abs(target), uprate)
            fscan_rtarget = np.abs(target[fscan_index, :])
            fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
            x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))

            plt.subplot(3, self.points_n, self.points_n + cnt+1)
            plt.plot(x_dr, 20*np.log10(fscan_rtarget))
            plt.grid()
            plt.ylim(-60, 0)
            plt.xlabel("距离向(m)", fontproperties=my_font)
            plt.ylabel("幅度(dB)", fontproperties=my_font)
            plt.title("({})".format(letter_mapping[self.points_n + cnt+1]))


            fscan_azimuth_res, fscan_index = self.get_azimuth_IRW(np.abs(target), uprate)
            fscan_atarget = np.abs(target[:, fscan_index])
            fscan_atarget = fscan_atarget/np.max(fscan_atarget)
            x_da = np.linspace(da[0], da[1], len(fscan_atarget))

            plt.subplot(3, self.points_n, 2*self.points_n + cnt+1)
            plt.plot(x_da, 20*np.log10(fscan_atarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("方位向(m)", fontproperties=my_font)
            plt.ylabel("幅度(dB)", fontproperties=my_font)
            plt.title("({})".format(letter_mapping[2*self.points_n + cnt+1]))

            print("fscan range irw: ", fscan_range_res)
            print("fscan azimuth irw: ", fscan_azimuth_res)
            print("fscan range pslr: ", self.get_pslr(fscan_rtarget))
            print("fscan azimuth pslr: ", self.get_pslr(fscan_atarget))
            print("fscan range islr: ", self.get_islr(fscan_rtarget))
            print("fscan azimuth islr: ", self.get_islr(fscan_atarget))

            image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2] = 0
            cnt = cnt+1
        
        plt.tight_layout()

        plt.savefig("../../../fig/dbf/dot_estimate.png", dpi=300)

 






