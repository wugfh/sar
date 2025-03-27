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
        self.H = 600e3                          #卫星高度
        self.fscan_tau = 8.8e-9                 #频扫通道时延
        self.Lr = 2                             #雷达距离向宽度
        self.fscan_N = 10                       #频扫通道数   
        self.fscan_d = self.Lr/self.fscan_N     #频扫通道间隔
        self.DBF_N = 10                         #DBF天线数
        self.Re = 6371.39e3                     #地球半径
        self.beta = cp.deg2rad(25)              #天线安装角
        self.dbf_d = self.Lr/self.DBF_N         #DBF子孔径间距
        self.c = 299792458                      #光速
        self.Fs = 120e6                         #采样率              
        self.Tp = 103e-6                        #脉冲宽度                        
        self.f0 = 30e+09                        #载频                     
        self.PRF = 1950                         #PRF                     
        self.Vr = 7062                          #雷达速度     
        self.B = 100e6                          #信号带宽
        self.fc = -1000                         #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        tmp_angle = cp.arcsin((self.H+self.Re)*cp.sin(self.beta)/self.Re)
        tmp_angle = tmp_angle - self.beta
        self.R0 = self.Re*cp.sin(tmp_angle)/cp.sin(self.beta)
        self.La = 15
        self.Ta = 1
        self.Kr = -self.B/self.Tp 
        self.fscan_beam_width = (0.886*self.lambda_/self.fscan_d)
        self.dbf_beam_width = (0.886*self.lambda_/self.dbf_d)
        self.focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0, self.Kr)
        print("fscan beam_width: ", cp.rad2deg(self.fscan_beam_width))
        print("dbf beam_width: ", cp.rad2deg(self.dbf_beam_width))

    def init_raw_data(self, data1):
        [Na, Nr] = cp.shape(data1)
        data = cp.zeros([int(cp.ceil(Na*1.5)), int(cp.ceil(Nr*1.5))], dtype=cp.complex128)
        [self.Na, self.Nr] = cp.shape(data)
        data[self.Na/2-Na/2:self.Na/2+Na/2, self.Nr/2-Nr/2:self.Nr/2+Nr/2] = data1
        return data
    
    ## 添加高斯白噪声
    def add_noise(self, data, snr):
        data = data + cp.random.randn(self.Na, self.Nr)*cp.sqrt(cp.var(data)/10**(snr/10))
        return data
    
    def calulate_R0(self, look_angle):
        tmp_angle = cp.arcsin((self.H+self.Re)*cp.sin(look_angle)/self.Re)
        tmp_angle = tmp_angle - look_angle
        R0 = self.Re*cp.sin(tmp_angle)/cp.sin(look_angle)
        return R0

    ## 计算接收窗口大小
    def calulate_re_window(self, beam_width):
        R_max = self.calulate_R0(self.beta+beam_width/2)
        R_min = self.calulate_R0(self.beta-beam_width/2)
        return (R_max-R_min)*2/self.c

    def init_simparams(self, mode = "fscan"):
        if(mode == "fscan"):
            self.Tr = self.calulate_re_window(self.fscan_beam_width)
            self.Nr = int(cp.ceil(self.Fs*self.Tr))
        elif(mode == "dbf"):
            self.Tr = self.calulate_re_window(self.dbf_beam_width)
            self.Nr = int(cp.ceil(self.Fs*self.Tr))

        print("receive window length Tr: ", self.Tr*1e6)   
        print("R_width: ", self.Tr*self.c/2) 
        print("Doppler bandwidth: ", 0.886*2*self.Vr*cp.cos(self.theta_c)/self.La)
        self.Na = int(cp.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+cp.linspace(-6000, 6000, self.points_n)
        self.points_a = cp.linspace(-3000, 3000, self.points_n)
        self.Ba = 2*0.886*self.Vr*cp.cos(self.theta_c)/self.La + 2*self.Vr*self.B*cp.sin(self.theta_c)/self.c
        self.Rc = self.R0/cp.cos(self.theta_c)

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
        return S_echo 

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
        return S_echo
    
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

            deci = self.c/(self.f0+self.Kr*(mat_tau-2*R_eta/self.c))
            pr = cp.sin(self.fscan_N*(cp.pi*(self.fscan_tau*self.c-self.fscan_d*cp.sin(doa-self.beta)))/deci) \
            /(cp.sin(cp.pi*(self.fscan_tau*self.c-self.fscan_d*cp.sin(doa-self.beta))/deci))
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tr/2
            # signal_r = Wr*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)*pr*pr
            # 发射机,地面目标接收到的信号为所有通道的汇总
            for j in range(self.fscan_N):
                tj = (j-(self.fscan_N-1)/2)*(self.fscan_tau - self.fscan_d*cp.sin(doa-self.beta)/self.c) 
                phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-tj-2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.f0*tj) ## 基带信号
                signal_t += phase_r*Wr
            signal_r = signal_t*pr

            # ## 接收机,每个通道的时延不同
            # signal_fftr = cp.fft.fft(signal_t, axis = 1)
            # for j in range(self.fscan_N):
            #     tj = (j-(self.fscan_N-1)/2)*(self.fscan_tau - self.fscan_d*cp.sin(doa-self.beta)/self.c)+2*R_eta/self.c
            #     signal_t_shift = cp.fft.ifft(signal_fftr*cp.exp(-2j*cp.pi*mat_f_tau*tj), axis = 1)
            #     signal_r += signal_t_shift

            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo
    
    ## 计算每个目标Doa对应的快时间
    def fscan_calulate_doaTx(self, doa):
        t_inv = self.fscan_tau-self.fscan_d*cp.sin(doa-self.beta)/self.c
        m = cp.floor((self.f0-self.B/2)*t_inv)
        t_peak = m/(self.Kr*t_inv) - (self.f0-self.B/2)/self.Kr
        interval = cp.abs(1/(self.fscan_N*self.Kr*t_inv))
        t_left = t_peak - interval
        t_right = t_peak + interval
        band_width = cp.abs(interval*self.Kr*2)

        return t_peak, t_left, t_right, band_width
    
    def fscan_range_estimate(self):
        print("target estimate:")
        target_doa = cp.arccos(((self.H+self.Re)**2+self.points_r**2-self.Re**2)/(2*(self.H+self.Re)*self.points_r)) ## DoA 信号到达角
        target_peak,target_left,target_right, target_bw= self.fscan_calulate_doaTx(target_doa)
        print("t_peak(us): ",target_peak*1e6)    
        print("t_left(us): ",(target_left)*1e6)
        print("t_right(us): ",(target_right)*1e6)
        print("target duration(us): ", (target_right-target_left)*1e6)
        print("target bandwidth(Mhz): ", (target_bw)/1e6)

        print("\nswath estimate:")
        doa = self.beta + cp.linspace(-self.fscan_beam_width/2 + self.fscan_beam_width/20, self.fscan_beam_width/2, self.Nr)
        tx_peak, _, _, _ = self.fscan_calulate_doaTx(doa)
        doaR = self.calulate_R0(doa)
        rx_peak = 2*doaR/self.c + tx_peak - 2*cp.min(doaR)/self.c

        plt.figure()
        plt.subplot(211)
        plt.plot(cp.rad2deg(doa).get(), rx_peak.get()*1e6)
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("Rx peak(us)")
        plt.title("look angle vs. Rx time")
        plt.subplot(212)
        plt.plot(cp.rad2deg(doa).get(), tx_peak.get()*self.Kr/1e6)
        plt.grid()
        plt.xlabel("look angle(degree)")
        plt.ylabel("Frequency(MHz)")
        plt.title("look angle vs. Frequency")
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/doa_t_f.png", dpi=300)

        print("swath width:", (cp.max(target_right)-cp.min(target_left))*1e6)
        return target_bw
    
    def fscan_focus(self, echo):
        image = self.focus.wk_focus(echo, self.R0)

        f_tau = cp.fft.fftshift(cp.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fs/self.Nr))
        f_eta = self.fc + (cp.linspace(-self.Na/2,self.Na/2-1,self.Na)*(self.PRF/self.Na))
        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)

        image_ftau_feta = cp.fft.fftshift(cp.fft.fft2(image), axes=1)

    def fscan_tpeak(self, doa, f0, tau):
        t_inv = tau-self.fscan_d*cp.sin(doa-self.beta)/self.c
        m = cp.floor((f0-self.B/2)*t_inv)
        t_peak = m/(self.Kr*t_inv) - (f0-self.B/2)/self.Kr
        return t_peak

    def fscan_rasr(self):
        pass
    def fsan_aasr(self):
        pass
    def fsan_nesz(self):
        pass

    def get_range_IRW(self, ehco):
        max_index = cp.argmax(cp.abs(cp.max(cp.abs(ehco), axis=1))) 
        max_value = cp.max(cp.abs(ehco[max_index,:]))
        half_max = max_value/cp.sqrt(2)
        valid = cp.abs(ehco[max_index,:]) > half_max
        irw = cp.sum(valid)
        irw = irw*self.c/(2*self.Fs)
        return irw, max_index
    
    def get_azimuth_IRW(self, ehco):
        max_index = cp.argmax(cp.abs(cp.max(cp.abs(ehco), axis=0))) 
        max_value = cp.max(cp.abs(ehco[:,max_index]))
        half_max = max_value/cp.sqrt(2)
        valid = cp.abs(ehco[:,max_index]) > half_max
        irw = cp.sum(valid)
        irw = irw*self.Vr/(self.PRF)
        return irw, max_index
    
    def get_pslr(self, target):
        target_np = cp.asnumpy(target)
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
    plt.contour(cp.abs(strip_image).get())
    plt.title("strip mode")
    plt.subplot(122)
    plt.contour(cp.abs(dbf_image).get())
    plt.title("dbf mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/sim_dbf.png", dpi=300)

    
    strip_fft = cp.abs((cp.fft.fft2(strip_image)))
    strip_fft = strip_fft/cp.max(cp.max(strip_fft))
    strip_fft = 20*cp.log10(strip_fft)  
    dbf_fft = cp.abs((cp.fft.fft2(dbf_image)))
    dbf_fft = dbf_fft/cp.max(cp.max(dbf_fft))
    dbf_fft = 20*cp.log10(dbf_fft)
    plt.figure()
    plt.subplot(211)
    plt.imshow((cp.asnumpy(strip_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.subplot(212)
    plt.imshow((cp.asnumpy(dbf_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/dbf_fft.png", dpi=300)

    dbf_res, dbf_index = dbf_sim.get_range_IRW(dbf_image)
    strip_res, strip_index = dbf_sim.get_range_IRW(strip_image)
    print("dbf irw: ", dbf_res)
    print("strip irw: ", strip_res)
    print("theoretical irw: ", dbf_sim.c/(2*dbf_sim.B))

    dbf_target = cp.abs(dbf_image[dbf_index, :])
    dbf_target = dbf_target/cp.max(dbf_target)
    strip_target = cp.abs(strip_image[strip_index, :])
    strip_target = strip_target/cp.max(strip_target)
    tau = range(dbf_sim.Nr) 
    plt.figure()
    plt.subplot(121)
    plt.plot(tau, 20*cp.log10(cp.abs(dbf_target)).get())
    plt.title("dbf mode")
    plt.subplot(122)
    plt.plot(tau, 20*cp.log10(cp.abs(strip_target)).get())
    plt.title("strip mode")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/range.png", dpi=300)
    print("dbf pslr: ", dbf_sim.get_pslr(dbf_target))
    print("strip pslr: ", dbf_sim.get_pslr(strip_target))

def fscan_simulation():
    fscan_sim = BeamScan()
    echo = fscan_sim.fscan_echogen()
    echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]
    Be = fscan_sim.fscan_range_estimate()

    plt.figure()
    plt.imshow(abs(cp.asnumpy(echo)), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    echo_shape = cp.shape(echo)
    tmp = cp.zeros([echo_shape[0], echo_shape[1]*4], dtype=cp.complex128)
    tmp[:, echo_shape[1]:echo_shape[1]*2] = echo
    echo = tmp
    image = fscan_sim.focus.wk_focus(echo, fscan_sim.R0)
    image_show = cp.abs(image)/cp.max(cp.max(cp.abs(image)))
    image_show = 20*cp.log10(image_show)

    image_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(image), axes=1))
    image_fft = image_fft/cp.max(cp.max(image_fft))
    image_fft = 20*cp.log10(image_fft)  
    plt.figure()
    plt.imshow((cp.asnumpy(image_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0, 
               extent=[-fscan_sim.Fs/2e6, fscan_sim.Fs/2e6, -fscan_sim.PRF/2, fscan_sim.PRF/2])
    plt.colorbar()

    plt.xlabel("Range Frequency(MHz)")
    plt.ylabel("Azimuth Frequency(Hz)")
    plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


    plt.figure()
    plt.imshow(image_show.get(), aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    fscan_range_res, fscan_index = fscan_sim.get_range_IRW(cp.abs(image))
    fscan_rtarget = cp.abs(image[fscan_index, :])
    fscan_rtarget = fscan_rtarget/cp.max(fscan_rtarget)
    x = range(cp.shape(fscan_rtarget)[0])
    plt.figure()
    plt.plot(x, 20*cp.log10(fscan_rtarget).get())
    plt.savefig("../../../fig/dbf/fscan_range.png", dpi=300)

    fscan_azimuth_res, fscan_index = fscan_sim.get_azimuth_IRW(cp.abs(image))
    fscan_atarget = cp.abs(image[:, fscan_index])
    fscan_atarget = fscan_atarget/cp.max(fscan_atarget)
    x = range(cp.shape(fscan_atarget)[0])
    plt.figure()
    plt.plot(x, 20*cp.log10(fscan_atarget).get())
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
    tau = cp.linspace(8.7, 8.9, 3000)*1e-9
    doa = cp.deg2rad(cp.array([25, 35]))
    t_swath = []
    t_swath.append(fscan_sim.fscan_tpeak(doa[1], 10e9, tau) - fscan_sim.fscan_tpeak(doa[0], 5e9, tau))
    t_swath.append(fscan_sim.fscan_tpeak(doa[1], 20e9, tau) - fscan_sim.fscan_tpeak(doa[0], 10e9, tau))
    t_swath.append(fscan_sim.fscan_tpeak(doa[1], 30e9, tau) - fscan_sim.fscan_tpeak(doa[0], 15e9, tau))
    t_swath.append(fscan_sim.fscan_tpeak(doa[1], 40e9, tau) - fscan_sim.fscan_tpeak(doa[0], 35e9, tau))

    t_swath = cp.abs(cp.array(t_swath))
    plt.figure()
    plt.plot(cp.asnumpy(tau)*1e9, cp.asnumpy(t_swath[0])*1e6, label="10e9", alpha=1)
    plt.plot(cp.asnumpy(tau)*1e9, cp.asnumpy(t_swath[1])*1e6, label="20e9", alpha=1)
    plt.plot(cp.asnumpy(tau)*1e9, cp.asnumpy(t_swath[2])*1e6, label="30e9", alpha=1)
    plt.plot(cp.asnumpy(tau)*1e9, cp.asnumpy(t_swath[3])*1e6, label="40e9", alpha=1)
    plt.legend()
    plt.grid()
    plt.xlabel("TTD(ns)")
    plt.ylabel("Tp(us)")
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_t_f.png", dpi=300)



if __name__ == '__main__':
    fscan_ka_estimate()
    # fscan_simulation()
    # dbf_simulation()





