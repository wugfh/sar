import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
import scipy.optimize as optimize
sys.path.append(r"../../")
from beam_scan import BeamScan
from sar_focus import SAR_Focus

cp.cuda.Device(0).use()


class FScanAzimuth(BeamScan):
    def __init__(self):
        super().__init__()
        self.theta_c = np.deg2rad(2)  # 斜视角
        self.theta_az = np.deg2rad(0.05)
        self.theta_sc = np.deg2rad(0.5)
        self.Br = 200e6
        self.Fr = self.Br*1.2
        self.Tr = self.Tp*10
        self.Nr = int(np.ceil(self.Fr*self.Tr))
        self.Kr = self.Br/self.Tp
        self.alpha = self.Br/self.theta_sc
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+self.theta_sc/2) - np.sin(self.theta_c-self.theta_sc/2))/self.lambda_
        self.Rc = self.R0/np.cos(self.theta_c)

        self.PRF = 2000
        self.Fa = self.PRF ## 初始采样率

        self.points_n = 3
        self.points_r = self.R0 + cp.array([0, 0, 0])
        self.points_a = cp.array([-3000, 0, 3000])

        self.Ta = self.R0*self.theta_sc/self.Vr*1.5 + (cp.max(self.points_a) - cp.min(self.points_a))/self.Vr 
        self.Na = int(np.ceil(self.PRF*self.Ta))

    
    def echogen(self):
        ##接收机时间窗
        self.tau_c = 2*self.Rc/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            doa = np.arctan((self.points_a[i] - self.Vr*mat_eta)/R0_tar) ## DoA 信号到达角,注意斜视角存在正负

            ftau_l = self.alpha*(-doa+self.theta_c-self.theta_az/2)
            ftau_u = self.alpha*(-doa+self.theta_c+self.theta_az/2)

            ## 接收机频率与时间的关系
            ftau_send = (self.Kr*(mat_tau - 2*R_eta/self.c)) 

            ## 脉宽限制
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            ## 从距离向描述fscan 限制
            Wr_fscan = (ftau_send>ftau_l) * (ftau_send<ftau_u)
            # Wr = 1
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            signal_r = phase_r*Wr*Wr_fscan

            Tstrip_tar = self.theta_sc*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + self.eta_c)) < Tstrip_tar/2

            ## 从方位向描述多普勒限制
            # feta_l = 2*self.Vr*cp.sin(ftau_send/self.alpha +self.theta_c-self.theta_az/2)/self.lambda_
            # feta_u = 2*self.Vr*cp.sin(ftau_send/self.alpha +self.theta_c+self.theta_az/2)/self.lambda_
            # feta = 2*self.Vr*cp.sin(doa)/self.lambda_
            # Wa_fscan = (feta > feta_l) * (feta < feta_u)
            phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        return S_echo

    def azimuth_mosaic(self, echo):
        [Na, Nr] = echo.shape
        echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo, axes=0), axis=0), axes=0)
        self.Fa = self.Bd*1.2
        uprate = self.Fa/self.PRF
        if uprate < 1:
            uprate = 1
            self.Fa = self.PRF
        uprate = int(cp.ceil(uprate))
        echo_tau_feta_up = cp.tile(echo_tau_feta, (uprate, 1))

        Na_up = Na * uprate
        tau = self.tau_c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_
        feta = self.feta_c + cp.arange(-Na_up/2, Na_up/2, 1)*(self.Fa/(Na_up))
        mat_tau, mat_feta = cp.meshgrid(tau, feta)
        
        mat_R = mat_tau*self.c/2
        doa = cp.arccos(self.R0/mat_R)
        feta_R0 = 2*self.Vr*cp.sin(doa)/self.lambda_-self.feta_c
        feta_R0 = feta_R0*1.1
        W_fscan = cp.abs(mat_feta-self.feta_c-feta_R0) < self.PRF/2
        shift = int(cp.round(self.PRF/(2*self.Fa/Na_up)))
        W_fscan = cp.roll(W_fscan, shift, axis=0)

        echo_tau_feta_up *= W_fscan
        return echo_tau_feta_up


if __name__ == "__main__":
    fscan = FScanAzimuth()
    echo = fscan.echogen()
    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(echo)), aspect='auto', cmap='jet', extent=((fscan.tau_c-fscan.Tr/2)*1e6, (fscan.tau_c+fscan.Tr/2)*1e6, (fscan.eta_c-fscan.Ta/2).get(),(fscan.eta_c+fscan.Ta/2).get()))
    plt.xlabel('Range time(us)')
    plt.ylabel('Azimuth time(s)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo.png")


    echo_tau_feta = fscan.azimuth_mosaic(echo)
    # echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo, axes=0), axis=0), axes=0)

    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(echo_tau_feta)), aspect='auto', cmap='jet', extent=(-fscan.Tr*1e6/2, fscan.Tr*1e6/2, -fscan.Fa/2 + fscan.feta_c.get(), fscan.Fa/2+fscan.feta_c.get()))
    plt.xlabel('Range Time(us)')
    plt.ylabel('Azimuth Frequency(Hz)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo_range_fft.png")

    echo_ftau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo_tau_feta, axes=1), axis=1), axes=1)
    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(echo_ftau_feta)), aspect='auto', cmap='jet', extent=(-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2, fscan.Fa/2))
    plt.xlabel('Range Frequency')
    plt.ylabel('Azimuth Frequency')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/echo_fft2.png")

    focus = SAR_Focus(fscan.Fr, fscan.Tp, fscan.f0, fscan.Fa, fscan.Vr, fscan.Br, fscan.feta_c, fscan.R0, fscan.Kr)
    image = focus.wk_focus(cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(echo_ftau_feta))), fscan.Rc)
    image_show = cp.abs(image)/cp.max(cp.max(cp.abs(image)))
    image_show = 20*cp.log10(image_show)
    plt.figure()
    plt.imshow(cp.asnumpy(image_show), aspect='auto', cmap='jet', extent=((fscan.tau_c-fscan.Tr/2)*1e6, (fscan.tau_c+fscan.Tr/2)*1e6, (fscan.eta_c-fscan.Ta/2).get(),(fscan.eta_c+fscan.Ta/2).get()))
    plt.xlabel('Range time(us)')
    plt.ylabel('Azimuth time(s)')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/image.png")

    image_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(image)))
    plt.figure()
    plt.imshow(cp.asnumpy(cp.abs(image_fft)), aspect='auto', cmap='jet', extent=(-fscan.Fr/2, fscan.Fr/2, -fscan.Fa/2, fscan.Fa/2))
    plt.xlabel('Range Frequency')
    plt.ylabel('Azimuth Frequency')
    plt.tight_layout()
    plt.savefig("../../../fig/fscan_azimuth/image_fft.png")
    
