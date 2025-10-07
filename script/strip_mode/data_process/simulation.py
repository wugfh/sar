
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
sys.path.append(r"../../")
from sar_focus import SAR_Focus
from sinc_interpolation import SincInterpolation
from autofocus import AutoFocus
import doppler_estimation as doppler
from dot_estimate import DotEstimator
from concurrent.futures import ThreadPoolExecutor

cp.cuda.Device(1).use()

class Fcous_Air:
    def __init__(self, Tp, Br, f0, R0, Fr, PRF, fc, Vr):
        self.c = 299792458
        self.Tr = Tp*5
        self.Tp = Tp
        self.Br = cp.abs(Br)
        self.f0 = f0
        self.Fr = Fr
        self.PRF = PRF
        self.fc = fc
        self.Vr = Vr
        self.lambda_ = self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))

        self.R0 = R0
        self.Kr = Br/self.Tp
        self.atheta_width = cp.deg2rad(2)
        self.Bd = 2*self.Vr*self.atheta_width/self.lambda_
        self.da = self.Vr/self.Bd
        print("da: ", self.da)
        self.auto_focus = AutoFocus(Fr, self.Tp, f0, PRF, Vr, Br, fc, self.R0, self.Kr)
        self.points_n = 3
        self.Ta = 5
        self.Nr = int(cp.ceil(self.Fr*self.Tr))
        self.Na = int(cp.ceil(self.PRF*self.Ta))
        self.points_r = cp.array([-150, -70, 50]) + self.R0
        self.points_a = cp.array([-50, 0, -50])
    
    def echo_generate(self):
               ##接收机时间窗
        self.tau_c = 2*self.R0/self.c
        tau = self.tau_c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        self.eta_c = -self.R0*cp.sin(self.theta_c)/self.Vr
        eta = self.eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        vr = cp.random.normal(loc=0, scale=5, size=mat_eta.shape)
        
        dR = cp.random.normal(loc=0, scale=5, size=mat_eta.shape)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]
            R_tar = R0_tar
            R_eta = cp.sqrt(R_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)


            ## 脉宽限制
            Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
            phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            signal_r = phase_r*Wr

            Tstrip_tar = self.atheta_width*R_tar/(self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + self.eta_c)) < Tstrip_tar/2

            phase_a = cp.exp(-4j*cp.pi*R_eta*(self.f0)/self.c)
            signal_a = Wa*phase_a
            S_echo += signal_r*signal_a
        # 添加白噪声
        snr_db = 10  # 信噪比(dB)，可根据需要调整
        signal_power = cp.mean(cp.abs(S_echo)**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = cp.sqrt(noise_power / 2) * (cp.random.standard_normal(S_echo.shape) + 1j * cp.random.standard_normal(S_echo.shape))
        S_echo = S_echo+noise

        return S_echo.get()
    
    def read_data(self, data_filename, pos_filename):
        with h5py.File(data_filename, "r") as data:
            sig = data['sig']
            sig = sig["real"] + 1j*sig["imag"]
            sig = np.array(sig)

        with h5py.File(pos_filename) as pos:    
            self.forward = np.squeeze(np.array(pos['forward']))
            self.right = np.squeeze(np.array(pos['right']))
            self.down = np.squeeze(np.array(pos['down']))
            self.frame_time = self.time2sec(np.array(pos['frame_time']))

        sig = sig[::3,:]
        self.sig = np.array(sig)
        self.Na, self.Nr = np.shape(self.sig)


    @staticmethod
    def time2sec(time):
        """
        Convert echo recorded time to pos seconds.
        
        Parameters:
        time (numpy array): Array of time values in hhmmss format.
        
        Returns:
        numpy array: Array of time values in seconds.
        """
        # Split hhmmss digits
        time = np.array(time, dtype=float)
        hours = np.floor(time / 1e4)
        minutes = np.floor((time - hours * 1e4) / 1e2)
        seconds = time - hours * 1e4 - minutes * 1e2

        # Add hours and minutes
        timezone = 8 
        seconds = seconds + (hours - timezone) * 3600 + minutes * 60  # time zone conversion

        # Add fractional part
        seconds = np.squeeze(seconds)
        sec_change_idx = np.where(np.diff(seconds) != 0)[0] + 1
        poly = np.polyfit(sec_change_idx, seconds[sec_change_idx], 1)
        seconds_new = np.polyval(poly, np.arange(len(seconds)))

        return seconds_new
    
    def rd_focus_rc(self, echo, squint_angle):
        [Na, Nr] = cp.shape(echo)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo, axes=1), axis=1), axes=1)
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hm = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        if(squint_angle > 2):
            data_fft_cr = data_fft_r*Hr*Hm
        else:
            data_fft_cr = data_fft_r*Hr
        data_cr = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_cr, axes=1), axis = 1), axes=1)
        return data_cr.get()
    
    def rd_focus_rcmc(self, data_rc):  
        [Na, Nr] = cp.shape(data_rc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rc, axes=0), axis=0), axes=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta = mat_R0/mat_D - mat_R0
        delta = delta*2/(self.c/self.Fr)
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag


        data_final = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)

        return data_final.get()
    
    def rd_focus_ac(self, data_rcmc):
        [Na, Nr] = cp.shape(data_rcmc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_a_rcmc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rcmc, axes=0), axis=0), axes=0)
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## 方位压缩
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        # Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)
        data_final = data_ca_rcmc
        return data_final.get()

    def dechirp(self, data):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        mat_R0 = mat_tau*self.c/2 + self.R0;  
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        data = data*cp.exp(1j*cp.pi*Ka*mat_eta**2)
        data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
        return data.get()
    
    def rechirp(self, data):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        mat_R0 = mat_tau*self.c/2 + self.R0;  
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        data = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data, axes=0), axis=0), axes=0)
        data = data*cp.exp(-1j*cp.pi*Ka*mat_eta**2)
        return data.get()

    def range2ground(self, data, H):
        Na,Nr = cp.shape(data)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        range_0 = tau*self.c/2 + self.R0
        mat_R0 = cp.tile(range_0, (Na,1))
        ground_0 = cp.sqrt((range_0**2 - H**2))
        ground_grid = cp.linspace(cp.min(ground_0), cp.max(ground_0), Nr)
        ground_R0 = cp.sqrt((ground_grid**2 + H**2))
        mat_gnd = cp.tile(ground_R0, (Na,1))
        delta = (mat_gnd - mat_R0)
        delta = delta*2/(self.c/self.Fr)
        sinc_intp = SincInterpolation()
        data_ground_imag = sinc_intp.sinc_interpolation(cp.imag(data), delta, Na, Nr, 8)
        data_ground_real = sinc_intp.sinc_interpolation(cp.real(data), delta, Na, Nr, 8)
        data_ground = data_ground_real + 1j*data_ground_imag
        return data_ground.get()
    
    def get_showimage(self, image):
        image_abs = np.abs(image)
        image_abs = image_abs/np.max(np.max(image_abs))
        image_abs = 20*np.log10(image_abs+1)
        # Perform histogram equalization on image_abs
        image_abs = np.flip(image_abs, axis=1)
        image_abs = cv2.normalize(image_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_abs = cv2.equalizeHist(image_abs)
        return image_abs
    
    def upsample(self, data, N):
        Na, Nr = cp.shape(data)
        data_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(data)))
        tmp = cp.zeros((N[0]*Na, N[1]*Nr), dtype=complex)
        tmp[N[0]*Na//2-Na/2:N[0]*Na//2+Na/2, N[1]*Nr//2-Nr/2:N[1]*Nr//2+Nr/2] = data_fft
        data_up = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(tmp)))
        return data_up.get()
    
    def get_azimuth_IRW(self, ehco, uprate, vr):
        max_val = cp.max(cp.abs(ehco), axis=0)
        max_index = cp.argmax(max_val)
        max_value = cp.max(cp.abs(ehco[:, max_index]))
        midx = cp.argmax(cp.abs(ehco[:,max_index]))
        half_max = max_value/cp.sqrt(2) # 3dB 宽度
        left_idx = cp.where(cp.abs(ehco[:midx, max_index]) < half_max)[0]
        right_idx = cp.where(cp.abs(ehco[midx:, max_index]) < half_max)[0]

        left_idx = left_idx[-1] if len(left_idx) > 0 else 0
        right_idx = midx + right_idx[0] if len(right_idx) > 0 else ehco.shape[0] - 1

        irw = right_idx - left_idx
        irw = irw*vr/(self.PRF*uprate)
        return irw.get(), max_index.get()
            
            
if __name__ == '__main__':

    focus_air = Fcous_Air(0.5e-6, 2e9, 37e9, 5256.3, 2.5e9, 4000/3, 0, 72.25)
    R0 =  3.46e-5*focus_air.c/2
    focus_air.R0 = R0
  
    focus_air.sig = focus_air.echo_generate()
    print("echo shape: ", np.shape(focus_air.sig))

    plt.figure(figsize=(9, 12))
    plt.imshow(np.abs(focus_air.sig), cmap='jet', aspect='auto')
    plt.title("Moco")
    plt.savefig("../../../fig/data_process/echo.png")

    focus_air.sig = focus_air.rd_focus_rc(cp.array((focus_air.sig)), squint_angle=0)

    # Apply Kaiser window along the y-axis
    kaiser_window = np.kaiser(focus_air.sig.shape[0], beta=5)[:, cp.newaxis]
    kaiser_window = np.tile(kaiser_window, (1, focus_air.sig.shape[1]))
    focus_air.sig = focus_air.sig * kaiser_window



    


    focus_air.sig = focus_air.rd_focus_rcmc(cp.array((focus_air.sig)))
    [Na, Nr] = cp.shape(focus_air.sig)
    phi_error = cp.linspace(-1, 1, Na)
    focus_air.sig = cp.array(focus_air.sig)
    phi_error = 20*phi_error**4+67*phi_error**3 + 50*phi_error**2 - 1000*phi_error + 5
    signal_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(focus_air.sig, axes=0), axis=0), axes=0)
    signal_fft = signal_fft*cp.tile(cp.exp(1j*phi_error)[:, cp.newaxis], (1,Nr))
    focus_air.sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_fft, axes=0), axis=0), axes=0).get()



    focus_air.sig = focus_air.dechirp(cp.array((focus_air.sig)))


    bsize = int(focus_air.Na)
    lstart = np.arange(0, focus_air.Na, bsize)
    afoucs = AutoFocus(focus_air.Fr, focus_air.Tr, focus_air.f0, focus_air.PRF, focus_air.Vr, focus_air.Br, focus_air.fc, focus_air.R0, focus_air.Kr)
    error_sum = np.zeros(focus_air.Na, dtype=np.float32)
    sum_num = np.zeros(focus_air.Na, dtype=np.float32)
    step = 1
    for start in lstart:
        print("step: ", step)
        step += 1
        start = int(start)
        end = int(np.minimum(start+bsize, focus_air.Na))
        block = focus_air.sig[start:end, :]
        error, rms, windata = afoucs.pga_autofocus(cp.array((block)), num_iter=50)
        error_sum[start:end] += (error)
        sum_num[start:end] += 1
    error_sum = error_sum/sum_num

    # error_sum = cp.array(error_sum)
    error_sum = cp.tile(error_sum[:, cp.newaxis], (1, focus_air.Nr))
    focus_air.sig  = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(focus_air.sig, axes=0), axis=0), axes=0)
    focus_air.sig = focus_air.sig*cp.exp(-1j*error_sum)
    focus_air.sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(focus_air.sig, axes=0), axis=0), axes=0).get()
    # focus_air.sig = focus_air.rechirp(cp.array((focus_air.sig)))
    # focus_air.sig = focus_air.rd_focus_ac(cp.array((focus_air.sig)))

    # windata = focus_air.get_showimage(windata)
    # plt.figure(figsize=(9, 12))
    # plt.imshow(np.abs(windata), cmap='gray', aspect='auto')
    # plt.title("Moco")
    # plt.savefig("../../../fig/data_process/windata.png")

    image_show = focus_air.get_showimage(focus_air.sig)
    plt.figure(figsize=(4.5, 8))
    plt.imshow(image_show, cmap='gray', aspect='auto')
    plt.title(" Moco PGA")
    plt.savefig("../../../fig/data_process/image.png")

    dot_estimate = DotEstimator(focus_air.points_n, focus_air.c, focus_air.Vr, focus_air.PRF, focus_air.Fr, "../../../fig/data_process/")
    dot_estimate.dot_estimate((focus_air.sig), (100, 50), 16)

    plt.figure()
    plt.subplot(211)
    plt.plot(cp.unwrap(phi_error).get(), label="true")
    plt.subplot(212)
    plt.plot(cp.unwrap(error_sum).get(), label="estimated")
    plt.savefig("../../../fig/data_process/pga_error.png")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    exit()