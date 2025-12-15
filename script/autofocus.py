import numpy as np
import cupy as cp
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation

class AutoFocus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0):                         
        self.Re = 6371.39e3                     #地球半径
        self.c = 299792458                      #光速
        self.Fs = Fs                                     
        self.Tp = Tp                            #脉冲宽度                        
        self.f0 = f0                            #载频                     
        self.PRF = PRF                          #PRF                     
        self.Vr = Vr                            #雷达速度     
        self.B = B                              #信号带宽
        self.fc = fc                            #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.R0 = R0
        self.Rc = self.R0/cp.cos(self.theta_c)

    def Moco_first(self, echo, right, down, forward, phi):
        """
        Motion compensation.
        
        Parameters:
        echo (cupy array): echo data before range compress.
        
        Returns:
        numpy array: Motion compensated echo data.
        """
        [Na, Nr] = cp.shape(echo)

        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        theta = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))

        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)
        down = down
        right = right
        r_los = forward*cp.sin(theta)+(down*cp.cos(phi) - right*cp.sin(phi))*cp.cos(theta)
        mat_r_los = cp.tile(r_los[:, cp.newaxis],(1,Nr))
        s_rfft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo, axes=1), axis=1), axes=1)
        H_mcl = cp.exp(4j*cp.pi*(mat_f_tau+self.f0)*mat_r_los/self.c)
        s_rfft_mcl = s_rfft * H_mcl
        echo_mcl = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(s_rfft_mcl, axes=1), axis=1), axes=1)
        return echo_mcl.get()
    def Moco_second(self, echo, right, down, phi):
        """
        Motion compensation.
        
        Parameters:
        echo (cupy array): echo data before range compress.
        
        Returns:
        numpy array: Motion compensated echo data.
        """
        [Na, Nr] = cp.shape(echo)
        R_r = self.R0 + cp.arange(-Nr/2, Nr/2, 1)*(self.c/(2*self.Fs))
        incident_c = phi + cp.pi/2
        incident = cp.arcsin(self.R0*cp.sin(incident_c)/R_r)
        phi_r = phi + cp.pi - incident_c - incident
        phi_r = phi*cp.ones((Nr,))
        
        mat_phi_r = cp.tile(phi_r[cp.newaxis, :], (Na, 1))
        down = cp.tile((down)[:, cp.newaxis], (1, Nr))
        right = cp.tile((right)[:, cp.newaxis], (1, Nr))
        delta_r_los =down*(cp.cos(mat_phi_r) - cp.cos(phi)) - right*(cp.sin(mat_phi_r)-cp.sin(phi))

        H_mcl = cp.exp(4j*cp.pi*delta_r_los/self.lambda_)
        echo_mcl = echo * H_mcl
        
        return echo_mcl.get()
    
    def pga_autofocus(self, corrupted_image, slrange, num_iter=10, rms_threshold=0.1, snr=0):
        """
        
        参数:
        corrupted_image (np.ndarray): 含相位误差的复数SAR图像
        num_iter (int): 最大迭代次数
        min_window_size (int): 最小窗口大小
        snr_threshold (float): 窗口阈值(dB)
        rms_threshold (float): 收敛阈值
        
        返回:
        np.ndarray: 校正后的复数图像
        float: RMS误差记录
        """
        rows, cols = corrupted_image.shape
        midpoint = rows // 2

        ## 估计SNR
        if snr == 0:
            max_power = cp.max(cp.abs(corrupted_image)**2)
            mean_power = cp.mean(cp.abs(corrupted_image)**2)
            snr = 20 * cp.log10((mean_power) / (max_power))*0.5
        print("Estimated SNR (dB):", snr)

        snr_threshold = 10**(snr/20)
        eps = cp.finfo(cp.float32).eps
        R_threshold = 1 / 10**(5/20)  
        error_sum = cp.zeros(rows, dtype=cp.float32)
        image_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(corrupted_image, axes=0), axis=0), axes=0)
        mat_r0 = cp.tile(slrange[cp.newaxis, :], (rows, 1))
        for iter in range(num_iter):

            # 1. 循环移位：对齐最强散射体至中心
            mat_error_sum = cp.tile(error_sum[:, cp.newaxis], (1, cols))
            mat_error_sum = mat_error_sum *mat_r0 /  slrange[cols//2]
            image_ffta_temp = image_ffta*cp.exp(-1j*mat_error_sum)
            
            image = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(image_ffta_temp, axes=0), axis=0), axes=0)
           
            centered = image.copy()
            shifted = cp.zeros(cols, dtype=cp.int32)
            for i in range(cols):
                bin = image[:, i]
                midx = cp.argmax(cp.abs(bin))
                shift = midpoint - midx
                centered[:, i] = cp.roll(bin, shift)
                shifted[i] = shift
            
            Sx = cp.sum(cp.abs(centered)**2, axis=1)
            winbool = Sx >= (cp.max(Sx)*snr_threshold)
            win_len = cp.sum(winbool)
            win_start = cp.maximum(midpoint - win_len//2, 0)
            win_end = cp.minimum(midpoint + win_len//2, rows-1)
            # win_indices = cp.where(winbool)[0]
            # if win_indices.size > 0:
            #     win_start = win_indices[0]
            #     win_end = win_indices[-1]
            #     win_len = win_end - win_start + 1
            # else:
            #     win_start = 0
            #     win_end = rows - 1
            #     win_len = rows


            x = cp.arange(0, rows)
            winbool = (x > win_start) & (x <win_end)

            centered = centered * cp.tile(winbool[:, cp.newaxis], (1, cols))
            # for i in range(cols):
            #     centered[:, i] = cp.roll(centered[:, i], -shifted[i])

            # dot_len = 1600
            # centered = cp.zeros((dot_len, cols*n_scatter), dtype=cp.complex64)
            # winlen = 800
            # for c in range(cols):
            #     bin = image[:, c].copy()
            #     for k in range(n_scatter):
            #         midx = cp.argmax(cp.abs(bin))
            #         mval = cp.max(cp.abs(bin))
            #         mask = cp.abs(bin) >= (mval*snr_threshold)
            #         l = cp.argmax(mask)
            #         r = cp.argmin(mask[l:])
            #         l = cp.maximum(l - 200, 0)
            #         r = cp.minimum(r + 200, len(bin)-1)
         

            #         temp = cp.zeros(dot_len, dtype=cp.complex64)
            #         bin_len = r - l
            #         temp[dot_len//2-bin_len/2:dot_len//2+bin_len/2] = bin[l: r]
            #         bin[l:r] = 0
            #         centered[:, c*n_scatter + k] = temp

            
            # 截取窗口数据
            # windowed_data = centered*cp.tile(WinBool[:, cp.newaxis], (1, cols))
            Gn = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(centered, axes=0), axis=0), axes=0)
            val = Gn * cp.roll(cp.conj(Gn), 1, axis=0)
            val_abs = cp.abs(val)
            phi_error = cp.angle(val)

            # 计算距离标度因子，主要看误差成分中运动误差是否占主导
            phi_error = phi_error*slrange[cols//2]/mat_r0

            ## 增强高信噪比部分的权重
            val = val_abs**2*cp.exp(1j*phi_error)
            

            ## qulity measurement
            # quality_threshold = 0.5
            # Gn_abs = cp.abs(Gn)
            # Gn_quality = cp.var(Gn_abs, axis=0) / (cp.var(Gn_abs, axis=0)  + cp.mean(Gn_abs, axis=0)**2 )
            # Gn_quality = cp.tile(Gn_quality[cp.newaxis, :], (Gn.shape[0], 1))
            # Gn = Gn * (Gn_quality <= quality_threshold)

            # # WML estimation
            c = cp.mean(cp.abs(Gn), axis=0)
            d = cp.mean(cp.abs(Gn)**2, axis=0)
            R = (4 * (2 * c**2 - d) - 4 * c * cp.sqrt(cp.maximum(0, 4 * c**2 - 3 * d)) + eps) / (d + eps)
            w = 1 / (0.5 * R + 5 / 24 * R**2 + eps)

            w = w * (cp.logical_and(R > 0, R < R_threshold))
            w = cp.tile(w[cp.newaxis, :], (Gn.shape[0], 1))
            w = w / cp.tile((cp.sum(w, axis=1) + eps)[:, cp.newaxis], (1, w.shape[1]))
            phi_error = cp.angle(cp.sum(w *  val, axis=1))
            # 计算RMS
            rms = cp.sqrt((cp.mean((phi_error)**2)))

            phi_error = cp.cumsum(phi_error, axis=0)
            phi_error = cp.unwrap(phi_error)

                        ## 误差不包含线性项
            poly_fit = cp.polyfit(cp.arange(phi_error.shape[0]), phi_error, 1)
            poly_value = cp.polyval(poly_fit, cp.arange(phi_error.shape[0]))
            phi_error = phi_error - poly_value

            error_sum += phi_error


            # rms = cp.sqrt(cp.mean((error-error_sum)**2))
            print("rms:{} winlen:{}".format(rms.get(), win_len))
            # if(rms < 0.1):
            #     break
            
            

        # 对 error_sum 做平滑处理
        # window_size = 21  # 可以根据需要调整窗口大小
        # if window_size > 1:
        #     kernel = cp.ones(window_size) / window_size
        #     error_sum = cp.convolve(error_sum, kernel, mode='same')
        
        return error_sum.get(), rms.get(), centered.get()

if __name__ == "__main__":
    Na = 40000
    Fa = 40000
    eta = cp.arange(-Na/2, Na/2)*(1/Fa)
    Ka = 40000
    signal = 1.8*cp.exp(1j*cp.pi*Ka*eta**2) + 2*cp.exp(1j*cp.pi*Ka*(eta-Na/(Fa*3))**2) 
    phi_error = cp.linspace(-10, 10, Na)
    phi_error = 10*phi_error**3 + 4*phi_error**2 - 10*phi_error + 5
    phi_error = cp.angle(cp.exp(1j*phi_error))
    signal_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal)))
    signal_fft = signal_fft*cp.exp(1j*phi_error)
    signal = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_fft)))

    # 添加0dB噪声
    snr_db = 0
    signal_power = cp.mean(cp.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = cp.sqrt(noise_power / 2) * (cp.random.standard_normal(signal.shape) + 1j * cp.random.standard_normal(signal.shape))
    # signal = signal + noise

    feta = cp.arange(-Na/2, Na/2)*(Fa/Na)

    
    signal_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal)))
    signal_fft = signal_fft*cp.exp(1j*cp.pi*(feta**2)/Ka)
    signal_no = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_fft)))

    ##dechirp
    # signal_dechirp = signal*cp.exp(-1j*cp.pi*Ka*(eta**2))
    # signal_dechirp = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal_dechirp)))
    signal_dechirp = signal_no
    # signal_dechirp = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal)))
    # signal_dechirp = signal_dechirp*cp.exp(1j*cp.pi*(feta**2)/Ka)
    # signal_dechirp = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_dechirp)))

    autofocus = AutoFocus(Fs=20e6, Tp=10e-6, f0=5.3e9, PRF=Fa, Vr=150, B=20e6, fc=0, R0=800e3)
    error, rms, centered = autofocus.pga_autofocus(signal_dechirp[:, cp.newaxis], num_iter=1, snr = -40)
    centered = cp.array(np.squeeze(centered))
    phi_dechirp = cp.angle(cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(centered))))
    
    error = cp.array(np.squeeze(error))



    signal_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal_dechirp)))
    signal_fft = signal_fft*cp.exp(-1j*error)
    signal_yes = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_fft)))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(eta.get(), 20*np.log10(np.abs(signal_yes.get())+1e-10), label="after pga")
    plt.legend()
    plt.subplot(212)
    plt.plot(eta.get(),20*np.log10(np.abs(signal_no.get())+1e-10), label="before pga")
    plt.legend()
    plt.savefig("./before_pga.png")
    plt.figure(2)
    # phi_error = cp.unwrap(phi_error)

    dis_phi = cp.angle(cp.exp(1j*(phi_error-error)))
    dis_phi = dis_phi - cp.mean(dis_phi)
    # phi_dechrip = cp.unwrap(phi_dechrip)
    plt.subplot(311)
    plt.plot(eta.get(), cp.unwrap(phi_error).get(), label="true")
    plt.subplot(312)
    plt.plot(eta.get(), cp.unwrap(error).get(), label="estimated")
    plt.subplot(313)
    plt.plot(eta.get(), cp.unwrap(phi_dechirp).get(), label="dechirp")
    print("RMS error:", cp.sqrt(cp.mean(dis_phi**2)).get())
    plt.savefig("./pga_error.png")