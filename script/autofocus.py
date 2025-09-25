import numpy as np
import cupy as cp
import sys
from tqdm import tqdm
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation

class AutoFocus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0, Kr):                         
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
        self.Kr = Kr

    def Moco_first(self, echo, right, down, phi):
        """
        Motion compensation.
        
        Parameters:
        echo (cupy array): echo data before range compress.
        
        Returns:
        numpy array: Motion compensated echo data.
        """
        [Na, Nr] = cp.shape(echo)

        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.fc + cp.fft.fftshift(cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)
        down = down
        right = right
        r_los = down*cp.cos(phi) - right*cp.sin(phi)
        mat_r_los = r_los[:, cp.newaxis] * cp.ones((1, Nr))
        s_rfft = cp.fft.fft(echo, axis=1)
        H_mcl = cp.exp(4j*cp.pi*(mat_f_tau+self.f0)*mat_r_los/self.c)
        s_rfft_mcl = s_rfft * H_mcl
        echo_mcl = cp.fft.ifft(s_rfft_mcl, axis=1)
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
        
        mat_phi_r = cp.tile(phi_r[cp.newaxis, :], (Na, 1))
        down = cp.tile((down)[:, cp.newaxis], (1, Nr))
        right = cp.tile((right)[:, cp.newaxis], (1, Nr))
        delta_r_los =down*(cp.cos(mat_phi_r) - cp.cos(phi)) - right*(cp.sin(mat_phi_r)-cp.sin(phi))

        H_mcl = cp.exp(4j*cp.pi*delta_r_los/self.lambda_)
        echo_mcl = echo * H_mcl
        
        return echo_mcl.get()
    
    def pga_autofocus(self, corrupted_image, num_iter=10, n_scatter = 4, snr_threshold=-10, rms_threshold=0.1):
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

        snr_threshold = 10**(snr_threshold/20)
        eps = cp.finfo(cp.float32).eps
        R_threshold = 1 / 10**(5/20)  
        error_sum = cp.zeros(rows, dtype=cp.float32)
        image_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(corrupted_image, axes=0), axis=0), axes=0)
        
        for iter in range(num_iter):

            # 1. 循环移位：对齐最强散射体至中心
            image_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(corrupted_image, axes=0), axis=0), axes=0)
            image_ffta = image_ffta*cp.exp(-1j*cp.tile(error_sum[:, cp.newaxis], (1, cols)))
            corrupted_image = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(image_ffta, axes=0), axis=0), axes=0)
           
            image = corrupted_image.copy()
            # centered = cp.zeros((rows, cols), dtype=cp.complex64)
            # for i in range(cols):
            #     bin = image[:, i]
            #     midx = cp.argmax(cp.abs(bin))
            #     shift = midpoint - midx
            #     centered[:, i] = cp.roll(bin, shift)

            # Sx = cp.sum(cp.abs(centered)**2, axis=1)
            # winbool = Sx >= (cp.max(Sx)*snr_threshold)
            # win_len = cp.sum(winbool)*1.5
            # window = slice(midpoint - int(win_len//2), midpoint + int(win_len//2)+1)
            # centered = centered[window, :]

            dot_len = 1000
            centered = cp.zeros((dot_len, cols*n_scatter), dtype=cp.complex64)
            window_len = 400
            for c in range(cols):
                bin = image[:, c].copy()
                for k in range(n_scatter):
                    midx = cp.argmax(cp.abs(bin))
                    mval = cp.max(cp.abs(bin))
                    l = cp.maximum(0, midx - window_len//2)
                    r = cp.minimum(rows-1, midx + window_len//2)
                    if l >= r:
                        continue

                    temp = cp.zeros(dot_len, dtype=cp.complex64)
                    bin_len = r - l
                    temp[dot_len//2-bin_len/2:dot_len//2+bin_len/2] = bin[l: r]
                    bin[l:r] = 0
                    centered[:, c*n_scatter + k] = temp

            
            # 截取窗口数据
            # windowed_data = centered*cp.tile(WinBool[:, cp.newaxis], (1, cols))
            Gn = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(centered, axes=0), axis=0), axes=0)

            
            # 3. 相位梯度估计（LUMV）
            # dGn =  cp.roll(Gn, -1, axis=0)
            # numerator = cp.sum(cp.imag(cp.conj(Gn) * dGn), axis=1)
            # denominator = cp.sum(cp.abs(Gn)**2, axis=1)
            # phi_error = numerator / (denominator) 
            


            ## qulity measurement
            quality_threshold = 0.5
            Gn_abs = cp.abs(Gn)
            Gn_quality = cp.var(Gn_abs, axis=0) / (cp.var(Gn_abs, axis=0)  + cp.mean(Gn_abs, axis=0)**2 )
            Gn_quality = cp.tile(Gn_quality[cp.newaxis, :], (Gn.shape[0], 1))
            Gn = Gn * (Gn_quality <= quality_threshold)

            # # WML estimation
            c = cp.mean(cp.abs(Gn), axis=0)
            d = cp.mean(cp.abs(Gn)**2, axis=0)
            R = (4 * (2 * c**2 - d) - 4 * c * cp.sqrt(cp.maximum(0, 4 * c**2 - 3 * d)) + eps) / (d + eps)
            w = 1 / (0.5 * R + 5 / 24 * R**2 + eps)

            w = w * (cp.logical_and(R > 0, R < R_threshold))
            w = cp.tile(w[cp.newaxis, :], (Gn.shape[0], 1))
            w = w / cp.tile((cp.sum(w, axis=1) + eps)[:, cp.newaxis], (1, w.shape[1]))
            phi_error = cp.angle(cp.sum(w * cp.conj(Gn) * cp.roll(Gn, -1, axis=0), axis=1))

            # 积分相位梯度并去除线性趋势
            phi_error = cp.cumsum(phi_error, axis=0)
            # x = cp.arange(len(phi_error))
            # linear_fit = cp.polyfit(x, phi_error, 1)
            # phi_error -= cp.polyval(linear_fit, x)

            # 将窗口内的相位误差映射回完整长度
            xn = cp.linspace(-1, 1, len(phi_error))
            polyfit = cp.polyfit(xn, phi_error, 3)
            x_row= cp.linspace(-1, 1, rows)
            full_phi = cp.polyval(polyfit, x_row)
            # 计算RMS
            rms = cp.sqrt((cp.mean(phi_error**2)))
            error_sum += full_phi
            print(rms.get())
            if(rms < 0.1):
                break
            
            # 4. 相位校正
  
            # corrupted_image = corrupted_image*compensation
            

        
        
        return error_sum.get(), rms.get(), corrupted_image.get()
