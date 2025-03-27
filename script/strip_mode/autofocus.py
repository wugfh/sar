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
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)
        down = down - cp.mean(down)
        right = right - cp.mean(right)
        r_los = down*cp.cos(phi) - right*cp.sin(phi)
        mat_r_los = r_los[:, cp.newaxis] * cp.ones((1, Nr))
        s_rfft = cp.fft.fft(echo, axis=1)
        H_mcl = cp.exp(4j*cp.pi*(mat_f_tau+self.f0)*mat_r_los/self.c)
        s_rfft_mcl = s_rfft * H_mcl
        echo_mcl = cp.fft.ifft(s_rfft_mcl, axis=1)
        return echo_mcl.get()

    
    def pga_autofocus(self, corrupted_image, num_iter=10, min_window_size=32, snr_threshold=10, rms_threshold=0.1):
        """
        
        参数:
        corrupted_image (np.ndarray): 含相位误差的复数SAR图像（距离向 * 方位向）
        num_iter (int): 最大迭代次数
        min_window_size (int): 最小窗口大小
        snr_threshold (float): 窗口阈值（dB）
        rms_threshold (float): 收敛阈值
        
        返回:
        np.ndarray: 校正后的复数图像
        float: RMS误差记录
        """
        rows, cols = corrupted_image.shape
        midpoint = cols // 2

        tau = cp.arange(-rows/2, rows/2, 1)*(1/self.Fs)
        eta = cp.arange(-cols/2, cols/2, 1)*(1/self.PRF)  
        mat_eta, mat_tau = cp.meshgrid(eta, tau)
        mat_R0 = mat_tau*self.c/2+self.R0
        H_deramp = cp.exp(2j*cp.pi*self.Vr**2*mat_eta**2/(mat_R0*self.lambda_))
        
        for iter in range(num_iter):
            # 1. 循环移位：对齐最强散射体至中心
            range_compressed = cp.fft.fft(corrupted_image*H_deramp, axis=1)
            max_indices = cp.argmax(cp.abs(range_compressed), axis=1)
            centered = cp.zeros_like(range_compressed)
            for r in range(rows):
                shift = midpoint - max_indices[r]
                centered[r] = cp.roll(range_compressed[r], shift)
            
            # 2. 加窗：动态确定窗口位置
            Sx = cp.sum(cp.abs(centered)**2, axis=0)
            Sx_dB = 20 * cp.log10(Sx)  # 避免log(0)
            peak = cp.max(Sx_dB)
            cutoff = peak - snr_threshold
            WinBool = Sx_dB >= cutoff
            
            # 计算窗口边界
            start = cp.argmax(WinBool)
            end = len(WinBool) - cp.argmax(WinBool[::-1])
            current_window_size = max(end - start, min_window_size)
            window_start = max(0, midpoint - current_window_size//2)
            window_end = min(cols, midpoint + current_window_size//2)
            window = slice(window_start, window_end)
            
            # 截取窗口数据
            windowed_data = centered[:, window]
            
            # 3. 相位梯度估计（Minimum Variance）
            Gn = cp.fft.ifft(windowed_data, axis=1)
            dGn = Gn - cp.roll(Gn, 1, axis=1)
            numerator = cp.sum(cp.imag(cp.conj(Gn) * dGn), axis=0)
            denominator = cp.sum(cp.abs(Gn)**2, axis=0)
            phi_error = numerator / (denominator)  
            
            # 积分相位梯度并去除线性趋势
            phi_error = cp.cumsum(phi_error)
            x = cp.arange(len(phi_error))
            linear_fit = cp.polyfit(x, phi_error, 1)
            phi_error -= cp.polyval(linear_fit, x)

            # 将窗口内的相位误差映射回完整长度
            full_phi = cp.zeros(cols)
            full_phi[window] = phi_error
            
            # 4. 相位校正
            compensation = cp.exp(-1j * full_phi)
            compensation = cp.tile(compensation[cp.newaxis, :], (rows, 1))
            corrupted_image = cp.fft.ifft(compensation * cp.fft.fft(corrupted_image, axis=1), axis=1)
            # corrupted_image = corrupted_image*compensation
            
            # 计算RMS
            rms = cp.sqrt(cp.mean(full_phi**2))
        
        
        return corrupted_image.get(), rms.get()
