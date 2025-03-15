import scipy.io as sci
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
from scipy import signal
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
        echo (numpy array): echo data before range compress.
        
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
        H_mcl = cp.exp(4j*(mat_f_tau+self.f0)*mat_r_los/self.c)
        s_rfft_mcl = s_rfft * H_mcl
        echo_mcl = cp.fft.ifft(s_rfft_mcl, axis=1)
        return echo_mcl.get()

    def pga(self, corrupted_image, num_iter=10, initial_window_ratio=0.5, snr_threshold=10, rms_threshold=0.1):
        """
        
        参数:
        corrupted_image (np.ndarray): 含相位误差的复数SAR图像（方位向×距离向）
        num_iter (int): 最大迭代次数
        initial_window_ratio (float): 初始窗口占全长的比例
        snr_threshold (float): 窗口阈值（dB）
        rms_threshold (float): 收敛阈值
        
        返回:
        np.ndarray: 校正后的复数图像
        np.ndarray: 估计的相位误差
        list: RMS误差记录
        """
        img = corrupted_image.copy()
        rows, cols = img.shape
        rms_history = []
        midpoint = cols // 2
        window_size = int(cols * initial_window_ratio)  # 初始窗口大小
        
        for iter in range(num_iter):
            # 1. 循环移位：对齐最强散射体至中心
            range_compressed = cp.fft.fft(img, axis=1)
            max_indices = cp.argmax(cp.abs(range_compressed), axis=1)
            centered = cp.zeros_like(range_compressed)
            for r in range(rows):
                shift = midpoint - max_indices[r]
                centered[r] = cp.roll(range_compressed[r], shift)
            
            # 2. 加窗：动态确定窗口位置
            Sx = cp.sum(cp.abs(centered)**2, axis=0)
            Sx_dB = 20 * cp.log10(Sx + 1e-10)  # 避免log(0)
            peak = cp.max(Sx_dB)
            cutoff = peak - snr_threshold
            WinBool = Sx_dB >= cutoff
            
            # 计算窗口边界
            start = cp.argmax(WinBool)
            end = len(WinBool) - cp.argmax(WinBool[::-1])
            current_window_size = min(window_size, end - start)
            window_start = max(0, midpoint - current_window_size//2)
            window_end = min(cols, midpoint + current_window_size//2)
            window = slice(window_start, window_end)
            
            # 截取窗口数据
            windowed_data = centered[:, window]
            
            # 3. 相位梯度估计（论文式4）
            Gn = cp.fft.ifft(windowed_data, axis=1)
            x_shift = cp.arange(windowed_data.shape[1]) - windowed_data.shape[1]//2
            dGn = cp.fft.ifft(1j * x_shift * windowed_data, axis=1)
            numerator = cp.sum(cp.imag(cp.conj(Gn) * dGn), axis=0)
            denominator = cp.sum(cp.abs(Gn)**2, axis=0)
            phi_grad = numerator / (denominator + 1e-10)  # 避免除零
            
            # 积分相位梯度并去除线性趋势
            phi_error = cp.cumsum(phi_grad)
            x = cp.arange(len(phi_error))
            linear_fit = cp.polyfit(x, phi_error, 1)
            phi_error -= cp.polyval(linear_fit, x)
            
            # 将窗口内的相位误差映射回完整长度
            full_phi = cp.zeros(cols)
            full_phi[window] = phi_error
            
            # 4. 相位校正
            compensation = cp.exp(-1j * full_phi)
            img = cp.fft.ifft(cp.fft.fft(img, axis=1) * compensation, axis=1)
            
            # 计算RMS
            rms = cp.sqrt(cp.mean(full_phi**2))
            rms_history.append(rms)
            print(f"Iter {iter+1}: RMS = {rms:.3f} rad")
            
            # 收敛检查
            if rms < rms_threshold:
                print(f"Converged at iteration {iter+1}")
                break
            
            # 缩小窗口
            window_size = int(window_size * 0.8)
        
        return img, full_phi, rms_history
