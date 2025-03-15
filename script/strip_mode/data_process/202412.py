import scipy.io as sci
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
sys.path.append(r"../")
from sar_focus import SAR_Focus
from sinc_interpolation import SincInterpolation

cp.cuda.Device(0).use()

class Fcous_Air:
    def __init__(self, Tr, Br, f0, t0, Fr, PRF, fc, Vr):
        self.c = 299792458
        self.Tr = Tr
        self.Br = cp.abs(Br)
        self.f0 = f0
        self.t0 = t0
        self.Fr = Fr
        self.PRF = PRF
        self.fc = fc
        self.Vr = Vr
        self.lambda_ = self.c/self.f0
        self.R0 = t0*self.c/2
        self.Kr = Br/Tr
        self.focus = SAR_Focus(Fr, Tr, f0, PRF, Vr, cp.abs(Br), fc, self.R0, self.Kr)
        
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

        sig = sig[::3]
        self.Na, self.Nr = np.shape(sig)
        tmp = np.zeros((self.Na*22//10, self.Nr), dtype=np.complex64)
        tmp[0:self.Na, 0:self.Nr] = np.array(sig)
        self.sig = np.array(tmp)
        self.Na, self.Nr = np.shape(self.sig)
        print(self.Na, self.Nr)

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
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fft(echo, Nr, axis = 1) 
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hm = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        if(squint_angle > 2):
            data_fft_cr = data_fft_r*Hr*Hm
        else:
            data_fft_cr = data_fft_r*Hr
        data_cr = cp.fft.ifft(data_fft_cr, Nr, axis = 1)
        return data_cr.get()
    
    def rd_focus_ac(self, data_rc, motion_R):  
        [Na, Nr] = cp.shape(data_rc)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_moiton_R = motion_R[:, cp.newaxis] @ cp.ones((1, Nr))


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fft(data_rc, Na, axis=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2 + mat_moiton_R;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta = mat_R0/mat_D - mat_R0
        delta = delta*2/(self.c/self.Fr)
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag

        ## 方位压缩
        # Ka = 2 * self.Vr**2/ (self.lambda_ * mat_R0)
        # Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifft(data_fft_a_rcmc, axis=0)

        data_final = data_ca_rcmc.get()
        # data_final = cp.abs(data_final)/cp.max(cp.max(cp.abs(data_final)))
        # data_final = 20*cp.log10(data_final)
        return data_final
    
    def pga_autofocus(self, corrupted_image, num_iter=10, initial_window_ratio=0.5, snr_threshold=10, rms_threshold=0.1):
        """
        精准PGA自动聚焦算法实现
        
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
            
            # 缩小窗口
            window_size = int(window_size * 0.8)
        
        return img, full_phi, rms_history


if __name__ == '__main__':
    focus_air = Fcous_Air(24e-6, 2e9, 37e9, 3.46e-5, 2.5e9, 5000/3, 0, 72.25)
    altitude = 353
    # focus_air = Fcous_Air(4.175000000000000e-05, -30.111e+06 , 5.300000000000000e+09 ,  6.5959e-03, 32317000, 1.256980000000000e+03, -6900, 7062)
    focus_air.read_data("../../../data/example_49_cropped_sig_rc_small.mat", "../../../data/pos.mat")

    # focus_air.read_data("../../../data/English_Bay_ships.mat", "../../../data/pos.mat")

    print((focus_air.forward[-1] - focus_air.forward[0])/(focus_air.frame_time[-1] - focus_air.frame_time[0]))
    # motion_R = np.sqrt(focus_air.forward**2 + focus_air.right**2 + (focus_air.down+altitude)**2) / np.cos(np.deg2rad(58))
    motion_R = cp.ones(focus_air.Na) * focus_air.R0
    print(motion_R)

    # focus_air.sig = focus_air.rd_focus_rc(cp.array((focus_air.sig)), 10)
    # image_pga = focus_air.sig
    image_pga, phi, rms = focus_air.pga_autofocus(cp.array((focus_air.sig)), 10)
    # print(rms)
    image = focus_air.rd_focus_ac(cp.array((focus_air.sig)), cp.array(motion_R))

    image_abs = np.abs(image)
    image_abs = image_abs/np.max(np.max(image_abs))
    image_abs = 20*np.log10(image_abs+1)
    image_abs = image_abs**0.3
    plt.figure(figsize=(4,12))
    plt.imshow(image_abs, cmap='gray', aspect='auto')
    plt.tight_layout()
    plt.savefig("../../../fig/data_202412/image.png")

        
        