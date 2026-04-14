import numpy as np
import cupy as cp
import sys
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
import scipy.io as sio
from inverse_conv import recover_dft_phase
from concurrent.futures import ThreadPoolExecutor

class AutoFocus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0, theta_width):                         
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
        self.Kr = self.B/self.Tp
        self.theta_width = theta_width

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

        R_eta = cp.sqrt(forward**2+self.R0**2)
        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)
        r_los = cp.sqrt(down**2 + right**2+forward**2)-R_eta
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

    def phase_reover(self, sig, win_spectrum, snr):

        ## wiener recovery
        win_toep = cp.array(scipy.linalg.toeplitz(win_spectrum.get()))
        est = win_toep.T.conj() @ sig
        return cp.angle(est)
    
    def line_pga(self, corrupted_image, block_len, num_iter=10, snr=0):
        rows, cols = corrupted_image.shape
        midpoint = rows // 2
        # print("Estimated SNR (dB):", snr)
        snr_threshold = 10**(snr/10)
        eps = cp.finfo(cp.float32).eps
        pre_win_len = rows/2
        pre_rms = 0.1
        phi_error = cp.zeros((rows, cols), dtype=cp.float32)
        range_res = self.c/(2*self.B)
        azimuth_res = self.lambda_/(self.theta_width*2)
        range_width = cp.ceil(range_res*30/(self.c/(2*self.Fs))).astype(cp.int32)
        azimuth_with = cp.ceil(azimuth_res*60/(self.Vr/self.PRF)).astype(cp.int32)
        error_sum = cp.zeros((rows,cols), dtype=cp.float32)
        image_iffta = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(corrupted_image, axes=0), axis=0), axes=0)

        eta = (cp.arange(0,rows)-rows//2)*(1/self.PRF)   

        for iter in range(num_iter):

            # 1. 循环移位：对齐最强散射体至中心
            image_iffta = image_iffta*cp.exp(-1j*error_sum)
            
            image = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image_iffta, axes=0), axis=0), axes=0)
            image_est = cp.abs(image)
            centered = []
            area = cp.zeros((rows,), dtype=cp.int32)
            pos = []
            while cp.sum(area) < rows*0.9:
                midx = cp.unravel_index(cp.argmax(cp.abs(image_est)), image.shape)
                pos.append(midx)
                bin = image[:, midx[1]].copy()
                bin = cp.roll(bin, midpoint - midx[0])
                centered.append(bin[:, cp.newaxis])
                left = cp.maximum(midx[0] - block_len//2, 0)
                right = cp.minimum(midx[0] + block_len//2, rows)
                area[left:right] =1
                left = cp.maximum(midx[0] - azimuth_with, 0)
                right = cp.minimum(midx[0] + azimuth_with, rows)
                up = cp.maximum(midx[1] - range_width, 0)
                down = cp.minimum(midx[1] + range_width, cols)  
                image_est[left:right, up:down] = 0
               
            centered = cp.concatenate(centered, axis=1)
            print(centered.shape)
            Sx = cp.sum(cp.abs(centered)**2, axis=1)
            winbool = Sx >= (cp.max(Sx)*(snr_threshold))
            win_len = cp.sum(winbool)
            x = cp.arange(rows) - midpoint
            window =  cp.exp(-0.5 * ((x) / win_len) ** 2)
            # window = cp.abs(x)<win_len//2
            win_spectrum = cp.real(cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(window))))
            centered = centered * cp.tile(window[:, cp.newaxis], (1, centered.shape[1]))

            if np.abs(1-win_len/pre_win_len) < 0.05 or win_len > pre_win_len*1.05:
                error_sum -= phi_error
                if win_len > 10:
                    win_len = cp.array(pre_win_len)
                    rms = cp.array(pre_rms)
                break
            Gn = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(centered, axes=0), axis=0), axes=0) 

            # for i in range(Gn.shape[1]):
            #     tmp, info =recover_dft_phase(Gn[:, i],win_spectrum, 1e-6,tol=1e-5, max_iter=1000)
            #     if info != 0:
            #         print("CG did not converge for column {}".format(i))
            #     Gn[:, i] = cp.array(tmp)

            val = Gn * cp.roll(cp.conj(Gn), 1, axis=0)
            # power = cp.sum(cp.abs(val)**2, axis=1)
            # thresh = cp.max(power)/100

            # # WLS estimation
            # c = cp.mean(cp.abs(Gn), axis=0)
            # d = cp.mean(cp.abs(Gn)**2, axis=0)
            # R = (4 * (2 * c**2 - d) - 4 * c * cp.sqrt(cp.maximum(0, 4 * c**2 - 3 * d)) + eps) / (d + eps)
            # w = 1 / (0.5 * R + 5 / 24 * R**2 + eps)

            # ## WPGA 权重计算
            # w = w * (cp.logical_and(R > 0, R < R_threshold))
            # w = cp.tile(w[cp.newaxis, :], (val.shape[0], 1))
            # w = w / cp.tile(cp.sqrt(cp.sum(abs(w)**2, axis=1) + eps)[:, cp.newaxis], (1, w.shape[1]))
            phi_error = cp.angle(cp.sum(val, axis=1))
            # phi_error = self.phase_reover(cp.exp(1j*phi_error), win_spectrum, 0.001)
            # phi_error = phi_error*(power>thresh)
            # 计算RMS
            rms = cp.sqrt(cp.mean(cp.mean((phi_error)**2)))
            phi_error = cp.cumsum(phi_error, axis=0)
            phi_error = cp.tile(phi_error[:, cp.newaxis], (1, cols)) 

            
            # phi_error = cp.unwrap(phi_error, axis=0)
            print("rms:{} winlen:{}".format(rms.get(), win_len))


            pre_win_len = win_len
            pre_rms = rms
            error_sum += phi_error

            # rms = cp.sqrt(cp.mean((error-error_sum)**2))

            # if(rms < 0.1):
            #     break
                # 去除error_sum每一列的线性项
        # x = cp.arange(rows)
        # error_sum = cp.unwrap(error_sum, axis=0)
        # y = error_sum[:, 0]
        # A = cp.vstack([x, cp.ones_like(x)]).T
        # # 最小二乘拟合直线
        # m, b = cp.linalg.lstsq(A, y, rcond=None)[0]
        # error_sum = error_sum - (m * cp.tile(x[:,cp.newaxis], (1, cols)) + b)

        return error_sum.get(), rms.get(), win_len.get()
    def mat_pga(self, corrupted_image,num_iter=10, snr=0, range_win = 30):
        rows, cols = corrupted_image.shape
        midpoint = rows // 2

        # print("Estimated SNR (dB):", snr)
        snr_threshold = 10**(snr/10)
        eps = cp.finfo(cp.float32).eps
        pre_win_len = 1e5
        R_threshold = 1 / 10**(5/20)  
        range_res = self.c/(2*self.B)
        error_sum = cp.zeros((rows,cols), dtype=cp.float32)
        image_iffta = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(corrupted_image, axes=0), axis=0), axes=0)
        phi_error = cp.zeros((rows, cols), dtype=cp.float32)
        for iter in range(num_iter):

            # 1. 循环移位：对齐最强散射体至中心
            image_iffta = image_iffta*cp.exp(-1j*error_sum)
            
            image = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image_iffta, axes=0), axis=0), axes=0)
           
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

            x = cp.arange(0, rows)
            winbool = (x > win_start) & (x <win_end)

            centered = centered * cp.tile(winbool[:, cp.newaxis], (1, cols))


            # 截取窗口数据
            # windowed_data = centered*cp.tile(WinBool[:, cp.newaxis], (1, cols))
            Gn = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(centered, axes=0), axis=0), axes=0) 
            val = Gn * cp.roll(cp.conj(Gn), 1, axis=0)
            power = cp.sum(cp.abs(val)**2, axis=1)
            thresh = cp.max(power)/10

            sinc_win_len = range_win*range_res/(self.c/(2*self.Fs)) 
            sinc_win_len = int(cp.ceil(sinc_win_len))
            if sinc_win_len % 2 == 0:
                sinc_win_len += 1  # 保证对称
            sigma = sinc_win_len / (2 * np.sqrt(2 * np.log(10)))  # 使窗外约为-10dB
            # 将循环转为矩阵运算
            # 构造Gabor窗矩阵，中心为每一列，窗长为sinc_win_len，窗外衰减到10dB
            x = cp.arange(sinc_win_len) - sinc_win_len//2
            sinc_window =  cp.exp(-0.5 * ((x) / sigma) ** 2)

            for i in range(rows):
                phi_error[i, :] = cp.angle(cp.convolve(val[i, :], sinc_window, mode='same'))
    
            # 计算RMS
            rms = cp.sqrt(cp.mean(cp.mean((phi_error)**2)))
            phi_error = cp.cumsum(phi_error, axis=0)
            phi_error = cp.tile(phi_error[:, cp.newaxis], (1, cols)) 

            
            # phi_error = cp.unwrap(phi_error, axis=0)
            print("rms:{} winlen:{}".format(rms.get(), win_len))
            if np.abs(1-win_len/pre_win_len) < 0.01 or win_len>pre_win_len*1.05 or rms < 1e-4:
                win_len = pre_win_len
                break

            pre_win_len = win_len
            error_sum += phi_error

            
            

        # 去除error_sum每一列的线性项
        x = cp.arange(rows)
        error_sum = cp.unwrap(error_sum, axis=0)
        for col in range(cols):
            y = error_sum[:, col]
            A = cp.vstack([x, cp.ones_like(x)]).T
            # 最小二乘拟合直线
            m, b = cp.linalg.lstsq(A, y, rcond=None)[0]
            error_sum[:, col] = y - (m * x + b)

        return error_sum.get(), rms.get(), win_len.get()
    

    def estimate_ape_2d(self,centered, snr_threshold, theta_width, pos):
        # ref,_= self.spga(cp.array(centered), 1, snr_threshold, 30, 10, method="line", range_win=30)
       ##接收机时间窗
        [Na,Nr]= cp.shape(centered)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fs)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((Na, Nr), dtype=cp.complex64)
        R0_tar = self.R0+pos[1]
        R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - pos[0])**2)

        signal_t = cp.zeros((Na, Nr), dtype=cp.complex64)
        signal_r = cp.zeros((Na, Nr), dtype=cp.complex64)

        Wr = cp.abs(mat_tau-(2*R_eta/self.c))<self.Tp/2 
        phase_r = cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
        ## 发送机到点目标
        signal_t = Wr*phase_r
        ## 点目标到接收机
        signal_r = signal_t

        Tstrip_tar = theta_width*R0_tar/(self.Vr*cp.cos(self.theta_c)**2)
        Wa =  cp.abs(mat_eta-(pos[0]/self.Vr + eta_c)) < Tstrip_tar/2
        # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
        phase_a = cp.exp(-4j*cp.pi*R_eta/self.lambda_)
        signal_a = Wa*phase_a
        ref = signal_r*signal_a

        focus = SAR_Focus(self.Fs, self.Tp, self.f0, self.PRF, self.Vr, self.B, self.fc, self.R0,self.Kr,theta_width)
        ref = focus.erma_rcmc(ref)
        ref = focus.erma_ac(ref)

        center_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(centered)))
        ref_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ref)))
        phase_diff = cp.angle(center_fft2*cp.conj(ref_fft2))
        phase_diff = phase_diff[:, centered.shape[1]//2]
        return phase_diff

    def compensate_R(self, sig, snr_threshold, theta_width):
        [rows, cols] = sig.shape
        centered = sig.copy()
        ## select point
        midx = cp.unravel_index(cp.argmax(cp.abs(centered)), centered.shape)
        midpoint_row = rows//2
        midpoint_col = cols//2
        centered = cp.roll(centered,(midpoint_row - midx[0]), axis=0)
        centered = cp.roll(centered,(midpoint_col - midx[1]), axis=1)
        midx = (midpoint_row, midpoint_col)

        S = cp.max(cp.max(cp.abs(centered)))
        snr_threshold = 10**(snr_threshold/20)
        win_cols = cp.abs(centered[midx[0],:])>snr_threshold*S
        win_rows = cp.abs(centered[:,midx[1]])>snr_threshold*S
        win_rows = cp.sum(win_rows)*1.5
        win_cols = cp.sum(win_cols)*1.5
        rows_slice = slice(max(midx[0] - win_rows//2, 0), min(midx[0] + win_rows//2, rows))
        cols_slice = slice(max(midx[1] - win_cols//2, 0), min(midx[1] + win_cols//2, cols))
        W = cp.zeros_like(centered)
        W[rows_slice, cols_slice] = 1
        centered = centered * W



        ## estimate APE
        pos = cp.array(midx)
        pos[0] = (pos[0]-rows/2)*self.Vr/self.PRF
        pos[1] = (pos[1]-cols/2)*self.c/(2*self.Fs)
        error = self.estimate_ape_2d(centered, snr_threshold, theta_width, pos)


        ## 2d spectrum compensation
        ftau = cp.arange(-cols/2, cols/2, 1)*(self.Fs/cols)+self.f0
        feta = cp.arange(-rows/2, rows/2, 1)*(self.PRF/rows)+self.fc
        error_2d = cp.zeros((rows, cols), dtype=cp.float32)
        mat_ftau = cp.tile(ftau[cp.newaxis, :], (rows, 1))
        for i in range(cols):
            error_2d[:,i] = cp.interp(self.f0/ftau[i]*feta, feta, cp.array(error))
        re_value = mat_ftau*error_2d/self.f0


        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(sig)))
        sig_fft2_compensated = sig_fft2 * cp.exp(-1j*re_value)

        ## compensate residual rcm
        error = cp.unwrap(error)
        error = error - cp.mean(error)
        phase_dR = -error/(4*cp.pi)*self.lambda_
        mat_phase_dR = cp.tile(phase_dR[:, cp.newaxis], (1, cols))

        ftau = cp.arange(-cols/2, cols/2, 1)*(self.Fs/cols)
        mat_ftau = cp.tile(ftau[cp.newaxis, :], (rows, 1))
        sig_compensated = sig_fft2_compensated * cp.exp(1j*4*cp.pi*(mat_ftau+self.f0)*mat_phase_dR/self.c)


        sig = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_compensated)))

        return sig.get(),phase_dR.get()

    def spga(self, sig, block_num, snr_threshold, num_iter=10, win_min=10, method = "mat", range_win = 30):
        [Na, Nr] = sig.shape
        # Ka = cp.tile(ka[cp.newaxis, :], (Na, 1))
        # sig = self.dechirp(sig, Ka)
        if block_num == 1:
            bsize = Na
            block_len = Na
        else:
            block_len = Na//block_num
            if block_len % 2 == 1:
                block_len += 1
            bsize = Na
        lmid = np.arange(0, Na, bsize) + bsize//2

        step = 0
        error_sum = cp.zeros((Na,Nr))
        sum_cnt = cp.zeros((Na,Nr))
        win_len_list = cp.zeros_like(snr_threshold)
        for mid in lmid:
            step += 1
            start = np.maximum(0, int(mid - bsize/2))
            end = int(np.minimum(start+bsize, Na))
            if end-mid < win_min:
                continue
            # print("step:{}, start:{}, end:{}".format(step, start, end))
            
            block = cp.zeros_like(sig)
            block[start:end, :] = cp.array(sig[start:end, :])
            # block,_ = self.compensate_R(block, -20, self.theta_width)

            print("block {}:start {}, end {}".format(step-1, start, end))

     
            if method == "mat":
                mat_error, rms, winlen = self.mat_pga(cp.array((block)), num_iter=num_iter, snr = snr_threshold[step-1], range_win=range_win)
            if method == "line":
                mat_error, rms, winlen = self.line_pga(cp.array((block)), block_len, num_iter=num_iter, snr = snr_threshold[step-1])
            print("RMS error:{}  winlen:{}\r\n".format(rms,winlen))
            mat_error = cp.array(mat_error)
            win_len_list[step-1] = winlen
            if start > 0:
                mat_error[start:end, :] = mat_error[start:end, :] + error_sum[start-1, :] - mat_error[start, :]
            error_sum[start:end, :] += mat_error[start:end, :]
            sum_cnt[start:end, :] += 1
        error_sum = error_sum / (sum_cnt)
        return error_sum.get(), win_len_list.get()

    def dechirp(self, data):
        Na, Nr = cp.shape(data)
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*self.R0)
        tau = (cp.arange(0,Nr)-Nr//2)*(1/self.Fs)
        eta =  -self.Rc*cp.sin(self.theta_c)/self.Vr + (cp.arange(0,Na)-Na//2)*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        mat_R0 = mat_tau*self.c/2 + self.R0;  
        R_eta = cp.sqrt(mat_R0**2 + (self.Vr*mat_eta)**2)
        data = data*cp.exp(1j*cp.pi*Ka*mat_eta**2)
        data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
        return data.get()
    
    def rechirp(self, data):
        Na, Nr = cp.shape(data)
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*self.R0)
        feta = (cp.arange(0,Na)-Na//2)*(self.PRF/Na)+self.fc
        mat_feta = cp.tile(feta[:, cp.newaxis], (1, Nr))
        data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
        data = data*cp.exp(1j*cp.pi*mat_feta**2/Ka)
        data = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data, axes=0), axis=0), axes=0)
        return data.get()
    
    def down_res(self, sig, down_rate):
        Na, Nr = cp.shape(sig)
        sig_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)
        W = cp.zeros_like(sig_fftr)
        new_width = int(self.B/down_rate)/(self.Fs/Nr)
        W[:, Nr//2-new_width//2:Nr//2+new_width//2] = 1
        sig_fftr = sig_fftr*W
        sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_fftr, axes=0), axis=0), axes=0)
        return sig.get()

if __name__ == "__main__":
    Na = 40000
    Fa = 40000
    eta = cp.arange(-Na/2, Na/2)*(1/Fa)
    Ka = 40000
    signal = 1*cp.exp(1j*cp.pi*Ka*eta**2) + 2*cp.exp(1j*cp.pi*Ka*(eta-Na/(Fa*3))**2) 
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
    signal = signal + noise

    feta = cp.arange(-Na/2, Na/2)*(Fa/Na)

    
    signal_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal)))
    signal_fft = signal_fft*cp.exp(1j*cp.pi*(feta**2)/Ka)
    signal_no = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(signal_fft)))

    ##dechirp
    # signal_dechirp = signal*cp.exp(-1j*cp.pi*Ka*(eta**2))
    # signal_dechirp = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(signal_dechirp)))
    # signal_dechirp = cp.conj(signal_dechirp)
    signal_dechirp = signal_no
    R0 = 1
    autofocus = AutoFocus(Fs=20e6, Tp=10e-6, f0=5.3e9, PRF=Fa, Vr=150, B=20e6, fc=0, R0=R0)
    error, _, _ = autofocus.line_pga(signal_dechirp[:, cp.newaxis], R0, num_iter=1, snr = -40)
    
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
    plt.subplot(211)
    plt.plot(eta.get(), cp.unwrap(phi_error).get(), label="true")
    plt.subplot(212)
    plt.plot(eta.get(), cp.unwrap(error).get(), label="estimated")
    plt.savefig("./pga_error.png")