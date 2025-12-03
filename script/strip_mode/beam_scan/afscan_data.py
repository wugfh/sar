import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys

sys.path.append(r"../../")
from autofocus import AutoFocus
from dot_estimate import DotEstimator
from sar_focus import SAR_Focus
sys.path.append(r"./")
from afscan_vehicle import FScanAzimuth
from multiprocessing import Process, Queue
import multiprocessing as mp
import cv2
from tqdm import tqdm
from sinc_interpolation import SincInterpolation
from joblib import Parallel, delayed
import scipy.optimize as op
import h5py
import numpy as np
import scipy.io as sio


class AFScanData(FScanAzimuth):
    def __init__(self, param_path, data_path, pos_path):
        super().__init__()
        self.read_data(data_path, pos_path, param_path)
        self.theta_c = np.deg2rad(0)
        self.theta_width = np.deg2rad(8.06)  # beam width in elevation
        self.theta_lowf = np.deg2rad(17.94) 
        self.theta_upf = np.deg2rad(10.9)
        self.theta_az = np.deg2rad(2.5) ## maximum azimuth beam angle width while scanning
        self.theta_sc = np.abs(self.theta_upf - self.theta_lowf)  # scanning angle width
        self.alpha = self.Br/(self.theta_upf - self.theta_lowf)
        self.R0 = (self.t0)*self.c/2
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Vr = np.mean(np.diff(self.forward)/np.diff(self.frame_time))

        print("imaging time:", self.frame_time.max()-self.frame_time.min(), len(self.frame_time))
        print("Vr: {}, R0: {}".format(self.Vr, self.R0))
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c
        self.inc = cp.arccos(self.R0*np.cos(self.phi)/self.Rc)
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+(self.theta_width)/2) - np.sin(self.theta_c-self.theta_width/2))/self.lambda_
        self.da = self.Vr/self.Bd
        self.dr = (self.theta_sc/self.theta_az)*self.c/(2*self.Br)
        self.T_ap = self.theta_sc*self.R0/self.Vr
        
        print("therical azimuth res:{}, therical range res:{}".format(self.Vr/self.Bd, (self.theta_sc/self.theta_az)*self.c/(2*self.Br)))



    def read_data(self, data_filename, pos_filename, param_filename):
        with h5py.File(data_filename, "r") as data:
            sig = data['sig']
            sig = sig["real"] + 1j*sig["imag"]
            self.sig = np.array(sig)

        [self.Na, self.Nr] = sig.shape

        with h5py.File(pos_filename) as pos:    
            self.forward = np.squeeze(np.array(pos['forward']))
            self.down = np.squeeze(np.array(pos['down']))
            self.right = np.squeeze(np.array(pos['right']))
            self.frame_time = self.time2sec(np.squeeze(np.array(pos['frame_time'])))


        param = sio.loadmat(param_filename) 
        grp = param['params'] if 'params' in param else param
        self.Fr = float(np.squeeze(grp['Fr']))
        self.t0 = float(np.squeeze(grp['t0']))
        self.Br = float(np.squeeze(grp['Br']))
        self.f0 = float(np.squeeze(grp['f0']))
        self.PRF = float(np.squeeze(grp['PRF']))
        self.Tp = float(np.squeeze(grp['Tr']))
        self.Kr = self.Br / self.Tp
        self.lambda_ = self.c / self.f0
        self.phi = np.deg2rad(20)


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
    
    def shift_doppler(self, sig, fd_shift):
        """
        Shift the Doppler frequency of the signal.
        
        Parameters:
        sig (cupy array): Input signal array.
        fd_shift (float): Doppler frequency shift value.
        
        Returns:
        cupy array: Doppler frequency shifted signal.
        """
        Na, Nr = sig.shape
        t_az = cp.arange(Na)/self.PRF
        shift_phase = cp.exp(1j*2*cp.pi*fd_shift*cp.reshape(t_az, (Na, 1)))
        sig_shifted = sig * shift_phase
        return sig_shifted

    def doppler_downsample(self, sig,Fa, Fa_down):
        """
        Downsample the Doppler frequency of the signal.
        
        Parameters:
        sig (cupy array): Input signal array.
        fd_shift (float): Doppler frequency shift value.
        
        Returns:
        cupy array: Doppler frequency downsampled signal.
        """
        Na, Nr = sig.shape
        sig = cp.array(sig)
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(sig)))
        new_width = int(Na * Fa_down / Fa)
        sig_downsampled = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2[(Na/2 - new_width//2):(Na/2 + new_width//2),:])))
        return sig_downsampled.get()
    
    def doppler_shift(self, sig, fa_shift):
        """
        Shift the Doppler frequency of the signal.
        
        Parameters:
        sig (cupy array): Input signal array.
        fa_shift (float): Doppler frequency shift value.
        
        Returns:
        cupy array: Doppler frequency shifted signal.
        """
        Na, Nr = sig.shape
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig))))
        shift = fa_shift * Na / self.PRF
        sig_shift_fft2 = cp.roll(sig_fft2, int(shift), axis=0)
        sig_shift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_shift_fft2)))
        return sig_shift.get()
    
    def squint_sm(self, sig):
        [Na, Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c+cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.feta_c + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        mat_tau, mat_eta = cp.meshgrid(tau, eta)

        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)
        H2 = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        # doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
        # tau_mid = self.alpha*(doa1-self.theta_c)/self.Kr
        H1 = cp.exp(2j*cp.pi*self.feta_c*(1+mat_f_tau/self.f0)*mat_eta)
        sig_ftau_eta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=1), axis=1), axes=1)
        # sig_ftau_eta = sig_ftau_eta*H1
        sig_ftau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig_ftau_eta, axes=0), axis=0), axes=0)
        sig_ftau_feta = sig_ftau_feta*H2
        sig_shift = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(sig_ftau_feta)))
        return sig_shift.get()

    def afscan_spectrum_orth(self, sig):
        [Na, Nr] = cp.shape(sig)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.feta_c + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = 2*self.Rc/self.c+cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = self.eta_c+cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(sig)))
        mat_ftau_center = (cp.arcsin(mat_f_eta*self.lambda_/(2*self.Vr))-self.theta_c)*self.alpha
        delta = -mat_ftau_center/(self.Fr/Nr)

        delta = delta-cp.mean(cp.mean(delta))
        sinc_N = 8
        sig_fft2 = cp.ascontiguousarray(sig_fft2)
        sig_fft2_real = cp.real(sig_fft2).astype(cp.double)
        sig_fft2_imag = cp.imag(sig_fft2).astype(cp.double)
        sinc_intp = SincInterpolation()
        sig_shift_fft2_real = sinc_intp.sinc_interpolation(sig_fft2_real, delta, Na, Nr, sinc_N)  
        sig_shift_fft2_imag = sinc_intp.sinc_interpolation(sig_fft2_imag, delta, Na, Nr, sinc_N)
        sig_shift_fft2 = sig_shift_fft2_real + 1j*sig_shift_fft2_imag
        sig_shift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_shift_fft2)))
        return sig_shift.get()

    def rd_focus_rcmc(self, data_rc):  
        data_rc = cp.array(data_rc)
        [Na, Nr] = cp.shape(data_rc)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        f_eta = self.feta_c + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rc, axes=0), Na, axis=0), axes=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta =  mat_R0/mat_D - mat_R0 
        delta = delta*2/(self.c/self.Fr)
        delta = delta-cp.mean(cp.mean(delta))
        # delta = cp.zeros_like(delta)  # disable RCMC for testing
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag

        data_final = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)

        return data_final.get()
    
    def rd_focus_ac(self, data_rcmc):
        [Na, Nr] = cp.shape(data_rcmc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.feta_c + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = 2*self.Rc/self.c+cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_R0 = mat_tau*self.c/2;  

        data_fft_a_rcmc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rcmc, axes=0), axis=0), axes=0)
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## 方位压缩
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*self.R0)
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        # Ha = cp.exp(4j*cp.pi*mat_D*self.R0*self.f0/self.c)
        offset = cp.exp(-2j*cp.pi*mat_f_eta*self.eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha*offset
        data_ca_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)
        data_final = data_ca_rcmc

        return data_final.get()

    def dechirp(self, data, Ka):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        # mat_R0 = mat_tau*self.c/2 + self.R0;  

        data = data*cp.exp(1j*cp.pi*Ka*mat_eta**2)
        data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
        return data.get()
    
    def rechirp(self, data, Ka):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        # mat_R0 = mat_tau*self.c/2 + self.R0;  
        data = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data, axes=0), axis=0), axes=0)
        data = data*cp.exp(-1j*cp.pi*Ka*mat_eta**2)
        return data.get()
    
    
    def spga(self, sig, ka):
        afoucs = AutoFocus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.fc, self.R0)
        [Na, Nr] = sig.shape
        

        bsize = int(Na/2)
        lstart = np.arange(0, Na, bsize)
        step = 0
        focus_image = cp.zeros((Na, Nr), dtype=cp.complex128)
        for start in lstart:
            step += 1
            start = int(start)
            end = int(np.minimum(start+bsize, Na))
            print("step:{}, start:{}, end:{}".format(step, start, end))
            block = sig[start:end, :]
            block_dechirp = self.dechirp(cp.array((block)), ka)
            error, rms, windata = afoucs.pga_autofocus(cp.array((block_dechirp)), num_iter=30)
            error_sum = cp.array(error)
            poly_fit = cp.polyfit(cp.arange(error_sum.shape[0]), error_sum, 1)
            poly_value = cp.polyval(poly_fit, cp.arange(error_sum.shape[0]))
            error_sum = error_sum - poly_value
            error_sum = cp.tile(error_sum[:, cp.newaxis], (1, Nr))
            
            block_dechirp_ffta  = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(block_dechirp, axes=0), axis=0), axes=0)
            block_dechirp_ffta = block_dechirp_ffta*cp.exp(-1j*error_sum)
            block_dechirp = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(block_dechirp_ffta, axes=0), axis=0), axes=0)
            block = block_dechirp
            if step == len(lstart):
                focus_image[start:, :] += block
            else:
                focus_image[start:lstart[step], :] += block[:lstart[step]-start, :]
        sig = focus_image.get()
        return sig,error

    def azimuth_interp(self, sig):
        [Na,Nr] = cp.shape(sig)
        min_forward = cp.min(cp.array(self.forward))
        da = min_forward + cp.arange(Na)*(self.Vr/self.PRF)
        for i in range(Nr):
            sig[:,i] = cp.interp(da, cp.array(self.forward), sig[:,i])
        return sig.get()

    def estimate_res(self, sig, v):
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig))))
        max_val = cp.max(cp.abs(sig_fft2), axis=1)
        midx = cp.argmax(cp.abs(max_val))
        max_slice = cp.abs(sig_fft2)[midx,:]
        win = max_slice>cp.max(max_slice)*0.1
        bandwidth = cp.sum(win)*(afscan.Fr/afscan.Nr)
        res = v/(bandwidth*2)
        return res
    

        
    def compensate_residual_rcm(self, sig):
        [Na, Nr] = cp.shape(sig)


        # T_block = self.T_ap/5
        # block_len = T_block/(1/self.PRF)
        N_b = 1

        blocks = cp.array_split(sig, N_b, axis=0)
        delta_tau = cp.zeros((Na))
        index = 1

        uprate = (1,8)
        max_rcm = int(cp.floor(Nr*uprate[1]//8))
        dis = 0
        for block in blocks:
            corr = cp.zeros((Na,max_rcm+10), dtype=cp.complex128)
            [bNa, bNr] = cp.shape(block)
            max_pos = cp.unravel_index(cp.argmax(block), block.shape)
            crop_size = (int(1.5/(afscan.Vr/afscan.Fa)), int(3/(afscan.c/(2*afscan.Fr))))
            W = cp.zeros_like(block)
            W[cp.maximum(0, max_pos[0]-crop_size[0]//2):cp.minimum(Na, max_pos[0]+crop_size[0]//2), cp.maximum(0, max_pos[1]-crop_size[1]//2):cp.minimum(Nr, max_pos[1]+crop_size[1]//2)] = 1
            sig_crop = block*W

            block_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift((sig_crop), axes=0), axis=0), axes=0)
            mean_power = cp.sqrt(cp.mean(cp.mean(cp.abs(block_ffta)**2)))
            block_ffta = (cp.abs(block_ffta)>mean_power*0.1)*block_ffta

            block_ffta = cp.array(self.upsample(block_ffta, uprate))
            G_hat = cp.zeros_like(block_ffta)
            G_hat[0,:] = block_ffta[0,:]
            
            for l in tqdm(range(0, bNa-1)):
                G = cp.squeeze(block_ffta[l+1,:])
         
                if l >=dis:
                    G_ref = G_hat[l-dis,:]
                else:
                    G_ref = G_hat[0,:]
                value = cp.sqrt(cp.sum(cp.abs(G_ref)**2)*cp.sum(cp.abs(G)**2))
                for i in range(-max_rcm//2, max_rcm//2):
                    corr[index, i + max_rcm//2] = cp.sum(cp.abs(G_ref)*cp.abs(cp.roll(G, i)))/value
                shift = cp.argmax(cp.abs(corr[index, :])) - max_rcm//2
                delta_tau[index] = -shift/(self.Fr*uprate[1]) 
                G_hat[l+1, :] = cp.roll(G, shift)
                index = index+1

        for i in range(dis+1):
            if i == dis:
                break
            if(dis == 0):
                delta = delta_tau
            # Smooth delta_tau (remove spikes and reduce noise)
            if Na > 1:
                delta[0] = delta[1]

            # Choose an odd window size based on azimuth length, clamp to reasonable bounds
            win = int(max(5, min(201, (Na // 100) * 2 + 1)))

            # Moving average using cumulative sum on GPU
            pad_l = win // 2
            pad_r = win - pad_l
            x_pad = cp.pad(delta, (pad_l, pad_r), mode='edge')
            cs = cp.cumsum(x_pad)
            delta = (cs[win:] - cs[:-win]) / win

            # Optional second pass for extra smoothing
            x_pad = cp.pad(delta, (pad_l, pad_r), mode='edge')
            cs = cp.cumsum(x_pad)
            delta = (cs[win:] - cs[:-win]) / win
            if dis == 0:
                delta_tau[i] = delta
            else:
                delta_tau[i::dis] = delta

        

        # Remove DC offset to avoid global range shift

        # Interpolate delta_tau from feta_crop grid to full f_eta grid
        # Use boundary values for extrapolation outside feta_crop range

 
        return delta_tau.get(), corr.get(), block_ffta.get()

    def compensate_residual_rcm_ancps(self, sig):
        [Na,Nr] = cp.shape(sig)
        sig_feta_tau = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift((sig), axes=0), axis=0), axes=0)
        G = cp.zeros_like(sig_feta_tau)
        S_pre = sig_feta_tau[0,:]
        delta_tau = cp.zeros((Na))
        index = 1
        dis = 10
        G[:dis] = sig_feta_tau[:dis,:]
        for i in range(dis, Na):
            S_pre = G[i-dis,:]
            S_curr = sig_feta_tau[i,:]
            F_pre = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(S_pre)))
            F_curr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(S_curr)))
            H = F_pre * cp.conj(F_curr)/cp.abs(F_pre * cp.conj(F_curr))
            H_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(H)))
            R = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(H_fft*cp.conj(H_fft))))
            phase = cp.angle(R)
            # 线性拟合 phase
            phase_unwrap = cp.unwrap(phase)
            x = cp.arange(-Nr/2, Nr/2, dtype=cp.float64)*(self.Fr/Nr)
            A = cp.stack([x, cp.ones_like(x)], axis=1)
            coef = cp.linalg.lstsq(A, phase_unwrap.astype(cp.float64), rcond=None)[0]
            slope = coef[0]  # rad/sample
           
            delta_tau[index] = -slope/(2*cp.pi)  # seconds
            index = index + 1
            F_curr = F_curr*cp.exp(1j*2*cp.pi*delta_tau[index-1]*cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr))
            G[i] = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(F_curr)))
        return delta_tau.get()

    def process_data_rd_pga(self):
        [Na,Nr] = cp.shape(self.sig)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        f_eta = self.feta_c + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*self.R0)
        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

             

        ## coarse compress
        self.sig = self.rd_focus_rcmc(cp.array(self.sig))
        sig_rcmc = cp.array((self.sig))
        sig_fft_value = (cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig_rcmc,axes = 0), axis=0), axes=0))
        rcm_error = cp.argmax(cp.abs(sig_fft_value), axis=1).get()
        rcm_error = rcm_error - cp.mean(rcm_error)

        self.sig, error = self.spga(self.sig, Ka)
    
        # delta_tau = cp.zeros((Na))
        # ## residual RCM compensation
        # delta,corr,block_ffta = self.compensate_residual_rcm(cp.array(self.sig))
        # delta_tau += cp.array(delta)
        # [mat_ftau, _] = cp.meshgrid(f_tau, f_eta)
        # sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig_rcmc))))
        # mat_delta_tau = cp.tile(delta_tau[:, cp.newaxis], (1, Nr))
        # sig_fft2 = sig_fft2*cp.exp(1j*2*cp.pi*mat_delta_tau*mat_ftau)
        # sig_corrected = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2)))

        # ## final focusing
        # self.sig, error = self.spga(sig_corrected, Ka)
        # self.sig = self.rd_focus_ac(cp.array((self.sig)))

        # delta_tau = delta_tau.get()
        # rcm_error_estimate = delta_tau*self.Fr
        # plt.figure()
        # plt.plot(rcm_error)
        # plt.plot(rcm_error_estimate)
        # plt.xlabel("Azimuth lines")
        # plt.ylabel("Residual RCM (samples)")
        # plt.legend()
        # plt.grid()
        # plt.savefig("../../../fig/afscan/rcm_error.png", dpi=300)

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(np.abs(corr), aspect='auto', cmap='jet')
        # plt.subplot(1,2,2)
        # plt.imshow(np.abs(block_ffta), aspect='auto', cmap='jet')
        # plt.savefig("../../../fig/afscan/rcm_error_corr.png", dpi=300)

      
   

       


        # self.sig = afscan.afscan_spectrum_orth(cp.array(self.sig))

        return self.sig

if __name__ == "__main__":
    cp.cuda.Device(0).use()
    param_path = r"../../../data/example_13_param.mat"
    data_path = r"../../../data/example_13_sig.mat"
    pos_path = r"../../../data/example_13_pos.mat"
    afscan = AFScanData(param_path, data_path, pos_path)
    afoucs = AutoFocus(afscan.Fr, afscan.Tr, afscan.f0, afscan.PRF, afscan.Vr, afscan.Br, afscan.fc, afscan.R0)

    afscan.sig = afoucs.Moco_first(cp.array(afscan.sig), cp.array(afscan.down), -cp.array(afscan.right), cp.array(afscan.forward), afscan.phi)
    afscan.sig = afscan.azimuth_interp(cp.array(afscan.sig))
    sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(afscan.sig))))
    # beta = 8.0
    # Na = afscan.sig.shape[0]
    # w = cp.kaiser(Na, beta)
    # sig_fft2 = sig_fft2 * cp.tile(w[:, cp.newaxis], (1, afscan.Nr))
    # w = cp.kaiser(afscan.sig.shape[1], beta)
    # sig_fft2 = sig_fft2 * cp.tile(w[cp.newaxis, :], (afscan.Na, 1))
    # afscan.sig = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2)))
    # afscan.sig = afscan.sig[8000:29000, 300:420]
    # afscan.sig = afscan.sig[14000:29000, 130:220]
    # afscan.sig = afscan.sig[:, 300:400]
    # afscan.forward = afscan.forward[14000:29000]
    # afscan.down = afscan.down[14000:29000]
    # afscan.right = afscan.right[14000:29000]
    # afscan.Na, afscan.Nr = afscan.sig.shape
    # temp = np.zeros((afscan.Na, afscan.Nr*3), dtype=np.complex64)
    # temp[:, afscan.Nr*3//2-afscan.Nr//2:afscan.Nr//2+afscan.Nr*3//2] = afscan.sig
    # afscan.sig = temp
    # afscan.Nr = afscan.Nr*3 

    afscan.sig = afscan.doppler_shift(afscan.sig, afscan.feta_c)
    print("After Doppler shift, feta_c:", afscan.feta_c)
    # # PRF = afscan.PRF
    afscan.sig = afscan.doppler_downsample(afscan.sig, afscan.PRF, 1000)
    # afscan.down = np.squeeze(afscan.doppler_downsample(afscan.down[:,np.newaxis], PRF, 500))
    # afscan.right = np.squeeze(afscan.doppler_downsample(afscan.right[:,np.newaxis], PRF, 500))
    afscan.PRF = 1000
 
    echo_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(afscan.sig)))).get()
    echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(cp.array(afscan.sig), axes=0), axis=0), axes=0).get()
    plt.figure()
    plt.imshow(np.abs(afscan.sig), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/echo_sig.png", dpi=300)

    plt.figure()
    plt.imshow(np.abs(echo_fft2), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/echo_sig_fft2.png", dpi=300)

    plt.figure()
    plt.imshow(np.abs(echo_tau_feta), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/echo_sig_tau_feta.png", dpi=300)



    

    # afscan.sig = afscan.squint_sm(cp.array(afscan.sig))
    focus = afscan.process_data_rd_pga()

    image = 20*np.log10(np.abs(focus))
    # focus = focus.get()
    focus_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(focus))))
    max_val = cp.max(cp.abs(focus_fft2), axis=1)
    midx = cp.argmax(cp.abs(max_val))
    max_slice = cp.abs(focus_fft2)[midx,:]
    win = max_slice>cp.max(max_slice)*0.1
    bandwidth = cp.sum(win)*(afscan.Fr/afscan.Nr)
    print("afscan bandwidth:", bandwidth)
    print("afscan range res:", afscan.c/(bandwidth*2))
    focus_fft2 = focus_fft2.get()
    focus_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(cp.array(focus), axes=0), axis=0), axes=0).get()
    plt.figure()
    plt.imshow(np.abs(focus), aspect='auto', cmap='gray')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/par_focus.tif", dpi=300)

    plt.figure()
    plt.imshow(np.abs(focus_fft2), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/par_focus_fft2.png", dpi=300)

    plt.figure()
    plt.imshow(np.abs(focus_tau_feta), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan/par_focus_tau_feta.png", dpi=300)

    dot_estimate = DotEstimator(3, afscan.c, afscan.Vr, afscan.PRF, afscan.Fr, "../../../fig/afscan/")
    dot_estimate.dot_estimate((focus), (int(1.5/(afscan.Vr/afscan.PRF)), int(3/(afscan.c/(2*afscan.Fr)))), 16)
