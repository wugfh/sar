import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys

sys.path.append(r"../../")
from autofocus import AutoFocus
from dot_estimate import DotEstimator
from sar_focus import SAR_Focus
sys.path.append(r"./")
from multiprocessing import Process, Queue
import multiprocessing as mp
from tqdm import tqdm
from sinc_interpolation import SincInterpolation
import scipy.optimize as op
import h5py
import numpy as np
import pywt
import scipy.io as sio
import imageio as iio
from pos_transform import PosReader

class Tradition():
    def __init__(self, param_path, data_path):
        super().__init__()
        self.read_data(data_path, param_path)
        print("parameters: Fr:{}, Br:{}, f0:{}, PRF:{}, Tr:{}".format(self.Fr, self.Br, self.f0, self.PRF, self.Tr))
        self.Rc = self.R0/np.cos(self.theta_c)
        self.Vr = np.mean(np.diff(self.forward)/np.diff(self.frame_time))

        print("Vr: {}, R0: {}".format(self.Vr, self.R0))
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c
        self.phi = np.deg2rad(60)
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_

        ## azimuth
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_bw/2) - np.sin(self.theta_c-self.theta_bw/2))/self.lambda_
        self.da = self.Vr/self.B_fov
        self.dr = self.c/(2*self.Br)
        
        print("therical azimuth res:{}, therical range res:{}".format(self.da, self.dr))



    def read_data(self, data_filename, param_filename):
        with h5py.File(data_filename, "r") as data:
            sig = data['sig'][()]
            # sig = sig["real"] + 1j*sig["imag"]
            self.sig = sig["real"] + 1j * sig["imag"]
            self.sig = self.sig[:, 20000:22500]
        print("original data shape: ", self.sig.shape)
        [self.Na, self.Nr] = self.sig.shape

        with h5py.File(param_filename) as param:    
            pos_reader = PosReader(param_filename)
            forward, right, down = pos_reader.get_coords(pos_reader.time)
            self.frame_time = pos_reader.time
            self.forward = forward
            self.right = right
            self.down = down
  
            param = param['parFocus']
            self.Fr = float(param['Fr'][()])
            self.Br = float(param['Br'][()])
            self.f0 = float(param['f0'][()])
            self.PRF = float(param['PRF'][()])
            self.Tr = float(param['Tr'][()])
            self.c = float(param['c'][()])
            self.lambda_ = float(param['lambda'][()])
            self.R0 = float(param['R0'][()])
            self.Kr = float(param['Kr'][()])
            self.Ta = float(param['Ta'][()])
            self.theta_c = float(param['theta_rc'][()])
            self.theta_bw = float(param['theta_bw'][()])
       
            self.H = -np.mean(self.down)-390
            self.down = self.down + self.H+ 390
            self.phi = np.arccos(np.abs(self.H)/self.R0)
            self.Y0 = self.R0*np.sin(self.phi)

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
        sig_shift_fft2 = cp.roll(sig_fft2, -int(shift), axis=0)
        sig_shift = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_shift_fft2)))
        return sig_shift.get()
    
    def squint_sm(self, sig):
        [Na, Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c+cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = self.eta_c+cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.feta_c + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        mat_tau, mat_eta = cp.meshgrid(tau, eta)

        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)
        H2 = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        # doa1 = cp.arctan((0 - self.Vr*mat_eta)/self.R0)
        # tau_mid = self.alpha*(doa1-self.theta_c)/self.Kr
        H1 = cp.exp(-2j*cp.pi*self.feta_c*(1+0.15*mat_f_tau/self.f0)*mat_eta)
        sig_ftau_eta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=1), axis=1), axes=1)
        # sig_ftau_eta = sig_ftau_eta*H1
        sig_ftau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig_ftau_eta, axes=0), axis=0), axes=0)
        sig_ftau_feta = sig_ftau_feta*H2
        sig_shift = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(sig_ftau_feta)))
        return sig_shift.get()

    def azimuth_interp(self, sig):
        [Na,Nr] = cp.shape(sig)
        min_forward = cp.min(cp.array(self.forward))
        da = min_forward + cp.arange(Na)*(self.Vr/self.PRF)
        for i in range(Nr):
            sig[:,i] = cp.interp(da, cp.array(self.forward), sig[:,i])
        return sig.get()
        
    def time_freq_analysis(self, sig):
        [Na, Nr] = sig.shape
        # Find the position of the strongest point in sig
        max_idx = np.unravel_index(np.abs(sig).argmax(), sig.shape)
        target = sig[:, 200]
        totalscale = 128
        wavelet = 'cmor1.5-1.0'
        fc = pywt.central_frequency(wavelet)
        print("central frequency of wavelet:", fc)
        cparam = 2* fc * totalscale
        scales = cparam / np.arange(totalscale, 0, -1)
        coeffs, freqs = pywt.cwt(target.get(), scales, wavelet, sampling_period=1/self.PRF)
        plt.figure(figsize=(10, 6))
        plt.contourf(np.arange(Na), freqs, np.abs(coeffs), cmap="jet")
        plt.colorbar(label="Magnitude")
        plt.xlabel("time")
        plt.ylabel("scale")
        plt.savefig("../../../fig/tradition/time_freq_analysis.png", dpi=300)

        # coeffs is a list of wavelet coefficients for each azimuth line

    def process_data_rd_pga(self):
        [Na,Nr] = cp.shape(self.sig)
        f_tau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        f_eta = self.feta_c + cp.arange(-Na/2, Na/2, 1)*(self.PRF/Na)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        R = tau*self.c/2

             

        # coarse compress

        # self.sig = self.rd_focus_rcmc(cp.array(self.sig))

        # self.sig = self.rd_focus_ac(cp.array(self.sig))
        sar_focus = SAR_Focus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.Kr, self.theta_bw)

        self.sig = sar_focus.erma_rcmc(cp.array(self.sig))
        self.sig = sar_focus.erma_ac(self.sig).get()

        afoucs = AutoFocus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.theta_bw)

        # final focusing
        block_size = self.sig.shape[1]
        step_len = block_size
        lmid = np.arange(step_len//2, self.sig.shape[1], step_len) 
        block_spga = np.zeros_like(self.sig, dtype=np.complex128)
        step = 0
        error_array = []
        bstart = []
        bend = []
        
        for mid in lmid:
            start = int(max(0, mid - block_size//2))
            end = int(min(start+block_size, self.sig.shape[1]))
            block = self.sig[:, start:end]
            bstart.append(start)
            bend.append(end)

            # for iter in range(2):
            #     pga_block,mat_error = afoucs.spga(((block)), R[start:end], 1, snr_threshold=-30, num_iter=30,  win_min=10, method = "line", range_win = 10)
            #     block = self.compensate_R2(block, mat_error)
            pga_block,_ = afoucs.spga(((block)), 3, snr_threshold=-25, num_iter=30,  win_min=10, method = "line", range_win = 10)
            # for i in range(0,-1,-1):
                # pga_block,_ = afoucs.spga(((pga_block)), R[start:end], 12, snr_threshold=-30, num_iter=30,  win_min=10, method = "mat", range_win = 30*2**i)
            #     mat_error = mat_error + error
            if np.abs(mid-start) <= np.abs(mid-end):
                bmid = np.abs(mid-start)
            else:
                bmid = pga_block.shape[1] - np.abs(mid-end)
            winlen = np.minimum(bmid*2, step_len)
            block_spga[:, mid-winlen//2:mid+winlen//2] += pga_block[:, bmid-winlen//2:bmid+winlen//2]
            step += 1
        self.sig = block_spga

        return self.sig

if __name__ == "__main__":
    cp.cuda.Device(0).use()
    prefix = '../../../data/2025_3_20/'
    experiment_tag = 'example_5_blk_6'
    param_path = f"{prefix}{experiment_tag}_parFocus.mat"
    data_path = f"{prefix}{experiment_tag}.mat"
    tradition = Tradition(param_path, data_path)
    afoucs = AutoFocus(tradition.Fr, tradition.Tr, tradition.f0, tradition.PRF, tradition.Vr, tradition.Br, tradition.feta_c, tradition.R0, tradition.theta_bw)

    tradition.sig = afoucs.Moco_first(cp.array(tradition.sig), cp.array(tradition.right-tradition.Y0), -cp.array(tradition.down-tradition.H), cp.array(tradition.forward), tradition.phi)


    # tradition.sig = tradition.squint_sm(cp.array(tradition.sig))
    tradition.sig = tradition.azimuth_interp(cp.array(tradition.sig))
    focus = tradition.process_data_rd_pga()

    image = np.abs(focus)
    image_abs = np.abs(image)
    image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
    iio.imwrite("../../../fig/tradition/par_focus.tif", image_norm)

