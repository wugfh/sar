import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys

sys.path.append(r"../../")
from autofocus import AutoFocus
from dot_estimate import DotEstimator
from sar_focus import SAR_Focus
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation
import scipy.optimize as op
import h5py
import numpy as np
import pywt
import scipy.io as sio
import imageio as iio
from pos_transform import PosReader
import scipy.interpolate as intp

class Tradition():
    def __init__(self, param_path, data_path, pos_path):
        super().__init__()
        self.read_data(data_path, param_path, pos_path)
        print("phi (deg):", np.rad2deg(self.phi))
        print("parameters: Fr:{}, Br:{}, f0:{}, PRF:{}, Tr:{}".format(self.Fr, self.Br, self.f0, self.PRF, self.Tr))
        self.Vr = np.mean(np.diff(self.forward)/np.diff(self.time))

        print("Vr: {}, R0: {}".format(self.Vr, self.R0))
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_

        ## azimuth
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_bw/2) - np.sin(self.theta_c-self.theta_bw/2))/self.lambda_
        self.da = self.Vr/self.B_fov
        self.dr = self.c/(2*self.Br)
        
        print("therical azimuth res:{}, therical range res:{}".format(self.da, self.dr))



    def read_data(self, data_filename, param_filename, pos_filename):
        with h5py.File(data_filename, "r") as data:
            sig = data['sig'][()]
            # sig = sig["real"] + 1j*sig["imag"]
            sig = sig["real"] + 1j * sig["imag"]
            self.sig_all = sig
            # self.sig = sig[:, 8000:12000]
            self.sig = sig[:, 20000:22500]
        [self.Na, self.Nr] = self.sig.shape
        self.image_start = 20000

        with h5py.File(param_filename) as param:    
            pos_reader = PosReader(pos_filename)

  
            param = param['parFocus']
            time = np.squeeze(np.array(param['time'][()]))
            forward, right, down = pos_reader.get_coords(time)
            self.time = time
            self.forward = forward
            self.right = right
            self.down = down

            self.Fr = float(param['Fr'][()])
            self.Br = float(param['Br'][()])
            self.f0 = float(param['f0'][()])
            self.PRF = float(param['PRF'][()])
            self.Tr = float(param['Tr'][()])
            self.c = float(param['c'][()])
            self.lambda_ = float(param['lambda'][()])
            self.Kr = float(param['Kr'][()])
            self.Ta = float(param['Ta'][()])
            self.theta_c = float(param['theta_rc'][()])
            self.theta_c = np.deg2rad(2.4650)
            self.theta_bw = float(param['theta_bw'][()])
            self.Rc = float(param['R0'][()])
            self.Rc = self.Rc-self.sig_all.shape[1]/2*self.c/(2*self.Fr) 
            self.Rc = self.Rc + (self.image_start+self.sig.shape[1]/2)*self.c/(2*self.Fr)
            self.R0 = self.Rc*np.cos(self.theta_c)
            self.forward = self.forward - np.median(self.forward) - self.Rc*np.sin(self.theta_c)

            self.H = -np.mean(self.down)-390
            self.down = self.down + self.H
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
    

    def azimuth_interp(self, sig):
        [Na,Nr] = cp.shape(sig)
        da = (np.arange(Na)-Na//2)*(self.Vr/self.PRF) + (self.eta_c*self.Vr).get()
        mat_eta = cp.tile(cp.array(self.forward[:,np.newaxis]), (1, Nr))/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*self.feta_c*mat_eta)

        linear_interp = intp.interp1d(self.forward, np.arange(Na), kind='linear', fill_value="extrapolate")
        new_index = cp.array(linear_interp(da))
        delta = new_index - cp.arange(Na)
        delta = cp.tile(delta[:, cp.newaxis], (1, Nr))
        sinc_interp = SincInterpolation()
        sig = cp.ascontiguousarray(sig)
        sig_real = sinc_interp.sinc_interpolation(cp.real(sig).T, delta.T, Nr, Na, 8).T
        sig_imag = sinc_interp.sinc_interpolation(cp.imag(sig).T, delta.T, Nr, Na, 8).T
        sig = sig_real + 1j*sig_imag
        mat_eta = cp.tile(da[:,cp.newaxis], (1, Nr))/self.Vr
        sig = sig*cp.exp(2j*cp.pi*self.feta_c*mat_eta)
        return sig
        
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
    def estimate_rcm(self, sig):
        midx = cp.argmax(cp.abs(sig), axis=1)
        ac_max = cp.max(cp.abs(sig), axis=1)
        midx[ac_max<cp.max(ac_max)*0.1] = cp.median(midx)
        midx = midx - cp.median(midx)

        dR_true = midx*(self.c/(2*self.Fr))
        return dR_true.get()

    def process_data_rd_pga(self):
        [Na,Nr] = cp.shape(self.sig)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = (cp.arange(Na)-Na//2)/self.PRF + self.eta_c
        mat_tau,mat_eta = cp.meshgrid(tau, eta)


        self.sig = cp.array(self.azimuth_interp(cp.array(self.sig)))
        # kaiser_win = cp.kaiser(Na, beta=8.6)
        # self.sig = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(self.sig, axes=0), axis=0), axes=0)
        # self.sig = self.sig*cp.tile(kaiser_win[:, cp.newaxis], (1, Nr))
        # self.sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(self.sig, axes=0), axis=0), axes=0)
        ## coarse compress

        # self.sig = self.rd_focus_rcmc(cp.array(self.sig))

        # self.sig = self.rd_focus_ac(cp.array(self.sig))
        sar_focus = SAR_Focus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.Kr, self.theta_bw)
        data_rc = self.sig
        rcmc = sar_focus.erma_rcmc(cp.array(data_rc))
        ac = sar_focus.erma_ac(cp.array(rcmc))

        # self.sig = sar_focus.erma_ac(self.sig).get()
        afocus = AutoFocus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.theta_bw)
        ftau = cp.arange(-self.Nr/2, self.Nr/2, 1)*(self.Fr/self.Nr)
        mat_ftau = cp.tile(ftau, (self.Na, 1))
        dR = cp.zeros(Na)

        snr = cp.array([-25.0, -2.0, -7.0, -12.5,-10.0,-10.0,-12.5])
        pre_win_len = np.zeros_like(snr)
        # da = self.eta_c*self.Vr + (cp.arange(Na)-Na//2)*(self.Vr/self.PRF)
        block_cnt = 3
        exit_flag = False
        for i in range(15,0,-1):
            ac_rechirp = afocus.rechirp(cp.array(ac))
            ac_rechirp = afocus.down_res(cp.array(ac_rechirp), 1)
            ac_down = afocus.dechirp(cp.array(ac_rechirp))
            error_line,win_len = afocus.spga(cp.array(ac_down), block_cnt, snr, 30, 10, method="line", range_win=30)
            error_line = cp.array(error_line)
            mat_dr = error_line/(4*np.pi)*self.lambda_
            dR += mat_dr[:, mat_dr.shape[1]//2]
            
            mat_dr = cp.tile(dR[:, cp.newaxis], (1, self.Nr))
            data_rc_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(self.sig, axes=1), axis=1), axes=1)
            data_rc_fftr = data_rc_fftr*cp.exp(-1j*4*cp.pi*mat_dr*(mat_ftau+self.f0)/self.c)
            data_rc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_rc_fftr, axes=1), axis=1), axes=1)
            rcmc = sar_focus.erma_rcmc(cp.array(data_rc))
            ac = sar_focus.erma_ac(cp.array(rcmc))
            for j in range(block_cnt):
                if pre_win_len[j] == 0:
                    pre_win_len[j] = win_len[j]
                    if win_len[j] < 100:
                        snr[j] -= 1
                elif win_len[j] < 50:
                    snr[j] -= 1
                elif win_len[j] < 100:
                    snr[j] -= 0.5
                else:
                    exit_flag = True
                pre_win_len[j] = win_len[j] 
            print("snr:", snr)
            if exit_flag:
                break

        self.sig = ac.get()
        plt.figure()
        plt.imshow(np.abs(rcmc.get()), aspect='auto', cmap='jet')
        plt.savefig("../../../fig/tradition/rcmc_after.png", dpi=300)

        # plt.figure()
        # plt.plot(-dR.get())
        # plt.savefig("../../../fig/tradition/dR_estimate.png", dpi=300)

        dot_estimator = DotEstimator(3, self.c, self.Vr, self.PRF, self.Fr, "../../../fig/tradition/")
        dot_estimator.dot_estimate(self.sig, (int(1/(self.Vr/self.PRF)), int(1/(self.c/(2*self.Fr)))), 16)
        return self.sig

if __name__ == "__main__":
    cp.cuda.Device(0).use()
    prefix = '../../../data/2025_3_20/'
    experiment_tag = 'example_5_blk_6'
    param_path = f"{prefix}{experiment_tag}_parFocus.mat"
    data_path = f"{prefix}{experiment_tag}.mat"
    pos_path = f"{prefix}0323sbet_Mission23.out"
    tradition = Tradition(param_path, data_path, pos_path)
    afoucs = AutoFocus(tradition.Fr, tradition.Tr, tradition.f0, tradition.PRF, tradition.Vr, tradition.Br, tradition.feta_c, tradition.R0, tradition.theta_bw)



    # tradition.sig = tradition.squint_sm(cp.array(tradition.sig))

    cnt = np.floor(tradition.sig_all.shape[1]/tradition.sig.shape[1])
    # for i in range(int(cnt)):
        # tradition.sig = tradition.sig_all[:, int(i*tradition.sig.shape[1]):int((i+1)*tradition.sig.shape[1])]
    tradition.sig = afoucs.Moco_first(cp.array(tradition.sig), cp.array(tradition.right-tradition.Y0), -cp.array(tradition.down-tradition.H), cp.array(tradition.forward), tradition.phi)
    focus = tradition.process_data_rd_pga()
        # tradition.sig_all[:, int(i*focus.shape[1]):int((i+1)*focus.shape[1])] = focus   

    image_abs = np.abs(focus)

    image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
    iio.imwrite("../../../fig/tradition/par_focus.tif", image_norm)

    threshold = np.percentile(image_abs, 99)
    image_abs[image_abs > threshold] = threshold
    plt.figure(figsize=(20*image_abs.shape[1]/image_abs.shape[0]+2, 20))
    plt.imshow(image_abs, cmap="gray")
    plt.savefig("../../../fig/tradition/par_focus.png", dpi=300)

