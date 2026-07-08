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
from tqdm import tqdm
from sinc_interpolation import SincInterpolation
import h5py
import pywt
import scipy.io as sio
import imageio as iio
from scipy import interpolate as intp
import gc
from inverse_conv import recover_dft_phase_batch
import tqdm
from multiprocessing import Process
from spectrum_recovery import build_annihilating_filter, solve_c_based_cg_batch

class AFScanData(FScanAzimuth):
    def __init__(self, param_path, data_path):
        super().__init__()
        self.read_data(data_path, param_path)
        print("signal shape:", self.sig.shape)

        print("parameters: Fr:{}, Br:{}, f0:{}, PRF:{}, t0:{}".format(self.Fr, self.Br, self.f0, self.PRF, self.t0))

        print("R0: {}, phi: {}, H: {}, Y0: {}, theta_c: {}".format(self.R0, np.rad2deg(self.phi), self.H, self.Y0, np.rad2deg(self.theta_c)))
        self.theta_width = np.deg2rad(7.04)  # beam width in elevation
        self.theta_lowf = -np.deg2rad(17.94) 
        self.theta_upf = -np.deg2rad(10.9)
        self.theta_az = np.deg2rad(2.5) ## maximum azimuth beam angle width while scanning
        self.theta_sc = np.abs(self.theta_upf - self.theta_lowf)  # scanning angle width
        self.alpha = self.Br/(self.theta_upf - self.theta_lowf)

        self.Ta = self.frame_time.max()-self.frame_time.min()
        print("imaging time:", self.frame_time.max()-self.frame_time.min(), len(self.frame_time))
        self.Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3*self.f0/(self.c*self.R0)

        self.scan_left = np.deg2rad(60-4)
        self.scan_right = np.deg2rad(60+4)
        self.Tswath = (self.H/cp.cos(self.scan_right) - self.H/cp.cos(self.scan_left))/self.c*2
        self.Kfscan = -self.Br/self.Tswath
   
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_
        print("Vr: {}, Ka: {}, feta_c: {}".format(self.Vr, self.Ka, self.feta_c))

        ## azimuth
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
    
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+(self.theta_width)/2) - np.sin(self.theta_c-self.theta_width/2))/self.lambda_

        print("azimuth bandwidth : {}".format(self.Bd))

        self.da = self.Vr/self.B_fov
        self.dr = (self.theta_sc/self.theta_az)*self.c/(2*self.Br)
        self.T_ap = self.theta_sc*self.R0/self.Vr
        
        print("therical azimuth res:{}, therical range res:{}".format(self.Vr/self.Bd, (self.theta_sc/self.theta_az)*self.c/(2*self.Br)))



    def read_data(self, data_filename, param_filename):
        param = sio.loadmat(param_filename) 
        grp = param['params'] if 'params' in param else param
        self.Fr = float(np.squeeze(grp['Fr'][0,0]))
        self.t0 = float(np.squeeze(grp['t0'][0,0]))
        self.Br = float(np.squeeze(grp['Br'][0,0]))
        # self.f0 = float(np.squeeze(grp['f0']))
        self.f0 = 35e9
        self.PRF = float(np.squeeze(grp['PRF'][0,0]))
        self.Tp = float(np.squeeze(grp['Tr'][0,0]))
        self.Kr = self.Br / self.Tp
        self.lambda_ = self.c / self.f0

        self.forward = np.squeeze(np.array(grp['forward'][0, 0]))
        self.right = np.squeeze(np.array(grp['right'][0, 0]))
        self.down = np.squeeze(np.array(grp['down'][0, 0]))
        self.frame_time = np.squeeze(np.array(grp['frame_time'][0, 0]))
        self.frame_time = self.time2sec(self.frame_time)

        with h5py.File(data_filename, "r") as data:
            self.sig_all = data['sig']
            # sig = sig["real"] + 1j*sig["imag"]
            self.sig_all = np.array(self.sig_all).T
            [self.Na_all, self.Nr_all] = self.sig_all.shape
            self.image_startr = 8500
            self.image_starta = 0
            slice_a = slice(self.image_starta, self.image_starta+self.Na_all)
            slice_r = slice(self.image_startr, self.image_startr+3000)
            self.sig = self.sig_all[slice_a, slice_r]
            print("signal element type:", self.sig[0, 0].dtype)
  

        [self.Na, self.Nr] = self.sig.shape
        self.forward = self.forward[slice_a]
        self.right = self.right[slice_a]
        self.down = self.down[slice_a]
        self.frame_time = self.frame_time[slice_a]

        self.Vr = np.diff(self.forward)/np.diff(self.frame_time)
        print("velocity:", np.max(self.Vr), np.min(self.Vr))
        self.Vr = np.mean(np.diff(self.forward)/np.diff(self.frame_time))
        
        self.theta_c = self.estimate_doppler_ceneter(self.sig)

        self.Rc = self.t0*self.c/2
        print("initial Rc:", self.Rc)
        self.Rc = self.Rc-self.Nr_all//2*(self.c/(2*self.Fr)) 
        self.Rc = self.Rc + (self.image_startr+self.sig.shape[1]//2)*(self.c/(2*self.Fr))
        self.forward = self.forward - np.median(self.forward) - self.Rc*np.sin(self.theta_c)
        self.R0 = self.Rc*np.cos(self.theta_c)
        self.H = -np.mean(self.down)-390
        self.down = self.down - np.mean(self.down)
        self.phi = np.arccos(np.abs(self.H)/self.R0)
        self.Y0 = self.R0*np.sin(self.phi)
        self.right = self.right - np.mean(self.right)

        
        self.forward = self.forward[:, np.newaxis]
        self.right = self.right[:, np.newaxis]
        self.down = self.down[:, np.newaxis]

        print("flight height:", np.max(self.down), np.min(self.down))

        self.frame_time = self.frame_time[:, np.newaxis]

    def set_image_start(self, image_startr):
        self.Rc = self.Rc - (self.image_startr+self.sig.shape[1]//2)*(self.c/(2*self.Fr))
        self.image_startr = image_startr
        self.Rc = self.Rc + (self.image_startr+self.sig.shape[1]//2)*(self.c/(2*self.Fr))
        self.R0 = self.Rc*np.cos(self.theta_c)
        self.forward = self.forward - np.median(self.forward) - self.Rc*np.sin(self.theta_c)
        self.phi = np.arccos(np.abs(self.H)/self.R0)
        self.Y0 = self.R0*np.sin(self.phi)
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c

    
    def estimate_doppler_ceneter(self, sig):
        [Na, Nr] = sig.shape
        sig_fftr = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig, axes=1), axis=1), axes=1)
        dphase = np.conj(sig_fftr[0:Na-1,:])*(sig_fftr[1:Na, :])
        feta_c = np.angle(dphase.mean())/(2*np.pi)*self.PRF
        theta_c = np.arcsin(feta_c*self.lambda_/(2*self.Vr))
        return theta_c

    def time2sec(self,time):
        """convert echo recorded time (HHMMSS[.sss]) to GNSS seconds (numpy array)"""
        t = np.asarray(time, dtype=float)
        hours = np.floor(t / 1e4)
        minutes = np.floor((t - hours * 1e4) / 1e2)
        seconds = t - hours * 1e4 - minutes * 1e2

        timezone = 8
        hours = hours - timezone
        seconds = seconds + hours * 3600 + minutes * 60
        seconds = seconds + 18  # leap seconds (as in original code)
        if seconds.size and seconds.flat[0] < 0:
            seconds = seconds + 3600 * 24

        # fit a line to index positions where seconds change to get smooth monotonic seconds
        if seconds.size > 1:
            sec_change_idx = np.nonzero(np.diff(seconds) != 0)[0] + 1
            if sec_change_idx.size == 0:
                seconds_new = seconds
            else:
                # include the first index to stabilize fit if needed
                idxs = sec_change_idx
                poly = np.polyfit(idxs, seconds[idxs], 1)
                seconds_new = np.polyval(poly, np.arange(seconds.size))
        else:
            seconds_new = seconds
        return seconds_new

    def doppler_downsample(self, sig,Fa, Fa_down):
        Na, Nr = sig.shape
        sig = cp.array(sig)
        sig_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)
        new_width = int(cp.floor(Na * Fa_down / Fa))
        if new_width % 2 == 1:
            new_width += 1
        sig_downsampled = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_ffta[(Na/2 - new_width//2):(Na/2 + new_width//2),:], axes=0), axis=0), axes=0)
        [self.Na, self.Nr] = sig_downsampled.shape
        return sig_downsampled
    
    def doppler_shift(self, sig, feta_c, da):
        [Na,Nr] = cp.shape(sig)
        eta = da/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*feta_c*eta)
        return sig

    def azimuth_interp(self, sig, forward):
        [Na,Nr] = cp.shape(sig)
        da = (np.arange(Na)-Na//2)*(self.Vr/self.PRF) - self.Rc*np.sin(self.theta_c)
        da = da[:,np.newaxis]
        eta = cp.array(forward)/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*self.feta_c*eta)

        linear_interp = intp.interp1d(np.squeeze(forward), np.arange(Na), kind='linear', fill_value="extrapolate")
        new_index = cp.array(linear_interp(da))
        delta = new_index - cp.arange(Na)[:, cp.newaxis]
        delta = cp.tile(delta[:, cp.newaxis], (1, Nr))
        sinc_interp = SincInterpolation()
        sig = cp.ascontiguousarray(sig)
        sig_real = sinc_interp.sinc_interpolation(cp.real(sig).T, delta.T, Nr, Na, 8).T
        sig_imag = sinc_interp.sinc_interpolation(cp.imag(sig).T, delta.T, Nr, Na, 8).T
        sig = sig_real + 1j*sig_imag
        eta = cp.array(da)/self.Vr
        sig = sig*cp.exp(2j*cp.pi*self.feta_c*eta)
        return sig

    def moco_resample(self, prf_down):
        da = (np.arange(self.Na)-self.Na//2)*(self.Vr/self.PRF) - self.Rc*np.sin(self.theta_c)
        new_width = int(np.floor(self.Na * prf_down / self.PRF))
        if new_width % 2 == 1:
            new_width += 1

        da_down = (np.arange(new_width)-new_width//2)*(self.Vr/prf_down) - self.Rc*np.sin(self.theta_c)

        right_interp = intp.CubicSpline( np.squeeze(self.forward), np.squeeze(self.right),bc_type='natural', extrapolate=True)
        right = (right_interp(np.squeeze(da_down)))
        right = right[:, np.newaxis]

        down_interp = intp.CubicSpline(np.squeeze(self.forward), np.squeeze(self.down), bc_type='natural', extrapolate=True)
        down = (down_interp(np.squeeze(da_down)))
        down = down[:, np.newaxis]

        forward_interp = intp.CubicSpline(np.squeeze(da), np.squeeze(self.forward), bc_type='natural', extrapolate=True)
        forward = (forward_interp(np.squeeze(da_down)))
        forward = forward[:, np.newaxis]
        return forward, right, down


    def process_data_rd_pga(self):
        [Na,Nr] = cp.shape(self.sig)
        # print("sig shape:", self.sig.shape)

        # kaiser_win = cp.kaiser(Na, beta=8.6)
        # self.sig = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(self.sig, axes=0), axis=0), axes=0)
        # self.sig = self.sig*cp.tile(kaiser_win[:, cp.newaxis], (1, Nr))
        # self.sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(self.sig, axes=0), axis=0), axes=0)
        ## coarse compress

        # self.sig = self.rd_focus_rcmc(cp.array(self.sig))

        # self.sig = self.rd_focus_ac(cp.array(self.sig))
        sar_focus = SAR_Focus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.Kr, self.theta_width)
        ac = sar_focus.erma_rcmc(cp.array(self.sig))
        ac = sar_focus.erma_ac(cp.array(ac))

        afocus = AutoFocus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br*self.theta_az/self.theta_sc, self.feta_c, self.R0, self.theta_width)
        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        ftau = ftau[cp.newaxis, :]
        dR = cp.zeros(Na)

        snr = cp.array([-12.0, -2.0, -7.0, -12.5,-10.0,-10.0,-12.5])
        pre_win_len = np.zeros_like(snr)
        # da = self.eta_c*self.Vr + (cp.arange(Na)-Na//2)*(self.Vr/self.PRF)
        block_cnt = 5
        exit_flag = False

        for i in range(7,0,-1):
            print("snr:", snr)
            ac_chirp = afocus.rechirp(cp.array(ac))
            ac_chirp = afocus.dechirp(cp.array(ac_chirp))

            # image_abs = np.abs(ac_chirp)
            # threshold = np.percentile(image_abs, 99)
            # image_abs[image_abs > threshold] = threshold
            # plt.figure()
            # plt.imshow(image_abs, cmap="gray")
            # plt.show()

            error_line,win_len = afocus.spga(cp.array(ac_chirp), block_cnt, snr, 30, 10, method="line", range_win=30)
            error_line = cp.array(error_line)
            for j in range(block_cnt):
                if pre_win_len[j] == 0:
                    pre_win_len[j] = win_len[j]
                    snr[j] -= 1
                elif win_len[j] < 500:
                    snr[j] -= 1
                elif win_len[j] > 500:
                    exit_flag = True
                pre_win_len[j] = win_len[j] 
     
            print("error value :", cp.abs(error_line).max())
            if cp.abs(error_line).max() < 1e-5:
                continue

            dR += error_line[:, error_line.shape[1]//2]/(4*np.pi)*self.lambda_
            
            ac = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(self.sig, axes=1), axis=1), axes=1)
            ac = ac*cp.exp(-1j*4*cp.pi*dR[:,cp.newaxis]*(ftau+self.f0)/self.c)
            ac = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(ac, axes=1), axis=1), axes=1)
            ac = sar_focus.erma_rcmc(cp.array(ac))
            ac = sar_focus.erma_ac(cp.array(ac))


            if exit_flag:
                break


        self.sig = ac.get()
        # plt.figure()
        # plt.imshow(np.abs(rcmc.get()), aspect='auto', cmap='jet')
        # plt.savefig("../../../fig/afscan/rcmc_after.png", dpi=300)

        # plt.figure()
        # plt.plot(-dR.get())
        # plt.savefig("../../../fig/afscan/dR_estimate.png", dpi=300)

        return self.sig
    

    def fscan_dramp(self, sig):
        [Na,Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fr)
        tau = tau[cp.newaxis, :]
        sig = sig*cp.exp(-1j*cp.pi*self.Kfscan*(tau)**2)

        return sig
    
    def fscan_reramp(self, sig):
        [Na,Nr] = cp.shape(sig)
        tau = 2*self.Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fr)
        tau = tau[cp.newaxis, :]
        sig = sig*cp.exp(1j*cp.pi*self.Kfscan*tau**2)

        return sig.get()
    
    def fscan_super_resolution(self, sig):
        [Na,Nr] = cp.shape(sig)
        batch_size = 256
        
        f_send = self.f0 + (cp.arange(Nr) - Nr // 2) * (self.Fr / Nr)
        win_len = self.theta_az/(np.abs(self.theta_upf-self.theta_lowf))*Nr*0.3
        print("win_len:", win_len)
        x = cp.arange(-Nr/2, Nr/2, 1)
        window =  cp.exp(-0.5 * ((x) / win_len) ** 2)
        max_window = cp.max(window)

        window[window < max_window * 0.08] = max_window*0.08
        window[f_send < self.f0 - self.Br / 2] = max_window
        window[f_send > self.f0 + self.Br / 2] = max_window


        window = window/cp.sqrt(cp.sum(window**2))
        
        plt.figure()
        plt.plot(window.get())
        plt.savefig("../../../fig/afscan/window.png", dpi=300)

     
        sig = cp.ascontiguousarray(sig)
        sig_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)

        num_batches = (Na + batch_size - 1) // batch_size
        for start in tqdm.tqdm(range(0, Na, batch_size), 
                            total=num_batches, 
                            desc="Super-resolution (batched GPU)"):
            
            end = min(start + batch_size, Na)
            # sig_ffta[start:end, :], info =  recover_dft_phase_batch(
            #     sig_ffta[start:end, :], window,
            #     Nr, lam=1/window.shape[0]**2, tol=1e-5, max_iter=1000
            # )
            # if info != 0:
            #     print(f"Warning: CG did not converge for batch {start}-{end}. Info: {info}")
            batch = sig_ffta[start:end, :]
            batch_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(batch, axes=1), axis=1), axes=1)
            batch_fft = batch_fft/window[cp.newaxis, :]
            sig_ffta[start:end, :] = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(batch_fft, axes=1), axis=1), axes=1)
        sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_ffta, axes=0), axis=0), axes=0)
        return sig
    
    def fscan_super_resolution_filter(self, sig, batch_size = 256):
        [Na,Nr] = sig.shape
        win_len = self.theta_az/(np.abs(self.theta_upf-self.theta_lowf))*Nr*0.3
        print("win_len:", win_len)
        x = cp.arange(-Nr/2, Nr/2, 1)
        window =  cp.exp(-0.5 * ((x) / win_len) ** 2)
        max_window = cp.max(window)
        f_send = self.f0 + (cp.arange(Nr) - Nr // 2) * (self.Fr / Nr)
        ## 注水
        factor = 0.08
        window[window < max_window * factor] = max_window*factor
        window[f_send < self.f0 - self.Br / 2] = max_window
        window[f_send > self.f0 + self.Br / 2] = max_window
        window = window/cp.sqrt(cp.sum(window ** 2))

        sig = cp.ascontiguousarray(sig)
        # sig_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)
        M = 32
        num_batches = (Na + batch_size - 1) // batch_size
        for start in tqdm.tqdm(range(0, Na, batch_size), 
                            total=num_batches, 
                            desc="Super-resolution (batched GPU)"):
            
            end = min(start + batch_size, Na)
            batch = sig[start:end, :]
            batch_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(batch, axes=1), axis=1), axes=1)
            hy = build_annihilating_filter(batch_fft[batch_fft.shape[0]//2, :], M)
            batch_fft = solve_c_based_cg_batch(batch_fft, window, hy, lam=1, eps=1e-2)

            sig[start:end, :] = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(batch_fft, axes=1), axis=1), axes=1)

        # sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_ffta, axes=0), axis=0), axes=0)
        return sig

    def estimate_fscan_center(self, sig):
        Na,Nr = sig.shape
        sig_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)
        dphase = cp.conj(sig_ffta[:,0:Nr-1])*(sig_ffta[:, 1:Nr])
        fc = (cp.angle(dphase.mean()))/(2*cp.pi)*self.Fr
        return fc
    
    def fscan_shift(self, sig, fc):
        [Na,Nr] = sig.shape
        tau = 2*self.Rc/self.c + (cp.arange(Nr)-Nr//2)*(1/self.Fr)
        tau = tau[cp.newaxis, :]
        sig = sig*cp.exp(-1j*2*cp.pi*fc*tau)
        return sig
    
    def estimate_kfscan(self, sig):
        [Na,Nr] = sig.shape
        range_res = self.c/(2*self.Br)
        azimuth_res = self.lambda_/(2*self.theta_width)
        area = (int(3/azimuth_res), int(3/range_res))
        max_index1 = cp.unravel_index(cp.argmax(np.abs(sig[:,0:Nr//3])), sig[:,0:Nr//3].shape)
        max_index2 = cp.unravel_index(cp.argmax(cp.abs(sig[:,2*Nr//3:])), sig[:,2*Nr//3:].shape)
        max_index2 = (max_index2[0], max_index2[1]+2*Nr//3)
        part1 = sig[:, max_index1[1]-area[1]//2:max_index1[1]+area[1]//2]
        part2 = sig[:, max_index2[1]-area[1]//2:max_index2[1]+area[1]//2]
        fc1 = self.estimate_fscan_center(part1)
        fc2 = self.estimate_fscan_center(part2)

        Tswath = (max_index2[1]/self.Fr - max_index1[1]/self.Fr)
        Kfscan = (fc2-fc1)/Tswath
        return Kfscan
    

def plot_raw_sig(sig):
    echo_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(sig)))).get()
    echo_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(cp.array(sig), axes=0), axis=0), axes=0).get()
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

def process(prefix, example_tag):
    param_path = f"{prefix}{example_tag}_param.mat"
    data_path = f"{prefix}{example_tag}_sig.mat"
    afscan = AFScanData(param_path, data_path)

    # cnt = np.floor(afscan.sig_all.shape[1]/afscan.sig.shape[1])
    cnt = 1
    process_len = afscan.sig.shape[1]
    mono_resample_flag = False

    focus_all = []
    for i in range(int(cnt)):
        image_start = int(1e4)
        print("processing image part: {}-{}".format(image_start, image_start+process_len))

        image_end = np.minimum(image_start + process_len, afscan.sig_all.shape[1])
        afscan.sig = afscan.sig_all[:, image_start:image_end]
        del afscan.sig_all
        gc.collect()
        afscan.PRF = 6000
        afscan.Na, afscan.Nr = afscan.sig.shape
        afscan.set_image_start(image_start)
        print("Rc:", afscan.Rc)  
        print("azimuth interpolation done, start azimuth downsample...")

        afoucs = AutoFocus(afscan.Fr, afscan.Tr, afscan.f0, afscan.PRF, afscan.Vr, afscan.Br, afscan.feta_c, afscan.R0, afscan.theta_width)


        prf_down = (afscan.Ka*afscan.Ta).get() 

        
        print("start motion compensation...")
        afscan.sig = afscan.azimuth_interp(cp.array(afscan.sig), afscan.forward)

        forward,right,down = afscan.moco_resample(prf_down)

        afscan.sig = afscan.doppler_shift(cp.array(afscan.sig), afscan.feta_c, cp.array(afscan.forward))

        afscan.sig = afscan.doppler_downsample(cp.array(afscan.sig), afscan.PRF, prf_down)
        afscan.PRF =  prf_down
        print("azimuth downsample to {}Hz".format(afscan.PRF))
        afscan.sig = afscan.doppler_shift(cp.array(afscan.sig), -afscan.feta_c, cp.array(forward))
        print("data loaded, start azimuth interpolation...")

        afscan.sig = afoucs.Moco_first(cp.array(afscan.sig), cp.array(right-afscan.Y0), -cp.array(down-afscan.H))

        print("motion compensation done, start focusing and pga...")
        tmp = np.zeros((afscan.sig.shape[0]*2, afscan.sig.shape[1]), dtype=np.complex128)
        tmp[tmp.shape[0]//2-afscan.sig.shape[0]//2:tmp.shape[0]//2+afscan.sig.shape[0]//2, :] = afscan.sig
        afscan.sig = tmp
        del tmp

        afscan.sig = afscan.process_data_rd_pga()

        afscan.Kfscan = afscan.estimate_kfscan(cp.array(afscan.sig))
        afscan.sig = afscan.fscan_dramp(cp.array(afscan.sig))
        fc = afscan.estimate_fscan_center(cp.array(afscan.sig))
        afscan.sig = afscan.fscan_shift(cp.array(afscan.sig), fc)
        # plot_raw_sig(afscan.sig)

        
        plt.figure()
        plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(afscan.sig.get())))), aspect='auto')
        plt.xlabel("Range frequency")
        plt.ylabel("Azimuth frequency")
        plt.colorbar()
        plt.savefig("../../../fig/afscan/par_focus_super_fft2_before.png", dpi=300)

        afscan.sig = afscan.fscan_super_resolution_filter(cp.array(afscan.sig))

        plt.figure()
        plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(afscan.sig.get())))), aspect='auto')
        plt.xlabel("Range frequency")
        plt.ylabel("Azimuth frequency")
        plt.colorbar()
        plt.savefig("../../../fig/afscan/par_focus_super_fft2_after.png", dpi=300)

        # afscan.sig = afscan.fscan_shift(cp.array(afscan.sig), -fc)
        # afscan.sig = afscan.fscan_reramp(cp.array(afscan.sig))

        # focus = afscan.sig
        
        focus_all.append(afscan.sig.get())
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    # del afscan.sig_all
    # gc.collect()

    focus_all = np.concatenate(focus_all, axis=1)
    sio.savemat("../../../fig/afscan/focus_all.mat", {"focus_all": focus_all})

    tif_path = f"../../../fig/afscan/part_focus_super.tif"
    image_abs = np.abs(focus_all)
    image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
    iio.imwrite(tif_path, image_norm)

    threshold = np.percentile(image_abs, 99)
    image_abs[image_abs > threshold] = threshold
    plt.figure(figsize=(20*image_abs.shape[1]*afscan.c/afscan.Fr/2/(image_abs.shape[0]*afscan.Vr/afscan.PRF)+2, 20))
    plt.imshow(image_abs, cmap="gray")
    plt.savefig("../../../fig/afscan/par_focus_super.png", dpi=300)

    dot_estimate = DotEstimator(10, afscan.c, afscan.Vr, afscan.PRF, afscan.Fr, "../../../fig/afscan/")
    dot_estimate.dot_estimate((focus_all), (int(1/(afscan.Vr/afscan.PRF)), int(1/(afscan.c/(2*afscan.Fr)))), 16)

if __name__ == "__main__":
    cp.cuda.Device(0).use()
    prefix = "F:/sar/data/2024_4_fs_data/"
    for i in range(6,7):
        example_tag = f'example_10_part{i}'
        p = Process(target=process, args=(prefix, example_tag))
        p.start()
        p.join()
    
