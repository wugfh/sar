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
import numpy as np
import scipy.io as sio
import imageio as iio
from scipy import interpolate as intp
from inverse_conv import recover_dft_phase_batch
import tqdm
import pandas as pd


class AFScanData(FScanAzimuth):
    def __init__(self, param_path, data_path, pos_path):
        super().__init__()
        self.read_data(data_path, pos_path, param_path)
        print("parameters: Fr:{}, Br:{}, f0:{}, PRF:{}, Tp:{}".format(self.Fr, self.Br, self.f0, self.PRF, self.Tp))
        # self.theta_c = -np.deg2rad(14.3)
        self.theta_c = -np.deg2rad(0)
        self.theta_width = np.deg2rad(8.06)  # beam width in elevation
        self.theta_lowf = -np.deg2rad(17.94) 
        self.theta_upf = -np.deg2rad(10.9)
        self.theta_az = np.deg2rad(2.5) ## maximum azimuth beam angle width while scanning
        self.theta_sc = np.abs(self.theta_upf - self.theta_lowf)  # scanning angle width
        self.alpha = self.Br/(self.theta_upf - self.theta_lowf)
        self.scan_left = self.phi - np.deg2rad(3.5)
        self.scan_right = self.phi + np.deg2rad(3.5)
        self.Tswath = (self.H/cp.cos(self.scan_right) - self.H/cp.cos(self.scan_left))/self.c*2
        self.Kfscan = -self.Br/self.Tswath*0.244

        print("imaging time:", self.frame_time.max()-self.frame_time.min(), len(self.frame_time))
        print("Vr: {}, R0: {}, theta_c: {}".format(self.Vr, self.R0, np.rad2deg(self.theta_c)))
        self.eta_c = -self.Rc*cp.sin(self.theta_c)/self.Vr
        self.tau_c = 2*self.Rc/self.c
        self.feta_c = 2*self.Vr*cp.sin(self.theta_c)/self.lambda_

        ## azimuth
        self.B_fov = 2*self.Vr*(np.sin(self.theta_c+self.theta_az/2) - np.sin(self.theta_c-self.theta_az/2))/self.lambda_
        ## range
        self.Bd = 2*self.Vr*(np.sin(self.theta_c+(self.theta_width)/2) - np.sin(self.theta_c-self.theta_width/2))/self.lambda_
        self.da = self.Vr/self.B_fov
        self.dr = (self.theta_sc/self.theta_az)*self.c/(2*self.Br)
        self.T_ap = self.theta_sc*self.R0/self.Vr
        
        print("therical azimuth res:{}, therical range res:{}".format(self.Vr/self.Bd, (self.theta_sc/self.theta_az)*self.c/(2*self.Br)))



    def read_data(self, data_filename, pos_filename, param_filename):
        with h5py.File(data_filename, "r") as data:
            sig = data['sig']
            # sig = sig["real"] + 1j*sig["imag"]
            self.sig = np.array(sig).T
            [self.Na_all, self.Nr_all] = self.sig.shape
            self.image_startr = 0

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

        self.Vr = np.mean(np.diff(self.forward)/np.diff(self.frame_time))

        self.theta_c = self.estimate_doppler_ceneter(self.sig)

        self.Rc = self.t0*self.c/2
        # self.Rc = self.Rc-self.Nr_all/2*self.c/(2*self.Fr) 
        self.Rc = self.Rc + (self.image_startr+self.sig.shape[1]/2)*self.c/(2*self.Fr)
        self.forward = self.forward - np.median(self.forward) - self.Rc*np.sin(self.theta_c)
        self.R0 = self.Rc*np.cos(self.theta_c)
        self.H = self.R0*np.cos(np.deg2rad(22))
        self.down = self.down - np.mean(self.down)
        self.phi = np.arccos(np.abs(self.H)/self.R0)
        self.Y0 = self.R0*np.sin(self.phi)
        self.right = self.right - np.mean(self.right)

        
        self.forward = self.forward[:, np.newaxis]
        self.right = self.right[:, np.newaxis]
        self.down = self.down[:, np.newaxis]
        self.frame_time = self.frame_time[:, np.newaxis]


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
        new_width = int(Na * Fa_down / Fa)
        sig_downsampled = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_ffta[(Na/2 - new_width//2):(Na/2 + new_width//2),:], axes=0), axis=0), axes=0)
        [self.Na, self.Nr] = sig_downsampled.shape
        return sig_downsampled
    
    def doppler_shift(self, sig, feta_c):
        [Na,Nr] = cp.shape(sig)
        da = (cp.arange(Na)-Na//2)*(self.Vr/self.PRF) + (self.eta_c*self.Vr)
        eta = da[:,cp.newaxis]/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*feta_c*eta)
        return sig

    def azimuth_interp(self, sig):
        [Na,Nr] = cp.shape(sig)
        da = (np.arange(Na)-Na//2)*(self.Vr/self.PRF) - self.Rc*np.sin(self.theta_c)
        da = da[:,np.newaxis]
        eta = cp.array(self.forward)/self.Vr
        sig = sig*cp.exp(-2j*cp.pi*self.feta_c*eta)

        linear_interp = intp.interp1d(np.squeeze(self.forward), np.arange(Na), kind='linear', fill_value="extrapolate")
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


    def estimate_doppler_ceneter(self, sig):
        [Na, Nr] = sig.shape
        sig_fftr = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig, axes=1), axis=1), axes=1)
        dphase = np.conj(sig_fftr[0:Na-1,:])*(sig_fftr[1:Na, :])
        feta_c = np.angle(dphase.mean())/(2*np.pi)*self.PRF
        theta_c = np.arcsin(feta_c*self.lambda_/(2*self.Vr))
        return theta_c


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
        data_rc = self.sig
        rcmc = sar_focus.erma_rcmc(cp.array(data_rc))
        ac = sar_focus.erma_ac(cp.array(rcmc))

        afocus = AutoFocus(self.Fr, self.Tr, self.f0, self.PRF, self.Vr, self.Br, self.feta_c, self.R0, self.theta_width)
        ftau = cp.arange(-Nr/2, Nr/2, 1)*(self.Fr/Nr)
        mat_ftau = cp.tile(ftau, (Na, 1))
        dR = cp.zeros(Na)

        snr = cp.array([-10.0, -2.0, -7.0, -12.5,-10.0,-10.0,-12.5])
        pre_win_len = np.zeros_like(snr)
        # da = self.eta_c*self.Vr + (cp.arange(Na)-Na//2)*(self.Vr/self.PRF)
        block_cnt = 3
        exit_flag = False
        for i in range(5,0,-1):
            print("snr:", snr)
            ac = afocus.rechirp(cp.array(ac))
            ac = afocus.dechirp(cp.array(ac))
            error_line,win_len = afocus.spga(cp.array(ac), block_cnt, snr, 30, 10, method="line", range_win=30)
            error_line = cp.array(error_line)
            mat_dr = error_line/(4*np.pi)*self.lambda_
            dR += mat_dr[:, mat_dr.shape[1]//2]
            
            mat_dr = cp.tile(dR[:, cp.newaxis], (1, Nr))
            data_rc_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(self.sig, axes=1), axis=1), axes=1)
            data_rc_fftr = data_rc_fftr*cp.exp(-1j*4*cp.pi*mat_dr*(mat_ftau+self.f0)/self.c)
            data_rc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_rc_fftr, axes=1), axis=1), axes=1)
            rcmc = sar_focus.erma_rcmc(cp.array(data_rc))
            ac = sar_focus.erma_ac(cp.array(rcmc))
            for j in range(block_cnt):
                if pre_win_len[j] == 0:
                    pre_win_len[j] = win_len[j]
                elif win_len[j] < 100:
                    snr[j] -= 2
                elif win_len[j] > pre_win_len[j]:
                    exit_flag = False
                pre_win_len[j] = win_len[j] 

            if exit_flag:
                break

        self.sig = ac.get()
        # plt.figure()
        # plt.imshow(np.abs(rcmc.get()), aspect='auto', cmap='jet')
        # plt.savefig("../../../fig/afscan_vehicle/rcmc_after.png", dpi=300)

        # plt.figure()
        # plt.plot(-dR.get())
        # plt.savefig("../../../fig/afscan_vehicle/dR_estimate.png", dpi=300)

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

    def read_ant_pattern(self, file_path_e, file_path_a=None):
        data = pd.read_excel(file_path_e, header=1)
        self.r_angle = np.deg2rad(np.array(data['Elevation (deg)'])) ## rad
        self.r_pattern = np.array(data['Amplitude (dB)']) ## dB
        if file_path_a:
            data_a = pd.read_excel(file_path_a, header=1)
            self.a_angle = np.deg2rad(np.array(data_a['Azimuth (deg)'])) ## rad
            self.a_pattern = np.array(data_a['Amplitude (dB)']) ## dB
        else:
            self.a_pattern = None
    
    def fscan_super_resolution(self, sig, batch_size=256):

        Na, Nr = sig.shape
        win_len = self.theta_az/(np.abs(self.theta_upf-self.theta_lowf))*Nr*0.3
        print("win_len:", win_len)
        x = cp.arange(-Nr/2, Nr/2, 1)
        window =  cp.exp(-0.5 * ((x) / win_len) ** 2)

        window = window / cp.sqrt(cp.sum(window ** 2))
        win_spectrum = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(window)))

        sig = cp.ascontiguousarray(sig)
        sig_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(sig, axes=0), axis=0), axes=0)

        num_batches = (Na + batch_size - 1) // batch_size
        for start in tqdm.tqdm(range(0, Na, batch_size), 
                            total=num_batches, 
                            desc="Super-resolution (batched GPU)"):
            
            end = min(start + batch_size, Na)
            sig_ffta[start:end, :], info =  recover_dft_phase_batch(
                sig_ffta[start:end, :], win_spectrum,
                Nr, lam=1, tol=1e-5, max_iter=1000
            )
            if info != 0:
                print(f"Warning: CG did not converge for batch {start}-{end}. Info: {info}")
        sig = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_ffta, axes=0), axis=0), axes=0)
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
        print("area:", area)
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
    
    
if __name__ == "__main__":
    cp.cuda.Device(1).use()
    prefix = "../../../data/"
    example_tag = "example_13"
    param_path = f"{prefix}{example_tag}_param.mat"
    data_path = f"{prefix}{example_tag}_sig.mat"
    pos_path = f"{prefix}{example_tag}_pos.mat"
    afscan = AFScanData(param_path, data_path, pos_path)
    afoucs = AutoFocus(afscan.Fr, afscan.Tr, afscan.f0, afscan.PRF, afscan.Vr, afscan.Br, afscan.fc, afscan.R0, afscan.theta_width)
    afscan.sig = afoucs.Moco_first(cp.array(afscan.sig), cp.array(afscan.right-afscan.Y0), -cp.array(afscan.down-afscan.H))

    afscan.sig = afscan.azimuth_interp(cp.array(afscan.sig))
    afscan.sig = afscan.doppler_shift(afscan.sig, afscan.feta_c)
    afscan.sig = afscan.doppler_downsample(afscan.sig, afscan.PRF, 500)
    afscan.sig = afscan.doppler_shift(afscan.sig, -afscan.feta_c)
    afscan.PRF = 500

    echo_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(afscan.sig)))).get()
    plt.figure()
    plt.imshow(np.abs(echo_fft2), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan_vehicle/par_echo_fft2.png", dpi=300)

    # afscan.sig = afscan.squint_sm(cp.array(afscan.sig))
    afscan.sig = afscan.process_data_rd_pga()
    max_value = np.abs(afscan.sig).max()
    noise_level = np.percentile(np.abs(afscan.sig), 80)
    snr = 20 * np.log10(max_value / noise_level)
    print(f"Estimated SNR: {snr:.2f} dB before super resolution")

    afscan.Kfscan = afscan.estimate_kfscan(cp.array(afscan.sig))
    afscan.sig = afscan.fscan_dramp(cp.array(afscan.sig))
    fc = afscan.estimate_fscan_center(cp.array(afscan.sig))
    afscan.sig = afscan.fscan_shift(cp.array(afscan.sig), fc)
    sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(afscan.sig)))).get()
    plt.figure()
    plt.imshow(np.abs(sig_fft2), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan_vehicle/par_focus_fft2_before.png", dpi=300)

    afscan.sig = afscan.fscan_super_resolution(cp.array(afscan.sig))

    sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(afscan.sig)))).get()
    plt.figure()
    plt.imshow(np.abs(sig_fft2), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan_vehicle/par_focus_fft2_after.png", dpi=300)

    afscan.sig = afscan.fscan_shift(cp.array(afscan.sig), -fc)
    afscan.sig = afscan.fscan_reramp(cp.array(afscan.sig))

    max_value = np.abs(afscan.sig).max()
    noise_level = np.percentile(np.abs(afscan.sig), 80)
    snr = 20 * np.log10(max_value / noise_level)
    print(f"Estimated SNR: {snr:.2f} dB after super resolution")

    


    focus = afscan.sig

    image = np.abs(focus)
    image_abs = np.abs(image)
    image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
    iio.imwrite("../../../fig/afscan_vehicle/par_focus.tif", image_norm)

    # focus = focus.get()
    focus_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(focus))))

    focus_fft2 = focus_fft2.get()
    focus_tau_feta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(cp.array(focus), axes=0), axis=0), axes=0).get()
    plt.figure()
    plt.imshow(image_abs, aspect='auto', cmap='gray')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan_vehicle/par_focus.png", dpi=300)



    plt.figure()
    plt.imshow(np.abs(focus_tau_feta), aspect='auto')
    plt.xlabel("Range samples")
    plt.ylabel("Azimuth lines")
    plt.colorbar()
    plt.savefig("../../../fig/afscan_vehicle/par_focus_tau_feta.png", dpi=300)

    dot_estimate = DotEstimator(14, afscan.c, afscan.Vr, afscan.PRF, afscan.Fr, "../../../fig/afscan_vehicle/")
    dot_estimate.dot_estimate((focus), (int(1/(afscan.Vr/afscan.PRF)), int(3/(afscan.c/(2*afscan.Fr)))), 16)
