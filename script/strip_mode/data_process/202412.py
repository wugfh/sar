
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
sys.path.append(r"../../")
from sar_focus import SAR_Focus
from sinc_interpolation import SincInterpolation
from autofocus import AutoFocus
import doppler_estimation as doppler
from concurrent.futures import ThreadPoolExecutor
import scipy.io as sio

class Fcous_Air:
    def __init__(self, Tr, Br, f0, R0, Fr, PRF, fc, Vr):
        self.c = 299792458
        self.Tr = Tr
        self.Br = cp.abs(Br)
        self.f0 = f0
        self.Fr = Fr
        self.PRF = PRF
        self.fc = fc
        self.Vr = Vr
        self.lambda_ = self.c/self.f0
        self.R0 = R0
        self.Kr = Br/Tr
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.auto_focus = AutoFocus(Fr, Tr, f0, PRF, Vr, Br, fc, self.R0)
        
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

        sig = sig[::3,:]
        self.sig = np.array(sig)
        self.Na, self.Nr = np.shape(self.sig)


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
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        Ksrc = 2*self.Vr**2*self.f0**3*mat_D**3/(self.c*self.R0*mat_f_eta**2)

        data_fft_r = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(echo, axes=1), axis=1), axes=1)
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        Hm = cp.exp(-1j*cp.pi*mat_f_tau**2/Ksrc)
        if(squint_angle > 2):
            data_fft_cr = data_fft_r*Hr*Hm
        else:
            data_fft_cr = data_fft_r*Hr
        data_cr = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_cr, axes=1), axis = 1), axes=1)
        return data_cr.get()
    
    def rd_focus_rcmc(self, data_rc):  
        [Na, Nr] = cp.shape(data_rc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rc, axes=0), axis=0), axes=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_a = cp.ascontiguousarray(data_fft_a)
        data_fft_a_real = cp.real(data_fft_a).astype(cp.double)
        data_fft_a_imag = cp.imag(data_fft_a).astype(cp.double)


        delta = mat_R0/mat_D - mat_R0
        delta = delta*2/(self.c/self.Fr)
        sinc_intp = SincInterpolation()
        data_fft_a_rcmc_real = sinc_intp.sinc_interpolation(data_fft_a_real, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc_imag = sinc_intp.sinc_interpolation(data_fft_a_imag, delta, Na, Nr, sinc_N)
        data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag


        data_final = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)

        return data_final.get()
    
    def rd_focus_ac(self, data_rcmc):
        [Na, Nr] = cp.shape(data_rcmc)
        f_tau = (cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_a_rcmc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rcmc, axes=0), axis=0), axes=0)
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## 方位压缩
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        # Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_fft_a_rcmc, axes=0), axis=0), axes=0)
        data_final = data_ca_rcmc
        return data_final.get()

    def dechirp(self, data):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        mat_R0 = mat_tau*self.c/2 + self.R0;  
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        data = data*cp.exp(1j*cp.pi*Ka*mat_eta**2)
        data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
        return data.get()
    
    def rechirp(self, data):
        Na, Nr = cp.shape(data)
        tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/self.Fr)
        eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta) 
        mat_R0 = mat_tau*self.c/2 + self.R0;  
        Ka = 2*self.Vr**2*cp.cos(self.theta_c)**3/(self.lambda_*mat_R0)
        data = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data, axes=0), axis=0), axes=0)
        data = data*cp.exp(-1j*cp.pi*Ka*mat_eta**2)
        return data.get()


    def range2ground(self, data, H):
        Na,Nr = cp.shape(data)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        range_0 = tau*self.c/2 + self.R0
        mat_R0 = cp.tile(range_0, (Na,1))
        ground_0 = cp.sqrt((range_0**2 - H**2))
        ground_grid = cp.linspace(cp.min(ground_0), cp.max(ground_0), Nr)
        ground_R0 = cp.sqrt((ground_grid**2 + H**2))
        mat_gnd = cp.tile(ground_R0, (Na,1))
        delta = (mat_gnd - mat_R0)
        delta = delta*2/(self.c/self.Fr)
        sinc_intp = SincInterpolation()
        data_ground_imag = sinc_intp.sinc_interpolation(cp.imag(data), delta, Na, Nr, 8)
        data_ground_real = sinc_intp.sinc_interpolation(cp.real(data), delta, Na, Nr, 8)
        data_ground = data_ground_real + 1j*data_ground_imag
        return data_ground.get()
    
    def get_showimage(self, image):
        threshold = np.percentile(np.abs(image), 75)
        image_abs = np.abs(image)
        image_abs[image_abs > threshold] = threshold
        return image_abs
    
    def upsample(self, data, N):
        Na, Nr = cp.shape(data)
        data_fft = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(data)))
        tmp = cp.zeros((N[0]*Na, N[1]*Nr), dtype=complex)
        tmp[N[0]*Na//2-Na/2:N[0]*Na//2+Na/2, N[1]*Nr//2-Nr/2:N[1]*Nr//2+Nr/2] = data_fft
        data_up = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(tmp)))
        return data_up.get()
    
    def get_azimuth_IRW(self, ehco, uprate, vr):
        max_val = cp.max(cp.abs(ehco), axis=0)
        max_index = cp.argmax(max_val)
        max_value = cp.max(cp.abs(ehco[:, max_index]))
        midx = cp.argmax(cp.abs(ehco[:,max_index]))
        half_max = max_value/cp.sqrt(2) # 3dB 宽度
        left_idx = cp.where(cp.abs(ehco[:midx, max_index]) < half_max)[0]
        right_idx = cp.where(cp.abs(ehco[midx:, max_index]) < half_max)[0]

        left_idx = left_idx[-1] if len(left_idx) > 0 else 0
        right_idx = midx + right_idx[0] if len(right_idx) > 0 else ehco.shape[0] - 1

        irw = right_idx - left_idx
        irw = irw*vr/(self.PRF*uprate)
        return irw.get(), max_index.get(), left_idx.get(), right_idx.get()
    
    def residual_rcm(self, sig):
        bsize = int(focus_air.Na)
        block_len = bsize
        lmid = np.arange(0, Na, block_len) + block_len//2
        afoucs = AutoFocus(focus_air.Fr, focus_air.Tr, focus_air.f0, focus_air.PRF, focus_air.Vr, focus_air.Br, focus_air.fc, focus_air.R0)
        step = 0
        for mid in lmid:
            step += 1
            start = np.maximum(0, int(mid - bsize/2))
            end = int(np.minimum(start+bsize, focus_air.Na))
            print("step:{}, start:{}, end:{}".format(step, start, end))
            W = cp.zeros_like(focus_air.sig)
            W[start:end, :] = 1
            block = cp.array(focus_air.sig)*W
            error, rms, windata = afoucs.pga_autofocus(cp.array((block)),R, num_iter=30, snr = -20)
        rcm = error*self.lambda_/(4*cp.pi)
        sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(sig)))

        ftau = (cp.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fr/self.Nr))
        mat_ftau = cp.tile(ftau, (self.Na,1))
        rcm_phase = cp.exp(-1j*4*cp.pi*rcm*mat_ftau/self.c)
        sig_fft2 = sig_fft2*rcm_phase
        sig_rcm = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2)))
        return sig_rcm.get()

if __name__ == '__main__':
    cp.cuda.Device(0).use()
    focus_air = Fcous_Air(24e-6, 2e9, 37e9, 5256.3, 2.5e9, 5000/3, 0, 72.25)
    R0 =  3.46e-5*focus_air.c/2
    focus_air.R0 = R0
    ## 地面海拔
    altitude = 337
    # focus_air = Fcous_Air(4.175000000000000e-05, -30.111e+06 , 5.300000000000000e+09 ,  6.5959e-03, 32317000, 1.256980000000000e+03, -6900, 7062)
    focus_air.read_data("../../../data/example_49_cropped_sig_rc_small.mat", "../../../data/pos1.mat")
    plt.figure()
    plt.imshow(np.abs(focus_air.sig), aspect='auto', cmap='jet')
    plt.colorbar()  
    plt.savefig("../../../fig/data_process/rc_small_sig.png", dpi=300)

    # focus_air.read_data("../../../data/English_Bay_ships.mat", "../../../data/pos.mat")
    # focus_air.sig = np.roll(focus_air.sig, focus_air.Nr*(37e9-focus_air.f0)//(2.5e9*2), axis=1)
    #数据相对于地面的高度
    phi = np.deg2rad(58)

    vr = (np.diff(focus_air.forward[::3])/np.diff(focus_air.frame_time[::3]))
    vr = np.append(vr, vr[-1])
    # focus_air.Vr = cp.array(np.tile(vr[:, np.newaxis], (1, focus_air.Nr)))
    # Apply Kaiser window along the y-axis
    # kaiser_window = np.kaiser(focus_air.sig.shape[0], beta=5)[:, cp.newaxis]
    # kaiser_window = np.tile(kaiser_window, (1, focus_air.sig.shape[1]))
    # focus_air.sig = focus_air.sig * kaiser_window

    H = np.mean(-focus_air.down[::3])
    focus_air.sig = focus_air.auto_focus.Moco_first(cp.array((focus_air.sig)), cp.array(focus_air.right[::3]), cp.array(-focus_air.down[::3]-H), cp.array(focus_air.forward[::3]), phi) 


    temp = cp.zeros((int(focus_air.Na*2), focus_air.Nr), dtype=cp.complex64)
    [Na, Nr] = cp.shape(focus_air.sig)
    temp[Na//2-focus_air.Na/2:Na//2+focus_air.Na/2, :] = cp.array(focus_air.sig)
    focus_air.sig = temp
    [focus_air.Na, focus_air.Nr] = cp.shape(focus_air.sig)
    [Na,Nr] = cp.shape(focus_air.sig)
    del temp


    focus_air.sig = focus_air.rd_focus_rcmc(cp.array((focus_air.sig)))
    # focus_air.sig = focus_air.rd_focus_ac(cp.array((focus_air.sig)))
    # focus_air.sig = focus_air.auto_focus.Moco_second(cp.array((focus_air.sig)), cp.array(focus_air.right[::3]), cp.array(-focus_air.down[::3]-H), phi) 
    focus_air.sig = focus_air.rd_focus_ac(cp.array((focus_air.sig)))
    # sig_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(focus_air.sig)))
    # kaiser_window = cp.kaiser(focus_air.Na, beta=5)[:, cp.newaxis]
    # kaiser_window = cp.tile(kaiser_window, (1, focus_air.Nr))
    # sig_fft2 = sig_fft2 * kaiser_window
    # focus_air.sig = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(sig_fft2)))

    # del temp,  sig_fft2
    # focus_air.sig = focus_air.dechirp(cp.array((focus_air.sig)))
  
    # image_show = focus_air.get_showimage(focus_air.sig)
    # plt.figure(figsize=(4.5, 16))
    # plt.imshow(image_show, cmap='gray', aspect='auto')
    # plt.title(" Moco PGA")
    # plt.savefig("../../../fig/data_process/test_pga.png")
    tau = cp.arange(-Nr/2, Nr/2, 1)*(1/focus_air.Fr) + focus_air.R0*2/focus_air.c
    R = tau*focus_air.c/2

    block_size = focus_air.sig.shape[1]
    step_len = block_size
    lmid = np.arange(step_len//2, focus_air.sig.shape[1], step_len) 
    block_spga = np.zeros_like(focus_air.sig, dtype=np.complex128)
    step = 0
    mat_R = cp.tile(R[cp.newaxis, :], (focus_air.Na,1))
    
    for mid in lmid:
        start = int(max(0, mid - block_size//2))
        end = int(min(start+block_size, focus_air.sig.shape[1]))
        block = focus_air.sig[:, start:end]

        block,_ = focus_air.auto_focus.spga(((block)), mat_R[:, start:end], 12, snr_threshold=-30, num_iter=30,  win_min=10, method="line")
        pga_block,_ = focus_air.auto_focus.spga(((block)), mat_R[:, start:end], 12, snr_threshold=-30, num_iter=30,  win_min=10, method="mat")
        if np.abs(mid-start) <= np.abs(mid-end):
            bmid = np.abs(mid-start)
        else:
            bmid = pga_block.shape[1] - np.abs(mid-end)
        winlen = np.minimum(bmid*2, step_len)
        block_spga[:, mid-winlen//2:mid+winlen//2] += pga_block[:, bmid-winlen//2:bmid+winlen//2]
        step += 1
    focus_air.sig = block_spga
    del block_spga

    image_show = np.abs(focus_air.sig)
    # sio.savemat("./focus_air_image.mat", {"image": focus_air.sig})
    image_show = focus_air.get_showimage(image_show)

    plt.figure(figsize=(4.5, 16))
    plt.imshow(image_show, aspect='auto', cmap='gray')
    plt.title(" Moco PGA")
    plt.savefig("../../../fig/data_process/image.png")

    reconstructed_image = focus_air.sig
    dot_image = reconstructed_image[6500:7500, 1500: 1600]
    image_show = focus_air.get_showimage(dot_image)
    plt.figure()
    plt.imshow(image_show, cmap='gray', aspect='auto')
    plt.title("Image")
    plt.savefig("../../../fig/data_process/dot_image.png")

    uprate = 32
    dot_image_up = focus_air.upsample(cp.array(dot_image), (uprate, 1))
    image_show = focus_air.get_showimage(dot_image_up)
    plt.figure()
    plt.imshow(image_show, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title("Image")
    plt.savefig("../../../fig/data_process/dot_image_upsample.png")

    irw, max_index, left_idx, right_idx = focus_air.get_azimuth_IRW(cp.array(dot_image_up), uprate, np.mean(vr))
    midx = (left_idx + right_idx)//2
    winlen = (right_idx - left_idx)*20
    left_idx = np.maximum(midx - winlen//2, 0)
    left_idx = int(left_idx)
    right_idx = np.minimum(midx + winlen//2, dot_image_up.shape[0]-1)
    right_idx = int(right_idx)
    irw_show = np.abs(dot_image_up[:,max_index])
    # midx = np.argmax(irw_show)
    # irw_show = irw_show[midx-2000:midx+2000]
    print("IRW: ", irw)
    plt.figure()
    plt.plot(20*np.log10(irw_show+np.finfo(np.float32).eps))
    plt.grid()
    plt.title("IRW")
    plt.savefig("../../../fig/data_process/IRW.png")

    cp.cuda.Device(0).synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    exit()