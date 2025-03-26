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
from autofocus import AutoFocus
import doppler_estimation as doppler
from joblib import Parallel, delayed

cp.cuda.Device(1).use()

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
        self.auto_focus = AutoFocus(Fr, Tr, f0, PRF, Vr, Br, fc, self.R0, self.Kr)
        
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
    
    def rd_focus_rcmc(self, data_rc):  
        [Na, Nr] = cp.shape(data_rc)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + cp.fft.fftshift(cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fft(data_rc, Na, axis=0)
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


        data_final = cp.fft.ifft(data_fft_a_rcmc, axis=0)

        return data_final.get()
    
    def rd_focus_ac(self, data_rcmc):
        [Na, Nr] = cp.shape(data_rcmc)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + cp.fft.fftshift(cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_a_rcmc = cp.fft.fft(data_rcmc, axis=0)
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## 方位压缩
        # Ka = 2 * self.Vr**2/ (self.lambda_ * mat_R0)
        # Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifft(data_fft_a_rcmc, axis=0)
        data_final = data_ca_rcmc
        return data_final.get()
    
    def rd_unfoucs_ac(self, data_ac):
        [Na, Nr] = cp.shape(data_ac)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + cp.fft.fftshift(cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))
        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)

        tau = cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)

        mat_R0 = mat_tau*self.c/2 + self.R0;  

        data_fft_ac = cp.fft.fft(data_ac, axis=0)
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子
        ## 方位解压缩
        # Ka = 2 * self.Vr**2/ (self.lambda_ * mat_R0)
        # Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        Ha = cp.exp(-4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_unac = data_fft_ac*Ha
        data_final = cp.fft.ifft(data_fft_unac, axis=0)
        return data_final.get()

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
        image_abs = np.abs(image)
        image_abs = image_abs/np.max(np.max(image_abs))
        image_abs = 20*np.log10(image_abs+1)
        # Perform histogram equalization on image_abs
        image_abs = np.flip(image_abs, axis=1)
        image_abs = cv2.normalize(image_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_abs = cv2.equalizeHist(image_abs)
        return image_abs



    

if __name__ == '__main__':
    focus_air = Fcous_Air(24e-6, 2e9, 37e9, 3.46e-5, 2.5e9, 5000/3, 0.04, 72.25)

    ## 地面海拔
    altitude = 337
    # focus_air = Fcous_Air(4.175000000000000e-05, -30.111e+06 , 5.300000000000000e+09 ,  6.5959e-03, 32317000, 1.256980000000000e+03, -6900, 7062)
    focus_air.read_data("../../../data/example_49_cropped_sig_rc_small.mat", "../../../data/pos.mat")

    # focus_air.read_data("../../../data/English_Bay_ships.mat", "../../../data/pos.mat")

    vr = (np.diff(focus_air.forward[::3])/np.diff(focus_air.frame_time[::3]))

    focus_air.sig = focus_air.auto_focus.Moco_first(cp.array((focus_air.sig)), cp.array(focus_air.right[::3]), cp.array(-focus_air.down[::3]), np.deg2rad(58)) 

    #飞机相对于地面的高度
    H = cp.mean(-focus_air.down[::3]) - altitude
    tmp = np.zeros((focus_air.Na*20//10, focus_air.Nr), dtype=complex)
    tmp[0:focus_air.Na,0:focus_air.Nr] = focus_air.sig
    focus_air.sig = tmp

    # print("Doppler center: ", doppler_center)
    focus_air.Na, focus_air.Nr = np.shape(focus_air.sig)
    print(focus_air.Na, focus_air.Nr)

    focus_air.sig = focus_air.rd_focus_rcmc(cp.array((focus_air.sig)))
    focus_air.sig = focus_air.rd_focus_ac(cp.array((focus_air.sig)))

    # Divide the image into 4 equal parts along the y-axis and 3 equal parts along the x-axis
    Nx = 3
    Ny = 4
    y_splits = np.array(np.array_split(focus_air.sig, Ny, axis=0))
    x_splits =np.array([np.array_split(y_split, Nx, axis=1) for y_split in y_splits])
    output = np.zeros(x_splits.shape, dtype=complex)
    print(output.shape)

    # Save each sub-image

    def process_sub_image(i, j, sub_image):
        cp.cuda.Device(1).use()
        sub_image = focus_air.rd_unfoucs_ac(cp.array(sub_image))
        # Apply Kaiser window along the y-axis
        kaiser_window = np.kaiser(sub_image.shape[0], beta=14)[:, cp.newaxis]
        kaiser_window = np.tile(kaiser_window, (1, sub_image.shape[1]))
        sub_image = sub_image * kaiser_window
        image, rms = focus_air.auto_focus.pga(cp.array(sub_image.T), 10)
        print("part {}{}  ;".format(i+1, j+1), " rms: ", rms)
        image = image.T
        image = focus_air.rd_focus_ac(cp.array((sub_image)))
        image_show = focus_air.get_showimage(image)
        return i, j, image, image_show

    results = Parallel(n_jobs=-1)(delayed(process_sub_image)(i, j, sub_image) 
                                  for i, y_split in enumerate(x_splits) 
                                  for j, sub_image in enumerate(y_split))
    
    plt.figure(figsize=(6, 8))
    for i, j, image, image_show in results:
        plt.subplot(Ny, Nx, i*Nx+j+1)
        plt.imshow(image_show, cmap='gray', aspect='auto')
        plt.title("Part {}, {}".format(i+1, j+1))
        output[i, j, :, :] = image
    # Concatenate the sub-images back together
    plt.tight_layout()
    path = "../../../fig/data_202412/image_part2.png"
    plt.savefig(path)

    reconstructed_image = np.block([[output[j, i, :, :] for i in range(Nx)] for j in range(Ny)])
    image_show = focus_air.get_showimage(reconstructed_image)
    plt.figure(figsize=(4.5, 12))
    plt.imshow(image_show, cmap='gray', aspect='auto')
    plt.title("Image")
    plt.savefig("../../../fig/data_202412/image3.png")

            
            