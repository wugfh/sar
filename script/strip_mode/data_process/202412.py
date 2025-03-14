import scipy.io as sci
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
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
        self.R0 = (self.t0)*self.c/2
        print(self.R0)
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

        self.Na, self.Nr = np.shape(sig)
        sig = sig[:, 0:int(self.Nr/4)]
        self.sig = np.array(sig)
        self.Na, self.Nr = np.shape(self.sig)
        print(self.Na, self.Nr)
    
    def inverse_rc(self):
        f_tau = np.fft.fftshift(np.linspace(-self.Nr/2,self.Nr/2-1,self.Nr)*(self.Fr/self.Nr))
        f_eta = self.fc + (np.linspace(-self.Na/2,self.Na/2-1,self.Na)*(self.PRF/self.Na))

        [mat_f_tau, _] = np.meshgrid(f_tau, f_eta)

        Hir = np.exp(-1j*np.pi*mat_f_tau**2/self.Kr)

        self.sig = np.fft.ifft(np.fft.fft(self.sig, axis=1)*Hir, axis=1)

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
    
    def rd_focus_ac(self, data_rc):  
        [Na, Nr] = cp.shape(data_rc)
        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fr/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [_, mat_f_eta] = cp.meshgrid(f_tau, f_eta)
        tau = 2*self.R0/self.c + cp.arange(-Nr/2, Nr/2, 1)*(1/self.Fr)
        eta = cp.arange(-Na/2, Na/2, 1)*(1/self.PRF)  
        mat_tau, _ = cp.meshgrid(tau, eta)


        ## 范围压缩
        mat_D = cp.sqrt(1-self.c**2*mat_f_eta**2/(4*self.Vr**2*self.f0**2))#徙动因子

        ## RCMC
        data_fft_a = cp.fft.fft(data_rc, Na, axis=0)
        sinc_N = 8
        mat_R0 = mat_tau*self.c/2;  

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
        Ha = cp.exp(4j*cp.pi*mat_D*mat_R0*self.f0/self.c)
        # ofself.Fset = cp.exp(2j*cp.pi*mat_f_eta*eta_c)
        data_fft_a_rcmc = data_fft_a_rcmc*Ha
        data_ca_rcmc = cp.fft.ifft(data_fft_a_rcmc, Na, axis=0)

        data_final = data_ca_rcmc
        # data_final = cp.abs(data_final)/cp.max(cp.max(cp.abs(data_final)))
        # data_final = 20*cp.log10(data_final)
        return data_final
    
    

if __name__ == '__main__':
    focus_air = Fcous_Air(24e-6, 2e9, 35e9, 3.46e-5, 2.5e9, 2000, 400, 72.24)
    # focus_air = Fcous_Air(4.175000000000000e-05, -30.111e+06 , 5.300000000000000e+09 ,  6.5959e-03, 32317000, 1.256980000000000e+03, -6900, 7062)
    focus_air.read_data("../../../data/security/202412/example_49_cropped_sig_rc_small.mat", "../../../data/security/202412/pos.mat")

    # focus_air.read_data("../../../data/security/202412/English_Bay_ships.mat", "../../../data/security/202412/pos.mat")

    print((focus_air.forward[-1] - focus_air.forward[0])/(focus_air.frame_time[-1] - focus_air.frame_time[0]))
    print(np.mean(focus_air.down))

    plt.figure()
    plt.imshow(np.abs((focus_air.sig)), cmap='jet', aspect='auto')
    plt.tight_layout()
    plt.savefig("../../../fig/data_202412/echo.png")
    
    image = focus_air.rd_focus_ac(cp.array((focus_air.sig)))
    image_abs = np.abs(image)
    image_abs = image_abs/np.max(np.max(image_abs))
    image_abs = 20*np.log10(image_abs+1)
    image_abs = image_abs**0.3
    plt.figure()
    plt.imshow(image_abs.get(), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("../../../fig/data_202412/image.png")

        
        