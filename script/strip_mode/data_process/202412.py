import scipy.io as sci
import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append(r"../")
from sar_focus import SAR_Focus

cp.cuda.Device(0).use()

class Fcous_Air:
    def __init__(self, Tr, Br, f0, t0, Fr, PRF, fc, Vr):
        self.c = 299792458
        self.Tr = Tr
        self.Br = Br
        self.f0 = f0
        self.t0 = t0
        self.Fr = Fr
        self.PRF = PRF
        self.fc = fc
        self.Vr = Vr
        self.lambda_ = self.c/self.f0
        self.R0 = (t0)*self.c/2
        self.Kr = self.Br/self.Tr
        self.focus = SAR_Focus(Fr, Tr, f0, PRF, Vr, Br, fc, self.R0, self.Kr)
        
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

        print(sig.shape)
        self.Na, self.Nr = np.shape(sig)
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
    

if __name__ == '__main__':
    focus_air = Fcous_Air(24e-6, 2e9, 3.5e10, 3.46e-5, 2.5e9, 5000.1, 0, 72.2475)
    # focus_air = Fcous_Air(4.175000000000000e-05, 30.111e+06 , 5.300000000000000e+09 ,  6.5959e-03, 32317000, 1.256980000000000e+03, -6900, 7062)
    focus_air.read_data("../../../data/security/202412/example_49_cropped_sig_rc_small.mat", "../../../data/security/202412/pos.mat")

    # focus_air.read_data("./English_Bay_ships.mat", "./pos.mat")

    print(np.mean(np.diff(focus_air.forward)/np.diff(focus_air.frame_time)))
    print(np.mean(np.diff(focus_air.right)/np.diff(focus_air.frame_time)))
    print(np.mean(np.diff(focus_air.down)/np.diff(focus_air.frame_time)))

    plt.figure()
    plt.imshow(np.abs(focus_air.sig), cmap='gray', aspect='auto')
    plt.tight_layout()
    plt.savefig("../../../fig/data_202412/echo.png")

    focus_air.inverse_rc()
    
    image = focus_air.focus.wk_focus(cp.array(focus_air.sig), focus_air.R0)
    image_abs = cp.abs(image)
    image_abs = image_abs/cp.max(cp.max(image_abs))
    image_abs = 20*cp.log10(image_abs+1)
    image_abs = image_abs**0.3
    plt.figure()
    plt.imshow(image_abs.get(), cmap='gray', aspect='auto')
    plt.tight_layout()
    plt.savefig("../../../fig/data_202412/wk_image.png")

        
        