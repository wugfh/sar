import scipy.io as sci
import numpy
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append(r"../")
from sar_focus import SAR_Focus


class Fcous_Air:
    def __init__(self, Tr, Br, f0, t0, Fr, PRF, fc):
        self.c = 299792458
        self.Tr = Tr
        self.Br = Br
        self.f0 = f0
        self.t0 = t0
        self.Fr = Fr
        self.PRF = PRF
        self.fc = fc
        self.Vr = 290
        self.lambda_ = self.c/self.f0
        self.R0 = (t0-Tr/2)*self.c/2
        self.Kr = self.Br/self.Tr
        self.focus = SAR_Focus(Fr, Tr, f0, PRF, self.Vr, Br, fc, self.R0, self.Kr)

    def read_data(self, filename):
        data = h5py.File(filename)
        sig = data['sig'][:]
        sig = sig["real"] + 1j*sig["imag"]
        self.sig = cp.array(sig)
        self.Na, self.Nr = sig.shape
        print(self.Na, self.Nr)


if __name__ == '__main__':
    focus_air = Fcous_Air(2.4e-5, 2e9, 3.5e10, 3.46e-5, 2.5e9, 6000.1, 0)
    focus_air.read_data("../../../data/security/202412/example_49_cropped_sig_rc_small.mat")
    image = focus_air.focus.wk_focus(focus_air.sig, focus_air.R0)
    image_abs = cp.abs(image)
    image_abs = cp.max(cp.max(image_abs))
    image_abs = 20*cp.log10(image_abs)

    plt.figure()
    plt.imshow(image_abs, cmap='gray')
    plt.colorbar()
    plt.savefig("../../../fig/data_202412/example_49_cropped_sig_rc_small.png")

        