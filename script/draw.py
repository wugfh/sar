import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
from tqdm import tqdm
import tqdm
from spectrum_recovery import build_annihilating_filter, solve_c_based_cg
import scipy.io as sio
import imageio as iio
from matplotlib import font_manager

c0 = 3e8
# --- Geometry (from fscan.py) ---
beta     = cp.deg2rad(45)         # antenna mounting angle
phi      = beta + cp.deg2rad(14.3)  # swath centre look angle
H_plat   = 3e3                    # platform height [m]
theta_c  = cp.deg2rad(0)          # squint angle

# --- Waveform (from fscan.py) ---
B        = 2e9                    # bandwidth [Hz]
Fs       = B * 1.25                # sampling rate
Tp       = 10e-7                  # pulse width [s] (from BeamScan)
Kr       = B / Tp                 # chirp rate [Hz/s]
f0       = 35e9 
# --- Derived ---
Nr       = int(cp.ceil(Fs * Tp*3))   
Nr       = 3000
print("Nr:{}".format(Nr))
Tr       = Nr / Fs                     # pulse duration [s]
df       = Fs / Nr                      # frequency resolution
freq     = f0 + (cp.arange(Nr) - Nr // 2) * df   # frequency axis [Hz]
N_inband = int(cp.ceil(B / df))         # samples inside bandwidth


def antenna_pattern_pr(f, doa, a, d, N_elem):
    lam_now = c0 / f
    # waveguide wavelength
    sin_arg = lam_now / (2 * a)
    sin_arg = cp.clip(sin_arg, 0, 0.999)        # stay below cutoff
    lam_g = lam_now / cp.sqrt(1 - sin_arg**2)

    # inter‑element phase progression
    u1 = (2 * cp.pi / lam_now) * d * cp.sin(doa) \
    - (2 * cp.pi / lam_g) * (d - lam_g / 2)

    # Dirichlet kernel (15 elements)
    pr = cp.sin(N_elem * u1 / 2) / (N_elem * cp.sin(u1 / 2))
    pr = cp.nan_to_num(pr, nan=1.0)             # lim_{u1→0} = 1
    return pr



a = 0.004871409163516
lambda_ = c0/f0
lambda_g=lambda_/np.sqrt(1-(lambda_/(2*a))**2)
shift_d = 2/3.717054305989132
d = lambda_g/2 +shift_d* lambda_g
P2 = antenna_pattern_pr(freq, phi-beta, 0.004871409163516, d, 16)
P2 = P2 ** 2
P2 = P2/cp.sqrt(cp.sum(P2**2))



plt.figure()
plt.plot(P2.get(), linewidth=2, c= "black")
plt.axis('off')                     # 只保留曲线
plt.savefig('clean_plot.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()