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

my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
from matplotlib import font_manager

plt.rc("font", family=my_font.get_name())
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14

path = "F:\\sar\\fig\\afscan\\focus_all.mat"
blk_num = 3

focus = sio.loadmat(path)['focus_all']

print(focus.shape)
if focus.shape[0] < focus.shape[1]:
    focus = focus.T

Fs = 2.5e9

B = 2e9

focus = focus[9000:10000, :]
[Na,Nr] = focus.shape
batch_size = 1

win_len = 320/3000*Nr
ax = cp.arange(-Nr/2, Nr/2, 1)
# shift = Nr//4
shift = 0
window =  cp.exp(-0.5 * ((ax+shift) / win_len) ** 2)
window = window / cp.sqrt(cp.sum(window**2)) 
M = Nr//2

hy1 = cp.ones(M)
hy2 = cp.ones(M)
dt = -0.5/(Fs/M)
fs = cp.arange(-M/2, M/2, 1)*Fs/M
angle = 2*cp.pi*fs*dt
hy1 = hy1*cp.exp(1j*angle)
hy2 = hy2*cp.exp(1j*angle)

for start in tqdm.tqdm(range(0, Na, batch_size), 
                    total=Na, 
                    desc="Super-resolution (batched GPU)"):
    
    end = min(start + batch_size, Na)
    batch = cp.squeeze(focus[start, :])

    batch_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(batch)))
    # hy1, hy2, _ = build_annihilating_filter(batch_fft, M)
    batch_fft = solve_c_based_cg(batch_fft, window, hy1, hy2, lam1=5e-3, lam2 = 5e-3, eps=0)
    batch_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(batch_fft)))
    focus[start, :] = batch_ifft.get()

tif_path = f"../fig/afscan/part_focus_super_130_cons.tif"
image_abs = np.abs(focus)
image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
iio.imwrite(tif_path, image_norm)

kaiser_win = np.kaiser(focus.shape[1], beta=5)
focus_all = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.array(focus), axes=1), axis=1), axes=1)
focus_all = focus_all*kaiser_win[np.newaxis, :]
focus_all = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(focus_all, axes=1), axis=1), axes=1)
image_abs = np.abs(focus_all)
image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
iio.imwrite("../fig/afscan/par_focus_super_kaiser_130_cons.tif", image_norm)

