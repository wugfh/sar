import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
from tqdm import tqdm
import tqdm
from spectrum_recovery import solve_c_based_cg, solve_c_based_cg_batch, build_annihilating_filter_batch, select_filter
import scipy.io as sio
import imageio as iio
from matplotlib import font_manager
from image_entropy import compute_image_entropy, compute_enl

my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
from matplotlib import font_manager

plt.rc("font", family=my_font.get_name())
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14

path = "F:\\sar\\fig\\afscan\\focus_all_100.mat"

focus = sio.loadmat(path)['focus_all']

print(focus.shape)
if focus.shape[0] < focus.shape[1]:
    focus = focus.T

Fs = 2.5e9

B = 2e9
blk_num = 6
inter = focus.shape[1]//blk_num

focus = focus[9500:10500, :]
[Na,Nr] = focus.shape
batch_size = 8

win_len = 320/3000*inter
ax = cp.arange(-inter/2, inter/2, 1)
# shift = Nr//4
shift = 0
window =  cp.exp(-0.5 * ((ax+shift) / win_len) ** 2)
window = window / cp.sqrt(cp.sum(window**2)) 
M = inter//2-1

filter_order = 200

# for start in tqdm.tqdm(range(0, Na, batch_size), 
#                     total=np.ceil(Na/batch_size), 
#                     desc="Super-resolution (batched GPU)"):
    
#     end = min(start + batch_size, Na)
#     for blk in range(blk_num):

#         batch = (focus[start:end, inter*blk:inter*(blk+1)])

#         batch_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(batch, axes=1), axis=1), axes=1)
#         hy, _ = build_annihilating_filter_batch(batch_fft, M, filter_order)
#         lam = 1e-5*cp.ones(hy.shape[1])
#         batch_fft = solve_c_based_cg_batch(batch_fft, window, hy, lam, eps=0)
#         batch_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(batch_fft, axes=1), axis=1), axes=1)
#         focus[start:end, inter*blk:inter*(blk+1)] = batch_ifft.get()

# tif_path = f"../fig/afscan/par_focus_super_100_{filter_order}.tif"
# image_abs = np.abs(focus)
# image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
# iio.imwrite(tif_path, image_norm)

# freq = np.arange(-Nr/2, Nr/2, 1) * Fs/Nr
# rect_win = np.abs(freq) < B/2
# focus_all = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.array(focus), axes=1), axis=1), axes=1)
# focus_all = focus_all*rect_win[np.newaxis, :]
# focus_all = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(focus_all, axes=1), axis=1), axes=1)
# image_abs = np.abs(focus_all)
# image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
# iio.imwrite(f"../fig/afscan/par_focus_super_rect_100_{filter_order}.tif", image_norm)

# mat_path = f"../fig/afscan/par_focus_super_100_{filter_order}.mat"
# sio.savemat(mat_path, {"focus": focus_all})

multi_10_entropy = compute_image_entropy("../fig/afscan/par_focus_super_rect_100_10.tif", bins=256)
multi_60_entropy = compute_image_entropy("../fig/afscan/par_focus_super_rect_100_60.tif", bins=256)
multi_200_entropy = compute_image_entropy("../fig/afscan/par_focus_super_rect_100_200.tif", bins=256)
bench_entropy = compute_image_entropy("../fig/afscan/par_focus_super_rect_100_0.tif", bins=256)

enl_10 = compute_enl("../fig/afscan/par_focus_super_100_10.mat")
enl_60 = compute_enl("../fig/afscan/par_focus_super_100_60.mat")
bench_enl = compute_enl("../fig/afscan/par_focus_super_100_0.mat")
enl_200 = compute_enl("../fig/afscan/par_focus_super_100_200.mat")



print(f"multi_10_entropy: {multi_10_entropy}")
print(f"multi_60_entropy: {multi_60_entropy}")
print(f"multi_200_entropy: {multi_200_entropy}")
print(f"bench_entropy: {bench_entropy}")

print("\r\n")
print(f"enl_10: {enl_10}")
print(f"enl_60: {enl_60}")
print(f"bench_enl: {bench_enl}")
print(f"enl_200: {enl_200}")