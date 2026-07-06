import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
import h5py
import imageio as iio
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")



plt.rc("font", family=my_font.get_name())
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14

pre_year_path = "F:/sar/data/snr/example_13_block_1_lastyear_original.mat"
this_year_path = "F:/sar/data/snr/example_11_block_2_crop_58000-117999_thisyear_original.mat"

blk_num = 3

with h5py.File(pre_year_path, "r") as data:
    pre_year = data['out_final_focus_pga']
    pre_year = pre_year["real"] + 1j*pre_year["imag"]
    pre_year = np.abs(np.array(pre_year)).T

with h5py.File(this_year_path, "r") as data:
    this_year = data['out_final_focus_pga']
    this_year = this_year["real"] + 1j*this_year["imag"]
    this_year = np.abs(np.array(this_year)).T

print("pre_year shape : {}".format(pre_year.shape))
print("this_year shape : {}".format(this_year.shape))

noise_pre = pre_year[4500:5000, 51000:52000].mean()
noise_this = this_year[18000:19000, 51000:52000].mean()

# pre_fig = pre_year[6000:8000,9000-2500:13000-2500]
# this_fig = this_year[6000:8000,9000:13000]
pre_fig = pre_year[13500:15500,10000:12000]
this_fig = this_year[13500-700:15500-700,10000+2900:12000+2900]

tif_path = f"../fig/super_conductor/"
image_abs = np.abs(pre_fig)
image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
iio.imwrite(tif_path + "pre.tif", image_norm)

image_abs = np.abs(this_fig)
image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
iio.imwrite(tif_path + "this.tif", image_norm)

# exit()

pre_year = pre_year[5000+1600:15000+1600, 0:-2500]
this_year = this_year[5000:15000, 2500:]


signal_pre = np.percentile(pre_year, 75, axis = 1)
signal_this = np.percentile(this_year, 75, axis = 1)

snr_pre = 20*np.log10(signal_pre/noise_pre)
snr_this = 20*np.log10(signal_this/noise_this)

PRF = 6000
t = np.arange(0, snr_pre.shape[0])/PRF*1000

plt.figure()
plt.plot(t, snr_pre, label = "无超导")
plt.plot(t, snr_this, label = "有超导")
plt.xlabel("飞行时间 (ms)", fontsize=16, fontweight='bold', fontproperties=my_font)
plt.ylabel("信噪比 (dB)", fontsize=16, fontweight='bold', fontproperties=my_font)
plt.grid()
plt.legend()
plt.savefig("../fig/super_conductor/snr_az_{}.pdf".format(blk_num))

diff = snr_this-snr_pre
diff = diff[0:40000]
diff_mean = np.mean(diff)
diff_std = np.std(diff)

plt.figure()
plt.plot(t[0:40000], diff)
plt.xlabel("飞行时间 (ms)", fontsize=16, fontweight='bold', fontproperties=my_font)
plt.ylabel("信噪比差值 (dB)", fontsize=16, fontweight='bold', fontproperties=my_font)
plt.grid()
plt.legend()
plt.savefig("../fig/super_conductor/snr_diff_az_{}.pdf".format(blk_num))

print("snr diff : {}-{}".format(diff_mean-diff_std, diff_mean+diff_std))


