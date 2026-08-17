import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
import h5py
import imageio as iio
from matplotlib import font_manager
import scipy.io as sio

my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")



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
focus_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(focus, axes=1), axis=1), axes=1)
freq = np.arange(-focus_fft.shape[1]//2, focus_fft.shape[1]//2)/focus_fft.shape[1]*Fs
plt.figure()
plt.plot(20*np.log10(np.abs(focus_fft[8700,:])/np.max(np.abs(focus_fft[8700,:]))))
plt.ylabel("Amplitude (dB)")
plt.xlabel("Range")
plt.ylim([-60, 0])
plt.grid(alpha=0.3)
plt.savefig("../fig/afscan/part_focus_super_upsample_freq.png", dpi=300)


rect = np.abs(freq)<B/2
focus_fft = focus_fft * rect[np.newaxis,:]

kaiser = np.kaiser(focus_fft.shape[1], beta=5)
focus_fft = focus_fft * kaiser[np.newaxis,:]

focus = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(focus_fft, axes=1), axis=1), axes=1)

tif_path = f"../fig/afscan/part_focus_super_upsample.tif"
image_abs = np.abs(focus)
image_norm = (image_abs / image_abs.max() * 65535).astype(np.uint16)
iio.imwrite(tif_path, image_norm)



