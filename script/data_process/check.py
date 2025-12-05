import sys

sys.path.append(r"./")
from pos_reader import time2sec
import scipy.io as sio
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')

data_filename = 'F:/sar/data/example_14_sig.mat'
pos_filename = 'F:/sar/data/example_14_pos.mat'
param_filename = 'F:/sar/data/example_14_param.mat'

data = h5py.File(data_filename, "r")
sig = data['sig']
# sig = sig["real"] + 1j*sig["imag"]

pos = h5py.File(pos_filename, "r")
forward = np.squeeze(pos['forward'])
right = np.squeeze(pos['right'])
down = np.squeeze(pos['down'])
frame_time = np.squeeze(pos['frame_time'])

plt.plot(frame_time)
plt.show()




plt.figure(figsize=(10, 8))
plt.imshow(np.abs(sig), aspect='auto')
plt.savefig('sig_amplitude.png')

plt.figure(figsize=(10, 8))
sig_fft2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sig)))
plt.imshow(np.abs(sig_fft2), aspect='auto')
plt.savefig('sig_fft2.png')


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot(forward, right, down, label='3D curve', color='blue')
plt.savefig('3d_curve.png')

v = np.mean(np.diff(forward)/np.diff(time2sec(frame_time)))
print("平台速度v =", v, "m/s")