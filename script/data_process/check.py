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

prefix_path = "F:/sar/data/2024_4_fs_data/"
exper_tar = 'example_5_part3'

data_filename = f'{prefix_path}{exper_tar}_sig.mat'
param_filename = f'{prefix_path}{exper_tar}_param.mat'

param = sio.loadmat(param_filename)
grp = param['params']
Fr = float(np.squeeze(grp['Fr']))
t0 = float(np.squeeze(grp['t0']))
Br = float(np.squeeze(grp['Br']))
f0 = float(np.squeeze(grp['f0']))
PRF = float(np.squeeze(grp['PRF']))
Tp = float(np.squeeze(grp['Tr']))

print("Fr =", Fr)
print("t0 =", t0)
print("Br =", Br)
print("f0 =", f0)
print("PRF =", PRF)
print("Tp =", Tp)


forward = np.squeeze(np.array(grp['forward'][0, 0]))
right = np.squeeze(np.array(grp['right'][0, 0]))
down = np.squeeze(np.array(grp['down'][0, 0]))
frame_time = np.squeeze(np.array(grp['frame_time'][0, 0]))


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot(forward, right, down, label='3D curve', color='blue')
plt.savefig('3d_curve.png')

v = np.mean(np.diff(forward)/np.diff(time2sec(frame_time)))
print("平台速度v =", v, "m/s")


data = h5py.File(data_filename, "r")
sig = data['sig']
sig = np.array(sig).T
# sig = sig["real"] + 1j*sig["imag"]



plt.figure(figsize=(10, 8))
plt.imshow(20*np.log10(np.abs(sig)), aspect='auto')
plt.savefig('sig_amplitude.png')


