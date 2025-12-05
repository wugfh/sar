import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append(r"./")
from read_data import read_echo
from tqdm import tqdm
from match_filter import MatchFilterBuilderMultiFile
from pos_reader import PosReader
import numpy as np
import h5py
import gc
from multiprocessing import Process
import time

import scipy.io as sio


# 记录开始时间
start_time = time.time()


path_prefix = "E:/data/"

# parameters
bands = [35e9]
Na_blk = 4096

# builders / readers
rc_builder = MatchFilterBuilderMultiFile([f'{path_prefix}example_8.dat'], bands, Na_blk)
pos_reader = PosReader(f'{path_prefix}sbet_Mission 2.out')

# build range-compression filters (one per band)
rc_filters = cp.array(np.stack([rc_builder.get_filter(b) for b in bands], axis=1))  # shape (Nr, Nb)

print("filters built.")

# read echo blocks, range-compress and collect
experiment_tag = 'example_13'
echo_file_name = f'{path_prefix}{experiment_tag}.dat'

sig_blocks = []
fcs_blocks = []
frame_time_blocks = []

offset_blocks = 0
params = None

print("Start reading echo data...")

while True:
    sig_blk, fcs_blk, frame_time_blk, params_blk = read_echo(echo_file_name, Na_blk, offset_blocks*Na_blk)
    # detect EOF / empty
    if sig_blk is None or getattr(sig_blk, "size", 0) == 0:
        break

    r_start = 1600 - 1  # convert 1-based to 0-based
    r_end = 2200        # python slice end (exclusive)
    # adjust params.t0 if available as dict or object
    if isinstance(params_blk, dict):
        params_blk['t0'] = params_blk.get('t0', 0) + (r_start) / params_blk.get('Fr', 1.0)
    else:
        if hasattr(params_blk, 't0') and hasattr(params_blk, 'Fr'):
            params_blk.t0 = params_blk.t0 + (r_start) / params_blk.Fr

    # range compression in frequency domain
    sig_fft = cp.fft.fft(cp.array(sig_blk), axis=0)
    for j, b in enumerate(bands):
        mask = cp.array(fcs_blk == b)
        if cp.any(mask):
            sig_fft[:, mask] *= rc_filters[:, j:j+1]
    sig_cc = cp.fft.ifft(sig_fft, axis=0)

    print(f"read block  {offset_blocks}, shape after RC: {sig_cc.shape}")

    sig_cc = sig_cc[r_start:r_end, :].get()  # back to CPU
    sig_blocks.append(sig_cc)
    fcs_blocks.append(fcs_blk)
    frame_time_blocks.append(frame_time_blk)
    offset_blocks += 1
    if params is None:
        params = params_blk
    # 释放所有中间变量
    del sig_blk, fcs_blk, frame_time_blk, params_blk, sig_fft, sig_cc, mask
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

    
    

print("Finished reading echo data.")

# concatenate azimuth blocks

if len(sig_blocks) == 0:
    raise RuntimeError("No signal blocks read")

sig = np.hstack(sig_blocks)
fcs = np.concatenate(fcs_blocks)
frame_time = np.concatenate(frame_time_blocks)

# cropping in azimuth
shift = int(1e4)
slcb = int(3e4 + shift)        # 40000
slce = int(7e4 + shift)        # 80000
slce = min(slce, sig.shape[1])  # ensure not exceed
start_idx = slcb            # convert to 0-based inclusive start
end_idx = slce                 # python slice end exclusive

sig = sig[:, start_idx:end_idx]
fcs = fcs[start_idx:end_idx]
frame_time = frame_time[start_idx:end_idx]

# imaging parameters
unique_bands = np.unique(fcs)
Nb = unique_bands.size
Nr, Na = sig.shape
params_dict = params if isinstance(params, dict) else {}
params_dict['PRF'] = 6000

# platform velocity from POS if available
forward, right, down = pos_reader.get_coords(frame_time)
params_dict['Vr'] = (forward[-1] - forward[0]) * (params_dict['PRF'] / (Na - 1))

# save outputs (MAT-file compatible)
save_path_prefix = 'F:/sar/data/'
sig_file = f'{save_path_prefix}{experiment_tag}_sig.mat'
pos_file = f'{save_path_prefix}{experiment_tag}_pos.mat'
param_file = f'{save_path_prefix}{experiment_tag}_param.mat'


with h5py.File(sig_file, 'w') as fh:
    fh.create_dataset('sig', data=sig, compression='gzip')
    fh.attrs['Nr'] = sig.shape[0]
    fh.attrs['Na'] = sig.shape[1]

with h5py.File(pos_file, 'w') as fh:
    fh.create_dataset('frame_time', data=frame_time, compression='gzip')
    fh.create_dataset('forward', data=forward, compression='gzip')
    fh.create_dataset('right', data=right, compression='gzip')
    fh.create_dataset('down', data=down, compression='gzip')

print(f"Saved HDF5: {sig_file}, {pos_file}")

sio.savemat(param_file, {'params': params_dict})

end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print("程序运行时间：", run_time, "秒")