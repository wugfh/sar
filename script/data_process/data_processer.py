import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation
from tqdm import tqdm
from read_data import read_echo
from match_filter import MatchFilterBuilderMultiFile
from pos_reader import PosReader
import numpy as np

import scipy.io as sio

# parameters
bands = [35e9]
Na_blk = 4096

# builders / readers
rc_builder = MatchFilterBuilderMultiFile(['./example_8.dat'], bands, Na_blk)
pos_reader = PosReader('./sbet_Mission 2.out')

# build range-compression filters (one per band)
rc_filters = np.stack([rc_builder.getFilter(b) for b in bands], axis=1)  # shape (Nr, Nb)

# read echo blocks, range-compress and collect
experiment_tag = 'example_13'
echo_file_name = f'./{experiment_tag}.dat'

sig_blocks = []
fcs_blocks = []
frame_time_blocks = []

offset_blocks = 0
params = None

while True:
    sig_blk, fcs_blk, frame_time_blk, params_blk = read_echo(echo_file_name, Na_blk, offset_blocks)
    # detect EOF / empty
    if sig_blk is None or getattr(sig_blk, "size", 0) == 0:
        break

    # initialize params and select range slice on first block
    if params is None:
        params = params_blk
        r_start = 1600 - 1  # convert 1-based to 0-based
        r_end = 2200        # python slice end (exclusive)
        # adjust params.t0 if available as dict or object
        if isinstance(params, dict):
            params['t0'] = params.get('t0', 0) + (r_start) / params.get('Fr', 1.0)
        else:
            if hasattr(params, 't0') and hasattr(params, 'Fr'):
                params.t0 = params.t0 + (r_start) / params.Fr

    # range compression in frequency domain
    sig_fft = np.fft.fft(sig_blk, axis=0)
    for j, b in enumerate(bands):
        mask = (fcs_blk == b)
        if np.any(mask):
            # rc_filters[:, j] has shape (Nr,)
            sig_fft[:, mask] *= rc_filters[:, j:j+1]
    sig_cc = np.fft.ifft(sig_fft, axis=0)

    # store cropped-in-range block
    sig_blocks.append(sig_cc[r_start:r_end, :])
    fcs_blocks.append(fcs_blk)
    frame_time_blocks.append(frame_time_blk)

    offset_blocks += 1

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
start_idx = slcb - 1           # convert to 0-based inclusive start
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
try:
    forward, right, down = pos_reader.getCoords(frame_time)
    params_dict['Vr'] = (forward[-1] - forward[0]) * (params_dict['PRF'] / (Na - 1))
except Exception:
    forward = right = down = None
    params_dict['Vr'] = 260.0 / 3.6

# save outputs (MAT-file compatible)
sig_file = f'./{experiment_tag}_sig.mat'
pos_file = f'./{experiment_tag}_pos.mat'
param_file = f'./{experiment_tag}_param.mat'

sio.savemat(sig_file, {'sig': sig})
sio.savemat(pos_file, {'frame_time': frame_time, 'forward': forward, 'right': right, 'down': down})
sio.savemat(param_file, {'params': params_dict})

# stop here (as in original workflow)
raise SystemExit("stop")