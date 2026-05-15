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
from multiprocessing import Queue
import time

import scipy.io as sio

def read_subprocess(sig_queue, fcs_queue, frame_time_queue, params_queue, echo_file_name, Na_blk):
    offset_blocks = 0
    
    while True:
        sig_blk, fcs_blk, frame_time_blk, params_blk = read_echo(echo_file_name, Na_blk, offset_blocks*Na_blk)
        sig_queue.put(sig_blk)
        fcs_queue.put(fcs_blk)
        frame_time_queue.put(frame_time_blk)
        params_queue.put(params_blk)
        if sig_blk is None or getattr(sig_blk, "size", 0) == 0:
            break
        del sig_blk, fcs_blk, frame_time_blk, params_blk
        gc.collect()
        offset_blocks += 1

def save_mat_file(sig_blocks, fcs_blocks, frame_time_blocks, params, pos_reader, experiment_tag):

    # concatenate azimuth blocks

    if len(sig_blocks) == 0:
        raise RuntimeError("No signal blocks read")

    sig = np.hstack(sig_blocks)
    fcs = np.concatenate(fcs_blocks)
    frame_time = np.concatenate(frame_time_blocks)

    # imaging parameters
    unique_bands = np.unique(fcs)
    Nb = unique_bands.size
    Nr, Na = sig.shape
    params_dict = params if isinstance(params, dict) else {}
    params_dict['PRF'] = 6000
    params_dict['theta_bw'] = 8
    params_dict["phi"] = 60

    # platform velocity from POS if available
    forward, right, down = pos_reader.get_coords(frame_time)
    params_dict['Vr'] = (forward[-1] - forward[0]) * (params_dict['PRF'] / (Na - 1))
    params_dict["forward"] = forward
    params_dict["right"] = right
    params_dict["down"] = down
    params_dict["frame_time"] = frame_time
    # save outputs (MAT-file compatible)
    save_path_prefix = "F:/sar/data/2024_4_fs_data/"
    sig_file = f'{save_path_prefix}{experiment_tag}_sig.mat'
    param_file = f'{save_path_prefix}{experiment_tag}_param.mat'
    sig = np.ascontiguousarray(sig)

    with h5py.File(sig_file, 'w') as fh:
        fh.create_dataset('sig', data=sig, compression='gzip')
        fh.attrs['Nr'] = sig.shape[0]
        fh.attrs['Na'] = sig.shape[1]

    print(f"Saved HDF5: {sig_file}")

    sio.savemat(param_file, {'params': params_dict})
    print(f"Saved param and pos: {param_file}")


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()


    path_prefix = "G:/2026_4_FS/"

    # parameters
    bands = [35e9]
    Na_blk = 2048

    # builders / readers
    rc_builder = MatchFilterBuilderMultiFile([f'{path_prefix}example_15.dat'], bands, Na_blk)
    pos_reader = PosReader(f'{path_prefix}export_Mission 1_0418.out')

    # build range-compression filters (one per band)
    rc_filters = cp.array(np.stack([rc_builder.get_filter(b) for b in bands], axis=1))  # shape (Nr, Nb)
    print("Range-compression filters built, shape:", rc_filters.shape)
    filter_time = (cp.fft.ifft(cp.squeeze(rc_filters)))
    t = np.arange(filter_time.size) / 2.5e9+36.2e-6
    plt.figure()
    plt.plot(t[1:], cp.diff(cp.unwrap(cp.angle(filter_time))).get()/(2*np.pi)*2.5e9)
    # plt.plot(cp.abs(filter_time).get())
    plt.savefig("./rc_filter.png", dpi=300)

    print("filters built.")

    # read echo blocks, range-compress and collect
    experiment_tag = 'example_10'
    echo_file_name = f'{path_prefix}{experiment_tag}.dat'

    sig_blocks = []
    fcs_blocks = []
    frame_time_blocks = []

    offset_blocks = 0
    params = None

    print("Start reading echo data...")
    sig_queue = Queue(maxsize=2)
    fcs_queue = Queue(maxsize=2)
    frame_time_queue = Queue(maxsize=2)
    params_queue = Queue(maxsize=2)

    sig_queue = Queue()
    fcs_queue = Queue()
    frame_time_queue = Queue()
    params_queue = Queue()

    p = Process(target=read_subprocess, args=(sig_queue, fcs_queue, frame_time_queue, params_queue, echo_file_name, Na_blk))
    p.start()

    file_block_count = 20

    while True:
        
        sig_blk = sig_queue.get()
        fcs_blk = fcs_queue.get()
        frame_time_blk = frame_time_queue.get()
        params_blk = params_queue.get()
        # detect EOF / empty
        if sig_blk is None or getattr(sig_blk, "size", 0) == 0:
            break

        r_start = 0         # convert 1-based to 0-based
        r_end = 30000        # python slice end (exclusive)
        # adjust params.t0 if available as dict or object
        if isinstance(params_blk, dict):
            params_blk['t0'] = params_blk.get('t0', 0) + (r_start+r_end)/2 / params_blk.get('Fr', 1)
        else:
            if hasattr(params_blk, 't0') and hasattr(params_blk, 'Fr'):
                params_blk.t0 = params_blk.t0 + (r_start+r_end)/2 / params_blk.Fr

        # range compression in frequency domain
        # plt.plot(cp.abs(filter_time).get())
        plt.savefig("./frame.png", dpi=300)
        sig_fft = cp.fft.fft(cp.array(sig_blk), axis=0)
        for j, b in enumerate(bands):
            mask = cp.array(fcs_blk == b)
            if cp.any(mask):
                sig_fft[:, mask] *= rc_filters[:, j:j+1]
        sig_cc = cp.fft.ifft(sig_fft, axis=0)
        if offset_blocks > 18:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(t*1e6, np.abs(sig_blk[:,-1]))
            plt.xlabel("time/μs")
            plt.ylabel("amplitude")
            plt.subplot(2,1,2)
            plt.plot(t*1e6, cp.abs(sig_cc[:,-1]).get())
            plt.xlabel("time/μs")
            plt.ylabel("amplitude")
            plt.tight_layout()
            plt.savefig("./raw_signal.png", dpi=300)
            plt.show()
            exit()

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

        if offset_blocks % file_block_count == 0:
            print(f"Processed {offset_blocks} blocks, saving intermediate results...")
            save_mat_file(sig_blocks, fcs_blocks, frame_time_blocks, params, pos_reader, f'{experiment_tag}_part{offset_blocks//file_block_count}')
            # clear blocks after saving
            sig_blocks.clear()
            fcs_blocks.clear()
            frame_time_blocks.clear()
            print("Intermediate results saved, continuing processing...")

    print(f"Processed lefted blocks, saving results...")
    save_mat_file(sig_blocks, fcs_blocks, frame_time_blocks, params, pos_reader, f'{experiment_tag}_part{offset_blocks//file_block_count}')
    print("Finished reading echo data.")
