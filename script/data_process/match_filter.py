import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append(r"./")
from read_data import read_echo

class MatchFilterBuilderMultiFile:
    """
    Build match filter from calibration signals across multiple subbands.
    Expects a function `read_echo(file_name, blk_len, flag)` returning
    (sig, fcs, _, params) where:
      - sig: 2D array (samples x pulses)
      - params: object with attributes Fr, Br, Tr, t0
    """

    def __init__(self, calib_file_names, bands, blk_len):
        self.bands = np.array(bands)
        self.params = None

        n_bands = len(self.bands)
        calibs_list = []

        for ii in range(n_bands):
            sig, fcs, _, params = read_echo(calib_file_names[ii], blk_len, 0)
            if ii == 0:
                self.params = params
                sig_len = sig.shape[0]
            # compute mean over pulses (axis=1)
            sig_mean = np.mean(sig, axis=1)
            # normalize
            max_abs = np.max(np.abs(sig_mean))
            if max_abs != 0:
                sig_mean = sig_mean / max_abs
            # ensure length matches first
            if sig_mean.shape[0] != sig_len:
                raise ValueError("Calibration signal lengths differ between files.")
            calibs_list.append(sig_mean.astype(np.complex128))

        self.calibs = np.stack(calibs_list, axis=1)
        # global normalize
        max_abs_all = np.max(np.abs(self.calibs))
        if max_abs_all != 0:
            self.calibs = self.calibs / max_abs_all


    def get_filter(self, subband):
        """
        Get frequency-domain match filter for selected sub-band (center freq).
        Returns a 1D complex numpy array or None if not found.
        """
        idxs = np.where(self.bands == subband)[0]
        if idxs.size == 0:
            print(f"Subband {subband/1e9:.1f} GHz not found.", file=sys.stderr)
            return None
        idx = idxs[0]
        filt = self.calibs[:, idx].copy()

        # convert pulse to match filter (conjugate FFT)
        filt = np.conj(np.fft.fft(filt, axis=0))

        # compensate for calibration signal start time difference
        Nr = filt.size
        n = np.arange(Nr) - np.floor(Nr / 2)
        f_r = np.fft.ifftshift((n / Nr) * self.params["Fr"])
        filt = filt * np.exp(2j * np.pi * self.params["t0"] * f_r)

        # remove DC-like components near ends
        winlen = int(np.floor(Nr / 100.0))
        filt = np.fft.fftshift(filt)
        if winlen > 0:
            filt[:winlen] = 0
            if winlen <= Nr:
                filt[-winlen:] = 0
        filt = np.fft.fftshift(filt)

        # (optional) spectrum envelope compensation - commented out in original
        # spe_comp = self._spe_env_comp(np.abs(filt))
        # filt = filt * spe_comp

        return filt

    # ----------------- private helpers -----------------
    def _find_pulse_by_phase(self, sig):
        Fs = self.params["Fr"]
        K = self.params["Br"] / self.params["Tr"]
        phase = np.unwrap(np.angle(sig))
        freq = np.diff(phase) * (Fs / (2 * np.pi))
        freq = _smooth(freq, 15)
        freq_diff = np.diff(freq) * Fs
        freq_diff = _smooth(freq_diff, 129)
        mask = (freq_diff > (K * -1)) & (freq_diff < (K * 10))
        pb, pe = find_longest_consecutive_true(mask)
        # compensate indices as in MATLAB (+2)
        return pb + 2, pe + 2

    def _find_spe(self, spe):
        N = spe.size
        f = (np.arange(N) - np.floor(N / 2.0)) * (self.params["Fr"] / N)
        spe_diff = np.concatenate(([0.0], np.diff(spe)))
        spe_diff[np.abs(f) > (self.params["Br"] / 2.0 * 1.05)] = 0
        idx_b = int(np.argmax(spe_diff))
        idx_e = int(np.argmin(spe_diff)) - 1
        return idx_b, idx_e

    def _spe_env_comp(self, spe_mag):
        spe_mag = np.fft.fftshift(spe_mag)
        spe_mag_comp = _smooth(spe_mag, 256)
        mean_mag = np.mean(spe_mag_comp)
        spe_b, spe_e = self._find_spe(spe_mag_comp)
        # clamp indices
        spe_b = max(0, min(spe_b, spe_mag_comp.size - 1))
        spe_e = max(0, min(spe_e, spe_mag_comp.size - 1))
        # avoid division by zero
        region = spe_mag_comp[spe_b:spe_e + 1]
        safe = np.where(region == 0, 1e-12, region)
        spe_mag_comp[spe_b:spe_e + 1] = (mean_mag / safe) ** 2
        spe_mag_comp[:spe_b] = 0
        spe_mag_comp[spe_e + 1:] = 0
        spe_mag_comp = np.fft.ifftshift(spe_mag_comp)
        return spe_mag_comp


# ----------------- standalone helper functions -----------------
def find_pulse_by_magnitude(sig, threshold):
    """
    Find start (pb) and end (pe) indices of the longest pulse by magnitude.
    Mirrors MATLAB helper in original code.
    """
    abs_sig = np.abs(sig)
    if abs_sig.size == 0:
        return 0, 0
    thresh_val = max(threshold * np.max(abs_sig), 2 * np.min(abs_sig))
    mask = abs_sig > thresh_val
    changes = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0]
    if changes.size < 2:
        return 0, 0
    # pairs of (start, end+1)
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    idx = int(np.argmax(lengths))
    pb = int(starts[idx])
    pe = int(ends[idx]) - 1
    return pb, pe


def find_longest_consecutive_true(seq):
    """
    Return start and end indices (inclusive) of the longest consecutive True run.
    If none, returns (0, 0).
    """
    seq = np.asarray(seq, dtype=bool)
    if seq.size == 0 or not seq.any():
        return 0, 0
    diff = np.diff(seq.astype(np.int8))
    change_idx = np.where(diff != 0)[0]
    # If sequence starts with True, prepend -1 to change boundaries to handle leading run
    if seq[0]:
        change_idx = np.concatenate(([-1], change_idx))
    # If sequence ends with True, append last index
    if seq[-1]:
        change_idx = np.concatenate((change_idx, [seq.size - 1]))
    if change_idx.size < 2:
        return 0, 0
    intervals = change_idx[1::2] - change_idx[0::2]
    idx_max = int(np.argmax(intervals))
    idx_b = int(change_idx[2 * idx_max] + 1)
    idx_e = int(change_idx[2 * idx_max + 1])
    return idx_b, idx_e


def _smooth(x, window_len):
    """
    Simple moving-average smoothing (returns same shape).
    If window_len <= 1, returns x unchanged.
    """
    x = np.asarray(x, dtype=float)
    if window_len is None or window_len <= 1:
        return x
    w = int(window_len)
    kernel = np.ones(w) / w
    # use 'same' convolution with symmetric padding to mimic MATLAB smooth
    pad = w // 2
    xp = np.pad(x, pad, mode='edge')
    y = np.convolve(xp, kernel, mode='valid')
    # ensure same length as original
    if y.size > x.size:
        start = (y.size - x.size) // 2
        y = y[start:start + x.size]
    return y