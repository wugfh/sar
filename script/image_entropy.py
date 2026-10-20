import numpy as np
from PIL import Image
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import cupy as cp
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

def compute_image_entropy(image_path: str, bins: int = 256) -> float:
    """
    Calculate the Shannon entropy of an image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    bins : int
        Number of histogram bins (default 256 for 8-bit images).

    Returns
    -------
    float
        Shannon entropy value in bits.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pixels = np.asarray(img, dtype=np.uint16)
  
    # Compute histogram and normalize to probabilities
    hist, _ = np.histogram(pixels, bins=bins, range=(0, bins))
    probs = hist / hist.sum()

    # Remove zero entries to avoid log2(0)
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def compute_rgb_entropy(image_path: str, bins: int = 256) -> dict:
    """
    Calculate Shannon entropy per channel for an RGB image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Entropy values for each channel ('R', 'G', 'B') and the average.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    entropies = {}
    for i, ch in enumerate(["R", "G", "B"]):
        channel = arr[:, :, i]
        hist, _ = np.histogram(channel, bins=bins, range=(0, bins))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropies[ch] = -np.sum(probs * np.log2(probs))

    entropies["average"] = np.mean(list(entropies.values()))
    return entropies

def compute_enl(image_path, area: tuple = (400, 500, 1850, 2000)):
    img = sio.loadmat(image_path)['focus']
    img = np.abs(img)
    img = img[area[0]:area[1], area[2]:area[3]]
    u = img.mean()
    sigma = img.var(ddof=1)
    enl = (u**2) / (sigma)
    return enl

def compute_psnr(x, x_true):
    x_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x)))
    x_true_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x_true)))
    x_ifft = x_ifft/cp.sqrt(cp.sum(cp.abs(x_ifft)**2))
    x_true_ifft = x_true_ifft/cp.sqrt(cp.sum(cp.abs(x_true_ifft)**2))
    mse = cp.mean(cp.abs(x_ifft - x_true_ifft)**2)
    psnr = 20 * cp.log10(cp.max(cp.abs(x_ifft)) / cp.sqrt(mse))
    return psnr

def ssim_1d(x, y, win=11, sigma=1.5, L=None, K1=0.01, K2=0.03,
            window="gaussian", trim_edges=True):
    """一维 SSIM。返回 (MSSIM, 逐点 SSIM 曲线)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape
    if L is None:                      # 动态范围：建议显式传入
        L = max(x.max() - x.min(), y.max() - y.min())
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    if window == "gaussian":
        trunc = ((win - 1) / 2) / sigma          # 支撑恰好 ≈ win
        f = lambda a: gaussian_filter1d(a, sigma, truncate=trunc, mode="reflect")
    else:
        f = lambda a: uniform_filter1d(a, win, mode="reflect")
    mx, my   = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    sxx, syy = np.maximum(sxx, 0), np.maximum(syy, 0)   # 数值误差
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx * mx + my * my + C1) * (sxx + syy + C2)
    smap = num / den
    if trim_edges:
        h = win // 2
        smap = smap[h:len(smap) - h] if len(smap) > 2 * h else smap
    return float(smap.mean()), smap


if __name__ == "__main__":
    path = "F:/sar/doc/Manuscript/first demo of fsar"
    fscan_file_prefix = "fscan_detail"
    single_file_prefix = "single_detail"
    for i in range(4):
        if i == 0:
            suffix = "A"
        elif i == 1:
            suffix = "B"
        elif i == 2:
            suffix = "C"
        else:
            suffix = "D"
        fscan_file = Path(path) / f"{fscan_file_prefix}{suffix}.png"
        single_file = Path(path) / f"{single_file_prefix}{suffix}.png"

        fscan_entropy = compute_image_entropy(fscan_file)
        single_entropy = compute_image_entropy(single_file)

        print(f"File: {fscan_file.name}, Entropy: {fscan_entropy}")
        print(f"File: {single_file.name}, Entropy: {single_entropy}")