import numpy as np
from PIL import Image
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import cupy as cp
import scipy.io as sio


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