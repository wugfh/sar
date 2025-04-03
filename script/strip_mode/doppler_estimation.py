import numpy as np
import cupy as cp
import time

def centroid_madsen2(echo, prf):
    """
    Calculate Doppler Centroid using Madsen Algorithm.
    
    Parameters:
        echo (cupy.ndarray): Input echo data.
        prf (float): Pulse repetition frequency.
    
    Returns:
        float: Calculated Doppler frequency.
    """
    num = 5
    ers = echo
    winsize = 1  # Window size
    num_lines1 = round(echo.shape[0] / num)  # Number of lines in a buffer
    p = 0
    fdc = []

    for m in range(1, num + 1):
        u = p * num_lines1
        data = ers[u:m * num_lines1, :]
        dnum = winsize * num_lines1
        fnum = int(3000 / winsize)
        sum_lines = cp.zeros(120)
        s = 0
        arg = []

        for i in range(1, fnum + 1):
            k = s * winsize
            sum_lines = cp.sum(
                cp.sign(data[1:, k:(i * winsize)]) *
                cp.conj(cp.sign(data[:-1, k:(i * winsize)])),
                axis=0
            )
            sum_lines1 = sum_lines / dnum
            sum_lines1 = cp.mean(sum_lines1)  # Correlator
            sum_lines2 = cp.sin(cp.pi * sum_lines1 / 2)  # Correlator coefficients
            x1 = cp.angle(sum_lines2)  # Argument value (angle)
            x = x1 * prf / (2 * cp.pi)  # Doppler centroid values along range direction
            arg.append(x1)
            fdc.append(x)
            s += 1
        p += 1

    fdc_avg = cp.mean(cp.array(fdc), axis=0)  # Average Doppler of each patch
    mean1 = cp.sum(fdc_avg)
    doppler = mean1 / fnum  # Doppler calculated
    return doppler

def doppler_center_estimation_ape(z, v, PRF, wavelength, Rs, DeltaR):
    """
    Estimate Doppler Center using the provided parameters.

    Parameters:
        z (cupy.ndarray): Input data.
        v (float): Velocity.
        PRF (float): Pulse repetition frequency.
        wavelength (float): Wavelength.
        Rs (float): Reference slant range.
        DeltaR (float): Range resolution.

    Returns:
        float: Estimated Doppler center.
    """
    x_zlt = cp.abs(cp.fft.fftshift(cp.fft.fft(z, axis=1)))
    x_zlt = cp.mean(x_zlt, axis=0)

    nrn, nan = z.shape
    rs = cp.arange(-nrn / 2 + 1, nrn / 2 + 1) * DeltaR
    ka = -2 * v**2 / wavelength / (Rs + rs)

    z_slt = cp.zeros_like(z, dtype=complex)
    for m in range(nrn):
        sref = cp.exp(-1j * cp.pi * ka[m] * (cp.arange(-nan / 2, nan / 2) / PRF)**2)
        z_slt[m, :] = z[m, :] * sref

    step = 2
    fy1 = z_slt[:, :nan // 2]
    fy2 = z_slt[:, nan // 2:]
    fy1 = cp.fft.fftshift(cp.fft.fft(fy1, n=nan // 2 * step, axis=1))
    fy2 = cp.fft.fftshift(cp.fft.fft(fy2, n=nan // 2 * step, axis=1))
    yf1 = cp.mean(cp.abs(fy1), axis=0)
    yf2 = cp.mean(cp.abs(fy2), axis=0)
    max_point1 = cp.argmin(yf1)
    max_point2 = cp.argmin(yf2)
    mm = (max_point1 + max_point2 + nan) / 2
    fd_center = (mm - nan / 4 * step) / (nan / 2 * step) * PRF

    return fd_center

