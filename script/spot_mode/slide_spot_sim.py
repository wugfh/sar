import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

cp.cuda.Device(1).use()
# 定义参数
f = 5.6e9  # 载波频率
La = 6
PRF = 2318
Tr = 4e-6
Br = 150e6
Fr = 200e6
vr = 7200
Rc = 600e3
omega = cp.deg2rad(0.2656)  # 波束旋转速度
theta_c = cp.deg2rad(12)
c = 299792458

R0 = Rc * cp.cos(theta_c)
lambda_ = c / f
Kr = Br / Tr
Nr = int(cp.ceil(1.2*Fr * Tr))

A = 1 - omega * R0 / (vr * cp.cos(theta_c)**2) # 放缩因子
Ta =  0.886*Rc*lambda_/(A*La*vr*cp.cos(theta_c))
Na = int(cp.ceil(PRF * Ta))
eta_c_strip = -R0 * cp.tan(theta_c) / vr

def equation(x):
    return -R0.get() * np.tan(theta_c.get() - omega.get()*x) / vr - x

etc_c_spot = fsolve(equation, eta_c_strip.get()) # 解出滑动聚焦的景中心时间
print(eta_c_strip, etc_c_spot)
etc_c_spot = cp.array(etc_c_spot)

tau_strip = 2 * cp.sqrt(R0**2+vr**2 * eta_c_strip**2) / c + cp.arange(-Nr/2, Nr/2) / Fr
tau_spot = 2 *cp.sqrt(R0**2+vr**2 * etc_c_spot**2) / c + cp.arange(-Nr/2, Nr/2) / Fr 
eta_strip = eta_c_strip + cp.arange(-Na/2, Na/2) / PRF
eta_spot = etc_c_spot + cp.arange(-Na/2, Na/2) / PRF

f_tau = cp.arange(-Nr/2, Nr/2) * Fr / Nr
feta_c = 2 * vr * cp.sin(theta_c) / lambda_
f_eta = feta_c + cp.arange(-Na/2, Na/2) * PRF / Na

mat_tau_strip, mat_eta_strip = cp.meshgrid(tau_strip, eta_strip)
mat_tau_spot, mat_eta_spot = cp.meshgrid(tau_spot, eta_spot)
mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

point = [0, 0]

# slide spot
R0_tar = cp.sqrt(point[0]**2 + R0**2)
R_eta_spot = cp.sqrt(R0_tar**2 + (vr * mat_eta_spot - point[1])**2)
Wr_spot = (cp.abs(mat_tau_spot - 2 * R_eta_spot / c) <= Tr / 2)
Wa_spot = cp.sinc(La * (cp.arccos(R0_tar / R_eta_spot) - (theta_c - omega * mat_eta_spot)) / lambda_)**2
Phase_spot = cp.exp(-4j * cp.pi * f * R_eta_spot / c) * cp.exp(1j * cp.pi * Kr * (mat_tau_spot - 2 * R_eta_spot / c)**2)
S_echo_spot = Wr_spot * Wa_spot * Phase_spot

# strip
R_eta_strip = cp.sqrt(R0_tar**2 + (vr * mat_eta_strip - point[1])**2)
Wr_strip = (cp.abs(mat_tau_strip - 2 * R_eta_strip / c) <= Tr / 2)
Wa_strip = cp.sinc(La * (cp.arccos(R0_tar / R_eta_strip) - theta_c) / lambda_)**2
Phase_strip = cp.exp(-4j * cp.pi * f * R_eta_strip / c) * cp.exp(1j * cp.pi * Kr * (mat_tau_strip - 2 * R_eta_strip / c)**2)
S_echo_strip = Wr_strip * Wa_strip * Phase_strip

S_tau_feta_spot = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(S_echo_spot, axes=0), Na, axis=0), axes=0)
S_tau_feta_strip = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(S_echo_strip, axes=0), Na, axis=0), axes=0)

S_ftau_feta_strip = cp.fft.fftshift(cp.fft.fft2(S_echo_strip))
S_ftau_feta_spot = cp.fft.fftshift(cp.fft.fft2(S_echo_spot))

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_tau_feta_spot).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_tau_feta_strip).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip_feta.png", dpi=300)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_echo_spot).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_echo_strip).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip.png", dpi=300)

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_ftau_feta_spot).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_ftau_feta_strip).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip_fft2.png", dpi=300)