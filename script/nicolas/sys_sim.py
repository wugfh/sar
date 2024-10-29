import cupy as cp
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import time

# 参数
cp.cuda.Device(1).use()

c = 299792458
H = 580e3
incidence = cp.deg2rad((34.9 + 41.9) / 2)
R_eta_c = H / cp.cos(incidence)
EarthRadius = 6.37e6
phi = incidence
R0 = H / cp.cos(incidence)
theta_rc = cp.arccos(R0 / R_eta_c)

Vs = 7560
Vg = Vs * EarthRadius / (EarthRadius + H)
Vr = cp.sqrt(Vs * Vg)
daz_rx = 1.6
Naz = 7
daz_tx = 2

f0 = 30e9
lambda_ = c / f0
Tr = 5e-6
Br = 262e6
Kr = Br / Tr
Fr = 1.2 * Br
Nr = int(cp.ceil(1.2 * Fr * Tr).astype(int))

B_dop = 0.886 * 2 * Vs * cp.cos(theta_rc) / daz_rx
Fa = 1350
Ta = 0.886 * R_eta_c * lambda_ / (daz_rx * Vg * cp.cos(theta_rc))
Na = int(cp.ceil(1.2 * Fa * Ta).astype(int))
fnc = 2 * Vr * cp.sin(theta_rc) / lambda_
eta_c = -R0 * cp.tan(theta_rc) / Vr

tau = 2 * R_eta_c / c + cp.arange(-Nr/2, Nr/2, 1) * (1 / Fr)
eta = eta_c + cp.arange(-Na/2, Na/2, 1) * (1 / Fa)

f_tau = cp.fft.fftshift(cp.arange(-Nr/2, Nr/2, 1) * (Fr / Nr))
f_eta = fnc + cp.fft.fftshift(cp.arange(-Na/2, Na/2, 1) * (Fa / Na))

mat_tau, mat_eta = cp.meshgrid(tau, eta)
mat_f_tau, mat_f_eta = cp.meshgrid(f_tau, f_eta)

# 生成
point = cp.array([0, 0])
S_echo = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
for i in range(Naz):
    ant_dx = (i) * daz_rx

    R_point = cp.sqrt((R0 * cp.sin(phi) + point[0])**2 + H**2)
    point_eta_c = (point[1] - R_point * cp.tan(theta_rc)) / Vr
    R_eta_tx = cp.sqrt(R_point**2 + (Vr * mat_eta - point[1])**2)

    mat_eta_rx = mat_eta + ant_dx / Vs
    R_eta_rx = cp.sqrt(R_point**2 + (Vr * mat_eta_rx - point[1])**2)

    Wr = (cp.abs(mat_tau - (R_eta_tx+R_eta_rx) / c) < Tr / 2)
    Wa = (daz_rx * cp.arctan(Vg * (mat_eta - point_eta_c) / (R0 * cp.sin(phi) + point[0]) / lambda_)**2) <= Ta / 2

    echo_phase_azimuth = cp.exp(-1j * 2 * cp.pi * f0 * (R_eta_rx + R_eta_tx) / c)
    echo_phase_range = cp.exp(1j * cp.pi * Kr * (mat_tau - (R_eta_tx + R_eta_tx) / c)**2)
    S_echo[i, :, :] = Wr * Wa * echo_phase_range * echo_phase_azimuth

# 成像处理
P = cp.zeros((Na, Naz, Naz), dtype=cp.complex128)
H = cp.zeros((Na, Naz, Naz), dtype=cp.complex128)
prf = Fa

# 计算方位向响应
for k in range(Naz):
    for n in range(Naz):
        H[:, k, n] = cp.exp(-1j * cp.pi * (n * daz_rx) / Vs * (f_eta + k * prf))

for j in range(Na): 
    P[j, :, :] = cp.linalg.inv(H[j])
# P = cp.linalg.inv_batch(H)


S_r_compress = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
for i in range(Naz):
    echo = cp.squeeze(S_echo[i, :, :])
    Hr = (cp.abs(mat_f_tau) <= Br / 2) * cp.exp(1j * cp.pi * mat_f_tau**2 / Kr)
    S1_ftau_eta = cp.fft.fft(echo, Nr, axis=1)
    S1_ftau_eta = S1_ftau_eta * Hr
  
    S1_tau_eta = cp.fft.ifft(S1_ftau_eta, Nr, axis=1)
    S2_tau_feta = cp.fft.fft(S1_tau_eta, Na, axis=0)
    S_r_compress[i, :, :] = S2_tau_feta


# 重构
uprate = Naz
f_eta_upsample = fnc + cp.fft.fftshift(cp.arange(-Na * uprate / 2, Na * uprate / 2, 1) * (Fa / Na))
t_eta_upsample = eta_c + cp.arange(-Na * uprate / 2, Na * uprate / 2, 1) * (1 / (Fa * uprate))

mat_tau_upsample, mat_eta_upsample = cp.meshgrid(tau, t_eta_upsample)
mat_f_tau_upsample, mat_f_eta_upsample = cp.meshgrid(f_tau, f_eta_upsample)


out_band = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
S_out = cp.zeros((Na*uprate, Nr), dtype=cp.complex128)

# 将 S_r_compress 和 P 进行适当的重塑
S_r_compress_reshaped = S_r_compress.transpose(1, 0, 2)  # 形状变为 (Na, M, N)
P_reshaped = P.transpose(0, 2, 1).conj()  # 形状变为 (Na, N, N)

# 批量矩阵乘法
tmp = cp.einsum('ijk,ikl->ijl', P_reshaped, S_r_compress_reshaped)

# 将结果存储到 out_band
out_band = tmp.transpose(1, 0, 2)  # 形状变为 (Naz, Na, N)

# for i in range(Na):
#     aperture = cp.squeeze(S_r_compress[:, i, :]) 
#     P_aperture = cp.squeeze(P[i, :, :]).conj().T    
#     tmp = P_aperture @ aperture
#     for j in range(Naz):
#         out_band[j, i, :] = tmp[j, :]


for j in range(Naz):
    S_out[j * Na: (j + 1) * Na, :] = cp.fft.fftshift(cp.squeeze(out_band[j, :, :]), axes=0) # 确保连续性

S_ref = cp.squeeze(S_r_compress[0, :, :])


plt.figure("孔径合成后的数据")
plt.subplot(1,2,1)
plt.imshow(cp.abs(S_out).get(), aspect="auto")
plt.title("after reconstruction")
plt.subplot(1,2,2)
plt.imshow(cp.abs(S_ref).get(), aspect="auto")
plt.title("no reconstruction")

plt.savefig("../../fig/nicolas/freq_image.png", dpi=300)

# 二次距离压缩

# 方位压缩
# RCMC
delta_R = lambda_**2 * R_point * mat_f_eta_upsample**2 / (8 * Vr**2)
G_rcmc = cp.exp(1j * 4 * cp.pi * mat_f_tau_upsample * delta_R / c)
Sout_ftau_feta = cp.fft.fft(S_out, Nr, axis=1)
Sout_ftau_feta = Sout_ftau_feta * G_rcmc
Sout_tau_feta_rcmc = cp.fft.ifft(Sout_ftau_feta, Nr, axis=1)

# 方位压缩
mat_R_upsample = mat_tau_upsample * c * cp.cos(theta_rc) / 2
Ka_upsample = 2 * Vr**2 * cp.cos(theta_rc)**2 / (lambda_ * mat_R_upsample)
Ha_upsample = cp.exp(-1j * cp.pi * mat_f_eta_upsample**2 / Ka_upsample)
Offset_upsample = cp.exp(-1j * 2 * cp.pi * mat_f_eta_upsample * eta_c)
out = Sout_tau_feta_rcmc * Ha_upsample * Offset_upsample
out = cp.fft.ifft(out, Na * uprate, axis=0)

# 参考方位压缩
delta_R = lambda_**2 * R_point * mat_f_eta**2 / (8 * Vr**2)
G_rcmc = cp.exp(1j * 4 * cp.pi * mat_f_tau * delta_R / c)
S3_ftau_feta = cp.fft.fft(S_ref, Nr, axis=1)
S3_ftau_feta = S3_ftau_feta * G_rcmc
S3_tau_feta_rcmc = cp.fft.ifft(S3_ftau_feta, Nr, axis=1)

mat_R = mat_tau * c * cp.cos(theta_rc) / 2
Ka = 2 * Vr**2 * cp.cos(theta_rc)**2 / (lambda_ * mat_R)
Ha = cp.exp(-1j * cp.pi * mat_f_eta**2 / Ka)
Offset = cp.exp(-1j * 2 * cp.pi * mat_f_eta * eta_c)

out_ref = S3_tau_feta_rcmc * Ha * Offset
out_ref = cp.fft.fft(out_ref, Na, axis=0)

plt.figure("RCMC后的数据")
plt.subplot(1,2,1)
plt.imshow(cp.abs(Sout_tau_feta_rcmc).get(), aspect="auto")
plt.title("after reconstruction")
plt.subplot(1,2,2)
plt.imshow(cp.abs(S3_tau_feta_rcmc).get(), aspect="auto")
plt.title("no reconstruction")
plt.savefig("../../fig/nicolas/rcmc_image.png", dpi=300)


# 显示
plt.figure("成像结果")
plt.imshow(cp.abs(out).get(), aspect="auto")
plt.title("image after reconstruction")
plt.savefig("../../fig/nicolas/out_image.png", dpi=300)


r_f_pos = cp.argmax(cp.max(cp.abs(S_out), axis=0))
r_pos = cp.argmax(cp.max(cp.abs(out), axis=0))

out_f_eta = Sout_tau_feta_rcmc[:, r_f_pos]
out_eta = out[:, r_pos]

plt.figure("重构后的切片")
plt.subplot(1, 2, 1)
plt.plot(cp.fft.fftshift(f_eta_upsample).get(), cp.abs(cp.fft.fftshift(out_f_eta)).get())
plt.title("slice in frequency")

plt.subplot(1, 2, 2)
plt.plot(t_eta_upsample.get(), cp.abs(out_eta).get())
plt.title("slice in imaging")
plt.savefig("../../fig/nicolas/out_slice.png", dpi=300)

r_f_pos_ref = cp.argmax(cp.max(cp.abs(S_ref), axis=0))
r_pos_ref = cp.argmax(cp.max(cp.abs(out_ref), axis=0))

ref_f_eta = S_ref[:, r_f_pos_ref]
ref_eta = out_ref[:, r_pos_ref]

plt.figure("各子带频谱")

f_eta_upsample = cp.fft.fftshift(f_eta_upsample)
for i in range(Naz):
    plt.plot(cp.asnumpy((f_eta_upsample[i*Na:(i+1)*Na])), cp.fft.fftshift(cp.abs(out_band[(i+(Naz+1)/2)%Naz, :, r_pos])).get(), label=f"suband {i+1}")

plt.legend()
plt.savefig("../../fig/nicolas/band_image.png", dpi=300)
