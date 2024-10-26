import cupy as cp
import matplotlib.pyplot as plt

# 参数
c = 299792458
H = 700e3
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

f0 = 5.4e10
lambda_ = c / f0
Tr = 5e-6
Br = 262e6
Kr = Br / Tr
Fr = 1.2 * Br
Nr = int(cp.ceil(1.2 * Fr * Tr).astype(int))

B_dop = 0.886 * 2 * Vr * cp.cos(theta_rc) / daz_rx
Fa = 1350
Ta = 0.886 * R_eta_c * lambda_ / (daz_rx * Vg * cp.cos(theta_rc))
Na = int(cp.ceil(1.2 * Fa * Ta).astype(int))
fnc = 2 * Vr * cp.sin(theta_rc) / lambda_
eta_c = -R0 * cp.tan(theta_rc) / Vr

tau = 2 * R_eta_c / c + cp.linspace(-Nr/2, Nr/2, Nr) * (1 / Fr)
eta = eta_c + cp.linspace(-Na/2, Na/2, Na) * (1 / Fa)

f_tau = cp.fft.fftshift(cp.linspace(-Nr/2, Nr/2, Nr) * (Fr / Nr))
f_eta = fnc + cp.fft.fftshift(cp.linspace(-Na/2, Na/2, Na) * (Fa / Na))

mat_tau, mat_eta = cp.meshgrid(tau, eta)
mat_f_tau, mat_f_eta = cp.meshgrid(f_tau, f_eta)

# 生成
point = cp.array([0, 0])
S_echo = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
for i in range(Naz):
    ant_dx = (i - 1) * daz_rx

    R_point = cp.sqrt((R0 * cp.sin(phi) + point[0])**2 + H**2)
    point_eta_c = (point[1] - R_point * cp.tan(theta_rc)) / Vr
    R_eta_tx = cp.sqrt(R_point**2 + (Vr * mat_eta - point[1])**2)

    mat_eta_rx = mat_eta + ant_dx / Vs
    R_eta_rx = cp.sqrt(R_point**2 + (Vr * mat_eta_rx - point[1])**2)

    Wr = (cp.abs(mat_tau - 2 * R_eta_c / c) < Tr / 2)
    Wa = (cp.abs(mat_eta - eta_c) < Ta / 2)

    echo_phase_azimuth = cp.exp(-1j * 2 * cp.pi * f0 * (R_eta_rx + R_eta_tx) / c)
    echo_phase_range = cp.exp(1j * cp.pi * Kr * (mat_tau - (R_eta_tx + R_eta_tx) / c)**2)
    S_echo[i, :, :] = Wr * Wa * echo_phase_range * echo_phase_azimuth

# 成像处理
P = cp.zeros((Naz, Naz, Na), dtype=cp.complex128)
H_matrix = cp.zeros((Naz, Naz, Na), dtype=cp.complex128)

prf = Fa
for k in range(Naz):
    for n in range(Naz):
        H_matrix[k, n, :] = cp.exp(-1j * cp.pi * (Vg / Vs) * ((n - 1) * daz_rx)**2 / (2 * lambda_ * R_point) - 1j * cp.pi * ((n - 1) * daz_rx) / Vs * (f_eta + (k - 1) * prf))
for j in range(Na):
    tmp = cp.linalg.inv(H_matrix[:,:,j])
    P[:, :, j] = tmp

S_tau_feta_rcmc = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
S_r_compress = cp.zeros((Naz, Na, Nr), dtype=cp.complex128)
for i in range(Naz):
    echo = S_echo[i, :, :]
    Hr = (cp.abs(mat_f_tau) <= Br / 2) * cp.exp(1j * cp.pi * mat_f_tau**2 / Kr)
    S1_ftau_eta = cp.fft.fft(echo, Nr, axis=1)
    S1_ftau_eta = S1_ftau_eta * Hr
    S_r_compress[i, :, :] = S1_ftau_eta

    S1_tau_eta = cp.fft.ifft(S1_ftau_eta, Nr, axis=1)
    S2_tau_feta = cp.fft.fft(S1_tau_eta, Na, axis=0)

    delta_R = lambda_**2 * R_point * mat_f_eta**2 / (8 * Vr**2)
    G_rcmc = cp.exp(1j * 4 * cp.pi * mat_f_tau * delta_R / c)
    S3_ftau_feta = cp.fft.fft(S2_tau_feta, Nr, axis=1)
    S3_ftau_feta = S3_ftau_feta * G_rcmc
    S3_tau_feta_rcmc = cp.fft.ifft(S3_ftau_feta, Nr, axis=1)
    S_tau_feta_rcmc[i, :, :] = S3_tau_feta_rcmc

# 重构
uprate = Naz
f_eta_upsample = fnc + cp.fft.fftshift(cp.linspace(-Na * uprate / 2, Na * uprate / 2, Na * uprate) * (Fa / Na))
t_eta_upsample = eta_c + cp.linspace(-Na * uprate / 2, Na * uprate / 2, Na * uprate) * (1 / (Fa * uprate))

mat_tau_upsample, mat_eta_upsample = cp.meshgrid(tau, t_eta_upsample)
mat_f_tau_upsample, mat_f_eta_upsample = cp.meshgrid(f_tau, f_eta_upsample)

S_out = cp.zeros((Na * uprate, Nr), dtype=cp.complex128)
for i in range(Na):
    aperture = S_tau_feta_rcmc[:, i, :]
    P_aperture = P[:, :, i].T
    tmp = P_aperture @ aperture
    for j in range(Naz):
        S_out[(j - 1) * Na + i, :] = tmp[j, :]

S_ref = S_tau_feta_rcmc[0, :, :]

# 绘图
plt.figure("rcmc后对比")
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_out).get())
plt.title("after reconstruction")

plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_ref).get())
plt.title("no reconstruction")
plt.savefig("../../fig/nicolas/freq_image.png")

# 方位压缩
mat_R_upsample = mat_tau_upsample * c * cp.cos(theta_rc) / 2
Ka_upsample = 2 * Vr**2 * cp.cos(theta_rc)**2 / (lambda_ * mat_R_upsample)
Ha_upsample = cp.exp(-1j * cp.pi * mat_f_eta_upsample**2 / Ka_upsample)
Offset_upsample = cp.exp(-1j * 2 * cp.pi * mat_f_eta_upsample * eta_c)
out = S_out * Ha_upsample * Offset_upsample
out = cp.fft.fft(out, Na * uprate, axis=0)

# 参考方位压缩
mat_R = mat_tau * c * cp.cos(theta_rc) / 2
Ka = 2 * Vr**2 * cp.cos(theta_rc)**2 / (lambda_ * mat_R)
Ha = cp.exp(-1j * cp.pi * mat_f_eta**2 / Ka)
Offset = cp.exp(-1j * 2 * cp.pi * mat_f_eta * eta_c)

out_ref = S_ref * Ha * Offset
out_ref = cp.fft.fft(out_ref, Na, axis=0)

# 显示
plt.figure("成像结果")
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(out).get())
plt.title("image after reconstruction")

plt.subplot(1, 2, 2)
plt.imshow(cp.abs(out_ref).get())
plt.title("image no reconstruction")
plt.savefig("../../fig/nicolas/out_image.png")


r_f_pos = cp.argmax(cp.max(cp.abs(S_out), axis=0))
r_pos = cp.argmax(cp.max(cp.abs(out), axis=0))

out_f_eta = S_out[:, r_f_pos]
out_eta = out[:, r_pos]

plt.figure("重构后的切片")
plt.subplot(1, 2, 1)
plt.plot(f_eta_upsample.get(), cp.abs(out_f_eta).get())
plt.title("slice in frequency")

plt.subplot(1, 2, 2)
plt.plot(t_eta_upsample.get(), cp.abs(out_eta).get())
plt.title("slice in imaging")
plt.savefig("../../fig/nicolas/out_slice.png")

r_f_pos_ref = cp.argmax(cp.max(cp.abs(S_ref), axis=0))
r_pos_ref = cp.argmax(cp.max(cp.abs(out_ref), axis=0))

ref_f_eta = S_ref[:, r_f_pos_ref]
ref_eta = out_ref[:, r_pos_ref]


plt.figure("未重构的切片")
plt.subplot(1, 2, 1)
plt.plot(f_eta.get(), cp.abs(ref_f_eta).get())
plt.title("slice in frequency")

plt.subplot(1, 2, 2)
plt.plot(eta.get(), cp.abs(ref_eta).get())
plt.title("slice in imaging")
plt.savefig("../../fig/nicolas/ref_slice.png")