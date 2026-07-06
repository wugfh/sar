"""
N−M Spectrum Recovery Using the fscan.py Antenna Pattern
=========================================================
Antenna model (from fscan.py echogen):
  pr(f) = sin(15·u₁(f)/2) / (15·sin(u₁(f)/2))
  u₁(f) = (2π/λ)·d·sin(θ_doa) − (2π/λ_g)·(d − λ_g/2)
  λ = c/f,   λ_g = λ / √(1 − (λ/(2a))²)

This is a 15‑element ULA factor → order‑14 trigonometric polynomial
→ exactly M=15 non‑zero components in the generalised delay domain.
The two‑way pattern pr² consumes 2M−1 ≈ 29 DoF, but null positions
are inherited from pr → we exclude the M=15 deepest nulls.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import hankel, svd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from scipy.linalg import solve, svd, norm
warnings.filterwarnings('ignore')

# =========================================================================
# 1.  System parameters  (identical to fscan.py __init__)
# =========================================================================
c0 = 3e8

# --- Waveguide / array ---
N_elem   = 15                     # sin(15·u₁/2) → 15 elements
a_wg     = 0.004871409163516      # waveguide broad‑wall [m]
shift    = 2 / 3.717054305989132
f0       = 35e9                   # carrier [Hz]  (Ka‑band, consistent with a≈4.87 mm)
lam0     = c0 / f0
lam_g0   = lam0 / np.sqrt(1 - (lam0 / (2 * a_wg))**2)
d_elem   = lam_g0 / 2 + shift * lam_g0   # element spacing [m]

# --- Geometry (from fscan.py) ---
beta     = np.deg2rad(45)         # antenna mounting angle
phi      = beta + np.deg2rad(14.3)  # swath centre look angle
H_plat   = 3e3                    # platform height [m]
theta_c  = np.deg2rad(0)          # squint angle

# --- Waveform (from fscan.py) ---
B        = 2e9                    # bandwidth [Hz]
Fs       = B * 1.2                # sampling rate ≈ 2.4 GHz
Tp       = 10e-7                  # pulse width [s] (from BeamScan)
Kr       = B / Tp                 # chirp rate [Hz/s]

# --- Derived ---
Nr       = int(np.ceil(Fs * Tp*3))   
df       = Fs / Nr                      # frequency resolution
freq     = f0 + (np.arange(Nr) - Nr // 2) * df   # frequency axis [Hz]
N_inband = int(np.ceil(B / df))         # samples inside bandwidth

M_exclude = N_elem*2                    # exclude M deepest nulls
print("=" * 60)
print(f"Carrier f0        = {f0/1e9:.2f} GHz")
print(f"Bandwidth B       = {B/1e9:.0f} GHz")
print(f"Sampling rate Fs  = {Fs/1e9:.2f} GHz")
print(f"Elements N_elem   = {N_elem}")
print(f"Element spacing d = {d_elem*1e3:.2f} mm")
print(f"Range samples Nr  = {Nr}")
print(f"In‑band samples   ≈ {N_inband}")
print(f"Excluded bins M   = {M_exclude}")
print(f"Recoverable       ≈ {N_inband - M_exclude}")
print("=" * 60)

# =========================================================================
# 2.  Antenna pattern pr(f) — exact replica of echogen()
# =========================================================================
def antenna_pattern_pr(f, doa, a=a_wg, d=d_elem):
    lam_now = c0 / f
    # waveguide wavelength
    sin_arg = lam_now / (2 * a)
    sin_arg = np.clip(sin_arg, 0, 0.999)        # stay below cutoff
    lam_g = lam_now / np.sqrt(1 - sin_arg**2)

    # inter‑element phase progression
    u1 = (2 * np.pi / lam_now) * d * np.sin(doa) \
       - (2 * np.pi / lam_g) * (d - lam_g / 2)

    # Dirichlet kernel (15 elements)
    with np.errstate(divide='ignore', invalid='ignore'):
        pr = np.sin(N_elem * u1 / 2) / (N_elem * np.sin(u1 / 2))
    pr = np.nan_to_num(pr, nan=1.0)             # lim_{u1→0} = 1
    return pr

# Compute pattern at swath‑centre DOA
doa_centre = phi - beta   # ≈ 14.3°
P1 = antenna_pattern_pr(freq, doa_centre)       # one‑way field
P2 = P1 ** 2                                     # two‑way voltage
P2 = P2 / np.max(P2) # normalise to unit energy

P_dB = 20 * np.log10(np.abs(P2) + 1e-30)

# Identify the M deepest nulls
null_idx_sorted = np.argsort(np.abs(P2))
null_idx = null_idx_sorted[:M_exclude]


def build_annihilating_filter(signal, M):
    N = len(signal)
    H_rows = N - M
    H_mat = np.zeros((H_rows, M + 1), dtype=complex)
    for i in range(H_rows):
        H_mat[i, :] = signal[i:i + M + 1]
    U, S, Vh = svd(H_mat, full_matrices=False)
    h = Vh[-1, :].conj()          # null‑space vector
    return h

h = build_annihilating_filter(P2, 30)
h = h/np.max(h) # normalise to unit energy
ph = np.convolve(P2, h, mode='full')[M_exclude:-(M_exclude)]
print("Annihilating filter value", np.max(np.abs(ph)))



# =========================================================================
# 4.  Synthetic scene
# =========================================================================
rng = np.random.default_rng(42)

# 4.1  Texture (distributed scatterers → smooth Doppler / spectral variation)
n_tex = 20
x_texture = np.zeros(Nr, dtype=complex)
for k in range(n_tex):
    phase = np.pi * (k + 1) * (freq - f0) / (B / 2)
    amp = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    x_texture += amp * np.exp(1j * phase) / (k + 1)
x_texture *= 0.3 / np.max(np.abs(x_texture))

# 4.2  Point targets  (sinusoids in frequency domain)
n_pts = 100
tau_pts = np.linspace(-Tp/100, Tp/100, n_pts)   # delays [s]
amp_pts = np.ones(n_pts)
x_sparse = np.zeros(Nr, dtype=complex)
for k in range(n_pts):
    x_sparse += amp_pts[k] * np.exp(-1j * 2 * np.pi * freq * tau_pts[k])

x_true = x_sparse

# 4.3  Observation
y_clean = P2 * x_true
SNR_dB = 10
sig_pow = np.mean(np.abs(y_clean)**2)
noise = np.sqrt(sig_pow / (10**(SNR_dB/10)) / 2) * (
    rng.normal(0, 1, Nr) + 1j * rng.normal(0, 1, Nr))
y = y_clean + noise
x_true = x_true+noise

def solve_c_based_direct(Y, P, h, lam, eps=1e-8):
    """
    C-based 正则化 (直接求解线性系统)
    min ||Y - D_p X||² + λ||C D_p X||² + ε||X||²
    →  A X = b
    A = D_p^T D_p + λ D_p^T C^T C D_p + ε I
    b = D_p^T Y
    """
    N = len(P)
    Dp = np.diag(P)                 # 实对角矩阵, Dp^T = Dp
    
    # 系统矩阵 (利用对称性)
    
    # 构造 h 的循环矩阵 (N x N)
    h = np.asarray(h).ravel()
    h_len = h.size
    # 将 h 放入长度为 N 的向量，超出部分截断，短于 N 的用零填充
    h_pad = np.zeros(N, dtype=complex)
    h_pad[:min(h_len, N)] = h[:min(h_len, N)]
    # 每一行是 h_pad 的循环右移
    C = np.zeros((N, N), dtype=complex)
    for i in range(N):
        C[i, :] = np.roll(h_pad, i)

    CTC = C.conj().T @ C

    A = np.diag(P**2) + lam * (Dp @ CTC @ Dp) + eps * np.eye(N)

    
    # 右端项
    b = Dp @ Y                       # = P ⊙ Y (逐元素)
    
    # 求解 (利用对称正定性)
    X = solve(A, b, assume_a='pos')  # Cholesky
    return X

P2 = P2 / np.max(P2)  # normalise to unit energy

x = solve_c_based_direct(y, P2, h, lam=1000, eps=1e-3)

# P = P2
# P[P < 3.5e-2] = np.max(P)
# P = P/np.max(P)
# x_tik = y / P

# x = x_tik


# =========================================================================
# 7.  Visualisation
# =========================================================================
plt.rcParams.update({'figure.figsize': (18, 20), 'font.size': 9})

f_GHz = freq / 1e9

# (d) Received signal
plt.figure()
plt.plot(f_GHz, 20*np.log10(np.abs(y)+1e-30), label='y')
plt.plot(f_GHz, 20*np.log10(np.abs(x)+1e-30), label='x')
plt.plot(f_GHz, 20*np.log10(np.abs(x_true)+1e-30), label='x_true')
# plt.plot(f_GHz, 20*np.log10(np.abs(x_tik)+1e-30), label='x_tik')
plt.legend()
plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
plt.grid(alpha=0.3)

upsample = 16
pad_y = np.zeros(Nr*upsample, dtype=complex)
pad_y[Nr*8:Nr*9] = y
pad_x = np.zeros(Nr*16, dtype=complex)
pad_x[Nr*8:Nr*9] = x
pad_x_true = np.zeros(Nr*16, dtype=complex)
pad_x_true[Nr*8:Nr*9] = x_true

y = pad_y
x = pad_x
x_true = pad_x_true

y_ifft = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(y)))
x_ifft = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x)))
x_true_ifft = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x_true)))

y_ifft = y_ifft/np.max(np.abs(y_ifft))
x_ifft = x_ifft/np.max(np.abs(x_ifft))
x_true_ifft = x_true_ifft/np.max(np.abs(x_true_ifft))


max_y = np.max(np.abs(y_ifft))
noise_y = np.mean(np.abs(y_ifft[0:4000]))
snr_y = 20*np.log10(max_y/noise_y)

max_x = np.max(np.abs(x_ifft))
noise_x = np.mean(np.abs(x_ifft[0:4000]))
snr_x = 20*np.log10(max_x/noise_x)

y_half = np.abs(y_ifft) > max_y/np.sqrt(2)
irw_y = np.sum(y_half)/Fs*c0/2/upsample

x_half = np.abs(x_ifft) > max_x/np.sqrt(2)
irw_x = np.sum(x_half)/Fs*c0/2/upsample

print("IRW of y_ifft : {:.3f} m".format(irw_y))
print("IRW of x_ifft : {:.3f} m".format(irw_x))

plt.figure()
plt.plot(20*np.log10(np.abs(y_ifft)+1e-30), label='y_ifft')
plt.plot(20*np.log10(np.abs(x_ifft)+1e-30), label='x_ifft')
# plt.plot(20*np.log10(np.abs(x_true_ifft)+1e-30), label='x_true_ifft')
plt.legend()
plt.xlabel('Sample index'); plt.ylabel('Magnitude')
plt.grid(alpha=0.3)

max_pos = np.argmax(np.abs(y_ifft))
y_ifft = y_ifft[max_pos-200:max_pos+200]
x_ifft = x_ifft[max_pos-200:max_pos+200]
x_true_ifft = x_true_ifft[max_pos-200:max_pos+200]

print("SNR of y_ifft : {:.2f} dB".format(snr_y))
print("SNR of x_ifft : {:.2f} dB".format(snr_x))

plt.figure()
plt.plot(20*np.log10(np.abs(y_ifft)+1e-30), label='y_ifft')
plt.plot(20*np.log10(np.abs(x_ifft)+1e-30), label='x_ifft')
plt.plot(20*np.log10(np.abs(x_true_ifft)+1e-30), label='x_true_ifft')
plt.legend()
plt.xlabel('Sample index'); plt.ylabel('Magnitude')
plt.grid(alpha=0.3)

plt.show()