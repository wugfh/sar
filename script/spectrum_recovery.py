import numpy as np
import cupy as cp
from scipy.interpolate import interp1d
from scipy.linalg import hankel, svd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from scipy.linalg import solve, svd, norm
from scipy import io as sio
from tqdm import tqdm
from scipy.linalg import hankel, eig, eigvals
from scipy.linalg import toeplitz
import scipy.interpolate as interpolate
import pandas as pd
import tv

plt.rc("font", family="Times New Roman")
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
warnings.filterwarnings('ignore')
cp.cuda.Device(0).use()

def solve_c_based_cg(Y, P, h, lam, eps=1e-8,
                     max_iter=30000, tol=1e-6, warm_start=None):
    N = len(P)
    dtype = Y.dtype
    h_pad = cp.zeros((h.shape[0], N), dtype=dtype)
    h_len = min(h.shape[1], N)
    if h_len % 2 == 0:
        h_pad[:, N//2 - h_len//2:N//2 + h_len//2] = cp.asarray(h[:, :h_len]) 
    else:
        h_pad[:, N//2 - h_len//2:N//2 + h_len//2 + 1] = cp.asarray(h[:, :h_len])  

    H_f = cp.fft.fft(h_pad, axis=1)            # C 的特征值
    H = cp.abs(H_f) ** 2              # C^H·C 的特征值 = |H(ω)|²
    P_sq = P ** 2
    mu = float(cp.mean(P_sq))          # T. Chan 最优参数

    temp = lam[0]*H[0, :]
    for i in range(1, h.shape[0]):
        temp += lam[i] * H[i, :]
    inv_M_denom = 1.0 / (mu + eps + temp)   # 用于 M_circ^{-1}
    # ---- 矩阵向量乘 A @ v (O(N log N)) ----
    def matvec(v):
        out = P_sq * v
        w_f = cp.fft.fft(v)  
        u = cp.fft.ifft(H * w_f, axis=1)  # (h.shape[0], N)   
        u = cp.sum(lam[:, None] * u, axis=0)  # (N,)       
        out += u
        out += eps * v        
        return out
    # ---- 预条件子 M⁻¹ @ v ----
    def precond(v):
        v_f = cp.fft.fft(v)
        return cp.fft.ifft(inv_M_denom * v_f)           # (M,N)
    # ---- 右端项 b = P ⊙ Y ----
    b = P * Y
    # ---- 初值 (热启动或零) ----
    if warm_start is not None:
        X = warm_start.astype(dtype).copy()
    else:
        X = cp.zeros(N, dtype=dtype)
    # ---- 预条件共轭梯度 (PCG) ----
    r = b - matvec(X)
    z = precond(r)
    p = z.copy()
    rz_old = cp.real(cp.vdot(r, z))
    b_norm_sq = cp.real(cp.vdot(b, b))
    tol_sq = max(tol ** 2 * b_norm_sq, 1e-30)
    for iteration in range(max_iter):
        Ap = matvec(p)
        pAp = cp.real(cp.vdot(p, Ap))
        if pAp <= 1e-30:          # 数值保护
            break
        alpha = rz_old / pAp
        X += alpha * p
        r -= alpha * Ap
        r_norm_sq = cp.real(cp.vdot(r, r))
        if r_norm_sq < tol_sq:
            # print(f"PCG converged in {iteration+1} iterations (residual norm²={r_norm_sq:.2e})")
            break
        z = precond(r)
        rz_new = cp.real(cp.vdot(r, z))
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        if iteration == max_iter - 1:
            print(f"PCG reached max iterations ({max_iter}) with residual norm²={r_norm_sq:.2e}")
    return X

def solve_c_based_cg_batch(Y, P, h, lam, eps=1e-8,
                           max_iter=30000, tol=1e-6, warm_start=None):
    """
    Y:   (B, N) 批量频谱
    P:   (N,)   窗函数
    h:   (B, K, L) 逐行零化滤波器（build_annihilating_filter_batch_* 的返回）
    lam: 标量 / (K,) / (B, K) 各滤波器权重
    """
    B, N = Y.shape
    dtype = Y.dtype
    K = h.shape[1]
    L = min(h.shape[2], N)

    h_pad = cp.zeros((B, K, N), dtype=dtype)              # 居中补零
    s0 = N // 2 - L // 2
    h_pad[:, :, s0:s0 + L] = h[:, :, :L]
    H = cp.abs(cp.fft.fft(h_pad, axis=-1)) ** 2           # |H(ω)|² : (B,K,N)

    lam = cp.asarray(lam)
    if lam.ndim == 0:
        lam = lam * cp.ones((1, 1), dtype=lam.dtype)
    elif lam.ndim == 1:
        lam = lam.reshape(1, K)
    lam = cp.broadcast_to(lam, (B, K))

    P_sq = P ** 2
    mu = float(cp.mean(P_sq))                             # T. Chan 最优参数
    inv_M_denom = 1.0 / (mu + eps + cp.sum(lam[:, :, None] * H, axis=1))   # (B,N)

    def matvec(v):
        out = P_sq * v
        w_f = cp.fft.fft(v, axis=-1)                      # (B,N)
        u = cp.fft.ifft(H * w_f[:, None, :], axis=-1)     # (B,K,N)
        out += cp.sum(lam[:, :, None] * u, axis=1)
        out += eps * v
        return out

    def precond(v):
        return cp.fft.ifft(inv_M_denom * cp.fft.fft(v, axis=-1), axis=-1)

    b = P * Y
    X = (warm_start.astype(dtype).copy() if warm_start is not None
         else cp.zeros((B, N), dtype=dtype))
    r = b - matvec(X)
    z = precond(r)
    p = z.copy()
    rz_old = cp.real(cp.sum(cp.conj(r) * z, axis=1))
    b_norm_sq = cp.real(cp.sum(cp.conj(b) * b, axis=1))
    tol_sq = cp.maximum(tol ** 2 * b_norm_sq, 1e-30)
    active = cp.ones(B, dtype=bool)
    for iteration in range(max_iter):
        if not cp.any(active):
            break
        Ap = matvec(p)
        pAp = cp.real(cp.sum(cp.conj(p) * Ap, axis=1))
        safe = active & (pAp > 1e-30)
        alpha = cp.where(safe, rz_old / cp.maximum(pAp, 1e-30), 0.0)
        X += alpha[:, None] * p
        r -= alpha[:, None] * Ap
        r_norm_sq = cp.real(cp.sum(cp.conj(r) * r, axis=1))
        active &= (r_norm_sq >= tol_sq)
        if not cp.any(active):
            break
        z = precond(r)
        rz_new = cp.real(cp.sum(cp.conj(r) * z, axis=1))
        beta = cp.where(active, rz_new / cp.maximum(rz_old, 1e-30), 0.0)
        p = z + beta[:, None] * p
        rz_old = rz_new
    return X

def build_annihilating_filter_batch(signals, M, filter_order):

    B, N = signals.shape
    H_rows = N - M

    idx = cp.arange(H_rows)[:, None] + cp.arange(M + 1)[None, :]   # (N-M, M+1)
    H_mat = signals[:, idx]

    _, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)

    return Vh[:, -filter_order:, :], S

def build_annihilating_filter(signal, M, filter_order):
    N = len(signal)
    signal_pad = signal
    # signal_pad = cp.zeros(N+M, dtype=complex)
    # off = (N + M) // 2 - N // 2
    # signal_pad[off:off + N] = signal
    H_rows = N-M
    # ---------- 向量化 Hankel：
    idx = cp.arange(H_rows)[:, None] + cp.arange(M + 1)[None, :]   # (H_rows, M+1)
    # idx = (idx-M)*(idx>=M)+idx*(idx<M)  # 防止索引越界
    H_mat = signal_pad[idx]                                   


    U, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)

    h1 = Vh[-1, :]
    h2 = Vh[-2, :]

    return Vh[-filter_order:], S


def esprit(signal, M):
    N = len(signal)
    H_rows = N-M
    idx = cp.arange(H_rows)[:, None] + cp.arange(M + 1)[None, :]   # (H_rows, M+1)
    H_mat = signal[idx]                                   

    U, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)
    U_M = U[:, :M]
    U_M1 = U_M[:-1, :]
    U_M2 = U_M[1:, :]

    # Solve the least squares problem: U_M1 * Phi ≈ U_M2
    Phi, residuals, rank, s = cp.linalg.lstsq(U_M1, U_M2)

    # Eigen decomposition of Phi
    eigvals, eigvecs = np.linalg.eig(Phi.get())

    # The DOAs are related to the angles of the eigenvalues
    # doa_estimates = np.angle(eigvals)
    # doa_estimates = np.sort(doa_estimates)  # Sort the DOA estimates
    # doa_estimates = cp.unwrap(doa_estimates*2)  # Unwrap the angles to avoid discontinuities
    # doa_estimates = doa_estimates / 2 - cp.mean(doa_estimates/2)  # Center around zero

    return eigvals

def antenna_pattern_pr(f, doa, a, d, N_elem):
    lam_now = c0 / f
    # waveguide wavelength
    sin_arg = lam_now / (2 * a)
    sin_arg = cp.clip(sin_arg, 0, 0.999)        # stay below cutoff
    lam_g = lam_now / cp.sqrt(1 - sin_arg**2)

    # inter‑element phase progression
    u1 = (2 * cp.pi / lam_now) * d * cp.sin(doa) \
    - (2 * cp.pi / lam_g) * (d - lam_g / 2)

    # Dirichlet kernel (15 elements)
    pr = cp.sin(N_elem * u1 / 2) / (N_elem * cp.sin(u1 / 2))
    pr = cp.nan_to_num(pr, nan=1.0)             # lim_{u1→0} = 1
    return pr

if __name__ == "__main__":
    c0 = 3e8
    # --- Geometry (from fscan.py) ---
    beta     = cp.deg2rad(45)         # antenna mounting angle
    phi      = beta + cp.deg2rad(14.3)  # swath centre look angle
    H_plat   = 3e3                    # platform height [m]
    theta_c  = cp.deg2rad(0)          # squint angle

    # --- Waveform (from fscan.py) ---
    B        = 2e9                    # bandwidth [Hz]
    Fs       = B * 1.25                # sampling rate
    Tp       = 10e-7                  # pulse width [s] (from BeamScan)
    Kr       = B / Tp                 # chirp rate [Hz/s]
    f0       = 35e9 
    # --- Derived ---
    Nr       = int(cp.ceil(Fs * Tp))   
    Nr       = 3000
    print("Nr:{}".format(Nr))
    Tr       = Nr / Fs                     # pulse duration [s]
    df       = Fs / Nr                      # frequency resolution
    freq     = f0 + (cp.arange(Nr) - Nr // 2) * df   # frequency axis [Hz]
    N_inband = int(cp.ceil(B / df))         # samples inside bandwidth

    data_path = "../data/250925KaAntenna/1-35-e.xlsx"
    data = pd.read_excel(data_path, header=1)
    r_angle = (cp.array(data['Elevation (deg)'])) ## rad
    r_pattern = cp.array(data['Amplitude (dB)']) ## dB
    ant_gain_func = interpolate.interp1d(r_angle.get(), r_pattern.get(), kind='cubic', fill_value="extrapolate")
    plt.figure()
    plt.plot(r_angle.get(), ant_gain_func(r_angle.get()), label = "Antenna gain")
    plt.xlabel('Elevation Angle (rad)'); plt.ylabel('Antenna gain (dB)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("../fig/spectrum_recovery/antenna_gain.png", dpi=300)

    fc = np.array([34e9,35e9,36e9])
    theta = np.array([17.9416367435066, 14.2959686223657, 10.9066262820108])  # 测量角度（度）
    param = np.polyfit(fc, theta, 2)
    doa_ant = param[0]*(freq)**2 + param[1]*freq + param[2] 
    print("doa_ant:{}-{}".format(cp.min(doa_ant).get(), cp.max(doa_ant).get()))
    P1 = cp.array(ant_gain_func(doa_ant.get()))
    P1 = 10**(P1/20)                           
    P2 = P1**2                                 # two‑way voltage
    P2 = P2 / cp.sqrt(cp.sum(P2**2))                            # normalise to unit energy

    plt.figure()
    plt.plot(freq.get()/1e9, 20*cp.log10(cp.abs(P1)).get())
    plt.xlabel('Frequency / GHz'); plt.ylabel('Antenna gain (dB)')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/window.png", dpi=300)      
    

    # win_len = 320/3000*Nr
    # ax = cp.arange(-Nr/2, Nr/2, 1)
    # # shift = Nr//4
    # shift = 0
    # window =  cp.exp(-0.5 * ((ax+shift) / win_len) ** 2)
    # P2 = window
    # P2 = P2/cp.sqrt(cp.sum(P2**2)) 
    # P2 = P2 *(cp.abs(freq - f0) < B/2)

    # a = 0.004871409163516
    # lambda_ = c0/f0
    # lambda_g=lambda_/np.sqrt(1-(lambda_/(2*a))**2)
    # shift_d = 2/3.717054305989132
    # d = lambda_g/2 +shift_d* lambda_g
    # P2 = antenna_pattern_pr(freq, phi-beta, 0.004871409163516, d, 16)
    # P2 = P2 ** 2
    # P2 = P2/cp.sqrt(cp.sum(P2**2))

            

    P_dB = 20 * cp.log10(cp.abs(P2) + 1e-30)

    rng = cp.random.default_rng(42)

    # 4.2  Point targets  (sinusoids in frequency domain)
    n_pts = 100
    tau_pts = cp.linspace(-Tp/2, Tp/2, n_pts)   # delays [s]    
    tau = cp.arange(-Nr/2, Nr/2)*1/Fs  # time axis [s]
    Kr = B / Tp
    # tau_pts = cp.random.uniform(-Nr/Fs/2, Nr/Fs/2, n_pts)
    # amp_pts = cp.random.uniform(1, 10, n_pts)
    amp_pts = cp.ones(n_pts)
    # amp_pts[cp.abs(tau_pts) < Tp/2] = 0
    x_sparse = cp.zeros(Nr, dtype=complex)
    W = cp.abs(freq - f0) < B/2
    for k in range(n_pts):
        x_sparse += amp_pts[k] * cp.exp(-1j * cp.pi * Kr * (tau - tau_pts[k])**2)
        # x_sparse += amp_pts[k] * cp.exp(-1j * 2 * cp.pi * freq * tau_pts[k])

    x_true = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(x_sparse)))
    # x_true = x_sparse

    # 4.3  Observation
    y_clean = P2 * x_true
    SNR_dB = 20
    sig_pow = cp.mean(cp.abs(y_clean)**2)/cp.sqrt(cp.sum(amp_pts**2))
    noise_power = sig_pow * 10**(-SNR_dB/20)
    noise = (cp.random.randn(*y_clean.shape) + 1j * cp.random.randn(*y_clean.shape)) * noise_power / cp.sqrt(2)

    y = y_clean + noise
    x_true = x_true/cp.sqrt(cp.sum(cp.abs(x_true)**2)) * cp.sqrt(cp.sum(cp.abs(y_clean)**2))
    x_true = x_true + noise

    Hm = cp.exp(-1j*cp.pi*(freq-f0)**2/Kr)
    y = y * Hm
    x_true = x_true * Hm
    y_clean = y_clean * Hm


    order = Nr//2-1
    filter_order = 10

    _, S_xt = build_annihilating_filter(x_true, order, filter_order)
    S_xt = S_xt/cp.max(S_xt)

    y = sio.loadmat("../fig/spectrum_recovery/test_130.mat")["test"]
    y = np.squeeze(y)
    y = cp.array(y)



    original_len = y.shape[0]

    pad_y = cp.zeros(Nr, dtype=complex)
    pad_y[Nr//2-original_len//2:Nr//2+original_len//2] = y
    y = pad_y

    y = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(y)))

    
    # doa = esprit(y, order)
    # plt.figure()
    # plt.scatter(np.abs(doa), np.unwrap(np.angle(doa)), label = "Estimated DOA")
    # plt.grid(alpha=0.3)
    # # plt.xlim([0.97, 0.99])
    # plt.xlabel('Mode index'); plt.ylabel('Estimated DOA (rad)')
    # plt.savefig("../fig/spectrum_recovery/esprit.png", dpi=300)

    max_pos_y = cp.argmax(cp.abs(cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(y)))))
    print("y shape:", y.shape)
    start = 0
    hy, S = build_annihilating_filter(y, order, filter_order)
    Sy = S/cp.max(S)


    # pre_down = cp.sqrt(cp.sum(cp.abs(x_true)**2))/cp.sqrt(cp.sum(cp.abs(noise)**2))
    # after_down = cp.sqrt(cp.sum(cp.abs(y)**2))/cp.sqrt(cp.sum(cp.abs(noise)**2))
    # down = []
    # down_y = []
    # hy_filter = []
    # for i in range(hy.shape[0]):
    #     h = hy[i, :]
    #     x_res = cp.convolve(x_true, h, mode='same')
    #     y_res = cp.convolve(y, h, mode='same')
    #     n_res = cp.convolve(noise, h, mode='same')
    #     x_down = cp.sqrt(cp.sum(cp.abs(x_res)**2))/cp.sqrt(cp.sum(cp.abs(n_res)**2))
    #     y_down = cp.sqrt(cp.sum(cp.abs(y_res)**2))/cp.sqrt(cp.sum(cp.abs(n_res)**2))
    #     down.append(x_down/pre_down)
    #     down_y.append(y_down/after_down)
    #     hy_filter.append(h[None,:])
    # down = cp.array(down)
    # down_y = cp.array(down_y)
    # plt.figure()
    # # plt.plot(down.get(), label = "down factor of x_true")
    # plt.plot(down_y.get(), label = "down factor of y")
    # plt.xlabel('Filter index'); plt.ylabel('factor')
    # plt.grid(alpha=0.3)    
    # plt.legend()
    # plt.savefig("../fig/spectrum_recovery/filter_down.png", dpi=300)

    # hy = hy[down_y < cp.percentile(down_y, 10/filter_order), :]

    ## 注水
    # factor = 1e-2
    # max_p2 = cp.max(P2)
    # P2[P2 < max_p2 * factor] = max_p2*factor
    
    P2 = P2/cp.sqrt(cp.sum(P2**2))
    # P2[freq < f0 - B/ 2] = 10*max_p2
    # P2[freq > f0 + B / 2] = 10*max_p2

    # lam = cp.ones(hy.shape[0]) * 1e-4
    # x1 = solve_c_based_cg(y, P2, hy, lam=lam, eps=0)
    lam = 1/noise.std()*1000
    y_ifft = cp.fft.ifft(y)
    p = cp.fft.ifft(P2)
    x, info= tv.solve_lasso_regularized(p, y_ifft, lam=lam, tol=1e-6)
    x = cp.fft.fft(x)
    x1 = x

    kaise_win = cp.kaiser(Nr, beta=5)
    # x = x * kaise_win

    _, S = build_annihilating_filter(x1, order, filter_order)
    Sx1 = S/cp.max(S)
    plt.figure()
    plt.plot(20*np.log10((Sx1).get()), label = "singular values of x1")
    plt.plot(20*np.log10((Sy).get()), label = "singular values of y")
    plt.plot(20*np.log10((S_xt).get()), label = "singular values of x_true")
    # plt.plot(20*np.log10((S_clean).get()), label = "singular values of y_clean")
    plt.legend(loc="best")
    plt.xlabel('Singular value index'); plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.ylim([-100, 0])
    plt.savefig("../fig/spectrum_recovery/singular_values_x.png", dpi=300)

    x = cp.squeeze(x1)
    # x_true = x2


    f_GHz = freq / 1e9

    # (d) Received signal
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(y)/cp.max(cp.abs(y))+1e-30).get(), label='y')
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(P2)/cp.max(cp.abs(P2))+1e-30).get(), label='window')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude (dB)')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    plt.subplot(2,1,2)
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x)/cp.max(cp.abs(x))+1e-30).get(), label='x')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude (dB)')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    # plt.subplot(3,1,3)
    # x_true = x_true/cp.max(cp.abs(x_true))
    # plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x_true)+1e-30).get(), label='x_true')
    # plt.legend()
    # plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    # plt.ylim([-60, 0])
    # plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../fig/spectrum_recovery/spectrum_recovery_freq.png", dpi=300)

    upsample = 16
    pad_y = cp.zeros(Nr*upsample, dtype=complex)
    pad_y[Nr*upsample//2-Nr//2:Nr*upsample//2+Nr//2] = y
    pad_x = cp.zeros(Nr*upsample, dtype=complex)
    pad_x[Nr*upsample//2-Nr//2:Nr*upsample//2+Nr//2] = x
    pad_x_true = cp.zeros(Nr*upsample, dtype=complex)
    pad_x_true[Nr*upsample//2-Nr//2:Nr*upsample//2+Nr//2] = x_true

    y = pad_y
    x = pad_x
    x_true = pad_x_true


    y_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(y)))
    x_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x)))
    x_true_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x_true)))
    x_ifft = x_ifft[x_ifft.shape[0]//2 - original_len*upsample//2 : x_ifft.shape[0]//2 + original_len*upsample//2]
    y_ifft = y_ifft[y_ifft.shape[0]//2 - original_len*upsample//2 : y_ifft.shape[0]//2 + original_len*upsample//2]
    x_true_ifft = x_true_ifft[x_true_ifft.shape[0]//2 - original_len*upsample//2 : x_true_ifft.shape[0]//2 + original_len*upsample//2]


    y_ifft = y_ifft/cp.max(cp.abs(y_ifft))
    x_ifft = x_ifft/cp.max(cp.abs(x_ifft))
    x_true_ifft = x_true_ifft/cp.max(cp.abs(x_true_ifft))

    x_true_ifft = x_true_ifft/cp.max(cp.abs(x_true_ifft))

    max_y = cp.max(cp.abs(y_ifft))
    noise_y = cp.sqrt(cp.mean(cp.abs(y_ifft[40000:41000])**2))
    snr_y = 20*cp.log10(max_y/noise_y)

    max_x = cp.max(cp.abs(x_ifft))
    noise_x = cp.sqrt(cp.mean(cp.abs(x_ifft[40000:41000])**2))
    snr_x = 20*cp.log10(max_x/noise_x)



    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(20*cp.log10(cp.abs(y_ifft)+1e-30).get(), label='y_ifft')
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    plt.subplot(2,1,2)
    plt.plot(20*cp.log10(cp.abs(x_ifft)+1e-30).get(), label='x_ifft')
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../fig/spectrum_recovery/spectrum_recovery_ifft.png", dpi=300)

    max_pos = cp.argmax(cp.abs(y_ifft)) 
    # print("max_pos:{}".format(max_pos))
    # max_pos = 25000
    y_ifft = y_ifft[max_pos-500:max_pos+500]
    x_ifft = x_ifft[max_pos-500:max_pos+500]
    y_ifft = y_ifft/cp.max(cp.abs(y_ifft))
    x_ifft = x_ifft/cp.max(cp.abs(x_ifft))
    x_true_ifft = x_true_ifft[max_pos-500:max_pos+500]
    x_true_ifft = x_true_ifft/cp.max(cp.abs(x_true_ifft))

    print("SNR of y_ifft : {:.2f} dB".format(snr_y))
    print("SNR of x_ifft : {:.2f} dB".format(snr_x))

    y_half = cp.abs(y_ifft) > cp.max(cp.abs(y_ifft))/cp.sqrt(2)
    irw_y = cp.sum(y_half)/Fs*c0/2/upsample

    x_half = cp.abs(x_ifft) > cp.max(cp.abs(x_ifft))/cp.sqrt(2)
    irw_x = cp.sum(x_half)/Fs*c0/2/upsample

    x_true_half = cp.abs(x_true_ifft) > cp.max(cp.abs(x_true_ifft))/cp.sqrt(2)
    irw_x_true = cp.sum(x_true_half)/Fs*c0/2/upsample


    print("IRW of y_ifft : {:.3f} m".format(irw_y))
    print("IRW of x_ifft : {:.3f} m".format(irw_x))
    print("IRW of x_true_ifft : {:.3f} m".format(irw_x_true))


    plt.figure()
    plt.plot(20*cp.log10(cp.abs(y_ifft)+1e-30).get(), label='y')
    plt.plot(20*cp.log10(cp.abs(x_ifft)+1e-30).get(), label='recovered x')
    # plt.plot(20*cp.log10(cp.abs(x_true_ifft)+1e-30).get(), label='true x')
    plt.legend(loc = "lower right")
    plt.xlabel('Sample index'); plt.ylabel('Magnitude (dB)')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)

    plt.savefig("../fig/spectrum_recovery/spectrum_recovery.png", dpi=300)