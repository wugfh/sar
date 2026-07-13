import numpy as np
import cupy as cp
from scipy.interpolate import interp1d
from scipy.linalg import hankel, svd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from scipy.linalg import solve, svd, norm
warnings.filterwarnings('ignore')

def solve_c_based_cg(Y, P, h, lam, eps=1e-8,
                     max_iter=30000, tol=1e-6, warm_start=None):
    N = len(P)
    dtype = Y.dtype

    h_pad = cp.zeros(N, dtype=dtype)
    h_len = min(len(h), N)
    h_pad[:h_len] = cp.asarray(h[:h_len])

    H_f = cp.fft.fft(h_pad)            # C 的特征值
    H2 = cp.abs(H_f) ** 2              # C^H·C 的特征值 = |H(ω)|²
    h_norm_sq = float(cp.sum(cp.abs(h_pad) ** 2))  # ||h||² (Parseval)
    P_sq = P ** 2
    diag_A = P_sq +  lam * h_norm_sq
    inv_diag_A = 1.0 / diag_A
    # ---- 矩阵向量乘 A @ v (O(N log N)) ----
    def matvec(v):
        out = P_sq * v
        w_f = cp.fft.fft(v)                
        u = cp.fft.ifft(H2 * w_f)           
        out += lam * u                  
        return out
    # ---- 预条件子 M^{-1} @ v = v ./ diag(A) ----
    def precond(v):
        return inv_diag_A * v
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

    M, N = Y.shape
    dtype = Y.dtype

    # ---- 预计算（所有行共用） ----
    h_pad = cp.zeros(N, dtype=dtype)
    h_len = min(len(h), N)
    h_pad[:h_len] = cp.asarray(h[:h_len])

    H_f = cp.fft.fft(h_pad)                         # (N,)
    H2 = cp.abs(H_f) ** 2                            # (N,)  |H(ω)|²
    h_norm_sq = float(cp.sum(cp.abs(h_pad) ** 2))    # ||h||²

    P_sq = P ** 2                                     # (N,)
    diag_A = P_sq + lam * h_norm_sq                   # (N,)
    inv_diag_A = 1.0 / diag_A                         # (N,)

    # ---- 批量矩阵向量乘 A @ v  (M,N) -> (M,N) ----
    def matvec(v):
        out = P_sq * v                                
        w_f = cp.fft.fft(v, axis=1)                 
        u = cp.fft.ifft(H2 * w_f, axis=1)           
        out += lam * u
        # out += eps * v 
        return out

    # ---- 预条件子 M⁻¹ @ v ----
    def precond(v):
        return inv_diag_A * v                         # (N,) 广播到 (M,N)

    # ---- 初始残差和内积 ----
    b = P * Y                                          # (M,N)

    if warm_start is not None:
        X = warm_start.astype(dtype).copy()
    else:
        X = cp.zeros((M, N), dtype=dtype)

    r = b - matvec(X)                                  # (M,N)
    z = precond(r)                                     # (M,N)
    p = z.copy()                                       # (M,N)

    # 行内积 → (M,) 实向量
    rz_old = cp.real(cp.sum(cp.conj(r) * z, axis=1))  # (M,)
    b_norm_sq = cp.real(cp.sum(cp.conj(b) * b, axis=1))  # (M,)
    tol_sq = cp.maximum(tol ** 2 * b_norm_sq, 1e-30)   # (M,)

    # ---- 活跃行掩膜（已收敛的行不再更新） ----
    active = cp.ones(M, dtype=bool)                     # (M,)

    for iteration in range(max_iter):
        if not cp.any(active):
            break

        Ap = matvec(p)                                  # (M,N)

        # pAp = <p, Ap> 逐行内积
        pAp = cp.real(cp.sum(cp.conj(p) * Ap, axis=1))  # (M,)

        # 安全除法：非活跃行或 pAp≈0 的行 α=0
        safe = active & (pAp > 1e-30)
        alpha = cp.where(safe, rz_old / cp.maximum(pAp, 1e-30), 0.0)  # (M,)

        # 更新（α=0 的行不受影响）
        X += alpha[:, None] * p                         # (M,N)
        r -= alpha[:, None] * Ap

        # 残差范数
        r_norm_sq = cp.real(cp.sum(cp.conj(r) * r, axis=1))  # (M,)

        # 标记新收敛的行
        active &= (r_norm_sq >= tol_sq)

        if not cp.any(active):
            break

        z = precond(r)
        rz_new = cp.real(cp.sum(cp.conj(r) * z, axis=1))  # (M,)

        # β：仅活跃行参与计算
        beta = cp.where(active,
                        rz_new / cp.maximum(rz_old, 1e-30),
                        0.0)                            # (M,)

        p = z + beta[:, None] * p                       # (M,N)
        rz_old = rz_new                                 # 非活跃行的值变为脏数据，但不再使用

        if iteration == max_iter - 1:
            n_active = int(cp.sum(active))
            if n_active > 0:
                max_res = float(cp.max(r_norm_sq[active]))
                print(f"CG batch: {n_active}/{M} rows did NOT converge "
                      f"(max residual²={max_res:.2e})")

    return X


def solve_c_based_direct(Y, P, h, lam, eps=1e-8):

        N = len(P)
        Dp = cp.diag(P)                 # 实对角矩阵, Dp^T = Dp
        
        # 系统矩阵 (利用对称性)
        
        # 构造 h 的循环矩阵 (N x N)
        h = cp.asarray(h).ravel()
        h_len = h.size
        # 将 h 放入长度为 N 的向量，超出部分截断，短于 N 的用零填充
        h_pad = cp.zeros(N, dtype=complex)
        h_pad[:min(h_len, N)] = h[:min(h_len, N)]
        # 每一行是 h_pad 的循环右移
        C = cp.zeros((N, N), dtype=complex)
        for i in range(N):
            C[i, :] = cp.roll(h_pad, i)
        CTC = C.conj().T @ C

        A = cp.diag(P**2) + lam * (CTC) + eps * cp.eye(N)

        
        # 右端项
        b = Dp @ Y                       # = P ⊙ Y (逐元素)
        
        # 求解 (利用对称正定性)
        X = cp.linalg.solve(A, b)
        return X

def build_annihilating_filter(signal, M):
    N = len(signal)
    H_rows = N - M
    H_mat = cp.zeros((H_rows, M + 1), dtype=complex)
    for i in range(H_rows):
        H_mat[i, :] = signal[i:i + M + 1]
    U, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)
    Sd = cp.abs(cp.diff(cp.diff(cp.log10(S))))
    mean_Sd = cp.mean(Sd)
    std_Sd = cp.std(Sd)
    threshold_up = mean_Sd + 2 * std_Sd
    idx = cp.where(Sd > threshold_up)[0]

    if idx.size > 0:
        pos = int(idx[0])
    else:
        # fallback to last index if all differences negative
        pos = int(Sd.size - 1)


    h = Vh[pos, :].conj()          # null‑space vector
    h = h/cp.sqrt(cp.sum(cp.abs(h)**2))  # normalise to unit energy
    return h,S

if __name__ == "__main__":
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
    lam_g0   = lam0 / cp.sqrt(1 - (lam0 / (2 * a_wg))**2)
    d_elem   = lam_g0 / 2 + shift * lam_g0   # element spacing [m]

    # --- Geometry (from fscan.py) ---
    beta     = cp.deg2rad(45)         # antenna mounting angle
    phi      = beta + cp.deg2rad(14.3)  # swath centre look angle
    H_plat   = 3e3                    # platform height [m]
    theta_c  = cp.deg2rad(0)          # squint angle

    # --- Waveform (from fscan.py) ---
    B        = 2e9                    # bandwidth [Hz]
    Fs       = B * 1.2                # sampling rate ≈ 2.4 GHz
    Tp       = 10e-7                  # pulse width [s] (from BeamScan)
    Kr       = B / Tp                 # chirp rate [Hz/s]

    # --- Derived ---
    Nr       = int(cp.ceil(Fs * Tp*3))   
    df       = Fs / Nr                      # frequency resolution
    freq     = f0 + (cp.arange(Nr) - Nr // 2) * df   # frequency axis [Hz]
    N_inband = int(cp.ceil(B / df))         # samples inside bandwidth

    M_exclude = N_elem*2                    # exclude M deepest nulls

    # =========================================================================
    # 2.  Antenna pattern pr(f) — exact replica of echogen()
    # =========================================================================
    def antenna_pattern_pr(f, doa, a=a_wg, d=d_elem):
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

    # Compute pattern at swath‑centre DOA
    doa_centre = phi - beta   # ≈ 14.3°
    P1 = antenna_pattern_pr(freq, doa_centre)       # one‑way field
    P2 = P1 ** 2                                     # two‑way voltage
    P2 = P2 / cp.sqrt(cp.sum(P2**2))                            # normalise to unit energy

    P_dB = 20 * cp.log10(cp.abs(P2) + 1e-30)

    # Identify the M deepest nulls
    null_idx_sorted = cp.argsort(cp.abs(P2))
    null_idx = null_idx_sorted[:M_exclude]

    # =========================================================================
    # 4.  Synthetic scene
    # =========================================================================
    rng = cp.random.default_rng(42)

    # 4.1  Texture (distributed scatterers → smooth Doppler / spectral variation)
    # n_tex = 20
    # x_texture = cp.zeros(Nr, dtype=complex)
    # for k in range(n_tex):
    #     phase = cp.pi * (k + 1) * (freq - f0) / (B / 2)
    #     amp = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    #     x_texture += amp * cp.exp(1j * phase) / (k + 1)
    # x_texture *= 0.3 / cp.max(cp.abs(x_texture))

    # 4.2  Point targets  (sinusoids in frequency domain)
    n_pts = 100
    tau_pts = cp.linspace(-Tp*1.5, Tp*1.5, n_pts)   # delays [s]   
    # amp_pts = cp.random.normal(0.1, 1, n_pts)
    amp_pts = cp.ones(n_pts)
    # amp_pts[cp.abs(tau_pts) < Tp/2] = 0
    x_sparse = cp.zeros(Nr, dtype=complex)
    for k in range(n_pts):
        x_sparse += amp_pts[k] * cp.exp(-1j * 2 * cp.pi * freq * tau_pts[k])

    x_true = x_sparse

    # 4.3  Observation
    y_clean = P2 * x_true
    SNR_dB = 10
    sig_pow = cp.mean(cp.abs(y_clean)**2)
    noise_power = sig_pow * 10**(-SNR_dB/20)
    noise = (cp.random.randn(*y_clean.shape) + 1j * cp.random.randn(*y_clean.shape)) * noise_power / cp.sqrt(2)

    y = y_clean + noise

    order = 300
    pos = 150
    hy, S = build_annihilating_filter(y, order, pos)
    hy_clean,S_clean = build_annihilating_filter(y_clean, order, pos)
    hx, S_x = build_annihilating_filter(x_true, order, pos)

    plt.figure()
    plt.plot(hy.get(), label = "with noise")
    # plt.plot(hy_clean.get(), label = "clean")
    # plt.plot(h.get(), label = "P2")
    plt.legend()
    plt.xlabel('Filter index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/annihilating_filter.png", dpi=300)


    plt.figure()
    plt.plot(20*np.log10(S.get()), label = "with noise")
    plt.plot(20*np.log10(S_clean.get()), label = "y clean")
    plt.plot(20*np.log10(S_x.get()), label = "x true")
    plt.legend()
    plt.xlabel('Singular value index'); plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/singular_values.png", dpi=300)

    hy_y = cp.convolve(y_clean, hy, mode='full')[M_exclude:-(M_exclude)]
    print("hy_y:{}".format(cp.max(cp.abs(hy_y))))

    hy_noise = cp.convolve(noise, hy, mode='full')[M_exclude:-(M_exclude)]
    print("hy_noise:{}".format(cp.max(cp.abs(hy_noise))))

    x_true = x_true+noise

    ## 注水
    factor = 0.06
    max_p2 = cp.max(P2)
    P2[P2 < max_p2 * factor] = max_p2*factor
    P2[freq < f0 - B/ 2] = max_p2
    P2[freq > f0 + B / 2] = max_p2
    P2 = P2/cp.sqrt(cp.sum(P2**2))

    plt.figure()
    plt.plot(freq.get()/1e9, 20*cp.log10(cp.abs(P2)+1e-30).get(), label='P2')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/P2.png", dpi=300)

    x = solve_c_based_cg(y, P2, hy, lam=0.01, eps=1e-3)

    # P = P2
    # P[P < 4.9e-2] = cp.max(P)
    # P = P/cp.max(P)
    # x_tik = y / P

    # x = x_tik


    # =========================================================================
    # 7.  Visualisation
    # =========================================================================
    plt.rcParams.update({'figure.figsize': (18, 20), 'font.size': 9})

    f_GHz = freq / 1e9

    # (d) Received signal
    plt.figure()
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(y)+1e-30).get(), label='y')
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x)+1e-30).get(), label='x')
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x_true)+1e-30).get(), label='x_true')
    # plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x_tik)+1e-30).get(), label='x_tik')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)

    upsample = 16
    pad_y = cp.zeros(Nr*upsample, dtype=complex)
    pad_y[Nr*8:Nr*9] = y
    pad_x = cp.zeros(Nr*16, dtype=complex)
    pad_x[Nr*8:Nr*9] = x
    pad_x_true = cp.zeros(Nr*16, dtype=complex)
    pad_x_true[Nr*8:Nr*9] = x_true

    y = pad_y
    x = pad_x
    x_true = pad_x_true

    y_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(y)))
    x_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x)))
    x_true_ifft = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(x_true)))

    y_ifft = y_ifft/cp.max(cp.abs(y_ifft))
    x_ifft = x_ifft/cp.max(cp.abs(x_ifft))
    x_true_ifft = x_true_ifft/cp.max(cp.abs(x_true_ifft))


    max_y = cp.max(cp.abs(y_ifft))
    noise_y = cp.sqrt(cp.mean(cp.abs(y_ifft[40000:41000])**2))
    snr_y = 20*cp.log10(max_y/noise_y)

    max_x = cp.max(cp.abs(x_ifft))
    noise_x = cp.sqrt(cp.mean(cp.abs(x_ifft[40000:41000])**2))
    snr_x = 20*cp.log10(max_x/noise_x)



    plt.figure()
    plt.plot(20*cp.log10(cp.abs(y_ifft)+1e-30).get(), label='y_ifft')
    plt.plot(20*cp.log10(cp.abs(x_ifft)+1e-30).get(), label='x_ifft')
    # plt.plot(20*cp.log10(cp.abs(x_true_ifft)+1e-30).get(), label='x_true_ifft')
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/spectrum_recovery_ifft.png", dpi=300)

    max_pos = cp.argmax(cp.abs(x_ifft))
    y_ifft = y_ifft[max_pos-500:max_pos+500]
    x_ifft = x_ifft[max_pos-500:max_pos+500]
    x_true_ifft = x_true_ifft[max_pos-500:max_pos+500]

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
    plt.plot(20*cp.log10(cp.abs(y_ifft)+1e-30).get(), label='y_ifft')
    plt.plot(20*cp.log10(cp.abs(x_ifft)+1e-30).get(), label='x_ifft')
    # plt.plot(20*cp.log10(cp.abs(x_true_ifft)+1e-30).get(), label='x_true_ifft')
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)

    plt.savefig("../fig/spectrum_recovery/spectrum_recovery.png", dpi=300)