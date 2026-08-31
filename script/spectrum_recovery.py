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

def solve_c_based_cg(Y, P, h1,h2, lam1,lam2, eps=1e-8,
                     max_iter=30000, tol=1e-6, warm_start=None):
    N = len(P)
    dtype = Y.dtype

    h_pad1 = cp.zeros(N, dtype=dtype)
    h_pad2 = cp.zeros(N, dtype=dtype)
    h_len1 = min(len(h1), N)
    h_len2 = min(len(h2), N)
    # h_pad[:h_len] = cp.asarray(h[:h_len])
    if h_len1 % 2 == 0:
        h_pad1[N//2 - h_len1//2:N//2 + h_len1//2] = cp.asarray(h1[:h_len1])  # 居中填充
    else:
        h_pad1[N//2 - h_len1//2:N//2 + h_len1//2 + 1] = cp.asarray(h1[:h_len1])  # 居中填充

    if h_len2 % 2 == 0:
        h_pad2[N//2 - h_len2//2:N//2 + h_len2//2] = cp.asarray(h2[:h_len2])  # 居中填充
    else:
        h_pad2[N//2 - h_len2//2:N//2 + h_len2//2 + 1] = cp.asarray(h2[:h_len2])  # 居中填充

    H_f1 = cp.fft.fft(h_pad1)            # C 的特征值
    H_f2 = cp.fft.fft(h_pad2)            # C 的特征值
    H1 = cp.abs(H_f1) ** 2              # C^H·C 的特征值 = |H(ω)|²
    H2 = cp.abs(H_f2) ** 2              # C^H·C 的特征值 = |H(ω)|²
    P_sq = P ** 2
    mu = float(cp.mean(P_sq))          # T. Chan 最优参数
    # ---- 循环预条件子的 FFT 分母 (一次性预计算) ----
    inv_M_denom = 1.0 / (mu + eps + lam1 * H1 + lam2 * H2)   # 用于 M_circ^{-1}
    # ---- 矩阵向量乘 A @ v (O(N log N)) ----
    def matvec(v):
        out = P_sq * v
        w_f = cp.fft.fft(v)                
        u1 = cp.fft.ifft(H1 * w_f) 
        u2 = cp.fft.ifft(H2 * w_f) 
        out += lam1 * u1           
        out += lam2 * u2
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

    M, N = Y.shape
    dtype = Y.dtype

    # ---- 预计算（所有行共用） ----
    h_pad = cp.zeros(N, dtype=dtype)
    h_len = min(len(h), N)
    if h_len % 2 == 0:
        h_pad[N//2 - h_len//2:N//2 + h_len//2] = cp.asarray(h[:h_len])  # 居中填充
    else:
        h_pad[N//2 - h_len//2:N//2 + h_len//2 + 1] = cp.asarray(h[:h_len])  # 居中填充

    H_f = cp.fft.fft(h_pad)                         # (N,)
    H2 = cp.abs(H_f) ** 2                            # (N,)  |H(ω)|²
    h_norm_sq = float(cp.sum(cp.abs(h_pad) ** 2))    # ||h||²

    P_sq = P ** 2                                     # (N,)
    mu = float(cp.mean(P_sq))          # T. Chan 最优参数
    # ---- 循环预条件子的 FFT 分母 (一次性预计算) ----
    inv_M_denom = 1.0 / (mu + eps + lam * H2)   # 用于 M_circ^{-1}

    # ---- 批量矩阵向量乘 A @ v  (M,N) -> (M,N) ----
    def matvec(v):
        out = P_sq * v                                
        w_f = cp.fft.fft(v, axis=1)                 
        u = cp.fft.ifft(H2 * w_f, axis=1)           
        out += lam * u
        out += eps * v 
        return out

    # ---- 预条件子 M⁻¹ @ v ----
    def precond(v):
        v_f = cp.fft.fft(v)
        return cp.fft.ifft(inv_M_denom * v_f)           # (M,N)

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

    return h1, h2, S

def build_annihilating_filter_fast(signal, M):
    N = len(signal)
    signal_pad = cp.zeros(N + M, dtype=complex)
    off = (N + M) // 2 - N // 2
    signal_pad[off:off + N] = signal
    # 零拷贝 Hankel 视图（省掉 fancy-indexing 的大索引数组和 gather）
    H = cp.lib.stride_tricks.sliding_window_view(signal_pad, M + 1)   # (N, M+1)
    # 若 cupy 版本较老，可改用：
    # H = cp.lib.stride_tricks.as_strided(
    #     signal_pad, shape=(N, M + 1),
    #     strides=(signal_pad.strides[0], signal_pad.strides[0]))
    # H^H H: 只算小矩阵，cuBLAS GEMM
    G = H.conj().T @ H                      # (M+1, M+1)
    w, V = cp.linalg.eigh(G)                # 升序特征值（= σ²）
    h1 = V[:, 0].conj()                     # 等价于 Vh[-1]，注意共轭
    h2 = V[:, 1].conj()                     # 等价于 Vh[-2]
    S = cp.sqrt(cp.maximum(w[::-1], 0.0))   # 奇异值降序；clip 防止极小负值
    return h1, h2, S

def cut_singular(signal, M):
    N = len(signal)
    signal_pad = cp.zeros(N+M, dtype=complex)
    signal_pad[(N+M)//2-N//2:(N+M)//2+N//2] = signal
    H_rows = N
    # ---------- 向量化 Hankel：
    idx = cp.arange(H_rows)[:, None] + cp.arange(M + 1)[None, :]   # (H_rows, M+1)
    # idx = (idx-M)*(idx>=M)+idx*(idx<M)  # 防止索引越界
    H_mat = signal_pad[idx]                                   


    U, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)

    Sd = cp.abs(cp.diff(cp.diff(cp.squeeze(((S))))))
    # pos = cp.argmax(Sd) + 10
    pos = 105
    # pos = 500
    # pos = min_pos
    S[pos:] = 0
    H_new = U @ cp.diag(S) @ Vh
    x = H_new[:,0]
    print("x shape:", x.shape)
    return x


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
    Nr       = int(cp.ceil(Fs * Tp*3))   
    Nr       = 3000
    print("Nr:{}".format(Nr))
    Tr       = Nr / Fs                     # pulse duration [s]
    df       = Fs / Nr                      # frequency resolution
    freq     = f0 + (cp.arange(Nr) - Nr // 2) * df   # frequency axis [Hz]
    N_inband = int(cp.ceil(B / df))         # samples inside bandwidth
    

    win_len = 320/3000*Nr
    ax = cp.arange(-Nr/2, Nr/2, 1)
    # shift = Nr//4
    shift = 0
    window =  cp.exp(-0.5 * ((ax+shift) / win_len) ** 2)
    P2 = window
    P2 = P2/cp.sqrt(cp.sum(P2**2)) 
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
    n_pts = 400
    # tau_pts = cp.linspace(-Tp/2, Tp/2, n_pts)   # delays [s]   
    tau_pts = cp.random.uniform(-Tp/2, Tp/2, n_pts)
    # amp_pts = 10**cp.random.normal(0.0, 1, n_pts)
    amp_pts = cp.ones(n_pts)
    # amp_pts[cp.abs(tau_pts) < Tp/2] = 0
    x_sparse = cp.zeros(Nr, dtype=complex)
    W = cp.abs(freq - f0) < B/2
    roots = []
    for k in range(n_pts):
        x_sparse += amp_pts[k] * cp.exp(-1j * 2 * cp.pi * freq * tau_pts[k])
        roots.append(cp.exp(-1j * 2 * cp.pi * tau_pts[k] * Fs/Nr))

    x_true = x_sparse

    # 4.3  Observation
    y_clean = P2 * x_true
    SNR_dB = -30
    sig_pow = cp.mean(cp.abs(y_clean)**2)/n_pts
    noise_power = sig_pow * 10**(-SNR_dB/20)
    noise = (cp.random.randn(*y_clean.shape) + 1j * cp.random.randn(*y_clean.shape)) * noise_power / cp.sqrt(2)

    y = y_clean + noise
    x_true = x_true + noise



    order = Nr//2
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

    start = 0
    hy1, hy2, S = build_annihilating_filter(y, order)
    Sy = S/cp.max(S)

    hy_order = order

    # hy = sio.loadmat("../fig/spectrum_recovery/hy_sim.mat")["h"]
    # hy1 = cp.array(np.squeeze(hy))
    # hy2 = hy1
    # hy = cp.squeeze(cp.array(hy))[:-1]
    # angle = cp.angle(hy)
    # dt = cp.mean(cp.unwrap(cp.diff(angle)) / (2 * cp.pi * (Fs/hy_order)))


    # hy = hy/cp.sqrt(cp.sum(cp.abs(hy)**2))
    dt = -0.5/(Fs/hy_order)
    # dt = -0.325/(Fs/hy_order)
    print("dt:{}".format(dt*(Fs/hy_order)))
    fs = cp.arange(-hy_order//2, hy_order//2) * (Fs/hy_order)
    angle = 2*cp.pi*fs*dt
    # hy = cp.kaiser(hy_order, beta=5)

    hy = cp.ones(hy_order, dtype=complex) * cp.exp(1j*angle)
    # hy2 = hy/cp.sqrt(cp.sum(cp.abs(hy)**2))
    # hy1 = hy2
    # hy1 = cp.abs(hy1[:-1])*cp.exp(1j*angle)
    # hy2 = cp.abs(hy2[:-1])*cp.exp(1j*angle)
    # hy = hy *(cp.abs(fs) < B/2)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(cp.abs(hy1).get(), label = "hy1")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.subplot(2,1,2)
    plt.plot(cp.abs(hy2).get(), label = "hy2")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../fig/spectrum_recovery/hy.png", dpi=300)

    x_res1 = cp.convolve(x_true, hy1, mode='same')
    x_res2 = cp.convolve(x_true, hy2, mode='same')
    n_res1 = cp.convolve(noise, hy1, mode='same')
    n_res2 = cp.convolve(noise, hy2, mode='same')
    x_down1 = cp.sqrt(cp.sum(cp.abs(x_res1)**2))/cp.sqrt(cp.sum(cp.abs(n_res1)**2))
    x_down2 = cp.sqrt(cp.sum(cp.abs(x_res2)**2))/cp.sqrt(cp.sum(cp.abs(n_res2)**2))
    pre_down = cp.sqrt(cp.sum(cp.abs(x_true)**2))/cp.sqrt(cp.sum(cp.abs(noise)**2))
    print("x1 down: {}  x2 down: {}, pre_down: {}".format(x_down1, x_down2, pre_down))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(cp.abs(x_down1+n_res1).get(), label = "x_res1")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(cp.abs(x_down2+n_res2).get(), label = "x_res2")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid()
    plt.tight_layout()
    plt.savefig("../fig/spectrum_recovery/x_res.png", dpi=300)
   

    Sd = cp.abs(cp.diff(cp.diff(S)))
    end_S = cp.minimum(n_pts*5, Sd.shape[0])
    pos = cp.argmax(Sd[start:]) + 2 + start

    # ---------- 第二幅大图（主图） ----------
    fig = plt.figure()
    ax_main = fig.add_subplot(111)                      # 主坐标轴
    ax_main.plot(20*np.log10(S.get()), label="singular values of y")
    # ax_main.legend()
    ax_main.set_xlabel('Singular value index')
    ax_main.set_ylabel('Magnitude (dB)')
    ax_main.grid(alpha=0.3)
    # ---------- 第一幅小图，嵌入主图右上角 ----------
    # [left, bottom, width, height] 均为相对整个 figure 的比例
    ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.30])
    ax_inset.plot(20*np.log10(S[:end_S].get()), label="y")
    ax_inset.legend(fontsize=7)
    ax_inset.set_xlabel('Singular value index', fontsize=7)
    ax_inset.set_ylabel('Magnitude (dB)', fontsize=7)
    ax_inset.grid(alpha=0.3)
    ax_inset.tick_params(labelsize=6)                   # 缩小刻度字体
    plt.savefig("../fig/spectrum_recovery/singular_values_inset.png", dpi=300)


    x_true = x_true+noise

    ## 注水
    # factor = 1e-2
    # max_p2 = cp.max(P2)
    # P2[P2 < max_p2 * factor] = max_p2*factor
    
    P2 = P2/cp.sqrt(cp.sum(P2**2))
    # P2[freq < f0 - B/ 2] = 10*max_p2
    # P2[freq > f0 + B / 2] = 10*max_p2
    line = cp.ones(Nr)
    line = line/cp.sqrt(cp.sum(line**2))


    sio.savemat("../fig/spectrum_recovery/hy_sim.mat", {"h": hy1.get()})
    lam1 = 1e-4
    lam2 = 1e-4
    x = solve_c_based_cg(y, P2, hy1, hy2, lam1=lam1, lam2=lam2, eps=0)
    # lam = 1/noise.std()/100
    # y_ifft = cp.fft.ifft(y)
    # p = cp.fft.ifft(P2)
    # h_pad = cp.zeros(Nr, dtype=complex)
    # h_pad[Nr//2-hy_order//2:Nr//2+hy_order//2] = hy
    # h_pad = cp.fft.ifft(h_pad)
    # h_pad = cp.abs(h_pad)**2
    # x, info= tv.solve_ew_l1_regularized(h_pad, p, y_ifft, lam, tol=1e-6,  verbose=True)
    # print("deconvolution info:", info)
    # x = cp.fft.fft(x)
    # print("x shape:{}".format(x.shape))
    # x = cut_singular(y, order)

    kaise_win = cp.kaiser(Nr, beta=5)
    # x = x * kaise_win

    hx, hx2, S = build_annihilating_filter(x, order)
    Sd = cp.abs(cp.diff(cp.diff(S)))
    plt.figure()
    Sx = S/cp.max(S)
    plt.plot(20*np.log10((Sx).get()), label = "singular values of x")
    plt.plot(20*np.log10((Sy).get()), label = "singular values of y")
    # plt.plot(20*np.log10((S_x[end_S]).get()), label = "singular values of x_true")
    # plt.plot(20*np.log10((S_clean).get()), label = "singular values of y_clean")
    plt.legend()
    plt.xlabel('Singular value index'); plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/singular_values_x.png", dpi=300)
 
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