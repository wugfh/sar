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

warnings.filterwarnings('ignore')
cp.cuda.Device(1).use()

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
    mu = float(cp.mean(P_sq))          # T. Chan 最优参数
    # ---- 循环预条件子的 FFT 分母 (一次性预计算) ----
    inv_M_denom = 1.0 / (mu + eps + lam * H2)   # 用于 M_circ^{-1}
    # ---- 矩阵向量乘 A @ v (O(N log N)) ----
    def matvec(v):
        out = P_sq * v
        w_f = cp.fft.fft(v)                
        u = cp.fft.ifft(H2 * w_f)        
        out += lam * u           
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
    h_pad[:h_len] = cp.asarray(h[:h_len])

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

def build_annihilating_filter(signal, M, min_pos, suffix):
    N = len(signal)
    H_rows = N-M
    # ---------- 向量化 Hankel：
    idx = cp.arange(H_rows)[:, None] + cp.arange(M + 1)[None, :]   # (H_rows, M+1)
    # idx = (idx-M)*(idx>=M)+idx*(idx<M)  # 防止索引越界
    H_mat = signal[idx]                                   


    U, S, Vh = cp.linalg.svd(H_mat, full_matrices=False)

    Sd = cp.abs(cp.diff(cp.diff(cp.squeeze((S)))))
    pos = 2*cp.argmax(Sd[min_pos:]) +3 + min_pos 
    # pos = min_pos
    print(f"{suffix} Annihilating filter order selected: {pos}")
    h = Vh[pos, :].conj()
    h = h / cp.sqrt(cp.sum(cp.abs(h) ** 2))

    tau = cp.diff(cp.unwrap(cp.angle(Vh), axis=1), axis=1)
    # tau = Vh

    # plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(tau[:,0].get())
    # plt.subplot(4,1,2)
    # plt.plot(tau[:,1].get())
    # plt.subplot(4,1,3)
    # plt.plot(tau[:,2].get())
    # plt.subplot(4,1,4)
    # plt.plot(tau[:,3].get())
    # plt.tight_layout()
    # plt.savefig(f"../fig/spectrum_recovery/annihilating_filter_{suffix}.png", dpi=300)
    return h, S

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
    # Nr       = int(cp.ceil(Fs * Tp*3))   
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


    P_dB = 20 * cp.log10(cp.abs(P2) + 1e-30)

    rng = cp.random.default_rng(42)

    # 4.2  Point targets  (sinusoids in frequency domain)
    n_pts = 100
    tau_pts = cp.linspace(-Tp, Tp, n_pts)   # delays [s]   
    # amp_pts = cp.random.normal(0.01, 1, n_pts)
    amp_pts = cp.ones(n_pts)
    # amp_pts[cp.abs(tau_pts) < Tp/2] = 0
    x_sparse = cp.zeros(Nr, dtype=complex)
    W = cp.abs(freq - f0) < B/2
    for k in range(n_pts):
        x_sparse += amp_pts[k] * cp.exp(-1j * 2 * cp.pi * freq * tau_pts[k])

    x_true = x_sparse

    # 4.3  Observation
    y_clean = P2 * x_true
    SNR_dB = -30
    sig_pow = cp.mean(cp.abs(y_clean)**2)/n_pts
    noise_power = sig_pow * 10**(-SNR_dB/20)
    noise = (cp.random.randn(*y_clean.shape) + 1j * cp.random.randn(*y_clean.shape)) * noise_power / cp.sqrt(2)

    y = y_clean + noise
    x_true = x_true + noise

    # doa = esprit(y, 1500)
    # plt.figure()
    # plt.plot(np.abs(doa), label = "Estimated DOA")
    # plt.grid(alpha=0.3)
    # plt.xlabel('Mode index'); plt.ylabel('Estimated DOA (rad)')
    # plt.savefig("../fig/spectrum_recovery/esprit.png", dpi=300)

    order = 1500
    y = sio.loadmat("../fig/spectrum_recovery/test.mat")["test"]
    y = np.squeeze(y)
    y = cp.array(y)

    original_len = y.shape[0]

    pad_y = cp.zeros(Nr, dtype=complex)
    pad_y[Nr//2-original_len//2:Nr//2+original_len//2] = y
    y = pad_y

    y = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(y)))

    win_len = 320/3000*Nr
    ax = cp.arange(-Nr/2, Nr/2, 1)
    window =  cp.exp(-0.5 * ((ax) / win_len) ** 2)
    P2 = window



    plt.figure()
    plt.plot(20*cp.log10(cp.abs(y)/cp.max(cp.abs(y))).get(), label = "y")
    plt.plot(20*cp.log10(cp.abs(P2)/cp.max(cp.abs(P2))).get(), label = "window")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/y.png", dpi=300)

    start = 0
    hy_new, S = build_annihilating_filter(y, order, start, "y")
    Sy = S/cp.max(S)
    hy = hy_new

    hy = sio.loadmat("../fig/spectrum_recovery/hy_sim.mat")["h"]
    hy = cp.squeeze(cp.array(hy))
    print("hy shape:{}".format(hy.shape))
    # hy = cp.abs(hy_new) * cp.exp(1j * cp.angle(hy))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(cp.abs(hy).get(), label = "hy")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    plt.subplot(2,1,2)
    plt.plot(cp.unwrap(cp.angle(hy)).get(), label = "hy phase")
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Phase (rad)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../fig/spectrum_recovery/hy.png", dpi=300)

    x_res = cp.convolve(x_true, hy, mode='same')
    y_res = cp.convolve(y, hy, mode='same')
    x_down = cp.sqrt(cp.sum(cp.abs(x_res)**2))/cp.sqrt(cp.sum(cp.abs(x_true)**2))
    y_down = cp.sqrt(cp.sum(cp.abs(y_res)**2))/cp.sqrt(cp.sum(cp.abs(y)**2))
    print("x,y down: {}".format(x_down/y_down))
   



    # hy_clean,S_clean = build_annihilating_filter(y_clean, order, 0)
    # hx, S_x = build_annihilating_filter(x_true, order, start, "x_true")

    Sd = cp.abs(cp.diff(cp.diff(S)))
    end_S = cp.minimum(n_pts*5, Sd.shape[0])
    pos = cp.argmax(Sd[start:]) + 2 + start
    plt.figure()
    plt.plot(20*np.log10((Sd[:end_S]).get()), label = "2nd derivative")
    plt.axvline(x=pos.get(), color='r', linestyle='--', label='Selected order')
    plt.legend()
    plt.xlabel('Singular value index'); plt.ylabel('Magnitude (dB)')
    plt.grid(alpha=0.3)
    plt.savefig("../fig/spectrum_recovery/singular_values.png", dpi=300)

    x_true = x_true+noise

    ## 注水
    factor = 0.06
    max_p2 = cp.max(P2)
    P2[P2 < max_p2 * factor] = max_p2*factor
    
    P2 = P2/cp.sqrt(cp.sum(P2**2))
    # P2[freq < f0 - B/ 2] = 10*max_p2
    # P2[freq > f0 + B / 2] = 10*max_p2

    # hy, Sn = find_hy(y, order, P2, n_pts)
    # Sn = cp.array(Sn)
    # plt.figure()
    # plt.plot(Sn.get(), label = "S[n]")
    # max_pos = cp.argmax(Sn)
    # plt.axvline(x=max_pos.get(), color='r', linestyle='--', label='Selected order')
    # plt.legend()
    # plt.xlabel('Order index'); plt.ylabel('S[n]')
    # plt.grid(alpha=0.3)
    # plt.savefig("../fig/spectrum_recovery/Sn.png", dpi=300)

    # sio.savemat("../fig/spectrum_recovery/hy_sim.mat", {"h": hy.get()})


    x = solve_c_based_cg(y, P2, hy, lam=0.1, eps=0) + solve_c_based_cg(y, P2, cp.conj(hy), lam=0.1, eps=0)

    print("x shape:{}".format(x.shape))

    kaise_win = cp.kaiser(Nr, beta=5)
    # x = x * kaise_win

    hx, S = build_annihilating_filter(x, order, start, "x")
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
    plt.subplot(3,1,1)
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(y)/cp.max(cp.abs(y))+1e-30).get(), label='y')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    plt.subplot(3,1,2)
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x)/cp.max(cp.abs(x))+1e-30).get(), label='x')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
    plt.subplot(3,1,3)
    x_true = x_true/cp.max(cp.abs(x_true))
    plt.plot(f_GHz.get(), 20*cp.log10(cp.abs(x_true)+1e-30).get(), label='x_true')
    plt.legend()
    plt.xlabel('Frequency / GHz'); plt.ylabel('Magnitude')
    plt.ylim([-60, 0])
    plt.grid(alpha=0.3)
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
    plt.savefig("../fig/spectrum_recovery/spectrum_recovery_ifft.png", dpi=300)

    max_pos = cp.argmax(cp.abs(y_ifft)) 
    print("max_pos:{}".format(max_pos))
    y_ifft = y_ifft[max_pos-500:max_pos+500]
    x_ifft = x_ifft[max_pos-500:max_pos+500]
    y_ifft = y_ifft/cp.max(cp.abs(y_ifft))
    x_ifft = x_ifft/cp.max(cp.abs(x_ifft))
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
    # print("IRW of x_true_ifft : {:.3f} m".format(irw_x_true))


    plt.figure()
    plt.plot(20*cp.log10(cp.abs(y_ifft)+1e-30).get(), label='y_ifft')
    plt.plot(20*cp.log10(cp.abs(x_ifft)+1e-30).get(), label='x_ifft')
    # plt.plot(20*cp.log10(cp.abs(x_true_ifft)+1e-30).get(), label='x_true_ifft')
    plt.legend()
    plt.xlabel('Sample index'); plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)

    plt.savefig("../fig/spectrum_recovery/spectrum_recovery.png", dpi=300)