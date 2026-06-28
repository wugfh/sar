import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cupy as cp 
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import scipy

def fast_toeplitz_mult(c, x):
    """快速计算实对称 Toeplitz 矩阵 H 与向量 x 的乘积 Hx"""
    N = len(c)
    c = cp.array(c)
    x = cp.array(x)
    c_ext = cp.concatenate([c, c[-2:0:-1]])
    x_ext = cp.concatenate([x, cp.zeros(len(c_ext) - N)])
    conv_circ = cp.fft.ifft(cp.fft.fft(c_ext) * cp.fft.fft(x_ext))
    return conv_circ[:N]


def fast_second_order_diff(x):
    """快速计算二阶差分矩阵 L 与向量 x 的乘积 Lx（循环边界条件）"""
    x = cp.array(x)
    x = cp.fft.fft(x)
    x_prev = cp.roll(x, 1)
    x_next = cp.roll(x, -1)
    return (x_prev - 2 * x + x_next)


def fast_regularized_A_mult(c, x, lam):
    """快速计算正则化矩阵 A = W^T W + λ L^T L 与向量 x 的乘积"""
    Wx = fast_toeplitz_mult(c, x)
    WWx = fast_toeplitz_mult(c, Wx)
    Lx = fast_second_order_diff(x)
    LLx = fast_second_order_diff(Lx)
    return (WWx + lam * LLx).get()

def recover_dft_phase(y_obs, c_window, lam, tol=1e-8, max_iter=1000):

    N = len(y_obs)
    b = fast_toeplitz_mult(c_window, y_obs)
    def _matvec(x):
        return fast_regularized_A_mult(c_window, x, lam)
    A_operator = LinearOperator(
        shape=(N, N), matvec=_matvec, rmatvec=_matvec, dtype=y_obs.dtype
    )
    x_recon, info = cg(A_operator, b.get(), rtol=tol, maxiter=max_iter)
    return cp.fft.fftshift(x_recon), info



def fast_toeplitz_mult_batch(X, C_fft, Nr):
    """
    批量 Toeplitz 乘法：对矩阵 X 的每一行应用相同的 Toeplitz 矩阵 H。
    
    参数:
        X: (batch_size, Nr)  输入矩阵，每行是一个向量
        C_fft: (2*Nr-2,)     预计算的扩展核 FFT（可跨批次复用）
        Nr: 原始信号长度
    返回:
        (batch_size, Nr)     H @ X^T 的转置
    """
    batch_size = X.shape[0]
    N_ext = len(C_fft)
    # 补零到循环卷积长度
    X_ext = cp.concatenate(
        [X, cp.zeros((batch_size, N_ext - Nr), dtype=X.dtype)], axis=1
    )
    X_fft = cp.fft.fft(X_ext, axis=1)
    conv = cp.fft.ifft(C_fft[cp.newaxis, :] * X_fft, axis=1)
    return conv[:, :Nr]


def fast_second_order_diff_batch(X):
    """
    批量二阶差分：对矩阵 X 的每一行计算 L @ F @ x（循环边界）。
    
    参数:
        X: (batch_size, N)
    返回:
        (batch_size, N)
    """
    X_prev = cp.roll(X, 1, axis=1)
    X_next = cp.roll(X, -1, axis=1)
    return X_prev - 2 * X + X_next
def apply_A_batch(X, C_fft, Nr, lam):
    """
    批量正则化算子 A = W^T W + λ F^TL^T LF 的矩阵乘法。
    
    参数:
        X: (batch_size, Nr)
        C_fft: (2*Nr-2,) 预计算扩展核 FFT
        Nr: 信号长度
        lam: 正则化参数 λ
    """
    # W @ X
    WX = fast_toeplitz_mult_batch(X, C_fft, Nr)
    # W @ W @ X
    WWX = fast_toeplitz_mult_batch(WX, C_fft, Nr)
    # F @ X
    # X = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(X, axes = 1), axis=1), axes =1)
    # L @ F @ X
    LX = fast_second_order_diff_batch(X)
    # L^H @ L @ F @ X
    LLX = fast_second_order_diff_batch(LX)
    # F^H @ L^H @ L @ F @ X
    # FLLX = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(LLX, axes = 1), axis=1), axes =1)
    return WWX + lam * LLX


def recover_dft_phase_batch(y_obs, c_window, Nr, lam=1e-6, tol=1e-5, max_iter=1000):

    c_ext = cp.concatenate([c_window, c_window[-2:0:-1]])
    C_fft = cp.fft.fft(c_ext)      
    B = fast_toeplitz_mult_batch(y_obs, C_fft, Nr)  # (batch_size, Nr)
    # 初始残差范数平方（用于相对容差判断）
    b_norm_sq = cp.sum(cp.abs(B) ** 2, axis=1).real  # (batch_size,)
    tol_sq = tol ** 2
    # ---- 共轭梯度下降法 ----
    X = cp.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rs_old = cp.sum(cp.abs(R) ** 2, axis=1).real
    info = 0
    for k in range(max_iter):
        AP = apply_A_batch(P, C_fft, Nr, lam)
        pAp = cp.sum(cp.conj(P) * AP, axis=1).real  # (batch_size,)
        alpha = rs_old / cp.maximum(pAp, 1e-30)
        X = X + alpha[:, cp.newaxis] * P
        R = R - alpha[:, cp.newaxis] * AP
        rs_new = cp.sum(cp.abs(R) ** 2, axis=1).real
        # 收敛判断：||r||^2 <= tol^2 * ||b||^2（所有行同时满足才退出）
        if cp.all(rs_new <= tol_sq * b_norm_sq):
            break
        beta = rs_new / cp.maximum(rs_old, 1e-30)
        P = R + beta[:, cp.newaxis] * P
        rs_old = rs_new
        if k == max_iter - 1:
            print("rs:", rs_new/b_norm_sq)
            info = 1  # 未收敛
  
    X = cp.fft.fftshift(X, axes=1)
    return X, info

def recover_dft_phase_sparse(
    y_obs, c_window,
    lam1=1e-3, lam2=1e-2,
    cg_tol=1e-8, out_tol=1e-6,
    max_iter=1000, outer_iter=20
):
    """L1 稀疏正则化版本"""
    N = len(y_obs)
    y_obs = cp.array(y_obs)
    c_window = cp.array(c_window)
    b_base = fast_toeplitz_mult(c_window, y_obs)
    x = cp.zeros(N, dtype=cp.complex128)
    for k in range(outer_iter):
        sig_real = cp.sign(x.real)
        sig_imag = cp.sign(x.imag)
        sign_x = sig_real + 1j * sig_imag
        b = b_base - lam2 * sign_x
        def _matvec(x):
            return fast_regularized_A_mult(c_window, x, lam1)
        A_op = LinearOperator(
            shape=(N, N), matvec=_matvec, rmatvec=_matvec, dtype=y_obs.dtype
        )
        x_np, info = cg(A_op, b.get(), x0=x.get(), rtol=cg_tol, maxiter=max_iter)
        x = cp.array(x_np, dtype=cp.complex128)
        if cp.linalg.norm(x_np - x.get()) < out_tol:
            break
    return cp.fft.fftshift(x), info

# ------------------------------ 测试示例 ------------------------------
if __name__ == "__main__":
    # 1. 参数设置
    N = 20572  # 信号长度
    sigma = 20  # 高斯模糊核的标准差
    noise_level = 0.01  # 观测噪声水平
    lam1 = 1e-6  # 正则化参数 λ（需根据实际情况调整）
    lam2 = 1e-3  # L1 稀疏正则化参数（需根据实际情况调整）
    
    # 2. 生成实对称 Toeplitz 矩阵的第一列 c（高斯模糊核）
    t = cp.arange(N) - N // 2  # 时间索引，中心对齐
    win = cp.exp(-t**2 / (2 * sigma**2))  # 高斯核（对称，对应实对称 Toeplitz 矩阵）
    win_spectrum = cp.real(cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(win))))  # 核的频谱（用于后续分析）
    # 3. 生成原始信号 x_true（阶跃 + 高斯峰）

    forward = cp.array(sio.loadmat("./strip_mode/beam_scan/pos.mat")["forward"].flatten())
    right = cp.array(sio.loadmat("./strip_mode/beam_scan/pos.mat")["right"].flatten())
    down = cp.array(sio.loadmat("./strip_mode/beam_scan/pos.mat")["down"].flatten())
    down = down - cp.mean(down)
    right = right - cp.mean(right)

    PRF = 2500
    eta_c = 0
    R0 = 4295.4
    H = 3000
    f0 = 35e9
    c = 299792458
    Vr = 66
    lambda_ = c/f0
    eta = eta_c + cp.arange(-N/2, N/2, 1)*(1/PRF)
    error_size = forward.shape[0]
    eta_error = eta_c + cp.arange(-error_size/2, error_size/2, 1)*(1/PRF) 
    right = cp.interp(eta, eta_error, right)
    down = cp.interp(eta, eta_error, down)
    down = down - cp.mean(down)
    right = right - cp.mean(right)
    Y = cp.sqrt(R0**2-H**2)

    eta = cp.min(forward) + cp.arange(N)*(Vr/PRF)
    R_eta = cp.sqrt(R0**2 + (Vr*eta)**2)
    R_error = cp.sqrt((right-Y)**2 + (down-H)**2 + (Vr*eta)**2)-cp.sqrt((Vr*eta)**2+R0**2)
    phase = -4*cp.pi*(R_error)/lambda_
    x_true = cp.exp(1j*phase)  # 原始信号（复数形式，包含相位信息）

    x_fft = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(x_true)))  # 原始信号的频谱（用于分析）
    x_win = win * x_fft  # 观测信号

    n = noise_level * cp.random.randn(N)  # 高斯白噪声
    y = x_win + n  # 含噪声观测
    y = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(y)))  # 转回时域（观测信号）
    # 6. 快速共轭梯度法求解
    # x_recon, info = recover_dft_phase(y, win_spectrum, lam1, tol=1e-6, max_iter=1000)
    x_recon, info = recover_dft_phase_sparse(y, win_spectrum, lam1, lam2, cg_tol=1e-6, out_tol=1e-6, max_iter=1000)
    if info == 0:
        print("CG 收敛成功！")
    else:
        print(f"CG 未收敛，info = {info}")

    phase_est = cp.unwrap((cp.angle((x_recon))))  # 恢复信号的相位
    
    # 7. 结果可视化
    plt.figure(figsize=(12, 8))
    
    plt.plot(phase.get(), label="original phase", color="blue")
    plt.plot(cp.unwrap((cp.angle(y))).get(), label="contaminated phase", color="green")
    plt.plot(phase_est.get(), label="estimated phase", color="orange")
   
    plt.legend()
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.show()