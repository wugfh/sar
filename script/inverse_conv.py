import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cupy as cp 
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import scipy

def fast_toeplitz_mult(c, x):
    """
    快速计算实对称 Toeplitz 矩阵 H 与向量 x 的乘积 Hx
    （利用 FFT 实现循环卷积加速，H 的第一列为 c）
    """
    N = len(c)
    c = cp.array(c)
    x = cp.array(x)
    # 构造循环嵌入向量：[c, reversed(c[1:-1])]
    c_ext = cp.concatenate([c, c[-2:0:-1]])
    # 补零 x 至与 c_ext 同长度
    x_ext = cp.concatenate([x, cp.zeros(len(c_ext) - N)])
    # FFT 循环卷积
    conv_circ = cp.fft.ifft(cp.fft.fft(c_ext) * cp.fft.fft(x_ext))
    # 取前 N 个元素（因 c 是实数，结果虚部为机器精度，取实部）
    return conv_circ[:N]


def fast_second_order_diff(x):
    """
    快速计算二阶差分矩阵 L 与向量 x 的乘积 Lx（循环边界条件）
    Lx[i] = x[i-1] - 2x[i] + x[i+1]
    """
    N = len(x)
    x = cp.array(x)
    # 循环移位：x_prev = x[-1, 0, 1, ..., N-2], x_next = x[1, 2, ..., N-1, 0]
    x_prev = cp.roll(x, 1)
    x_next = cp.roll(x, -1)
    return (x_prev - 2 * x + x_next)


def fast_regularized_A_mult(c, x, lam):
    """
    快速计算正则化矩阵 A 与向量 x 的乘积 Ax
    A = W^T W + λ L^T L（W 是实对称 Toeplitz 矩阵，故 W^T = W）
    """
    # 计算 W^T W x = W(Wx)
    Wx = fast_toeplitz_mult(c, x)
    WWx = fast_toeplitz_mult(c, Wx)
    # 计算 L^T L x = L(Lx)（二阶差分的转置等于自身，因 L 是对称循环矩阵）
    Lx = fast_second_order_diff(x)
    LLx = fast_second_order_diff(Lx)
    # 组合正则化项
    return (WWx + lam * LLx).get()


# ------------------------------ 主恢复算法 ------------------------------
def recover_dft_phase(y_obs, c_window, lam, tol=1e-8, max_iter=1000):
    """
    从观测 DFT 恢复原始信号的 DFT（幅度+相位）
    参数:
        y_obs: 观测信号的 DFT（时域加窗后的 DFT）
        c_window: 高斯窗的 DFT（实对称 Toeplitz 矩阵的第一列）
        lam: 二阶差分正则化参数 λ
        tol: CG 收敛阈值
        max_iter: CG 最大迭代次数
    返回:
        x_recon: 恢复的原始信号 DFT
        info: CG 收敛信息（0=成功）
    """
    N = len(y_obs)
    # 构造正则化方程右端项 b = W^T y_obs = W y_obs（因 W 对称）
    b = fast_toeplitz_mult(c_window, y_obs)
    # 定义线性算子 A(x) = fast_regularized_A_mult(c_window, x, lam)
    def _matvec(x):
        return fast_regularized_A_mult(c_window, x, lam)
    
    # 3. 【关键】显式构建 LinearOperator 对象
    A_operator = LinearOperator(
        shape=(N, N),          # 必须指定形状：(N, N)
        matvec=_matvec,         # 矩阵向量乘法函数
        rmatvec=_matvec,        # 对于 Hermitian 矩阵，转置乘法等于自身
        dtype=y_obs.dtype       # 必须指定数据类型：与输入一致的复数类型
    )             # 应为 comple
    # 调用 scipy 的 CG 迭代求解
    x_recon, info = cg(A_operator, b.get(), rtol=tol, maxiter=max_iter)
    return cp.fft.fftshift(x_recon), info


# ====================== 新增：带 L1 稀疏正则的求解器 ======================
def recover_dft_phase_sparse(
    y_obs, 
    c_window, 
    lam1=1e-3,    # 二阶差分正则（防振荡）
    lam2=1e-2,    # L1 稀疏正则（加强稀疏性）
    cg_tol=1e-8,
    out_tol=1e-6, 
    max_iter=1000,
    outer_iter=20 # 外迭代次数（处理L1非光滑项）
):
    """
    新增 L1 稀疏正则化版本
    lam1：二阶差分（平滑、抑震荡）
    lam2：L1 稀疏强度（越大越稀疏）
    """
    N = len(y_obs)
    y_obs = cp.array(y_obs)
    c_window = cp.array(c_window)

    # 1. 固定不变的右端基础项 b = W^T y
    b_base = fast_toeplitz_mult(c_window, y_obs)

    # 2. 初始化 x
    x = cp.zeros(N, dtype=cp.complex128)

    # -------------------- 外层迭代：处理 L1 稀疏项 --------------------
    for k in range(outer_iter):
        # ===== 核心：L1 次梯度 sign(x) （复数域分实部虚部）=====
        sig_real = cp.sign(x.real)
        sig_imag = cp.sign(x.imag)
        sign_x = sig_real + 1j * sig_imag

        # 新的右端项：b = b_base - λ2 * sign(x_old)
        b = b_base - lam2 * sign_x

        # ===== 定义线性算子=====
        def _matvec(x):
            res = fast_regularized_A_mult(c_window, x, lam1)
            return res

        A_op = LinearOperator(
            shape=(N, N),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=y_obs.dtype
        )

        # ===== 内层 CG 求解 =====
        x_np, info = cg(A_op, b.get(), x0=x.get(), rtol=cg_tol, maxiter=max_iter)
        x = cp.array(x_np, dtype=cp.complex128)

        # 收敛判断
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