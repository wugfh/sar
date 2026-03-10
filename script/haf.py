import numpy as np
import math
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
# matplotlib.rcParams['axes.unicode_minus'] = False # 正常显示负号

def haf_operator(y, p, tau):
    """
    计算信号y的p阶高阶模糊函数（HAF）操作符 P_p[y(n); τ]
    对应文献公式(4)：P_p[y(n); τ] ≜ ∏_{k=0}^{p-1} [y^(*k)(n + (p-1-k)τ)]^C(p-1, k)
    
    参数:
        y: 输入信号（复数数组）
        p: HAF的阶数
        tau: 延迟参数（正整数）
    
    返回:
        P: p阶HAF操作符的输出序列
    """
    N = len(y)
    # 输出序列的长度为 N - (p-1)*tau
    P_len = N - (p - 1) * tau
    if P_len <= 0:
        raise ValueError(f"tau={tau}过大或阶数p={p}过高，导致有效序列长度非正。")
    
    P = np.ones(P_len, dtype=complex)
    
    for k in range(p):
        # 计算二项式系数 C(p-1, k)
        binom_coeff = math.comb(p - 1, k)
        # 计算信号索引：n + (p-1-k)*tau
        start_idx = (p - 1 - k) * tau
        end_idx = N - k * tau
        
        # 获取信号段
        y_segment = y[start_idx:end_idx]
        
        # 根据k的奇偶性决定取原信号还是共轭信号 (公式5)
        if k % 2 == 0:  # k为偶数
            signal_factor = y_segment
        else:  # k为奇数
            signal_factor = np.conj(y_segment)
        
        # 应用二项式系数次幂
        P *= signal_factor ** binom_coeff
    
    return P

def haf_algorithm(y, M, nfft=None):
    """
    实现文献中描述的标准HAF算法（第I节）
    算法步骤：
    1) 设 p = M, x(n) = y(n)
    2) 设 τ_p = floor(N / p) [公式(6)]
    3) 估计 α_p [公式(7)]
    4) 去调频 [公式(8)]，p = p - 1，若p>0则回到第2步
    5) 估计 α_0 [公式(9)]
    
    参数:
        y: 观测信号 (y(n) = z(n) + w(n))
        M: 多项式相位的阶数
        nfft: FFT点数（可选，用于提高频率分辨率）
    
    返回:
        alpha_hat: 估计的系数列表 [α_0, α_1, ..., α_M]
        errors: 各阶系数估计的渐进方差（如果可计算）
    """
    N = len(y)
    if nfft is None:
        nfft = N
    
    t = np.arange(N)
    x = y.copy()  # 初始信号
    alpha_hat = np.zeros(M + 1)  # 存储估计的系数
    
    # 步骤1-4：从高阶到低阶递归估计
    for p in range(M, 0, -1):
        # 步骤2：设置延迟参数 τ_p
        tau_p = math.floor(N / p)  # 文献公式(6)的选择
        
        # 步骤3：计算p阶HAF并估计α_p
        P_p = haf_operator(x, p, tau_p)
        
        # 计算HAF的DFT（即高阶模糊函数P_p[y; ω, τ]）
        P_p_fft = fft(P_p, n=nfft)
        freqs = fftfreq(nfft)
        
        # 找到最大幅度对应的频率 ω_hat
        mag = np.abs(P_p_fft)
        peak_idx = np.argmax(mag)
        omega_hat = 2 * np.pi * freqs[peak_idx]  # 转换为角频率
        
        # 估计 α_p [公式(7)]
        # 注意：文献中 ω ∈ [-π, π]，对应数字角频率
        # 我们的频率freqs范围是[-0.5, 0.5)，需要乘以2π
        factorial_p = math.factorial(p)
        alpha_hat[p] = omega_hat / (factorial_p * (tau_p ** (p - 1)))
        
        # 步骤4：去调频
        x = x * np.exp(-1j * alpha_hat[p] * (t ** p))
    
    # 步骤5：估计常数项 α_0 [公式(9)]
    alpha_hat[0] = np.imag(np.log(np.mean(x)))
    
    return alpha_hat

def compute_crb(N, M, sigma2=1.0):
    """
    计算多项式相位信号参数估计的Cramér-Rao下界（CRB）
    基于文献第IV节公式(58)-(59)
    
    参数:
        N: 信号长度
        M: 多项式阶数
        sigma2: 噪声方差（默认信号幅度A=1，因此σ^2即噪声功率）
    
    返回:
        CRB_matrix: (M+1)×(M+1)的CRB矩阵
        CRB_diag: 各参数方差下界的对角线元素 [CRB(α_0), ..., CRB(α_M)]
    """
    # 构建矩阵C [公式(59)]
    C = np.zeros((M + 1, M + 1))
    for p in range(M + 1):
        for q in range(M + 1):
            if (p + q) % 2 == 0:  # p+q为偶数
                factor = ((-1) ** (p + q) * (M + p + 1) * (M + q + 1) /
                          (2 * (p + q + 1)))
                binom1 = math.comb(M + p, p) * math.comb(M, p)
                binom2 = math.comb(M + q, q) * math.comb(M, q)
                C[p, q] = factor * binom1 * binom2
            else:
                C[p, q] = 0.0
    
    # 根据文献公式(58): lim_{N→∞} D * CRB * D = σ^2 * C
    # 其中 D = diag(N^{0.5}, N^{1.5}, ..., N^{M+0.5})
    D = np.diag([N ** (m + 0.5) for m in range(M + 1)])
    D_inv = np.linalg.inv(D)
    
    # 有限N下的渐进CRB矩阵
    CRB_matrix = sigma2 * D_inv @ C @ D_inv.T
    
    return CRB_matrix, np.diag(CRB_matrix)

def compute_haf_variance(N, M, sigma2, alpha_hat):
    """
    计算HAF估计的渐进方差（基于文献第III-IV节分析）
    注意：这是简化实现，完整实现需要计算复杂的Ξ和E{εε^T}矩阵
    
    参数:
        N: 信号长度
        M: 多项式阶数
        sigma2: 噪声方差
        alpha_hat: HAF估计的系数
    
    返回:
        var_approx: 各系数渐进方差的近似
        are: 渐进相对效率（相对于CRB）
    """
    # 简化估计：使用文献中M=2的显式公式(61)作为示例
    # 对于一般M，需要实现文献中第III节的完整矩阵计算
    var_approx = np.zeros(M + 1)
    are = np.zeros(M + 1)
    
    if M == 2:
        # 文献公式(61)给出的具体表达式
        var_approx[0] = (28/27 + (8/27)*sigma2) / (N**1)  # 注意阶次
        var_approx[1] = (17/16 + 0.5*sigma2) / (N**3)
        var_approx[2] = (16/15 + (8/15)*sigma2) / (N**5)
        
        # 计算对应的CRB（简化）
        _, crb_diag = compute_crb(N, M, sigma2)
        
        # 计算ARE
        for m in range(M + 1):
            if crb_diag[m] > 0:
                are[m] = var_approx[m] / crb_diag[m]
    
    return var_approx, are

def analyze_estimation_performance(y, M, true_coeffs=None, sigma2=None):
    """
    完整的性能分析：执行HAF估计并分析结果
    
    参数:
        y: 观测信号
        M: 多项式阶数
        true_coeffs: 真实系数（如有，用于计算误差）
        sigma2: 噪声方差估计（如有）
    
    返回:
        result: 包含所有结果的字典
    """
    N = len(y)
    result = {}
    
    # 1. 使用HAF算法估计系数
    alpha_hat = haf_algorithm(y, M)
    result['estimates'] = alpha_hat
    
    # 2. 计算估计误差（如果有真实值）
    if true_coeffs is not None:
        true_array = np.array(true_coeffs)
        error = alpha_hat - true_array
        mse = np.mean(error ** 2)
        result['error'] = error
        result['mse'] = mse
    
    # 3. 计算CRB（如果知道噪声方差）
    if sigma2 is not None:
        CRB_matrix, CRB_diag = compute_crb(N, M, sigma2)
        result['CRB'] = CRB_diag
        
        # 4. 计算HAF的渐进方差和ARE（简化版）
        var_approx, are = compute_haf_variance(N, M, sigma2, alpha_hat)
        result['variance_approx'] = var_approx
        result['ARE'] = are
    
    return result

def visualize_performance(y, M, result, true_coeffs=None):
    """
    可视化HAF估计性能
    
    参数:
        y: 原始信号
        M: 多项式阶数
        result: analyze_estimation_performance的输出结果
        true_coeffs: 真实系数
    """
    N = len(y)
    t = np.arange(N)
    alpha_hat = result['estimates']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 信号幅度和相位
    ax1 = axes[0, 0]
    ax1.plot(t, np.abs(y), 'b-', alpha=0.7, label='|y(n)|')
    ax1.set_xlabel('Time n')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax1b = ax1.twinx()
    phase = np.unwrap(np.angle(y))
    ax1b.plot(t, phase, 'r-', alpha=0.5, linewidth=0.5, label='Phase (unwrapped)')
    ax1b.set_ylabel('Phase (rad)', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    # 2. 系数估计对比
    ax2 = axes[0, 1]
    indices = np.arange(M + 1)
    width = 0.35
    
    ax2.bar(indices - width/2, alpha_hat, width, label='HAF Estimate', alpha=0.8)
    if true_coeffs is not None:
        ax2.bar(indices + width/2, true_coeffs, width, label='True Value', alpha=0.8)
    
    ax2.set_xlabel('Coefficient Index m')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Polynomial Coefficients (M={M})')
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 相位拟合对比
    ax3 = axes[1, 0]
    # 原始相位
    phase_original = np.unwrap(np.angle(y))
    ax3.plot(t, phase_original, 'b-', alpha=0.5, label='Original Phase')
    
    # 重构相位
    phase_recon = np.zeros(N)
    for m in range(M + 1):
        phase_recon += alpha_hat[m] * (t ** m)
    ax3.plot(t, phase_recon, 'r-', label='Reconstructed Phase', linewidth=2)
    
    ax3.set_xlabel('Time n')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase Fitting Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分析
    ax4 = axes[1, 1]
    if true_coeffs is not None and 'error' in result:
        error = result['error']
        ax4.bar(indices, np.abs(error), alpha=0.7, color='g')
        ax4.set_xlabel('Coefficient Index m')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Estimation Error')
        ax4.set_xticks(indices)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 如果有CRB，添加为参考线
        if 'CRB' in result:
            crb_std = np.sqrt(result['CRB'])
            ax4.plot(indices, crb_std, 'ro-', label='CRB Std', markersize=8)
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No true coefficients provided\nfor error analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Error Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("=" * 60)
    print("HAF ALGORITHM PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Signal length N = {N}")
    print(f"Polynomial order M = {M}")
    print("\nEstimated Coefficients:")
    print("-" * 40)
    print(f"{'Index m':<8} {'Estimate α_m':<20} {'True Value':<20} {'Error':<15}")
    print("-" * 40)
    
    for m in range(M + 1):
        true_val = true_coeffs[m] if true_coeffs is not None else np.nan
        error = alpha_hat[m] - true_val if true_coeffs is not None else np.nan
        print(f"{m:<8} {alpha_hat[m]:<20.6e} {true_val:<20.6e} {error:<15.6e}")
    
    if 'mse' in result:
        print(f"\nMean Squared Error: {result['mse']:.6e}")
    
    if 'ARE' in result:
        print("\nAsymptotic Relative Efficiency (ARE):")
        for m in range(M + 1):
            if not np.isnan(result['ARE'][m]):
                print(f"  ARE(α_{m}) = {result['ARE'][m]:.4f}")

# 使用示例
if __name__ == "__main__":
    # 参数设置
    np.random.seed(42)
    N = 512
    M = 3
    SNR_db = 20  # 信噪比
    
    # 生成真实系数
    true_coeffs = np.array([0.1, 0.5, 0.001, 0])
    
    # 生成多项式相位信号
    t = np.arange(N)
    phase = np.zeros(N)
    for m in range(M + 1):
        phase += true_coeffs[m] * (t ** m)
    
    z = np.exp(1j * phase)  # 干净信号
    
    # 添加复高斯白噪声
    signal_power = np.mean(np.abs(z) ** 2)
    noise_power = signal_power * (10 ** (-SNR_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power/2), N) + 1j * np.random.normal(0, np.sqrt(noise_power/2), N)
    y = z + noise
    
    # 执行HAF估计和性能分析
    result = analyze_estimation_performance(y, M, true_coeffs, noise_power)
    
    # 可视化结果
    visualize_performance(y, M, result, true_coeffs)
    
    # 打印高信噪比ARE参考值（来自文献表I）
    print("\n" + "=" * 60)
    print("REFERENCE: High-SNR ARE from Paper (Table I)")
    print("=" * 60)
    print("For M=3, sqrt[ARE(α_m, 0)] values are approximately:")
    print("  m=0: 1.13, m=1: 1.22, m=2: 1.24, m=3: 1.25")
    print("\nNote: ARE = 1 corresponds to CRB (optimal performance)")
    print("      ARE > 1 indicates performance loss relative to CRB")