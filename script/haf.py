import numpy as np
import math
from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from scipy.special import factorial

def haf_operator(y, p, tau):
    """
    计算信号y的p阶高阶模糊函数（HAF）操作符 P_p[y(n); τ]
    """
    N = len(y)
    P_len = N - (p - 1) * tau
    if P_len <= 0:
        raise ValueError(f"tau={tau}过大或阶数p={p}过高，导致有效序列长度非正。")
    
    P = np.ones(P_len, dtype=complex)
    
    for k in range(p):
        binom_coeff = math.comb(p - 1, k)
        start_idx = (p - 1 - k) * tau
        end_idx = N - k * tau
        
        y_segment = y[start_idx:end_idx]
        
        if k % 2 == 0:
            signal_factor = y_segment
        else:
            signal_factor = np.conj(y_segment)
        
        P *= signal_factor ** binom_coeff
    
    return P

def haf_algorithm(y, M, nfft=None):
    """
    标准HAF算法
    """
    N = len(y)
    if nfft is None:
        nfft = N
    
    t = np.arange(N)
    x = y.copy()
    alpha_hat = np.zeros(M + 1)
    
    for p in range(M, 0, -1):
        tau_p = math.floor(N / p)
        P_p = haf_operator(x, p, tau_p)
        
        P_p_fft = fft(P_p, n=nfft)
        freqs = fftfreq(nfft)
        
        mag = np.abs(P_p_fft)
        peak_idx = np.argmax(mag)
        omega_hat = 2 * np.pi * freqs[peak_idx]
        
        factorial_p = math.factorial(p)
        alpha_hat[p] = omega_hat / (factorial_p * (tau_p ** (p - 1)))
        
        x = x * np.exp(-1j * alpha_hat[p] * (t ** p))
    
    alpha_hat[0] = np.angle(np.mean(x))
    
    return alpha_hat

def polynomial_phase_signal(t, coeffs):
    """
    根据系数生成多项式相位信号
    模型: s(t) = exp(j * Σ_{m=0}^M coeffs[m] * t^m)
    """
    phase = np.zeros_like(t, dtype=float)
    for m, coeff in enumerate(coeffs):
        phase += coeff * (t ** m)
    return np.exp(1j * phase)

def mle_cost_function(coeffs, t, y_observed):
    """
    最大似然估计的成本函数（负对数似然）
    对于复高斯白噪声模型，MLE等价于最小化：
    J(α) = ∑|y(n) - exp(jφ(n;α))|^2
    其中 φ(n;α) = Σ α_m * n^m
    
    参数:
        coeffs: 多项式系数 [α_0, α_1, ..., α_M]
        t: 时间索引数组
        y_observed: 观测到的复数信号
    
    返回:
        残差向量（复数信号的实部和虚部）
    """
    # 生成模型信号
    s_model = polynomial_phase_signal(t, coeffs)
    
    # 计算残差：观测值 - 模型值
    residuals = y_observed - s_model
    
    # 返回实部和虚部拼接的残差向量（用于最小二乘优化）
    return np.concatenate([residuals.real, residuals.imag])

def mle_phase_unwrapping(coeffs, t, y_observed):
    """
    基于相位解缠绕的MLE成本函数
    这种方法直接匹配相位，但需要注意相位缠绕问题
    
    参数:
        coeffs: 多项式系数
        t: 时间索引
        y_observed: 观测信号
    
    返回:
        相位残差
    """
    # 计算观测相位（解缠绕）
    phase_observed = np.unwrap(np.angle(y_observed))
    
    # 计算模型相位
    phase_model = np.zeros_like(t, dtype=float)
    for m, coeff in enumerate(coeffs):
        phase_model += coeff * (t ** m)
    
    # 相位残差
    phase_residual = phase_observed - phase_model
    
    # 将残差映射到[-π, π]区间
    phase_residual = (phase_residual + np.pi) % (2 * np.pi) - np.pi
    
    return phase_residual

def refine_with_mle(y, initial_coeffs, method='least_squares', max_iter=100, tol=1e-8):
    """
    使用最大似然估计（MLE）细化HAF估计结果
    
    参数:
        y: 观测信号
        initial_coeffs: HAF估计的初始系数
        method: 优化方法
            'least_squares': 非线性最小二乘法（推荐）
            'phase_match': 相位匹配法
            'gradient': 梯度下降法
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        refined_coeffs: 细化后的系数
        optimization_result: 优化结果信息
    """
    N = len(y)
    M = len(initial_coeffs) - 1
    t = np.arange(N)
    
    if method == 'least_squares':
        # 方法1: 非线性最小二乘法（最稳健）
        def cost_func(coeffs):
            return mle_cost_function(coeffs, t, y)
        
        # 设置参数边界（可选）
        bounds = [(-np.pi, np.pi)] + [(-10, 10) for _ in range(M)]
        
        # 使用最小二乘法优化
        result = least_squares(
            cost_func, 
            initial_coeffs,
            method='trf',  # 信任域反射法
            max_nfev=max_iter*100,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=0
        )
        
        refined_coeffs = result.x
        success = result.success
        message = result.message
        nfev = result.nfev
        cost = result.cost
        
    elif method == 'phase_match':
        # 方法2: 相位匹配法（对高信噪比有效）
        def phase_cost(coeffs):
            return mle_phase_unwrapping(coeffs, t, y)
        
        result = least_squares(
            phase_cost,
            initial_coeffs,
            method='trf',
            max_nfev=max_iter*100,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=0
        )
        
        refined_coeffs = result.x
        success = result.success
        message = result.message
        nfev = result.nfev
        cost = result.cost
        
    elif method == 'gradient':
        # 方法3: 基于梯度的方法（牛顿法/拟牛顿法）
        def neg_log_likelihood(coeffs):
            """负对数似然函数"""
            s_model = polynomial_phase_signal(t, coeffs)
            residuals = y - s_model
            # 负对数似然（忽略常数项）
            return 0.5 * np.sum(np.abs(residuals) ** 2)
        
        # 使用拟牛顿法
        result = minimize(
            neg_log_likelihood,
            initial_coeffs,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': tol,
                'gtol': tol,
                'disp': False
            }
        )
        
        refined_coeffs = result.x
        success = result.success
        message = result.message
        nfev = result.nfev
        cost = result.fun
        
    else:
        raise ValueError(f"未知的优化方法: {method}")
    
    optimization_result = {
        'success': success,
        'message': message,
        'nfev': nfev,
        'cost': cost,
        'method': method
    }
    
    return refined_coeffs, optimization_result

def iterative_haf_mle(y, M, max_mle_iter=3, nfft=None, verbose=True):
    """
    迭代HAF-MLE算法：交替使用HAF和MLE进行细化
    
    参数:
        y: 观测信号
        M: 多项式阶数
        max_mle_iter: 最大MLE迭代次数
        nfft: FFT点数
        verbose: 是否显示迭代信息
    
    返回:
        final_coeffs: 最终系数估计
        history: 迭代历史记录
    """
    N = len(y)
    history = []
    
    # 步骤1: 初始HAF估计
    haf_coeffs = haf_algorithm(y, M, nfft)
    history.append({
        'iteration': 0,
        'method': 'HAF',
        'coeffs': haf_coeffs.copy(),
        'mse': None
    })
    
    if verbose:
        print(f"迭代 0 (HAF): 初始估计完成")
        print(f"  系数: {haf_coeffs}")
    
    # 步骤2: 迭代MLE细化
    current_coeffs = haf_coeffs.copy()
    
    for iter_idx in range(1, max_mle_iter + 1):
        # MLE细化
        refined_coeffs, opt_result = refine_with_mle(
            y, 
            current_coeffs, 
            method='least_squares',
            max_iter=100,
            tol=1e-10
        )
        
        # 计算改进量
        improvement = np.linalg.norm(refined_coeffs - current_coeffs)
        
        # 更新历史
        history.append({
            'iteration': iter_idx,
            'method': 'MLE',
            'coeffs': refined_coeffs.copy(),
            'improvement': improvement,
            'opt_success': opt_result['success'],
            'opt_cost': opt_result['cost'],
            'opt_nfev': opt_result['nfev']
        })
        
        if verbose:
            print(f"迭代 {iter_idx} (MLE):")
            print(f"  系数: {refined_coeffs}")
            print(f"  改进量: {improvement:.2e}")
            print(f"  优化成本: {opt_result['cost']:.2e}")
            print(f"  函数调用次数: {opt_result['nfev']}")
            print(f"  成功: {opt_result['success']}")
        
        # 检查收敛
        if improvement < 1e-8:
            if verbose:
                print(f"在迭代 {iter_idx} 收敛")
            break
        
        current_coeffs = refined_coeffs.copy()
    
    return current_coeffs, history

def analyze_estimation_performance(y, M, true_coeffs=None, sigma2=None, use_mle=True):
    """
    完整的性能分析：HAF + 可选的MLE细化
    
    参数:
        y: 观测信号
        M: 多项式阶数
        true_coeffs: 真实系数
        sigma2: 噪声方差
        use_mle: 是否使用MLE细化
    
    返回:
        result: 包含所有结果的字典
    """
    N = len(y)
    t = np.arange(N)
    result = {}
    
    # 1. 使用HAF算法估计系数
    haf_coeffs = haf_algorithm(y, M)
    result['haf_estimates'] = haf_coeffs
    result['haf_mse'] = None
    
    # 2. 可选的MLE细化
    if use_mle:
        final_coeffs, mle_history = iterative_haf_mle(y, M, max_mle_iter=3, verbose=False)
        result['mle_estimates'] = final_coeffs
        result['mle_history'] = mle_history
        result['final_estimates'] = final_coeffs
    else:
        result['final_estimates'] = haf_coeffs
    
    # 3. 计算估计误差（如果有真实值）
    if true_coeffs is not None:
        true_array = np.array(true_coeffs)
        
        # HAF误差
        haf_error = haf_coeffs - true_array
        haf_mse = np.mean(haf_error ** 2)
        result['haf_error'] = haf_error
        result['haf_mse'] = haf_mse
        
        # 最终估计误差
        final_error = result['final_estimates'] - true_array
        final_mse = np.mean(final_error ** 2)
        result['final_error'] = final_error
        result['final_mse'] = final_mse
        
        # 改进比例
        if haf_mse > 0:
            improvement_ratio = haf_mse / final_mse
            result['improvement_ratio'] = improvement_ratio
    
    # 4. 生成模型信号用于评估
    if use_mle:
        model_signal = polynomial_phase_signal(t, result['final_estimates'])
    else:
        model_signal = polynomial_phase_signal(t, haf_coeffs)
    
    residuals = y - model_signal
    result['residual_power'] = np.mean(np.abs(residuals) ** 2)
    result['model_signal'] = model_signal
    
    return result

def visualize_comparison(y, M, result, true_coeffs=None):
    """
    可视化HAF和MLE的对比结果
    """
    N = len(y)
    t = np.arange(N)
    
    haf_coeffs = result['haf_estimates']
    final_coeffs = result['final_estimates']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 信号幅度和相位
    ax1 = axes[0, 0]
    ax1.plot(t, np.abs(y), 'b-', alpha=0.7, label='|y(n)|')
    ax1.set_xlabel('Time n')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 系数估计对比
    ax2 = axes[0, 1]
    indices = np.arange(M + 1)
    width = 0.25
    
    ax2.bar(indices - width, haf_coeffs, width, label='HAF Estimate', alpha=0.8, color='blue')
    ax2.bar(indices, final_coeffs, width, label='HAF+MLE Estimate', alpha=0.8, color='red')
    if true_coeffs is not None:
        ax2.bar(indices + width, true_coeffs, width, label='True Value', alpha=0.8, color='green')
    
    ax2.set_xlabel('Coefficient Index m')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Coefficient Comparison (M={M})')
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 相位拟合对比
    ax3 = axes[0, 2]
    phase_original = np.unwrap(np.angle(y))
    ax3.plot(t, phase_original, 'k-', alpha=0.3, label='Original Phase', linewidth=1)
    
    # HAF重构相位
    phase_haf = np.zeros(N)
    for m in range(M + 1):
        phase_haf += haf_coeffs[m] * (t ** m)
    ax3.plot(t, phase_haf, 'b--', label='HAF Phase', linewidth=1.5)
    
    # MLE重构相位
    phase_final = np.zeros(N)
    for m in range(M + 1):
        phase_final += final_coeffs[m] * (t ** m)
    ax3.plot(t, phase_final, 'r-', label='HAF+MLE Phase', linewidth=2)
    
    ax3.set_xlabel('Time n')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase Fitting Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分析
    ax4 = axes[1, 0]
    if true_coeffs is not None and 'haf_error' in result:
        haf_error = np.abs(result['haf_error'])
        final_error = np.abs(result['final_error'])
        
        x = np.arange(len(haf_error))
        ax4.bar(x - 0.2, haf_error, 0.4, label='HAF Error', alpha=0.7, color='blue')
        ax4.bar(x + 0.2, final_error, 0.4, label='HAF+MLE Error', alpha=0.7, color='red')
        
        ax4.set_xlabel('Coefficient Index m')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Estimation Error Comparison')
        ax4.set_xticks(x)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加MSE信息
        if 'haf_mse' in result and 'final_mse' in result:
            ax4.text(0.05, 0.95, f'HAF MSE: {result["haf_mse"]:.2e}\nFinal MSE: {result["final_mse"]:.2e}',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. 残差分析
    ax5 = axes[1, 1]
    if 'model_signal' in result:
        residuals = y - result['model_signal']
        ax5.plot(t, np.abs(residuals), 'g-', alpha=0.7)
        ax5.set_xlabel('Time n')
        ax5.set_ylabel('Residual Magnitude')
        ax5.set_title(f'Residuals (Power: {result["residual_power"]:.2e})')
        ax5.grid(True, alpha=0.3)
    
    # 6. 迭代历史（如果有MLE）
    ax6 = axes[1, 2]
    if 'mle_history' in result:
        history = result['mle_history']
        iterations = [h['iteration'] for h in history]
        
        # 计算每次迭代的MSE（如果有真实值）
        if true_coeffs is not None:
            mse_values = []
            for h in history:
                error = h['coeffs'] - np.array(true_coeffs)
                mse = np.mean(error ** 2)
                mse_values.append(mse)
            
            ax6.plot(iterations, mse_values, 'ro-', linewidth=2, markersize=8)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('MSE')
            ax6.set_title('MSE Convergence')
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        else:
            # 显示成本函数值
            costs = [h.get('opt_cost', 0) for h in history if h['method'] == 'MLE']
            mle_iterations = [h['iteration'] for h in history if h['method'] == 'MLE']
            if costs:
                ax6.plot(mle_iterations, costs, 'bo-', linewidth=2, markersize=8)
                ax6.set_xlabel('MLE Iteration')
                ax6.set_ylabel('Cost Function')
                ax6.set_title('MLE Cost Convergence')
                ax6.grid(True, alpha=0.3)
                ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("../fig/haf/error.png")
    
    # 打印详细结果
    print("=" * 70)
    print("HAF + MLE ITERATIVE REFINEMENT RESULTS")
    print("=" * 70)
    print(f"Signal length N = {N}")
    print(f"Polynomial order M = {M}")
    
    print("\nCoefficient Estimates:")
    print("-" * 70)
    print(f"{'Index m':<8} {'HAF Estimate':<20} {'HAF+MLE Estimate':<20} {'True Value':<20} {'Improvement':<15}")
    print("-" * 70)
    
    for m in range(M + 1):
        true_val = true_coeffs[m] if true_coeffs is not None else np.nan
        haf_val = haf_coeffs[m]
        final_val = final_coeffs[m]
        
        if true_coeffs is not None:
            haf_err = abs(haf_val - true_val)
            final_err = abs(final_val - true_val)
            if haf_err > 0:
                improvement = (haf_err - final_err) / haf_err * 100
            else:
                improvement = 0
            impr_str = f"{improvement:+.1f}%"
        else:
            impr_str = "N/A"
        
        print(f"{m:<8} {haf_val:<20.6e} {final_val:<20.6e} {true_val:<20.6e} {impr_str:<15}")
    
    if 'haf_mse' in result and 'final_mse' in result:
        print(f"\nMean Squared Error:")
        print(f"  HAF only:      {result['haf_mse']:.6e}")
        print(f"  HAF + MLE:     {result['final_mse']:.6e}")
        if 'improvement_ratio' in result:
            print(f"  Improvement:   {result['improvement_ratio']:.2f}x (MSE reduced by {100*(1-1/result['improvement_ratio']):.1f}%)")
    
    if 'residual_power' in result:
        print(f"\nResidual power: {result['residual_power']:.6e}")

# 测试示例
if __name__ == "__main__":
    # 参数设置
    np.random.seed(42)
    N = 512
    M = 2  # 测试高阶情况
    SNR_db = 20  # 中等信噪比
    
    # 生成真实系数（包含高次项）
    true_coeffs = np.array([0.5, 0.1, 0.005])
    # 生成多项式相位信号
    t = np.arange(N)
    phase = np.zeros(N)
    for m in range(M + 1):
        phase += true_coeffs[m] * (t ** m)
    
    z = np.exp(1j * phase)  # 干净信号
    
    # 添加复高斯白噪声
    signal_power = 1.0
    noise_power = signal_power * (10 ** (-SNR_db / 10))
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), N)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), N)
    noise = noise_real + 1j * noise_imag
    y = z + noise
    
    print(f"测试参数: N={N}, M={M}, SNR={SNR_db}dB")
    print(f"真实系数: {true_coeffs}")
    print()
    
    # 执行HAF + MLE估计
    result = analyze_estimation_performance(y, M, true_coeffs, noise_power, use_mle=True)
    
    # 可视化对比结果
    visualize_comparison(y, M, result, true_coeffs)
    
    # 单独测试常数项估计的改进
    print("\n" + "=" * 70)
    print("CONSTANT TERM (α₀) IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    haf_alpha0 = result['haf_estimates'][0]
    mle_alpha0 = result['final_estimates'][0]
    true_alpha0 = true_coeffs[0]
    
    haf_error = abs(haf_alpha0 - true_alpha0)
    mle_error = abs(mle_alpha0 - true_alpha0)
    
    print(f"True α₀:        {true_alpha0:.6f}")
    print(f"HAF estimate:   {haf_alpha0:.6f} (error: {haf_error:.6f})")
    print(f"MLE refined:    {mle_alpha0:.6f} (error: {mle_error:.6f})")
    
    if haf_error > 0:
        improvement = (haf_error - mle_error) / haf_error * 100
        print(f"Improvement:    {improvement:+.1f}%")