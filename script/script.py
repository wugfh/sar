"""
=============================================================================
 求解 min_X ||Y - diag(P)X||^2 + λ ||C·diag(X)·P||^2
 基于 fscan.py 的 15 阵元波导缝隙天线参数
=============================================================================

推导:
  C·diag(X)·P = C·(X⊙P) = C·diag(P)·X              (⊙ 为 Hadamard 积)
  
  目标函数: J(X) = ||Y - D_p X||^2 + λ ||C D_p X||^2
  
  梯度 → 正规方程:
    (D_p^H D_p + λ D_p^H C^H C D_p + εI) X = D_p^H Y
  
  解: X = A^{-1} b, 其中
    A = diag(|P|^2) + λ D_p^H C^H C D_p + εI
    b = diag(P)^H Y

关键性质 (Cp=0):
  C D_p 1 = C p = 0  →  平坦信号在正则化项中"不可见"
  →  λ 可以任意大而不损失平坦信号分量
  →  噪声被选择性抑制 (投影到 C 的值域中衰减)
=============================================================================
"""

import numpy as np
from scipy.linalg import solve, svd, norm
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 12, 'figure.dpi': 120})

# ===========================================================================
# 1. 参数设置 (来自 fscan.py)
# ===========================================================================
f0   = 35e9                           # 中心频率 [Hz]
B    = 2e9                            # 信号带宽 [Hz]
Fs   = B * 1.2                        # 采样率 [Hz]
c0   = 3e8                            # 光速 [m/s]
a_wg = 0.004871409163516              # 波导宽边 [m]
N_elem = 15                           # 阵元数
Nf   = 512                            # 频率采样点数

# 天线间距
lam0   = c0 / f0
lam_g0 = lam0 / np.sqrt(1 - (lam0 / (2 * a_wg))**2)
shift  = 2 / 3.717054305989132
d_el   = lam_g0 / 2 + shift * lam_g0  # 阵元间距

# 波束指向
beta = np.deg2rad(45)                 # 天线安装角
phi  = beta + np.deg2rad(14.3)        # 条带中心视角
doa  = np.deg2rad(14.3)               # 相对法向的到达角

# 频率网格
f_vec = f0 + np.arange(-Nf//2, Nf//2) * (Fs / Nf)

# ===========================================================================
# 2. 天线方向图 P(f) (来自 fscan.py 的公式)
# ===========================================================================
lam   = c0 / f_vec
lam_g = lam / np.sqrt(1 - (lam / (2 * a_wg))**2)

# u₁ 参数 (波导缝隙阵的相位项)
u1 = (2 * np.pi / lam * d_el * np.sin(doa)
      - 2 * np.pi / lam_g * (d_el - lam_g / 2))

# 双程功率方向图 (15阵元均匀加权)
p_1way = np.sin(N_elem * u1 / 2) / (N_elem * np.sin(u1 / 2))
P = p_1way ** 2                     # |p(f)|², 实向量, Nf 点

print("P 统计: min={:.4e}, max={:.4f}, mean={:.4f}".format(
    P.min(), P.max(), P.mean()))

# ===========================================================================
# 3. 构造零化矩阵 C (rank = Nf/2)
# ===========================================================================
# p_2way(u) ∈ span{e^{j k u₁} : k = -14,...,14} → 29 维子空间
m_vals = np.arange(-14, 15)         # k = -14, -13, ..., 14
V_fourier = np.exp(1j * np.outer(u1, m_vals))  # Nf × 29, 列即 Fourier 基

# SVD: V = U Σ Vh, 零空间 = U[:, 29:]
U_full, S_full, Vh_full = svd(V_fourier, full_matrices=True)
C_full = U_full[:, 29:].T.conj()    # (Nf-29) × Nf, 满足 C_full @ P ≈ 0

# 截断到 rank = Nf/2 = 256
rank_C = Nf // 2
C = C_full[:rank_C, :]              # 256 × Nf
print(f"C 形状: {C.shape}, rank = {np.linalg.matrix_rank(C)}")
print(f"||C·P|| / ||P|| = {norm(C @ P) / norm(P):.2e}  (应为 ~机器精度)")

# ===========================================================================
# 4. 核心求解器
# ===========================================================================

def solve_tikhonov(Y, P, lam):
    """
    Tikhonov 正则化 (逐元素闭式解)
    min ||Y - diag(P)X||² + λ||X||²
    →  X_k = P_k·Y_k / (P_k² + λ)
    """
    return P * Y / (P**2 + lam)

def solve_c_based_direct(Y, P, C, lam, eps=1e-8):
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
    CTC = C.T @ C                    # Nf × Nf, 实对称
    A = np.diag(P**2) + lam * (Dp @ CTC @ Dp) + eps * np.eye(N)
    
    # 右端项
    b = Dp @ Y                       # = P ⊙ Y (逐元素)
    
    # 求解 (利用对称正定性)
    X = solve(A, b, assume_a='pos')  # Cholesky
    return X

def solve_c_based_cg(Y, P, C, lam, eps=1e-8, max_iter=500, tol=1e-8):
    """
    C-based 正则化 (CG 迭代, 适用于大 N)
    """
    N = len(P)
    Dp = np.diag(P)
    CTC = C.T @ C
    A_mat = np.diag(P**2) + lam * (Dp @ CTC @ Dp) + eps * np.eye(N)
    b = Dp @ Y
    
    # 用 LinearOperator 避免显式存储大矩阵
    def matvec(x):
        return (P**2) * x + lam * P * (CTC @ (P * x)) + eps * x
    
    A_op = LinearOperator((N, N), matvec=matvec)
    X, info = cg(A_op, b, tol=tol, maxiter=max_iter)
    if info != 0:
        print(f"  CG 警告: 未完全收敛 (info={info}), 迭代 {max_iter}")
    return X

# ===========================================================================
# 5. 性能指标计算
# ===========================================================================

def compute_irw_from_response(g_eff):
    """
    从有效频率响应 g_eff 计算 IRW [cm]
    g_eff = H @ P  (平坦信号通过滤波器的输出)
    """
    N = len(g_eff)
    irf = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(g_eff)))
    irf_pow = np.abs(irf)**2
    peak = np.argmax(irf_pow)
    half = irf_pow[peak] / 2
    
    # 左 -3dB 点
    left = peak
    while left > 0 and irf_pow[left] > half:
        left -= 1
    if left > 0:
        frac = (half - irf_pow[left]) / (irf_pow[left+1] - irf_pow[left] + 1e-15)
        left = left + frac
    
    # 右 -3dB 点
    right = peak
    while right < N-1 and irf_pow[right] > half:
        right += 1
    if right < N-1:
        frac = (half - irf_pow[right]) / (irf_pow[right-1] - irf_pow[right] + 1e-15)
        right = right - frac
    
    irw_samples = right - left
    irw_m = irw_samples * c0 / (2 * Fs)
    return irw_m * 100  # → cm


def compute_snr_metrics(H, P):
    """
    计算 SNR 恶化 [dB]
    
    H: 滤波器矩阵 (Nf × Nf)
    P: 天线方向图
    
    信号增益:  ||H P||² / ||P||²
    噪声增益:  tr(H H^H) / Nf  (白噪声通过滤波器的平均功率增益)
    SNR 恶化:  10·log10(信号增益 / 噪声增益)  [dB]
    """
    Nf = len(P)
    g_sig = H @ P
    sig_gain = np.sum(np.abs(g_sig)**2) / np.sum(P**2)
    noise_gain = np.trace(H @ H.T).real / Nf  # H 是实矩阵时 H^H = H^T
    snr_loss = 10 * np.log10(sig_gain / (noise_gain + 1e-15))
    return snr_loss


def matched_filter_irw(P):
    """匹配滤波 IRW (基准)"""
    g_mf = P  # 匹配滤波: H = diag(P), g_eff = P⊙P = P² (归一化后)
    return compute_irw_from_response(g_mf)

# ===========================================================================
# 6. λ 扫描与对比
# ===========================================================================
print("\n" + "="*70)
print("λ 扫描: Tikhonov vs C-based (annihilation, rank=Nf/2)")
print("="*70)

lam_list = np.logspace(-6, 1.5, 60)
results_tikh = []   # (λ, IRW_cm, SNR_loss_dB)
results_cbas = []

# 参考值
irw_mf = matched_filter_irw(P)
irw_ideal = 0.886 * c0 / (2 * B) * 100  # 理论极限
print(f"匹配滤波 IRW = {irw_mf:.1f} cm")
print(f"理论极限 IRW = {irw_ideal:.1f} cm (0.886·c/(2B))")
print(f"\n{'λ':>10s}  {'Tikh IRW':>10s}  {'Tikh SNR':>10s}  {'C IRW':>10s}  {'C SNR':>10s}  {'改善':>8s}")
print("-"*60)

for lam in lam_list:
    # --- Tikhonov ---
    # 滤波器: H_tikh = diag(P / (P² + λ))
    H_tikh_diag = P / (P**2 + lam)
    H_tikh = np.diag(H_tikh_diag)
    g_tikh = H_tikh_diag * P       # = P² / (P² + λ)
    irw_t = compute_irw_from_response(g_tikh)
    snr_t = compute_snr_metrics(H_tikh, P)
    results_tikh.append((lam, irw_t, snr_t))
    
    # --- C-based ---
    H_c = solve_c_based_direct(np.ones(Nf), P, C, lam, eps=1e-8)
    # 注: 上面用 Y=1 是为了得到滤波器矩阵 H = A^{-1} D_p
    # 实际 H = A^{-1} D_p, 即 H @ Y = A^{-1} (D_p Y)
    # 对 Y = e_k (单位向量), X = H @ e_k = H 的第 k 列
    # 更高效: 直接构造 H = A^{-1} D_p
    Dp = np.diag(P)
    CTC = C.T @ C
    A_c = np.diag(P**2) + lam * (Dp @ CTC @ Dp) + 1e-8 * np.eye(Nf)
    H_c = solve(A_c, Dp, assume_a='pos')  # Nf × Nf 滤波器矩阵
    g_c = H_c @ P
    irw_c = compute_irw_from_response(g_c)
    snr_c = compute_snr_metrics(H_c, P)
    results_cbas.append((lam, irw_c, snr_c))
    
    if len(results_tikh) % 10 == 0:
        print(f"{lam:10.2e}  {irw_t:10.2f}  {snr_t:+9.2f}  {irw_c:10.2f}  {snr_c:+9.2f}  {snr_t-snr_c:+7.2f}")

results_tikh = np.array(results_tikh)
results_cbas = np.array(results_cbas)

# ===========================================================================
# 7. 找到 IRW=8cm 对应的 λ 和 SNR
# ===========================================================================
def interp_at_irw(results, target=8.0):
    """插值找到目标 IRW 对应的 λ 和 SNR loss"""
    valid = ~np.isnan(results[:, 1])
    lams = results[valid, 0]
    irws = results[valid, 1]
    snrs = results[valid, 2]
    
    # 找到跨越目标的区间
    for i in range(len(irws)-1):
        if (irws[i] - target) * (irws[i+1] - target) <= 0:
            log_lam = np.interp(target, [irws[i], irws[i+1]],
                               [np.log10(lams[i]), np.log10(lams[i+1])])
            snr_val = np.interp(target, [irws[i], irws[i+1]],
                               [snrs[i], snrs[i+1]])
            return 10**log_lam, snr_val
    return None, None

lam_t8, snr_t8 = interp_at_irw(results_tikh, 8.0)
lam_c8, snr_c8 = interp_at_irw(results_cbas, 8.0)

print("\n" + "="*70)
print("IRW = 8 cm 时的结果")
print("="*70)
if lam_t8:
    print(f"Tikhonov:   λ = {lam_t8:.4f},  SNR 恶化 = {snr_t8:+.2f} dB")
if lam_c8:
    print(f"C-based:    λ = {lam_c8:.4f},  SNR 恶化 = {snr_c8:+.2f} dB")
if lam_t8 and lam_c8:
    print(f"\n★ C-based 相比 Tikhonov 改善: {snr_t8 - snr_c8:.2f} dB")

# ===========================================================================
# 8. 可视化
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) IRW vs λ
ax = axes[0, 0]
ax.loglog(results_tikh[:, 0], results_tikh[:, 1], 'b-', lw=2, label='Tikhonov')
ax.loglog(results_cbas[:, 0], results_cbas[:, 1], 'r-', lw=2, label='C-based (annihilation)')
ax.axhline(irw_ideal, color='gray', ls='--', alpha=0.7, label=f'Ideal ({irw_ideal:.1f} cm)')
ax.axhline(8.0, color='green', ls=':', lw=2, label='Target (8 cm)')
ax.axhline(irw_mf, color='blue', ls='--', alpha=0.4, label=f'Matched filter ({irw_mf:.1f} cm)')
ax.set_xlabel('λ')
ax.set_ylabel('IRW (cm)')
ax.set_title('(a) 分辨率 vs 正则化参数')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) SNR loss vs λ
ax = axes[0, 1]
ax.semilogx(results_tikh[:, 0], results_tikh[:, 2], 'b-', lw=2, label='Tikhonov')
ax.semilogx(results_cbas[:, 0], results_cbas[:, 2], 'r-', lw=2, label='C-based (annihilation)')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('λ')
ax.set_ylabel('SNR 恶化 (dB)')
ax.set_title('(b) 信噪比恶化 vs 正则化参数')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) SNR loss vs IRW (核心权衡曲线)
ax = axes[1, 0]
ax.plot(results_tikh[:, 1], results_tikh[:, 2], 'b-', lw=2, label='Tikhonov')
ax.plot(results_cbas[:, 1], results_cbas[:, 2], 'r-', lw=2, label='C-based (annihilation)')
ax.axvline(8.0, color='green', ls=':', lw=2, label='Target (8 cm)')
ax.axvline(irw_ideal, color='gray', ls='--', alpha=0.5, label=f'Ideal ({irw_ideal:.1f} cm)')
if lam_t8:
    ax.plot(8.0, snr_t8, 'bo', ms=8, label=f'Tikhonov @8cm: {snr_t8:+.1f} dB')
if lam_c8:
    ax.plot(8.0, snr_c8, 'ro', ms=8, label=f'C-based @8cm: {snr_c8:+.1f} dB')
ax.set_xlabel('IRW (cm)')
ax.set_ylabel('SNR 恶化 (dB)')
ax.set_title('(c) SNR-分辨率权衡 (核心结果)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# (d) 有效频率响应对比 (在 IRW=8cm 的 λ 处)
ax = axes[1, 1]
f_mhz = f_vec / 1e6
ax.plot(f_mhz, P, 'gray', lw=1.5, alpha=0.6, label='天线方向图 P(f)')
if lam_t8:
    g_t8 = P**2 / (P**2 + lam_t8)
    ax.plot(f_mhz, g_t8, 'b-', lw=2, alpha=0.8, label=f'Tikhonov (λ={lam_t8:.3f})')
if lam_c8:
    Dp = np.diag(P)
    CTC = C.T @ C
    A_c8 = np.diag(P**2) + lam_c8 * (Dp @ CTC @ Dp) + 1e-8 * np.eye(Nf)
    H_c8 = solve(A_c8, Dp, assume_a='pos')
    g_c8 = H_c8 @ P
    ax.plot(f_mhz, np.abs(g_c8), 'r-', lw=2, alpha=0.8, label=f'C-based (λ={lam_c8:.1f})')
ax.set_xlabel('频率 (MHz)')
ax.set_ylabel('有效增益 |g(f)|')
ax.set_title('(d) 有效频率响应 @ IRW=8cm')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c_based_optimization_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ===========================================================================
# 9. 详细分析: C-based 为何优于 Tikhonov
# ===========================================================================
print("\n" + "="*70)
print("分析: C-based 方法的优势来源")
print("="*70)

# 在 λ → ∞ 极限下
print("\n--- λ → ∞ 极限行为 ---")

# Tikhonov: H → 0, 信号完全消失
print("Tikhonov:   X_k = P_k Y_k / (P_k² + λ) → 0  (信号被完全压制)")

# C-based: 
# A = diag(P²) + λ D_p C^T C D_p → λ D_p C^T C D_p (主导项)
# 但对于 X=1: C D_p 1 = C P = 0 → 正则项 = 0
# 所以平坦方向不受 λ 影响
print("C-based:    C D_p 1 = C P ≈ 0 → 平坦谱分量 λ 不敏感")
print("            噪声投影到 C 值域 → 被因子 1/(1+λ) 衰减")

# 验证: 检查 C D_p 1 的确切范数
residual_flat = norm(C @ (P * np.ones(Nf))) / norm(P)
print(f"\n验证 ||C D_p 1|| / ||P|| = {residual_flat:.2e} (应 ≈ 0)")

# 检查 C^T C 的结构
CTC = C.T @ C
# C^T C 的特征值分布
eigvals_CTC = np.linalg.eigvalsh(CTC)
print(f"\nC^T C 特征值: λ_max/λ_min = {eigvals_CTC[-1]/eigvals_CTC[0]:.1f}")
print(f"非零特征值数量: {np.sum(eigvals_CTC > 1e-10)} (理论值: {rank_C})")
print(f"零特征值数量:   {np.sum(eigvals_CTC < 1e-10)} (这些方向对应平坦谱)")

# 解释
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                       核心机制对比                                   │
├──────────────────┬──────────────────┬───────────────────────────────┤
│                  │   Tikhonov       │   C-based (Cp=0)              │
├──────────────────┼──────────────────┼───────────────────────────────┤
│ 惩罚项           │   λ||X||²        │   λ||C D_p X||²               │
│ 平坦信号 X=1     │   被惩罚 ✗       │   零惩罚 ✓ (C D_p 1 = C p = 0)│
│ 非平坦分量       │   均匀惩罚       │   按"非平坦度"惩罚            │
│ λ → ∞ 行为       │   X → 0          │   X_flat 无损, X_noise → 0   │
│ 物理含义         │   惩罚信号本身    │   惩罚信号的不平坦度          │
│ 分辨率-噪声权衡  │   必须折中       │   可同时优化 (λ 可任意大)     │
└──────────────────┴──────────────────┴───────────────────────────────┘
""")

print("Done. 图片已保存为 c_based_optimization_results.png")