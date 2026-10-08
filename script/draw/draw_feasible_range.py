# -*- coding: utf-8 -*-
"""
LWA-FSAR 设计可行区（第一张图）：距离波束宽度 theta_bw - 系统带宽 B_r
依据：Zeng et al., "LWA-based Compact Ka-band FSAR", Section III.A + Table I

三条约束线：
  1) 式(33)： B_r * theta_bw   >= C1    (测绘带-分辨率需求 W/rho_r = 2km/0.2m)
  2) 式(22)： B_r * theta_bw^2 <= C2    (NESZ <= -25 dB 链路预算)
  3) 截止频率：B_r <= 5 GHz             (Sec. III.A / Fig.6)
可行域 A = 三者交集；原型设计点 (2.0 GHz, 2.58 deg) 落在约束 1) 上。
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'legend.fontsize': 8.8,
})
rad2deg = 180.0 / np.pi

# ----------------------------------------------------------------------
# 1. 系统需求 (Table I)
# ----------------------------------------------------------------------
f0        = 35e9                 # 载频 [Hz]
H         = 3.0e3                # 飞行高度 [m]
vr        = 70.0                 # 平台速度 [m/s]
W_req     = 2.0e3                # 测绘带宽度 [m]
theta_in  = np.deg2rad(60.0)     # 测绘带中心视角
rho_r_req = 0.20                 # 距离分辨率需求 [m]
rho_a_req = 0.05                 # 方位分辨率需求 [m]
Pu        = 280.0                # 峰值发射功率 [W]
PRF       = 6000.0               # 脉冲重复频率 [Hz]（由 AASR 确定）
NESZ_req  = -25.0                # NESZ 需求 [dB]

# ----------------------------------------------------------------------
# 2. 链路预算 / 设计参数（论文正文取值）
# ----------------------------------------------------------------------
kn     = 1.380649e-23            # 玻尔兹曼常数
Tn     = 320.0                   # 系统噪声温度 [K]
Ln_dB  = 5.0                     # 系统总损耗 [dB]
Tp     = 20e-6                   # 脉宽 [s]（Fig.9(b)）
eta    = 0.5                     # 天线总效率（Appendix A）
phi_az = np.deg2rad(8.0)         # 方位波束宽度（设计值）
G_dB   = 30.0                    # LWA 增益（由 NESZ 链路预算得）
c      = 3e8
Br_max = 5.0                     # 截止频率限制：B_r < 5 GHz [GHz]
theta_nesz_eval = theta_in       # NESZ 在测绘带中心评估（可换成远端视角）
F_pat  = 2.285                   # 双程方向图积分因子 (φ_az/∫|P_az|^4)·(θ_bw/∫|P_el|^4)
                                 # ≈1.51²；取此值时式(22)在设计点给出 ≈-30 dB（与Fig.9(b)一致）
G_lin  = 10.0**(G_dB / 10.0)
lam    = c / f0

# ----------------------------------------------------------------------
# 3. 总角度宽度 Theta = theta_sc + theta_bw（式13，由测绘带确定）
# ----------------------------------------------------------------------
def swath_of_extent(ext):
    return H * (np.tan(theta_in + ext/2) - np.tan(theta_in - ext/2))

lo, hi = 1e-6, np.deg2rad(30.0)
for _ in range(100):
    mid = 0.5*(lo + hi)
    if swath_of_extent(mid) < W_req:
        lo = mid
    else:
        hi = mid
Theta = 0.5*(lo + hi)                     # rad（≈9.336°）

# 设计距离波束宽度（式34）
theta_bw_d = 4*np.pi*eta/(G_lin*phi_az)   # rad（≈2.578°）
theta_sc_d = Theta - theta_bw_d           # 设计扫描宽度

# ----------------------------------------------------------------------
# 4. 三条约束线
# ----------------------------------------------------------------------
# (1) 式(33)：B_r * theta_bw 的下界
#     theta_bw*Br = c*W*cos^2(theta_in)*theta_sc / (2*H*rho_r*Theta)
#     其中 θ_sc/Θ 取设计分配值 -> 集总常数 C1
A_dg = (c*W_req*np.cos(theta_in)**2/(2.0*H*rho_r_req)) * 1e-9 * rad2deg   # deg*GHz
C1   = A_dg*(theta_sc_d/Theta)                                            # deg*GHz
Br_design = C1/(theta_bw_d*rad2deg)                                       # GHz

# (2) 式(22)：NESZ 预算 -> B_r * theta_bw^2 的上界
#     NESZ = K_n * Br * theta_bw^2 ，其中 G = 4*pi*eta/(theta_bw*phi_az) （式34）
theta_n = theta_nesz_eval
R_n     = H/np.cos(theta_n)
K_n = (256.0*np.pi**3*kn*Tn*10.0**(Ln_dB/10.0)*vr/(Pu*lam**3*c)) \
      * (R_n**3*np.sin(theta_n)/(Tp*PRF)) \
      * F_pat*phi_az**2/(4.0*np.pi*eta)**2        # 单位: 1/(Hz*rad^2)
C2  = (10.0**(NESZ_req/10.0)/K_n) * 1e-9 * rad2deg**2                    # deg^2*GHz

# (3) 截止频率限制：B_r <= Br_max（竖线）

# 关键点
Br_pinch = C1**2/C2
th_pinch = C1/Br_pinch

print('--- derived constants -------------------------------------------')
print(f'Theta    = {Theta*rad2deg:7.3f} deg     (总角度宽度, 式13)')
print(f'theta_bw*= {theta_bw_d*rad2deg:7.3f} deg     (设计波束宽度, 式34)')
print(f'theta_sc*= {theta_sc_d*rad2deg:7.3f} deg')
print(f'C1 = {C1:6.3f} deg*GHz     (Br*theta_bw  下界, 式33)')
print(f'C2 = {C2:6.2f} deg^2*GHz   (Br*theta_bw^2 上界, NESZ)')
print(f'Br at design point = {Br_design:.3f} GHz  (论文: 2.00 GHz)')
print(f'region pinch point = ({Br_pinch:.3f} GHz, {th_pinch:.2f} deg)')
print(f'at Br=5 GHz -> theta_bw in [{C1/Br_max:.3f}, {np.sqrt(C2/Br_max):.3f}] deg')
print(f'NESZ at design point = {10*np.log10(K_n*Br_design*1e9*theta_bw_d**2):.2f} dB')

# ----------------------------------------------------------------------
# 5. 绘图
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 5.3), dpi=120)
Br_min = c/(2*rho_r_req)*1e-9
# 可行域 A
BrA = np.linspace(Br_min, Br_max, 800)
ax.fill_between(BrA, C1/BrA, np.sqrt(C2/BrA), color='tab:green', alpha=0.16,
                lw=0, zorder=1, label=r'Design feasible region $\mathcal{A}$')

# 约束 (1)：式(33) 下界
th1 = np.linspace(1.00, 8.60, 800)
ax.plot(C1/th1, th1, color='tab:blue', lw=2.0,
        label=rf'Eq. (33): $B_r\,\theta_{{bw}} \geq {C1:.2f}$ deg$\cdot$GHz')

# 约束 (2)：NESZ 上界
th2 = np.linspace(2.72, 8.60, 800)
ax.plot(C2/th2**2, th2, color='tab:red', lw=2.0,
        label=rf'NESZ $\leq$ {NESZ_req:.0f} dB: $B_r\,\theta_{{bw}}^2 \leq {C2:.1f}$ deg$^2\!\cdot$GHz')

# 约束 (3)：截止频率
ax.axvline(Br_max, color='k', ls='--', lw=1.6,
           label=r'Cutoff limit: $B_r \leq 5$ GHz')

ax.axvline(Br_min, color='k', ls=':', lw=1.6,
           label=r'$\rho_r > 0.2$ m: $B_r \geq 0.75$ GHz')

# （可选）严格式(33)边界（不做乘积集总近似）：theta_bw = A_dg/(Br + A_dg/Theta)
DRAW_EXACT_EQ33 = False
if DRAW_EXACT_EQ33:
    Theta_deg = Theta*rad2deg
    Br_ex = np.linspace(0.5, 5.4, 600)
    ax.plot(Br_ex, A_dg/(Br_ex + A_dg/Theta_deg), color='tab:blue', lw=1.2,
            ls='-.', label='Exact Eq. (33) boundary')

# 可行域角点 + 设计点
ax.plot([Br_max, Br_max], [C1/Br_max, np.sqrt(C2/Br_max)], 'o', color='k',
        ms=4.0, zorder=5)
ax.plot([Br_pinch], [th_pinch], 'o', color='k', ms=4.0, zorder=5)
ax.plot([Br_design], [theta_bw_d*rad2deg], marker='o', ms=8, mfc='gold',
        mec='k', mew=0.8, ls='none', zorder=6,
        label=f'Prototype design point ({Br_design:.2f} GHz, '
              f'{theta_bw_d*rad2deg:.2f}$^\\circ$)')

# 非可行区文字标注
ax.text(2.5, 8.45, 'NESZ > -25 dB', fontsize=14, color='0.35',
        style='italic', va='top', fontweight='bold')
ax.text(1, 1.35, r'$N_{fsar}$ requirement not met', fontsize=14,
        color='0.35', style='italic', va='center', fontweight='bold')
ax.text(5.08, 5.4, 'beyond cutoff', fontsize=14, color='0.35',
        style='italic', rotation=90, va='center', fontweight='bold')
ax.text(0.5, 5.4, r'$\rho_r$ requirement not met', fontsize=14, color='0.35',
        style='italic', rotation=90, va='center', fontweight='bold')

ax.set_xlim(0.5, 5.5)
ax.set_ylim(1.0, 8.8)
ax.set_xlabel(r'$B_r$  (GHz)', fontsize=18, fontweight= "bold")
ax.set_ylabel(r'$\theta_{bw}$  ($^\circ$)', fontsize=18, fontweight= "bold")
# ax.set_title('LWA-FSAR design feasible region (Table I requirements)')
ax.grid(alpha=0.3, ls=':')
ax.legend(loc='upper right', framealpha=0.95)
plt.tight_layout()
plt.savefig('lwa_fsar_design_region.png', dpi=300, bbox_inches='tight')
plt.show()