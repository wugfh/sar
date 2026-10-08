# -*- coding: utf-8 -*-
"""
LWA-FSAR 三维设计可行区：(扫描率 zeta, 系统带宽 Br, 距离波束宽度 theta_bw)
依据：Zeng et al., Section III.A + Table I（与图1、图2同一套常数）

区域 = 以下约束的交集：
 (1) 幅宽      : zeta*Br >= Theta - theta_bw            (zeta*Br 乘积下界, Theta=9.34deg)
 (2) 分辨率    : theta_bw/zeta >= c/(2*rho_r) = 0.75 GHz (theta_bw/zeta 比值下界)
 (3) 式(33)    : theta_bw*Br >= C1 = 5.18 deg*GHz       (图1 乘积下界, 可用 INCLUDE_EQ33 关闭)
 (4) NESZ      : theta_bw^2*Br <= C2 = 40.2 deg^2*GHz   (NESZ <= -25 dB)
 (5) 截止频率  : Br <= 5 GHz

设计点 (2.58 deg, 2.01 GHz, 3.44 deg/GHz)：约束(2)(3)同时取等号。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 10.5, 'axes.labelsize': 11,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'legend.fontsize': 7.8,
})
rad2deg = 180.0/np.pi

# ---------------- 1. Table I 需求 + 设计参数 ----------------
f0, H, vr = 35e9, 3.0e3, 70.0
W_req     = 2.0e3
theta_in  = np.deg2rad(60.0)
rho_r_req = 0.20
c = 3e8
eta, phi_az, G_dB = 0.5, np.deg2rad(8.0), 30.0
G_lin = 10.0**(G_dB/10.0)

theta_bw_deg = 4*np.pi*eta/(G_lin*phi_az)*rad2deg      # theta_bw* ≈ 2.578 deg (式34)

def swath_of_extent(ext):                              # 式(13)
    return H*(np.tan(theta_in+ext/2) - np.tan(theta_in-ext/2))
lo, hi = 1e-6, np.deg2rad(30.0)
for _ in range(100):
    mid = 0.5*(lo+hi)
    if swath_of_extent(mid) < W_req: lo = mid
    else:                            hi = mid
Theta_deg  = 0.5*(lo+hi)*rad2deg                       # ≈9.336 deg
theta_sc_d = Theta_deg - theta_bw_deg                  # ≈6.758 deg

# 统一常数（与图1/图2一致）
A_dg      = (c*W_req*np.cos(theta_in)**2/(2.0*H*rho_r_req))*1e-9*rad2deg
C1        = A_dg*(theta_sc_d/Theta_deg)     # 5.18  deg*GHz    (图1: theta_bw*Br 下界)
C2        = 40.2                            # 40.2  deg^2*GHz  (图1: NESZ 上界)
Br_max    = 5.0                             # GHz              (截止频率)
ratio_min = 0.75                            # GHz = c/(2*rho_r) (分辨率下界)

Br_design   = C1/theta_bw_deg               # ≈2.011 GHz
zeta_design = theta_bw_deg/ratio_min        # ≈3.438 deg/GHz

INCLUDE_EQ33 = True      # False = 不并入式(33)乘积下界面（纯"需求"可行域）

# ---------------- 2. 区域边界函数 ----------------
def Br_lo(th):
    """(theta_bw, Br) 投影的下边界: 式(33)乘积下界 与 幅宽∩分辨率(zeta存在性) 取严"""
    pin = ratio_min*(Theta_deg - th)/th          # 幅宽面与分辨率面相交（zeta 区间非空）
    return np.maximum(C1/th, pin) if INCLUDE_EQ33 else pin

def Br_hi(th):
    """投影的上边界: NESZ 与截止频率取严"""
    return np.minimum(Br_max, C2/np.square(th))

th_grid = np.linspace(0.5, 9.33, 6000)
mask    = Br_lo(th_grid) <= Br_hi(th_grid)
th_min, th_max = th_grid[mask][0], th_grid[mask][-1]

# ---------------- 3. 三维网格化 ----------------
Nt, Nv, Ns = 150, 56, 36
th  = np.linspace(th_min, th_max, Nt)
blo = Br_lo(th); bhi = Br_hi(th)
v   = np.linspace(0, 1, Nv)

TH  = np.tile(th[:, None], (1, Nv))
BR  = blo[:, None] + np.tile(v[None, :], (Nt, 1))*((bhi - blo)[:, None])
ZLO = (Theta_deg - TH)/BR        # 底面： zeta = (Theta - theta_bw)/Br   (幅宽面)
ZHI = TH/ratio_min               # 顶面： zeta = theta_bw/ratio_min      (分辨率面)

# ---------------- 4. 绘图 ----------------
fig = plt.figure(figsize=(9.2, 6.8), dpi=110)
ax  = fig.add_subplot(111, projection='3d')

# 底面：幅宽约束面
ax.plot_surface(TH, BR, ZLO, color='tab:blue', alpha=0.42,
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
# 顶面：分辨率约束面（过原点的平面）
ax.plot_surface(TH, BR, ZHI, color='tab:red', alpha=0.20,
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

# 侧壁 Br = Br_lo（式(33)乘积面；theta_bw<2.42 段退化为棱线）
s  = np.linspace(0, 1, Ns)
X0 = np.tile(th[:, None], (1, Ns))
Y0 = np.tile(blo[:, None], (1, Ns))
Z0 = ZLO[:, [0]] + s[None, :]*(ZHI[:, [0]] - ZLO[:, [0]])
ax.plot_surface(X0, Y0, Z0, color='tab:green', alpha=0.22,
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

# 侧壁 Br = Br_hi（截止频率 + NESZ 面）
X1 = np.tile(th[:, None], (1, Ns))
Y1 = np.tile(bhi[:, None], (1, Ns))
Z1 = ZLO[:, [-1]] + s[None, :]*(ZHI[:, [-1]] - ZLO[:, [-1]])
ax.plot_surface(X1, Y1, Z1, color='tab:orange', alpha=0.22,
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

# 关键棱线
ax.plot(th, blo, th/ratio_min, color='k', lw=1.5, alpha=0.7)              # 设计临界线
ax.plot(th, blo, (Theta_deg-th)/blo, color='tab:blue', lw=1.3, alpha=0.9) # 底面外缘
ax.plot(th, bhi, (Theta_deg-th)/bhi, color='0.3', lw=1.0, alpha=0.7)      # 外侧底缘
ax.plot([th_max]*2, [bhi[-1]]*2,
        [(Theta_deg-th_max)/bhi[-1], th_max/ratio_min], color='k', lw=1.5, alpha=0.7)

# 设计点 + 垂线
ax.scatter([theta_bw_deg], [Br_design], [zeta_design], marker='*', s=300,
           c='gold', edgecolors='k', linewidths=0.9, depthshade=False)
ax.plot([theta_bw_deg]*2, [Br_design]*2, [0, zeta_design], ls=':', c='k', lw=1.0)
ax.text(theta_bw_deg+0.15, Br_design+0.25, zeta_design+0.55,
        'Prototype design point\n(2.58°, 2.01 GHz, 3.44°/GHz)', fontsize=8.0)

# 各约束面标注
ax.text(3.1, 3.8, 2.05, 'Swath lower bound:\n' r'$\zeta B_r \geq 9.34-\theta_{bw}$',
        fontsize=8.0, color='tab:blue')
ax.text(5.1, 1.15, 7.35, 'Resolution lower bound:\n' r'$\theta_{bw}/\zeta \geq 0.75$ GHz',
        fontsize=8.0, color='tab:red')
ax.text(6.35, 0.72, 9.15, 'Eq. (33) product:\n' r'$\theta_{bw}B_r \geq 5.18$',
        fontsize=8.0, color='tab:green')
ax.text(2.25, 5.42, 4.30, 'NESZ & cutoff wall:\n' r'$B_r \leq 5$ GHz,  $\theta_{bw}^2B_r \leq 40.2$',
        fontsize=8.0, color='darkorange')

# 坐标轴
ax.set_xlim(0.5, 8.5); ax.set_ylim(0.4, 5.6); ax.set_zlim(0, 10.8)
ax.set_xticks([1,2,3,4,5,6,7,8]); ax.set_yticks([1,2,3,4,5]); ax.set_zticks([0,2,4,6,8,10])
ax.set_xlabel(r'Range beamwidth $\theta_{bw}$  (deg)', labelpad=9)
ax.set_ylabel(r'System bandwidth $B_r$  (GHz)', labelpad=9)
ax.set_zlabel(r'Scan rate $\zeta$  (deg/GHz)', labelpad=9)
ax.view_init(elev=24, azim=-55)
ax.set_box_aspect((1.3, 1.0, 1.45))
ax.set_title('LWA-FSAR 3-D design feasible region (Table I requirements)', fontsize=11)

handles = [
    Patch(facecolor='tab:blue',   alpha=0.5, label=r'Swath: $\zeta B_r \geq 9.34-\theta_{bw}$'),
    Patch(facecolor='tab:red',    alpha=0.4, label=r'Range resolution: $\theta_{bw}/\zeta \geq 0.75$ GHz'),
    Patch(facecolor='tab:green',  alpha=0.4, label=r'Eq. (33): $\theta_{bw}B_r \geq 5.18$ deg$\cdot$GHz'),
    Patch(facecolor='tab:orange', alpha=0.4, label=r'NESZ & cutoff: $\theta_{bw}^2B_r \leq 40.2$, $B_r \leq 5$ GHz'),
    Line2D([], [], ls='none', marker='*', ms=13, mfc='gold', mec='k',
           label='Prototype design point (2.58°, 2.01 GHz, 3.44°/GHz)'),
]
ax.legend(handles=handles, loc='upper left', framealpha=0.95)
plt.tight_layout()
plt.savefig('lwa_fsar_design_region_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------- 5. 校核输出 ----------------
print('--- 3-D feasible region ---------------------------------------')
print(f'Theta = {Theta_deg:.3f} deg,  theta_bw* = {theta_bw_deg:.3f} deg')
print(f'C1 = {C1:.3f} deg*GHz    (式33 乘积下界)')
print(f'C2 = {C2:.1f} deg^2*GHz  (NESZ 上界)')
print(f'ratio_min = {ratio_min:.2f} GHz  (theta_bw/zeta 下界 = c/(2*rho_r))')
print(f'theta_bw range: [{th_min:.3f}, {th_max:.3f}] deg')
print(f'left apex  = ({th_min:.3f} deg, {float(Br_lo(th_min)):.3f} GHz, {th_min/ratio_min:.3f} deg/GHz)')
print(f'right end  = ({th_max:.3f} deg, {float(Br_hi(th_max)):.3f} GHz, '
      f'zeta in [{(Theta_deg-th_max)/float(Br_hi(th_max)):.3f}, {th_max/ratio_min:.3f}])')
print(f'design point = ({theta_bw_deg:.3f} deg, {Br_design:.3f} GHz, {zeta_design:.3f} deg/GHz)')
print(f'  theta_bw*Br      = {theta_bw_deg*Br_design:.3f}  (= C1)')
print(f'  theta_bw/zeta    = {theta_bw_deg/zeta_design:.3f} GHz (= ratio_min)')
print(f'  zeta*Br+theta_bw = {zeta_design*Br_design+theta_bw_deg:.3f} deg (>= {Theta_deg:.3f})')
print(f'slice theta_bw=2.58 deg: Br in [{float(Br_lo(theta_bw_deg)):.3f}, 5.0] GHz, '
      f'zeta in [6.76/Br, 3.44] deg/GHz  (与图2一致)')