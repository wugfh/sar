import numpy as np
from scipy import stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 物理常数与固定参数
# ============================================================================
c      = 3.0e8
a_fixed = 7.11e-3           # 波导宽边 [m]
f1_fixed = c / (2 * a_fixed) # 截止频率 ≈ 21.097 GHz
f0_fixed = 35e9              # 载频 [Hz]
G_dB_fixed = 30.0            # 天线增益 [dB]
G_lin_fixed = 10 ** (G_dB_fixed / 10)  # ≈ 1000

print("=" * 70)
print("  周期 LWA 参数")
print("=" * 70)
print(f"  波导宽边 a      = {a_fixed*1e3:.2f} mm (R-320)")
print(f"  截止频率 f₁     = {f1_fixed/1e9:.4f} GHz")
print(f"  载频 f₀         = {f0_fixed/1e9:.1f} GHz")
print(f"  天线增益 G      = {G_dB_fixed:.0f} dB (线性 {G_lin_fixed:.0f})")
print(f"  增益约束: θ_bw·φ_az = 4π·η / G")

def g_periodic(f_r, f0, f2, a):

    f1 = c / (2 * a)
    f = f0 + f_r
    f_safe = np.maximum(f, f1 * 1.0001)

    # Term 1: sqrt(1 - (f₁/f)²)
    t1 = np.sqrt(np.clip(1.0 - (f1 / f_safe)**2, 0, None))

    # Term 2: (f₂/f)·sqrt(1 - (f₁/f₂)²)
    f2_safe = np.maximum(f2, f1 * 1.001)
    t2 = (f2 / f_safe) * np.sqrt(np.clip(1.0 - (f1 / f2_safe)**2, 0, None))

    sin_theta = np.clip(t1 - t2, -1.0, 1.0)
    return np.arcsin(sin_theta)


def g_periodic_inverse(theta_target, f0, f2, a, fr_bounds=(-5e9, 5e9)):
    """
    Inverse of Eq.8: given θ, find f_r via Brent's method.

    Returns np.nan if root not found.
    """
    f1 = c / (2 * a)
    def func(fr):
        return g_periodic(fr, f0, f2, a) - theta_target

    lo, hi = fr_bounds
    try:
        # Expand bounds if needed
        for _ in range(12):
            if func(lo) * func(hi) <= 0:
                break
            lo, hi = lo * 1.8, hi * 1.8
        else:
            return np.nan
        return brentq(func, lo, hi, xtol=1e3, maxiter=120)
    except (ValueError, RuntimeError):
        return np.nan



def compute_f2_bounds_eq10(f0, Br, a):

    f1 = c / (2 * a)
    f_min = f0 - Br / 2
    f_max = f0 + Br / 2

    # Eq.10 lower bound (theoretical): f₂ > f·(sqrt(1-(f₁/f)²) - 1) for all f
    # Since sqrt ≤ 1, lower bound ≤ 0 → practical: f₂ > f₁
    f2_lower_theo = f1

    # Eq.10 upper bound: f₂ < f·(sqrt(1-(f₁/f)²) + 1) for all f
    # h(f) = f·(1+sqrt(1-(f₁/f)²)) is monotonic increasing → min at f_min
    h_fmin = f_min * (1.0 + np.sqrt(np.clip(1.0 - (f1 / f_min)**2, 0, None)))
    f2_upper_theo = h_fmin

    # Practical FSAR design: f₂ inside operating band for broadside crossing
    f2_inband_lower = np.maximum(f_min, f1)
    f2_inband_upper = f_max

    return f2_lower_theo, f2_upper_theo, f2_inband_lower, f2_inband_upper



def compute_performance(Br, f0, f2, a, phi_az, eta, H, theta_in):

    res = {'valid': True}

    # ---- Scan width θ_sc (Eq.13) ----
    th_u = g_periodic(Br/2, f0, f2, a)
    th_l = g_periodic(-Br/2, f0, f2, a)
    theta_sc = np.abs(th_u - th_l)

    # ---- Scan rate ξ (Eq.21) ----
    xi = theta_sc / Br if Br > 0 else 0.0

    # ---- Range beamwidth θ_bw from gain constraint (Eq.27) ----
    # G = 4π·η / (φ_az · θ_bw)  →  θ_bw = 4π·η / (G · φ_az)
    theta_bw = 4.0 * np.pi * eta / (G_lin_fixed * phi_az) if phi_az > 0 else np.inf

    # ---- Dwell bandwidth fbw (Eq.24): fbw ≈ θ_bw / ξ ----
    if xi > 1e-16:
        fbw_linear = theta_bw / xi
        # Refined via exact inverse (Eq.16)
        th_c = g_periodic(0.0, f0, f2, a)
        f_up  = g_periodic_inverse(th_c + theta_bw/2, f0, f2, a)
        f_lo  = g_periodic_inverse(th_c - theta_bw/2, f0, f2, a)
        if np.isfinite(f_up) and np.isfinite(f_lo):
            half = Br / 2
            f_lo_eff = np.maximum(f_lo, -half)
            f_up_eff = np.minimum(f_up,  half)
            fbw = np.maximum(f_up_eff - f_lo_eff, 0.0)
        else:
            fbw = fbw_linear
    else:
        fbw = np.inf

    if fbw <= 0 or not np.isfinite(fbw):
        res['valid'] = False
        fbw = 1.0

    # ---- Range resolution ρr (Eq.17) ----
    rho_r = c / (2.0 * fbw)

    # ---- Azimuth resolution ρa (Eq.26) ----
    rho_a = c / (2.0 * f0 * phi_az) if phi_az > 0 else np.inf

    # ---- Swath width W (Eq.14) ----
    th_far  = theta_in + theta_sc/2 + theta_bw/2
    th_near = theta_in - theta_sc/2 - theta_bw/2
    th_far  = np.clip(th_far,  0.02, np.pi/2 - 0.02)
    th_near = np.clip(th_near, 0.02, np.pi/2 - 0.02)
    W = H * (np.tan(th_far) - np.tan(th_near))

    if W <= 0 or not np.isfinite(W):
        res['valid'] = False

    res.update({
        'theta_sc': theta_sc, 'xi': xi, 'theta_bw': theta_bw,
        'fbw': fbw, 'rho_r': rho_r, 'rho_a': rho_a, 'W': W
    })
    return res


def run_monte_carlo(n_samples=10000, seed=42):
    """
    Monte Carlo simulation with:
      - Br        : Uniform [0.5, 6] GHz
      - f₂        : Uniform [f0-Br/2, f0+Br/2] (Eq.10 + in-band design)
      - φ_az      : Truncated normal 8° ± 1.5° (3σ)
      - η         : Uniform [0.50, 0.72]
      - H         : Truncated normal 3000 ± 30 m (3σ)
      - θ_in      : Truncated normal 60° ± 1.5° (3σ)
    """
    rng = np.random.default_rng(seed)

    # ---- Sample independent variables ----
    Br       = rng.uniform(0.5e9, 6.0e9, n_samples)

    phi_az_nom = np.deg2rad(8.0)

    # phi_az   = rng.normal(phi_az_nom, np.deg2rad(1.5)/3, n_samples)
    # phi_az   = np.clip(phi_az, np.deg2rad(3.0), np.deg2rad(14.0))
    phi_az   = rng.uniform(np.deg2rad(2.0), np.deg2rad(14.0), n_samples)  

    eta      = rng.uniform(0.50, 0.72, n_samples)

    H_nom    = 3000.0
    H        = rng.normal(H_nom, 30.0/3, n_samples)
    H        = np.clip(H, 2800, 3200)

    theta_in_nom = np.deg2rad(60.0)
    theta_in = rng.normal(theta_in_nom, np.deg2rad(1.5)/3, n_samples)
    theta_in = np.clip(theta_in, np.deg2rad(50), np.deg2rad(70))

    # ---- Output arrays ----
    f2_arr    = np.zeros(n_samples)
    th_sc     = np.zeros(n_samples)
    xi_arr    = np.zeros(n_samples)
    th_bw     = np.zeros(n_samples)
    fbw_arr   = np.zeros(n_samples)
    rho_r     = np.zeros(n_samples)
    rho_a     = np.zeros(n_samples)
    W_arr     = np.zeros(n_samples)
    valid     = np.ones(n_samples, dtype=bool)
    f2_upper_theo_arr = np.zeros(n_samples)

    # ---- Loop over samples (f₂ depends on Br) ----
    for i in range(n_samples):
        # Eq.10 bounds
        _, f2_up_theo, f2_in_lo, f2_in_up = compute_f2_bounds_eq10(
            f0_fixed, Br[i], a_fixed)

        # Sample f₂ uniformly in the in-band range (satisfies Eq.10)
        f2_arr[i] = rng.uniform(f2_in_lo, f2_in_up)
        f2_upper_theo_arr[i] = f2_up_theo

        # Compute performance
        res = compute_performance(
            Br[i], f0_fixed, f2_arr[i], a_fixed,
            phi_az[i], eta[i], H[i], theta_in[i])

        valid[i]    = res['valid']
        th_sc[i]    = res['theta_sc']
        xi_arr[i]   = res['xi']
        th_bw[i]    = res['theta_bw']
        fbw_arr[i]  = res['fbw']
        rho_r[i]    = res['rho_r']
        rho_a[i]    = res['rho_a']
        W_arr[i]    = res['W']

    # ---- Pack results ----
    results = {
        # Inputs
        'Br_GHz':       Br / 1e9,
        'f2_GHz':       f2_arr / 1e9,
        'phi_az_deg':   np.rad2deg(phi_az),
        'eta':          eta,
        'H':            H,
        'theta_in_deg': np.rad2deg(theta_in),
        # Outputs
        'theta_sc_deg': np.rad2deg(th_sc),
        'xi_deg_ghz':   np.rad2deg(xi_arr) / 1e-9,
        'theta_bw_deg': np.rad2deg(th_bw),
        'fbw_MHz':      fbw_arr / 1e6,
        'rho_r':        rho_r,
        'rho_a':        rho_a,
        'W_km':         W_arr / 1e3,
        'valid':        valid,
        # Aux
        'f2_upper_theo_GHz': f2_upper_theo_arr / 1e9,
    }

    pct = 100 * np.mean(valid)
    print(f"\n  有效样本: {np.sum(valid)}/{n_samples} ({pct:.1f}%)")
    return results


def binned_stats(x, y, n_bins=25):
    """Compute mean, median, ±1σ per bin."""
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]
    if len(xv) < 50:
        return None
    bins = np.linspace(np.percentile(xv, 0.5), np.percentile(xv, 99.5), n_bins+1)
    bc = (bins[:-1] + bins[1:]) / 2
    m, med, s, cnt = [np.zeros(n_bins) for _ in range(4)]
    for j in range(n_bins):
        mask = (xv >= bins[j]) & (xv < bins[j+1])
        cnt[j] = np.sum(mask)
        if cnt[j] >= 20:
            m[j]   = np.mean(yv[mask])
            med[j] = np.median(yv[mask])
            s[j]   = np.std(yv[mask])
        else:
            m[j] = med[j] = s[j] = np.nan
    return {'bc': bc, 'mean': m, 'median': med, 'std': s, 'count': cnt}


def plot_fig1_rho_r_vs_Br(res):
    """Fig.1: Range resolution vs System bandwidth."""
    ok = res['valid']
    Br, rr = res['Br_GHz'][ok], res['rho_r'][ok]
    phi = res['phi_az_deg'][ok]
    # ra = res['rho_a'][ok]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(Br, rr, c=phi, cmap='viridis', alpha=0.3, s=4, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax); cb.set_label('azimuth beamwidth (°)', fontsize=20)

    # bs = binned_stats(Br, rr, 30)
    # if bs is not None:
    #     ok2 = np.isfinite(bs['mean'])
    #     ax.plot(bs['bc'][ok2], bs['mean'][ok2], 'r-', lw=2.5, label='Mean trend')
    #     ax.fill_between(bs['bc'][ok2],
    #                     bs['mean'][ok2]-bs['std'][ok2],
    #                     bs['mean'][ok2]+bs['std'][ok2],
    #                     alpha=0.2, color='red', label='±1σ')
        
    ax.set_xlabel('System Bandwidth (GHz)', fontsize=20)
    ax.set_ylabel('Range Resolution (m)', fontsize=20)

    ax.legend(fontsize=20); ax.grid(alpha=0.3); plt.tight_layout()
    return fig


def plot_fig2_W_vs_Br(res):
    """Fig.2: Swath width vs System bandwidth."""
    ok = res['valid']
    Br, W_km = res['Br_GHz'][ok], res['W_km'][ok]
    Hv = res['H'][ok]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(Br, W_km, c=Hv, cmap='plasma', alpha=0.3, s=4, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax); cb.set_label('Flight Height (m)', fontsize=20)

    bs = binned_stats(Br, W_km, 30)
    if bs is not None:
        ok2 = np.isfinite(bs['mean'])
        ax.plot(bs['bc'][ok2], bs['mean'][ok2], 'darkorange', lw=2.5, label='Mean trend')
        ax.fill_between(bs['bc'][ok2],
                        bs['mean'][ok2]-bs['std'][ok2],
                        bs['mean'][ok2]+bs['std'][ok2],
                        alpha=0.2, color='darkorange', label='±1σ')
    ax.set_xlabel('System Bandwidth (GHz)', fontsize=20)
    ax.set_ylabel('Swath Width (km)', fontsize=20)
    ax.legend(fontsize=20); ax.grid(alpha=0.3); plt.tight_layout()
    return fig


def plot_fig3_rho_r_vs_W(res):
    """Fig.3: Core trade-off — ρr vs W."""
    ok = res['valid']
    rr, W_km = res['rho_r'][ok], res['W_km'][ok]
    Br = res['Br_GHz'][ok]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _, _, _, im = ax.hist2d(rr, W_km, bins=70, cmap='jet',
            range=[[np.percentile(rr,0.5), np.percentile(rr,99.5)],
                   [np.percentile(W_km,0.5), np.percentile(W_km,99.5)]])
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Count', fontsize=20)

    bs = binned_stats(rr, W_km, 22)
    if bs is not None:
        ok2 = np.isfinite(bs['mean'])
        # Average Br per bin
        br_bin = np.zeros(len(bs['bc']))
        for j in range(len(bs['bc'])-1):
            mask = (rr >= (bs['bc'][j]-np.diff(bs['bc'][:2])[0]/2)) & \
                   (rr <  (bs['bc'][j]+np.diff(bs['bc'][:2])[0]/2))
            if np.sum(mask) > 5:
                br_bin[j] = np.mean(Br[mask])
        sc2 = ax.scatter(bs['bc'][ok2], bs['mean'][ok2], c=br_bin[ok2],
                         cmap='coolwarm', s=35, edgecolors='k', lw=0.4, zorder=6)
        cb = plt.colorbar(sc2, ax=ax)
        cb.set_label("System Bandwidth (GHz)", fontsize=20)
    ax.set_xlabel('Range Resolution (m)', fontsize=20)
    ax.set_ylabel('Swath Width  W (km)', fontsize=20)
    ax.grid(alpha=0.25); plt.tight_layout()
    return fig


def plot_fig4_rho_r_vs_rho_a(res):
    """Fig.4: Joint resolution — ρr vs ρa (coupled via gain constraint)."""
    ok = res['valid']
    rr, ra = res['rho_r'][ok], res['rho_a'][ok]
    xi_d = res['xi_deg_ghz'][ok]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(rr, ra, c=xi_d, cmap='Spectral_r', alpha=0.35, s=4,
                    edgecolors='none',
                    vmin=np.percentile(xi_d,2), vmax=np.percentile(xi_d,98))
    cb = plt.colorbar(sc, ax=ax); cb.set_label('Scan R[°/GHz]')

    # Theory overlay: ρr·ρa = (c²·ξ·G) / (16π·f₀·η)
    eta_ref = 0.6
    ra_grid = np.logspace(np.log10(0.012), np.log10(0.18), 120)
    for xi_r, col in zip([0.8, 1.8, 3.5, 7.0], ['blue','green','orange','red']):
        xi_rad = np.deg2rad(xi_r)/1e-9
        rr_ref = (c**2 * xi_rad * G_lin_fixed) / (16*np.pi*f0_fixed*eta_ref*ra_grid)
        ax.plot(rr_ref, ra_grid, '--', color=col, lw=1.4,
                alpha=0.7, label=f'ξ={xi_r}°/GHz')
        
    ax.set_xlabel('Range Resolution (m)', fontsize=20)
    ax.set_ylabel('Azimuth Resolution (m)', fontsize=20)

    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)
    ax.set_xlim(0.01, 0.5); ax.set_ylim(0.008, 0.18); plt.tight_layout()
    return fig


def plot_fig5_rho_a_vs_W(res):
    """Fig.5: Core trade-off — ρa vs W."""
    ok = res['valid']
    ra, W_km = res['rho_a'][ok], res['W_km'][ok]
    Br = res['Br_GHz'][ok]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _, _, _, im = ax.hist2d(ra, W_km, bins=70, cmap='jet',
            range=[[np.percentile(ra,0.5), np.percentile(ra,99.5)],
                   [np.percentile(W_km,0.5), np.percentile(W_km,99.5)]])
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Count', fontsize=20)

    bs = binned_stats(ra, W_km, 22)
    if bs is not None:
        ok2 = np.isfinite(bs['mean'])
        # Average Br per bin
        br_bin = np.zeros(len(bs['bc']))
        for j in range(len(bs['bc'])-1):
            mask = (ra >= (bs['bc'][j]-np.diff(bs['bc'][:2])[0]/2)) & \
                   (ra <  (bs['bc'][j]+np.diff(bs['bc'][:2])[0]/2))
            if np.sum(mask) > 5:
                br_bin[j] = np.mean(Br[mask])
        sc2 = ax.scatter(bs['bc'][ok2], bs['mean'][ok2], c=br_bin[ok2],
                         cmap='coolwarm', s=35, edgecolors='k', lw=0.4, zorder=6)
        plt.colorbar(sc2, ax=ax, label='Mean Br in bin [GHz]')
    
    ax.set_xlabel('Azimuth Resolution (m)', fontsize=20)
    ax.set_ylabel('Swath Width  W (km)', fontsize=20)
    ax.grid(alpha=0.25); plt.tight_layout()
    return fig


def plot_bonus_f2_distribution(res):
    """Bonus: f₂ distribution and Eq.10 verification."""
    ok = res['valid']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) f₂ histogram
    ax = axes[0]
    ax.hist(res['f2_GHz'][ok], bins=50, density=True, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.set(xlabel='Broadside Frequency  f₂ [GHz]', ylabel='Density',
           title='(a) f₂ Distribution (Uniform in operating band)')

    # (b) f₂ vs Br with Eq.10 upper bound
    ax = axes[1]
    ax.scatter(res['Br_GHz'][ok], res['f2_GHz'][ok], c='steelblue',
               alpha=0.25, s=3, edgecolors='none', label='Samples')
    # Eq.10 upper bound
    Br_sort = np.sort(res['Br_GHz'][ok])
    _, ub, _, _ = compute_f2_bounds_eq10(f0_fixed, Br_sort*1e9, a_fixed)
    ax.plot(Br_sort, ub/1e9, 'r-', lw=2, label="Eq.10 upper bound")
    ax.fill_between(Br_sort, 20, ub/1e9, alpha=0.06, color='red')
    ax.set(xlabel='System Bandwidth  Br [GHz]', ylabel='f₂ [GHz]',
           title='(b) f₂ vs Br  —  all samples well within Eq.10 bound')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle('Bonus: f₂ Constraint Analysis (Eq.10)', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_fig_pearson_heatmap(res):
    ok = res['valid']
    N_valid = np.sum(ok)
    # Variable dictionary — display name : data array
    var_dict = {
        'Br':       res['Br_GHz'][ok],        # System bandwidth [GHz]
        # 'f₂':       res['f2_GHz'][ok],        # Broadside frequency [GHz]
        # 'φ_az':     res['phi_az_deg'][ok],    # Azimuth beamwidth [°]
        # 'η':        res['eta'][ok],           # Antenna efficiency
        # 'H':        res['H'][ok],             # Flight altitude [m]
        # 'θ_in':     res['theta_in_deg'][ok],  # Incidence angle [°]
        # 'θ_sc':     res['theta_sc_deg'][ok],  # Scan width [°]
        # 'ξ':        res['xi_deg_ghz'][ok],    # Scan rate [°/GHz]
        # 'θ_bw':     res['theta_bw_deg'][ok],  # Range beamwidth [°]
        # 'fbw':      res['fbw_MHz'][ok],       # Dwell bandwidth [MHz]
        'ρr':       res['rho_r'][ok],         # Range resolution [m]
        'ρa':       res['rho_a'][ok],         # Azimuth resolution [m]
        'W':        res['W_km'][ok],          # Swath width [km]
    }
    labels = list(var_dict.keys())
    n = len(labels)
    # Compute correlation matrix
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(var_dict[labels[i]],
                                      var_dict[labels[j]])[0, 1]
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='equal',
                   interpolation='nearest')
    # Annotate each cell with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            # White text on dark cells, black on light
            if abs(val) > 0.55:
                text_color = 'white'
            else:
                text_color = 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=20, color=text_color,
                    fontweight='bold' if abs(val) > 0.7 else 'normal')
    # Axis ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=20, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=20)
    # Highlight diagonal cells
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='black',
                                    linewidth=1.8, linestyle='-'))
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Pearson  r', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    # Title & caption
    # fig.text(0.5, 0.005,
    #          'Blue = negative (r ≈ −1)    White = uncorrelated (r ≈ 0)    '
    #          'Red = positive (r ≈ +1)',
    #          ha='center', fontsize=20, style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def main():
    print("\n▶ 运行 Monte Carlo 仿真 (1,000,000 样本)...")
    res = run_monte_carlo(n_samples=1000000, seed=42)

    print("\n" + "=" * 90)
    print("  典型 Br 下的性能统计 (有效样本 mean ± std)")
    print("=" * 90)
    print(f"{'Br [GHz]':<10}{'ρr [m]':<14}{'ρa [m]':<14}{'W [km]':<14}{'ξ [°/GHz]':<14}{'θ_bw [°]':<12}")
    print("-" * 90)
    ok = res['valid']
    for br0 in [1, 2, 3, 4, 5, 6]:
        mask = (np.abs(res['Br_GHz'] - br0) < 0.35) & ok
        if np.sum(mask) >= 15:
            print(f"{br0:<10.1f}"
                  f"{np.mean(res['rho_r'][mask]):.3f}±{np.std(res['rho_r'][mask]):.3f}  "
                  f"{np.mean(res['rho_a'][mask]):.3f}±{np.std(res['rho_a'][mask]):.3f}  "
                  f"{np.mean(res['W_km'][mask]):.2f}±{np.std(res['W_km'][mask]):.2f}    "
                  f"{np.mean(res['xi_deg_ghz'][mask]):.2f}±{np.std(res['xi_deg_ghz'][mask]):.2f}    "
                  f"{np.mean(res['theta_bw_deg'][mask]):.2f}±{np.std(res['theta_bw_deg'][mask]):.2f}")
    print("-" * 90)

    # ---- 相关系数 ----
    print("\n关键变量 Pearson 相关系数 (有效样本):")
    vars_ = {
        'Br': res['Br_GHz'][ok], 'ρr': res['rho_r'][ok], 'ρa': res['rho_a'][ok],
        'W': res['W_km'][ok], 'ξ': res['xi_deg_ghz'][ok],
        'θ_bw': res['theta_bw_deg'][ok], 'f₂': res['f2_GHz'][ok],
        'φ_az': res['phi_az_deg'][ok], 'η': res['eta'][ok],
    }
    names = list(vars_.keys()); nv = len(names)
    for i, ni in enumerate(names):
        row = f"{ni:<8}"
        for j, nj in enumerate(names):
            if i == j:
                row += f"{'—':>8}"
            else:
                row += f"{np.corrcoef(vars_[ni], vars_[nj])[0,1]:8.3f}"
        print(row)

    # ---- 可视化 ----
    print("\n▶ 生成图表...")
    fig1 = plot_fig1_rho_r_vs_Br(res)
    plt.savefig("res_Br_mc.png", dpi=300)
    fig2 = plot_fig2_W_vs_Br(res)
    plt.savefig("W_Br_mc.png", dpi=300)
    fig3 = plot_fig3_rho_r_vs_W(res)
    plt.savefig("res_W_mc.png", dpi=300)
    fig4 = plot_fig4_rho_r_vs_rho_a(res)
    plt.savefig("rr_ra_mc.png", dpi=300)
    fig5 = plot_fig5_rho_a_vs_W(res)
    plt.savefig("ra_W_mc.png", dpi=300)
    # fig5 = plot_bonus_f2_distribution(res)
    fig6 = plot_fig_pearson_heatmap(res)
    plt.savefig("pearson_heatmap_mc.pdf", dpi=300)

    print("\n✓ 完成 — 共 6 组图表。")
    plt.show()
    return res


if __name__ == "__main__":
    results = main()