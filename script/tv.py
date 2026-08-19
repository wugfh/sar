import cupy as cp
import matplotlib.pyplot as plt


def first_column_from_row(r):
    """若循环矩阵由第一行 r 给出，转成第一列 p。
    第一行 r = [c0, c_{N-1}, ..., c_1]。
    """
    r = cp.asarray(r).reshape(-1)
    p = cp.empty_like(r)
    p[0] = r[0]
    p[1:] = r[:0:-1]          # r[N-1], r[N-2], ..., r[1]
    return p


def _soft_threshold(v, thresh):
    """复数软阈值: prox_{thresh ||.||_1}(v)"""
    v = cp.asarray(v)
    mag = cp.abs(v)
    scale = cp.maximum(mag - thresh, 0.0) / cp.maximum(mag, 1e-30)
    return scale * v


# ------------------------------------------------------------------
# 形式一: 正则化问题  min ||Dx||_1 + (lam/2)||Px - y||^2
# ------------------------------------------------------------------
def solve_tv_regularized(p, y, lam, rho=1.0, max_iter=60000, tol=1e-8,
                         dtype=cp.complex128, verbose=False):
    """
    参数
    ----
    p    : (N,) 循环矩阵 P 的第一列 (cupy 或 numpy 数组)
    y    : (N,) 含噪观测
    lam  : 数据保真项权重 (lam 越大越贴近数据/噪声, 越小越平滑)
    rho  : ADMM 罚参数
    dtype: cp.complex64 或 cp.complex128

    返回 x_hat (cupy 数组), info(dict)
    """
    p = cp.asarray(p, dtype=dtype).reshape(-1)
    y = cp.asarray(y, dtype=dtype).reshape(-1)
    N = p.size
    if y.size != N:
        raise ValueError(f"P 为 {N}x{N} 循环矩阵, y 长度应为 {N}, 实际 {y.size}")

    # ---- 频域预计算 ----
    p_hat = cp.fft.fft(p)
    y_hat = cp.fft.fft(y)
    k = cp.arange(N, dtype=cp.float64)
    d_hat = cp.exp(2j * cp.pi * k / N) - 1.0          # 周期差分算子 D 的频谱

    den = lam * (p_hat * cp.conj(p_hat)) + rho * (d_hat * cp.conj(d_hat))
    den = den + 1e-12                                  # 防止除零

    # ---- 初始解: 匹配滤波伪逆 ----
    x = cp.fft.ifft(cp.conj(p_hat) * y_hat /
                    cp.maximum(p_hat * cp.conj(p_hat), 1e-12))
    x = x.astype(dtype)

    def Dv(v):
        return cp.roll(v, -1) - v
        # return v

    z = Dv(x)                             # z = Dx
    u = cp.zeros(N, dtype=dtype)

    r_pri = r_dual = cp.inf
    it = -1
    for it in range(max_iter):
        # 1) x 更新: 频域对角
        q = z - u
        num = (lam * cp.conj(p_hat) * y_hat +
               rho * cp.conj(d_hat) * cp.fft.fft(q))
        x = cp.fft.ifft(num / den)

        Dx = Dv(x)

        # 2) z 更新: 复数软阈值
        z_old = z
        z = _soft_threshold(Dx + u, 1.0 / rho)

        # 3) 对偶更新
        u = u + Dx - z

        # 4) 残差
        r_pri = cp.linalg.norm(Dx - z)
        dz = z - z_old
        r_dual = rho * cp.linalg.norm(cp.roll(dz, 1) - dz)   # D^H dz

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"iter {it:4d}: r_pri={float(r_pri):.3e}, "
                  f"r_dual={float(r_dual):.3e}")

        if float(r_pri) <= tol and float(r_dual) <= tol:
            if verbose:
                print(f"iter {it:4d}: 收敛")
            break

    Px = cp.fft.ifft(p_hat * cp.fft.fft(x))
    info = {
        "iter": it + 1,
        "primal_residual": float(r_pri),
        "dual_residual": float(r_dual),
        "tv": float(cp.sum(cp.abs(Dx))),
        "data_fidelity": float(cp.linalg.norm(Px - y)),
    }
    return x, info


# ------------------------------------------------------------------
# 形式二: 正则化问题  min ||x||_1 + (lam/2)||Px - y||^2
# ------------------------------------------------------------------
def solve_lasso_regularized(p, y, lam, rho=1.0, max_iter=60000, tol=1e-8,
                            dtype=cp.complex128, verbose=False):
    """
    min_x ||x||_1 + (lam/2) ||Px - y||^2,  P 为循环矩阵(第一列为 p)
    """
    p = cp.asarray(p, dtype=dtype).reshape(-1)
    y = cp.asarray(y, dtype=dtype).reshape(-1)
    N = p.size
    if y.size != N:
        raise ValueError(f"P 为 {N}x{N} 循环矩阵, y 长度应为 {N}, 实际 {y.size}")
    # ---- 频域预计算 ----
    p_hat = cp.fft.fft(p)
    y_hat = cp.fft.fft(y)
    # x 子问题: (lam P^H P + rho I) x = lam P^H y + rho (z - u)
    # 频域分母: lam |p_hat|^2 + rho  
    den = lam * (p_hat * cp.conj(p_hat)) + rho
    den = den + 1e-12
    # ---- 初始解: 岭型解(不含 z 项), 比匹配滤波伪逆更稳 ----
    x = cp.fft.ifft(lam * cp.conj(p_hat) * y_hat / den).astype(dtype)
    z = x.copy()                       # 约束 z = x
    u = cp.zeros(N, dtype=dtype)
    r_pri = r_dual = cp.inf
    it = -1
    for it in range(max_iter):
        # 1) x 更新: 频域对角, O(N log N)
        q = z - u
        num = lam * cp.conj(p_hat) * y_hat + rho * cp.fft.fft(q)
        x = cp.fft.ifft(num / den)
        # 2) z 更新: 复数软阈值
        z_old = z
        z = _soft_threshold(x + u, 1.0 / rho)
        # 3) 对偶更新
        u = u + x - z
        # 4) 残差 (约束 x - z = 0)
        r_pri = cp.linalg.norm(x - z)
        r_dual = rho * cp.linalg.norm(z - z_old)   # rho * (z^{k+1} - z^k)
        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"iter {it:4d}: r_pri={float(r_pri):.3e}, "
                  f"r_dual={float(r_dual):.3e}")
        if float(r_pri) <= tol and float(r_dual) <= tol:
            if verbose:
                print(f"iter {it:4d}: 收敛")
            break
    Px = cp.fft.ifft(p_hat * cp.fft.fft(x))
    info = {
        "iter": it + 1,
        "primal_residual": float(r_pri),
        "dual_residual": float(r_dual),
        "l1": float(cp.sum(cp.abs(x))),
        "data_fidelity": float(cp.linalg.norm(Px - y)),
        "objective": float(cp.sum(cp.abs(x)) +
                           0.5 * lam * cp.linalg.norm(Px - y) ** 2),
    }
    return x, info


# ------------------------------------------------------------------
# 形式三: 正则化问题  min ||h \otimes x||_1 + (lam/2)||Px - y||^2
# ------------------------------------------------------------------

def solve_conv_l1_regularized(h, p, y, lam, rho=1.0, max_iter=60000, tol=1e-8,
                              dtype=cp.complex128, den_floor=1e-12, verbose=False):
    """
    min_x  ||h ⊗ x||_1  +  (lam/2) ||P x - y||^2
    
    """
    h = cp.asarray(h, dtype=dtype).reshape(-1)
    p = cp.asarray(p, dtype=dtype).reshape(-1)
    y = cp.asarray(y, dtype=dtype).reshape(-1)
    N = p.size
    if h.size != N or y.size != N:
        raise ValueError(f"h、p、y 长度必须都为 N={N}, "
                         f"实际 h={h.size}, y={y.size}")
    # ---------- 频域预计算 ----------
    h_hat = cp.fft.fft(h)
    p_hat = cp.fft.fft(p)
    y_hat = cp.fft.fft(y)
    hc = cp.conj(h_hat)                    # H^H 的频域核
    pc = cp.conj(p_hat)                    # P^H 的频域核
    h2 = cp.real(h_hat * hc)               # |h_hat|^2
    p2 = cp.real(p_hat * pc)               # |p_hat|^2
    # x 子问题分母: lam |p_hat|^2 + rho |h_hat|^2
    den = lam * p2 + rho * h2
    safe = den > den_floor                 # 可辨识频率掩码
    den_safe = cp.where(safe, den, 1.0)
    lam_pc_yhat = lam * pc * y_hat         # x 子问题常数项(与迭代无关)
    def Hv(v):
        return cp.fft.ifft(h_hat * cp.fft.fft(v))
    # ---------- 初始解: 仅含可辨识频率的岭型解 ----------
    x = cp.fft.ifft(cp.where(safe, lam_pc_yhat / den_safe,
                             cp.zeros(N, dtype=dtype)))
    z = Hv(x)
    u = cp.zeros(N, dtype=dtype)
    # ---------- ADMM 主循环 ----------
    r_pri = r_dual = cp.inf
    it = -1
    for it in range(max_iter):
        # 1) x 更新: 频域对角, O(N log N)
        q = z - u
        num = lam_pc_yhat + rho * hc * cp.fft.fft(q)
        x = cp.fft.ifft(cp.where(safe, num / den_safe,
                                 cp.zeros(N, dtype=dtype)))
        Hx = Hv(x)
        # 2) z 更新: 复数软阈值
        z_old = z
        z = _soft_threshold(Hx + u, 1.0 / rho)
        # 3) 对偶更新
        u = u + Hx - z
        # 4) 残差 (约束 Hx - z = 0)
        r_pri = cp.linalg.norm(Hx - z)
        dz = z - z_old
        # 对偶残差: rho * ||H^H dz||,  H^H dz = ifft(conj(h_hat) * fft(dz))
        r_dual = rho * cp.linalg.norm(cp.fft.ifft(hc * cp.fft.fft(dz)))
        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"iter {it:4d}: r_pri={float(r_pri):.3e}, "
                  f"r_dual={float(r_dual):.3e}")
        if float(r_pri) <= tol and float(r_dual) <= tol:
            if verbose:
                print(f"iter {it:4d}: 收敛")
            break
    # ---------- 统计信息 ----------
    Hx = Hv(x)
    Px = cp.fft.ifft(p_hat * cp.fft.fft(x))
    info = {
        "iter": it + 1,
        "primal_residual": float(r_pri),
        "dual_residual": float(r_dual),
        "reg_l1": float(cp.sum(cp.abs(Hx))),          # ||Hx||_1
        "data_fidelity": float(cp.linalg.norm(Px - y)),
        "objective": float(cp.sum(cp.abs(Hx))
                           + 0.5 * lam * cp.linalg.norm(Px - y) ** 2),
    }
    return x, info
