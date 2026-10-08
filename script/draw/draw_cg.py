# -*- coding: utf-8 -*-
r"""
pcg_flowchart_latex.py
======================
PCG 求解器（spectrum_recovery.py -> solve_c_based_cg）流程图 —— matplotlib 版。
所有数学符号均用 LaTeX 书写、由 matplotlib mathtext 渲染（本地无需安装 LaTeX）。

本版更新：
    - 删去 warm start 分支（"warm_start 非空？"菱形框及 X 初始化两框）
    - 前三个框改为简化版（输入 / 预条件分母 / 定义算子）
    - 后续节点整体上移，画幅由 7.5x13.2 压缩为 7.5x11.7

关于 $ 与 $$ :
    Python 绘图库(matplotlib mathtext)中，数学模式用成对【单】$ 界定，
    例如 r"$\lambda_1$"；$$ ... $$ 是 Markdown/LaTeX 文档里的“显示公式”，
    两者公式内容完全一致。本文件每个 $...$ 内即 LaTeX 源码，
    粘进论文时直接换成 $$ ... $$ 即可。

依赖: matplotlib
运行: python pcg_flowchart_latex.py
输出: pcg_solver_flowchart_latex.png / .pdf（300 dpi）
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

# ---------------- 全局风格 ----------------
plt.rcParams["font.family"] = ["Microsoft YaHei", "Times New Roman"]
#                             ↑ 中文字体（Linux 可改 "Noto Sans CJK SC"）
plt.rcParams["mathtext.fontset"] = "stix"        # 公式：Times 风格

FS, FS_E = 8.5, 7.5                              # 节点字号 / 边标签字号
C_TERM, C_PROC, C_DEC, C_IO, C_LOOP = ("#d9ead3", "#e8f0fe", "#fce8b2",
                                       "#f4cccc", "#d0e0e3")

fig = plt.figure(figsize=(7.5, 11.7), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(34.32, 0)                            # y 轴朝下
ax.axis("off")
CX = 11.0                                        # 主列中心线

# ---------------- 图元 ----------------
def box(x, y, w, h, text, fc=C_PROC):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.3",
                 fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, zorder=3,
            linespacing=1.5)

def diamond(x, y, w, h, text, fc=C_DEC):
    pts = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, zorder=3)

def ellipse(x, y, w, h, text, fc=C_TERM):
    ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, zorder=3,
            linespacing=1.5)

def para(x, y, w, h, text, fc=C_IO, skew=0.7):
    pts = [(x - w / 2 + skew, y - h / 2), (x + w / 2, y - h / 2),
           (x + w / 2 - skew, y + h / 2), (x - w / 2, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, zorder=3)

def arrow(p1, p2, ls="-"):
    ax.annotate("", xy=p2, xytext=p1, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=11))

def elbow(pts, ls="-"):
    """折线路径，最后一段带箭头"""
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, ls=ls, color="black", lw=1.1, zorder=1)
    arrow(pts[-2], pts[-1], ls=ls)

def elabel(x, y, text, ha="left", rot=0):
    ax.text(x, y, text, fontsize=FS_E, ha=ha, va="center",
            rotation=rot, zorder=4)

# ---------------- 节点 ----------------
ellipse(CX, 2.1, 12.0, 1.6,
        r"输入 $Y, P, h_1, h_2$；参数 $\lambda_1, \lambda_2, \varepsilon$ 等")

box(CX, 4.85, 11.5, 2.6,
    r"预条件分母：$q = (\mu + \varepsilon + \lambda_1 H_1 + \lambda_2 H_2)^{-1}$")

box(CX, 8.1, 12.0, 2.6,
    r"定义算子" "\n"
    r"$A(v) = P^2 \odot v + \lambda_1 \mathcal{F}^{-1}(H_1 \odot \mathcal{F}v)$"
    r"$+ \lambda_2 \mathcal{F}^{-1}(H_2 \odot \mathcal{F}v) + \varepsilon v$" "\n"
    r"$M^{-1}(v) = \mathcal{F}^{-1}(q \odot \mathcal{F}v)$")

box(CX, 10.5, 8.0, 0.9, r"右端项：$b = P \odot Y$")

box(CX, 12.85, 11.0, 2.6,
    "初始化残差与搜索方向：\n"
    r"$r \leftarrow b - A(X)$；$z \leftarrow M^{-1}(r)$；$p \leftarrow z$" "\n"
    r"$rz \leftarrow \langle r, z \rangle$" "\n"
    r"$tol_{sq} \leftarrow \max\{tol^2 \|b\|^2, 10^{-30}\}$")

# 注：图中 rz 即代码变量 rz_old，ρ 即代码变量 r_norm_sq
box(CX, 15.2, 10.0, 0.9,
    r"PCG 主循环：$k = 0, 1, \ldots$（最多 max_iter 次）", fc=C_LOOP)
box(CX, 17.05, 9.0, 1.6,
    r"$Ap \leftarrow A(p)$" "\n" r"$pAp \leftarrow \langle p, Ap \rangle$")
diamond(CX, 19.25, 7.5, 1.6, r"$pAp \leq 10^{-30}$ ？")
box(CX, 21.7, 10.5, 2.1,
    r"$\alpha \leftarrow rz / pAp$" "\n"
    r"$X \leftarrow X + \alpha p$；$r \leftarrow r - \alpha Ap$" "\n"
    r"$\rho \leftarrow \|r\|^2$")
diamond(CX, 24.15, 7.5, 1.6, r"$\rho < tol_{sq}$ ？")
box(CX, 26.6, 11.0, 2.1,
    r"$z \leftarrow M^{-1}(r)$；$rz_{new} \leftarrow \langle r, z \rangle$" "\n"
    r"$\beta \leftarrow rz_{new} / rz$" "\n"
    r"$p \leftarrow z + \beta p$；$rz \leftarrow rz_{new}$")
diamond(CX, 29.05, 7.5, 1.6, "已达最后一次迭代？")

para(CX, 30.95, 8.0, 1.0, r"返回 $X$（恢复的频谱）")
ellipse(CX, 32.55, 5.0, 1.0, "结束")

# ---------------- 连线 ----------------
arrow((CX, 2.9), (CX, 3.55))
arrow((CX, 6.15), (CX, 6.8))
arrow((CX, 9.4), (CX, 10.05))
arrow((CX, 10.95), (CX, 11.55))      # 右端项 → 初始化（原 warm start 分支已删）

arrow((CX, 14.15), (CX, 14.75))
arrow((CX, 15.65), (CX, 16.25))
arrow((CX, 17.85), (CX, 18.45))
arrow((CX, 20.05), (CX, 20.65))      # 否：数值保护
arrow((CX, 22.75), (CX, 23.35))
arrow((CX, 24.95), (CX, 25.55))      # 否：未收敛
arrow((CX, 27.65), (CX, 28.25))
arrow((CX, 29.85), (CX, 30.45))      # 是：跳出循环
arrow((CX, 31.45), (CX, 32.05))

elbow([(14.75, 19.25), (21.0, 19.25), (21.0, 30.95), (15.2, 30.95)],
      ls="dashed")                    # break ①（数值保护）
elbow([(14.75, 24.15), (20.3, 24.15), (20.3, 30.7), (15.3, 30.7)],
      ls="dashed")                    # break ②（已收敛）

elbow([(7.25, 29.05), (1.5, 29.05), (1.5, 15.2), (6.0, 15.2)])   # 否：回到主循环

# ---------------- 边标签 ----------------
elabel(11.4, 20.35, "否")
elabel(11.4, 25.25, "否")
elabel(11.4, 30.15, "是（未收敛则打印提示）")
elabel(15.35, 18.85, "是（break）")
elabel(15.35, 23.75, "是（break）")
elabel(1.13, 22.1, "否：继续下一轮", ha="center", rot=90)

# ---------------- 输出 ----------------
fig.savefig("pcg_solver_flowchart_latex.png", dpi=1000, facecolor="white")
fig.savefig("pcg_solver_flowchart_latex.pdf", facecolor="white")
print("已输出 pcg_solver_flowchart_latex.png / .pdf")
# plt.show()