# -*- coding: utf-8 -*-
"""
Redrawn design-strategy flowchart (Fig. 9) for
"LWA-based Compact Ka-band Frequency-Scanning Airborne SAR"
Run : python fsar_design_flowchart.py
Out : fsar_design_flowchart.pdf / .svg / .png  (vector)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------------ palette
STEP_F, STEP_E = "#E9F1F8", "#2E5E8E"    # design steps   (light blue)
IO_F,   IO_E   = "#F2F2F2", "#5A5A5A"    # input / output (grey)
DEC_F,  DEC_E  = "#FCEEDB", "#A9741B"    # decision       (amber)
LOOP_F, LOOP_E = "#F3F3F3", "#8A8A8A"    # feedback box   (grey)
ARR            = "#333333"
# pure B/W variant: STEP_F = IO_F = DEC_F = LOOP_F = "white"

fig = plt.figure(figsize=(4.4, 6.3))          # ~ column-width design
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.10, 10.35)
ax.set_ylim(0.40, 15.40)
ax.set_aspect("equal")
ax.axis("off")

# ------------------------------------------------------------------ helpers
def box(x, y, w, h, text, fc, ec, fs=8.7, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.5)

def arrow(p1, p2, lw=1.1, ms=9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=ms,
        color=ARR, lw=lw, shrinkA=0, shrinkB=0, zorder=2))

def elbow(pts, lw=1.1, ms=9):
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
            color=ARR, lw=lw, zorder=2, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], lw=lw, ms=ms)

def badge(x, y, n):
    ax.add_patch(Circle((x, y), 0.155, facecolor=STEP_E,
                        edgecolor="none", zorder=5))
    ax.text(x, y, str(n), color="white", ha="center", va="center",
            fontsize=7.2, fontweight="bold", zorder=6)

CX, BW = 4.00, 6.40                          # centre line / box width

# ------------------------------------------------------------------ nodes
box(CX, 14.45, BW, 1.30,
    "Mission requirements\n"
    r"$\rho_a,\ \rho_r,\ W$;   $H,\ \theta_{{in}}$;   $v_r$" "\n"
    "AASR / RASR / NESZ constraints",
    IO_F, IO_E)

box(CX, 12.70, BW, 1.30,
    "Azimuth beamwidth and PRF\n"
    r"$\phi_{{az}}\leftarrow\rho_a$;   PRF set from AASR",
    STEP_F, STEP_E)
badge(1.30, 12.95, 1)

box(CX, 10.90, BW, 1.40,
    "Range beamwidth\n"
    r"$\theta_{{bw}}\leftarrow$ RASR & NESZ constraints" "\n"
    r"(PRF, link budget $\rightarrow G \rightarrow \theta_{bw}$)",
    STEP_F, STEP_E)
badge(1.30, 11.20, 2)

box(CX, 9.05, BW, 1.40,
    "Bandwidth\n"
    r"$B_r \leftarrow N_{fsar} \leftarrow W/\rho_{r}$ " "\n"
    r"$\theta_{sc} \leftarrow W, \theta_{in}$",
    STEP_F, STEP_E)
badge(1.30, 9.35, 3)

box(CX, 6.95, BW, 2.00,
    "Scan Rate\n"
    r"the average scan rate $\zeta \leftarrow \theta_{sc} / B_r$ ",
    STEP_F, STEP_E)
badge(1.30, 7.55, 4)


# decision diamond
ax.add_patch(Polygon([(CX, 5.50), (CX + 2.70, 4.35),
                      (CX, 3.20), (CX - 2.70, 4.35)], closed=True,
                     facecolor=DEC_F, edgecolor=DEC_E, lw=1.1, zorder=3))
ax.text(CX, 4.35, "NESZ, RASR,\nAASR met?", ha="center", va="center",
        fontsize=8.4, zorder=4, linespacing=1.45)

# feedback box
box(8.75, 4.35, 2.60, 1.30,
    "Readjust within\navailable margins",
    LOOP_F, LOOP_E, fs=8.0)

# output box
box(CX, 1.85, BW, 1.90,
    r"FSAR design output: $B_r$, PRF, $\zeta$, $\theta_{{bw}}$, $\phi_{az}$, $G$" "\n"
    r"LWA design: $\beta(f_r)$, $\alpha$, $L$" "\n"
    r"measure $g(f_r),\ w(\theta)$",
    IO_F, IO_E)

# ------------------------------------------------------------------ edges
arrow((CX, 13.80), (CX, 13.35))                    # input -> step 1
arrow((CX, 12.05), (CX, 11.60))                    # 1 -> 2
ax.text(4.25, 11.82, "PRF", fontsize=8, ha="left", va="center")

arrow((CX, 10.20), (CX, 9.75))                     # 2 -> 3
ax.text(4.25, 9.97, r"$\theta_{{bw}}$", fontsize=8,
        ha="left", va="center")

arrow((CX, 8.35), (CX, 7.95))                      # 3 -> 4
ax.text(4.25, 8.15, r"$B_r, \theta_{sc}$", fontsize=8, ha="left", va="center")

arrow((CX, 5.95), (CX, 5.50))                      # 4 -> decision
ax.text(4.25, 5.72, r"$\zeta$", fontsize=8,
        ha="left", va="center")

arrow((CX, 3.20), (CX, 2.80))                      # decision -> output (Yes)
ax.text(4.26, 2.97, "Yes", fontsize=8, ha="left", va="center")

arrow((6.70, 4.35), (7.45, 4.35))                  # decision -> readjust (No)
ax.text(7.06, 4.66, "No", fontsize=8, ha="center", va="center")

elbow([(8.75, 5.00), (8.75, 12.70), (7.20, 12.70)])  # loop back to step 1

# ------------------------------------------------------------------ export
plt.savefig("fsar_design_flowchart.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("fsar_design_flowchart.svg", bbox_inches="tight", pad_inches=0.02)
plt.savefig("fsar_design_flowchart.png", dpi=600, bbox_inches="tight",
            pad_inches=0.02)
# plt.show()