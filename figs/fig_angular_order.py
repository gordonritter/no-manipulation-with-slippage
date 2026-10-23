"""Approved Appendix A illustration of edge angles and angular ordering.

Run with MPLCONFIGDIR=/private/tmp/no-manipulation-mpl python3
figs/fig_angular_order.py. Outputs PDF and PNG beside this script.
Exact Fraction calculations check round-trip volume, time, A, B, F(1),
the line integral K, and the signed polygon area in all 24 orderings.
"""

from fractions import Fraction as Q
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATES = [1, 9, -4, -16]
PHI = [1, 3, -2, -4]
DURATIONS = [Q(144, 205), Q(16, 205), Q(36, 205), Q(9, 205)]
ORIGINAL = [0, 1, 3, 2]
ORDERED = [1, 0, 3, 2]
BLUE, ORANGE, GRAY = "#2266a0", "#bb572b", "#646b73"


def path(order):
    points = [(Q(0), Q(0))]
    integral = Q(0)
    for i in order:
        z, f = points[-1]
        dz, df = DURATIONS[i]*RATES[i], DURATIONS[i]*PHI[i]
        integral += (z + dz/2)*df
        points.append((z+dz, f+df))
    area = sum(x*y2-x2*y for (x, y), (x2, y2)
               in zip(points, points[1:] + points[:1]))/2
    assert integral == area
    return points, area


def invariants(order):
    return tuple(sum(DURATIONS[i]*value[i] for i in order)
                 for value in ([1]*4, RATES, PHI,
                               [v*v for v in RATES], list(map(abs, RATES))))


def check():
    inv = invariants(ORIGINAL)
    assert inv[0] == 1 and inv[1] == 0 and inv[2] > 0
    _, maximum = path(ORDERED)
    for order in permutations(range(4)):
        assert invariants(order) == inv
        assert path(order)[1] <= maximum
    points, _ = path(ORDERED)
    vectors = [(x2-x, y2-y) for (x, y), (x2, y2)
               in zip(points, points[1:] + points[:1])]
    assert all(a*d-b*c > 0 for (a, b), (c, d)
               in zip(vectors, vectors[1:] + vectors[:1]))
    print("Exact checks passed: all 24 orderings; time, volume, F(1), A, B;")
    print("line integral equals signed area; ordered polygon is convex.")
    print(f"Original K = {path(ORIGINAL)[1]}; ordered K = {maximum}")


def arrow(ax, a, b, color, dashed=False, width=1.8):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=width,
                                linestyle="--" if dashed else "-",
                                mutation_scale=12, shrinkA=0, shrinkB=0))


def axes_at_origin(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")
    ax.plot(xlim, [0, 0], color="#bdc2c7", lw=.7, zorder=0)
    ax.plot([0, 0], ylim, color="#bdc2c7", lw=.7, zorder=0)
    ax.text(xlim[1], -.09, "$Z$", ha="right", va="top", fontsize=13)
    ax.text(-.07, ylim[1], "$F$", ha="right", va="top", fontsize=13)


def angle_arc(ax, degrees, radius, color):
    angles = np.deg2rad(np.linspace(0, degrees, 150))
    points = radius*np.column_stack([np.cos(angles), np.sin(angles)])
    ax.plot(points[:, 0], points[:, 1], color=color, lw=1.2)
    arrow(ax, points[-5], points[-1], color, width=1.2)


def directions(ax):
    axes_at_origin(ax, (-1.55, 1.55), (-1.25, 1.45))
    buy_angle, sell_angle = 38, 208
    # The positive magnitudes are cot(angle)^2, so the displayed vectors
    # have exactly the slopes 1/sqrt(u) and 1/sqrt(w).
    buy = 1.25*np.array([np.cos(np.deg2rad(buy_angle)),
                         np.sin(np.deg2rad(buy_angle))])
    sell = 1.25*np.array([np.cos(np.deg2rad(sell_angle)),
                          np.sin(np.deg2rad(sell_angle))])
    arrow(ax, (0, 0), buy, BLUE, width=2.1)
    arrow(ax, (0, 0), sell, ORANGE, width=2.1)
    angle_arc(ax, buy_angle, .44, BLUE)
    angle_arc(ax, sell_angle, .73, ORANGE)
    ax.text(.79, .13, r"$\arctan(1/\sqrt{u})$", color=BLUE, fontsize=12)
    ax.text(-1.48, .95, r"$\pi+\arctan(1/\sqrt{w})$",
            color=ORANGE, fontsize=12)
    ax.text(.70, 1.20, r"buy rate $u>0$", color=BLUE, ha="center", fontsize=12)
    ax.text(.78, .99, r"$\Delta t\,(u,\sqrt{u})$", color=BLUE,
            ha="center", fontsize=12)
    ax.text(-.83, -.94, r"sell rate $-w$, $w>0$", color=ORANGE,
            ha="center", fontsize=12)
    ax.text(-.83, -1.17, r"$\Delta t\,(-w,-\sqrt{w})$", color=ORANGE,
            ha="center", fontsize=12)
    ax.text(.5, -.12, "Angles measured counterclockwise from the positive $Z$ axis",
            transform=ax.transAxes, ha="center", fontsize=10)


def ordered_polygon(ax):
    points = np.array(path(ORDERED)[0], dtype=float)
    axes_at_origin(ax, (-.62, 1.67), (-.18, 1.22))
    ax.fill(points[:, 0], points[:, 1], color=BLUE, alpha=.075, zorder=0)
    labels = [r"$u_1$", r"$u_2$", r"$-w_1$", r"$-w_2$"]
    for j, label in enumerate(labels):
        a, b = points[j:j+2]
        color = BLUE if j < 2 else ORANGE
        arrow(ax, a, b, color, width=2.1)
        delta = b-a
        offset = np.array([-delta[1], delta[0]])
        offset = offset/np.linalg.norm(offset)*.11
        ax.text(*((a+b)/2+offset), label, color=color,
                ha="center", va="center", fontsize=14)
    arrow(ax, points[-1], points[0], GRAY, dashed=True, width=1.8)
    ax.scatter(*points[-1], color=GRAY, s=16, zorder=4)
    ax.text(-.045, points[-1, 1]+.045, r"$(0,F(1))$",
            ha="right", va="bottom", fontsize=12)
    ax.text(-.10, .20, "closing edge\n"+r"angle $3\pi/2$",
            color=GRAY, ha="right", va="center", fontsize=11)
    ax.text(-.035, -.075, "$(0,0)$", ha="right", va="top", fontsize=11)
    ax.text(.5, 1.03, r"$u_1>u_2>0,\qquad w_1>w_2>0$",
            transform=ax.transAxes, ha="center", fontsize=13)
    ax.text(.5, -.12, r"$F(1)>0$: the downward closing edge comes last",
            transform=ax.transAxes, ha="center", fontsize=11)


def main():
    check()
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "mathtext.fontset": "cm", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    fig.subplots_adjust(left=.035, right=.98, top=.86, bottom=.18, wspace=.25)
    directions(axes[0])
    ordered_polygon(axes[1])
    fig.text(.035, .94, "(a) Buy and sell edge angles", fontsize=13)
    fig.text(.56, .94, "(b) Counterclockwise angular order", fontsize=13)
    fig.text(.76, .04, r"If $F(1)=0$, the closing edge has zero length.",
             ha="center", fontsize=11)
    base = Path(__file__).with_suffix("")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
