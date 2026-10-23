"""Plot the sufficient concession bound for assumed market parameters in v2."""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.linalg import toeplitz
from scipy.sparse.linalg import eigsh


DAILY_VOLATILITY = 0.02
Y0_VALUES = (0.25, 0.5, 0.75, 1.0)
SPREAD_BP = np.linspace(10.0, 150.0, 1000)


def required_b_eta(spread_bp, y0, ratio=1.0):
    """Return the bound in bp of cost per one percent of ADV."""
    spread_return = spread_bp * 1.0e-4
    return ratio * 25.0 * y0**2 * DAILY_VOLATILITY**2 / spread_return


def power_kernel_ratio(gamma=0.25, n=2048):
    """Galerkin estimate using exact cell integrals, not a certified value.

    The symmetric kernel matrix has diagonal 2*F(h)/h and off-diagonal
    second differences of F divided by h, where F''(u) = u**(-gamma).
    """
    h = 1.0 / n
    grid = np.arange(n + 1) * h
    primitive = grid ** (2 - gamma) / ((1 - gamma) * (2 - gamma))
    col = np.empty(n)
    col[0] = 2 * primitive[1] / h
    col[1:] = np.diff(primitive, n=2) / h
    matrix = toeplitz(col)
    eigenvalue = eigsh(matrix, k=1, which="LA", v0=np.ones(n),
                       return_eigenvectors=False, tol=1e-12)[0]
    return 0.5 * eigenvalue * (1 - gamma)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

fig, (ax, kernel_ax) = plt.subplots(1, 2, figsize=(11.0, 4.3))
colors = ("#0072B2", "#009E73", "#E69F00", "#000000")
linestyles = ("-", "--", "-.", ":")
labels = (r"$Y_0 = 1/4$", r"$Y_0 = 1/2$", r"$Y_0 = 3/4$", r"$Y_0 = 1$")

for y0, color, linestyle, label in zip(
    Y0_VALUES, colors, linestyles, labels, strict=True
):
    ax.plot(
        SPREAD_BP,
        required_b_eta(SPREAD_BP, y0),
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )

ax.set_xlim(10.0, 150.0)
ax.set_ylim(0.0, 10.25)
ax.set_xticks((10, 20, 50, 100, 150))
ax.set_yticks((0, 2.5, 5, 7.5, 10))
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
ax.set_xlabel(r"Full bid-ask spread $s_{\mathrm{BA}}$ (basis points)")
ax.set_ylabel(r"Lower bound on $b_\eta$ (bp per 1% of ADV)")
ax.grid(which="major", color="0.85", linewidth=0.6)
ax.legend(frameon=False, ncol=4)

ratio_lower = 4.0 / 7.0
ratio_upper = 2.0 ** (-0.75)
ratio_estimate = power_kernel_ratio()
assert ratio_lower <= ratio_estimate <= ratio_upper
print(f"G(t)=t^(-1/4): c_G={4/3:.12f}, "
      f"nu_G estimate={ratio_estimate*4/3:.12f}, "
      f"ratio estimate={ratio_estimate:.12f}")
print(f"Analytic ratio interval: [{ratio_lower:.12f}, {ratio_upper:.12f}]")
kernel_ax.fill_between(
    SPREAD_BP, required_b_eta(SPREAD_BP, 0.5, ratio_lower),
    required_b_eta(SPREAD_BP, 0.5, ratio_upper),
    color="#56B4E9", alpha=0.4, label="Analytic interval",
)
kernel_ax.plot(SPREAD_BP, required_b_eta(SPREAD_BP, 0.5),
               color="0.45", linestyle=":", label="Worst-case ratio 1")
kernel_ax.plot(SPREAD_BP, required_b_eta(SPREAD_BP, 0.5, ratio_upper),
               color="#0072B2", linewidth=2,
               label="Guaranteed sufficient bound")
kernel_ax.plot(SPREAD_BP, required_b_eta(SPREAD_BP, 0.5, ratio_estimate),
               color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"Numerical ratio {ratio_estimate:.3f}")
kernel_ax.set_xlim(10, 150)
kernel_ax.set_ylim(0, 2.6)
kernel_ax.set_xticks((10, 20, 50, 100, 150))
kernel_ax.set_xlabel(r"Full bid-ask spread $s_{\mathrm{BA}}$ (basis points)")
kernel_ax.grid(which="major", color="0.85", linewidth=0.6)
kernel_ax.legend(frameon=False, fontsize=9)
ax.set_title("(a) Worst-case kernel ratio")
kernel_ax.set_title(r"(b) $G(t)=t^{-1/4}$, $Y_0=1/2$")
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_b_eta_spread_v2.pdf", bbox_inches="tight")
