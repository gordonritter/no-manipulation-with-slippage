"""Plot the market-data lower bound on rate-proportional concessions."""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DAILY_VOLATILITY = 0.02
Y0_VALUES = (0.25, 0.5, 0.75, 1.0)
SPREAD_BP = np.linspace(10.0, 150.0, 1000)


def required_b_eta(spread_bp, y0):
    """Return the bound in bp of cost per one percent of ADV."""
    spread_return = spread_bp * 1.0e-4
    return 25.0 * y0**2 * DAILY_VOLATILITY**2 / spread_return


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
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
ax.set_ylabel(r"Lower bound on $b_\eta$ (bp per 1\% of ADV)")
ax.grid(which="major", color="0.85", linewidth=0.6)
ax.legend(frameon=False, ncol=4)

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_b_eta_spread.pdf", bbox_inches="tight")
