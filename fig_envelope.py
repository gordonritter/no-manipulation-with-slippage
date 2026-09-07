"""The slippage envelope around the square-root law, the adversarial
impact function of the envelope-ceiling theorem, and its witness round trip.

Units: lambda = v_* = 1, kernel G(t) = t^{-1/4} (c_G = 4/3, nu_G = 0.763),
slippage eta = varsigma = s with s chosen so that inf_m Theta_m^* = 1,
i.e. 8 s^2 = nu_G c_G (the boundary of the sufficient condition).

(a) f(x) = sqrt(x), the optimal chord m_* x, and the band
    m_* x +/- Theta^* (eta x / nu_G + varsigma / c_G) with Theta^* = 1.
(b) The adversary f~(x) = theta (varsigma/c_G) g_delta(x) + (m - theta eta/nu_G)^+ x
    at theta = 2.5 > 2, drawn in the band of width theta.
(c) The witness round trip of the lemma on mu_G (h = 0.1, tau = 0.8,
    epsilon = 2): sell everything at once, trickle of sales, buy back.

Output: ../figs/fig_envelope.pdf, relative to this script.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM, VSTAR = 1.0, 1.0
C_G, NU_G = 4.0 / 3.0, 0.763
S = np.sqrt(NU_G * C_G / 8.0)          # eta = varsigma = s on the boundary
ETA = SIG = S
ALPHA, BETA = ETA / NU_G, SIG / C_G     # weights in the denominator of Theta


def sqrt_law(x):
    return LAM * np.sqrt(x * VSTAR)


def m_star():
    a = LAM * np.sqrt(VSTAR)
    return a * np.sqrt(ALPHA) / (2.0 * np.sqrt(2.0) * np.sqrt(BETA))


def tent(t, p, l):
    return 2.0 / l * np.maximum(0.0, 1.0 - np.abs(2.0 * (t - p) / l - 1.0))


plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
navy, red = "#141440", "#B03030"

# (a) envelope around the square-root law at Theta^* = 1
x = np.linspace(0.0, 3.0, 600)
m = m_star()
theta = 1.0
half = theta * (ALPHA * x + BETA)
ax[0].fill_between(x, m * x - half, m * x + half, color="0.88", lw=0)
ax[0].plot(x, m * x, "--", lw=1.0, color="0.4", label=r"$m_* x$")
ax[0].plot(x, sqrt_law(x), "-", lw=1.8, color=navy, label=r"$f(x) = \sqrt{x}$")
# the square root touches the upper edge at y0 = sqrt(1+p^2) - p, p = 1/(2 sqrt 2),
# and approaches the lower edge as x -> infinity (ratio -> -p)
y0 = np.sqrt(1.0 + (1.0 / (2.0 * np.sqrt(2.0))) ** 2) - 1.0 / (2.0 * np.sqrt(2.0))
x0 = BETA / ALPHA * y0 ** 2
ax[0].plot([x0], [sqrt_law(x0)], "o", ms=4, color=red)
ax[0].set_xlim(0, 3.0)
ax[0].set_ylim(-0.4, 2.2)
ax[0].set_xlabel(r"$x / v_*$")
ax[0].set_ylabel(r"impact / $(\lambda v_*)$")
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
ax[0].set_title("(a)", loc="left", fontsize=10)

# (b) adversary in the band of width theta = 2.5
theta = 2.5
delta = 0.02
mm = m  # same comparison slope
half = theta * (ALPHA * x + BETA)
g = np.minimum(x / delta, 1.0)
f_adv = theta * BETA * g + max(mm - theta * ALPHA, 0.0) * x
ax[1].fill_between(x, mm * x - half, mm * x + half, color="0.88", lw=0)
ax[1].plot(x, mm * x, "--", lw=1.0, color="0.4", label=r"$m x$")
ax[1].plot(x, sqrt_law(x), "-", lw=1.0, color="0.6", label=r"$\sqrt{x}$")
ax[1].plot(x, f_adv, "-", lw=1.8, color=red, label=r"$\tilde f(x)$")
ax[1].axhline(0.0, lw=0.6, color="0.6")
ax[1].set_xlim(0, 3.0)
ax[1].set_ylim(-3.0, 4.0)
ax[1].set_xlabel(r"$x / v_*$")
ax[1].legend(frameon=False, fontsize=9, loc="upper left")
ax[1].set_title("(b)", loc="left", fontsize=10)

# (c) witness round trip
t = np.linspace(0.0, 1.0, 4001)
h, tau, eps = 0.1, 0.8, 2.0
phi = np.clip(np.minimum(t / h, (tau - t) / h), 0.0, 1.0) * (t <= tau)
q = 1.0 + eps * (tau - h)
v = -tent(t, 0.0, h) - eps * phi + q * tent(t, 1.0 - h, h)
ax[2].axhline(0.0, lw=0.6, color="0.6")
ax[2].plot(t, v, "-", lw=1.6, color=navy)
ax[2].set_xlim(0, 1)
ax[2].set_xlabel(r"$t$")
ax[2].set_ylabel(r"$v(t)$")
ax[2].set_title("(c)", loc="left", fontsize=10)

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_envelope.pdf", bbox_inches="tight")
print(f"s = {S:.4f}, m_* = {m:.4f}, touching point x0 = {x0:.4f}")
