"""
Extremal manipulation constant Xi_eps for the power-law impact model with
transient propagator G(t) = t^{-gamma} on the unit horizon.

For f(x) = lambda * x * (x/v_star)^{-eps} the cost of a round trip v is
positively homogeneous piecewise in the scale c > 0,

    C[c v] = eta c^2 ||v||_2^2 + varsigma c ||v||_1 - c^{2-eps} K(v),
    K(v)   = -<v, P f(v)>,

so absence of price manipulation is equivalent to

    eta^{1-eps} varsigma^eps >= eps^eps (1-eps)^{1-eps} Xi_eps,
    Xi_eps = sup { K(v) / ( ||v||_2^{2(1-eps)} ||v||_1^eps ) : int v = 0 }.

This script computes Xi_eps by multistart optimization over piecewise
constant paths, reports the operator norms entering the analytic bound,
and draws the extremal round trip together with its scaled cost.

Outputs: ../figs/fig_extremal_round_trip.pdf and extremal_round_trip.csv,
relative to this script.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

GAMMA = 0.25
EPS = 0.5
LAM = 1.0
VSTAR = 1.0
SEED = 3


def propagator_matrix(n, gamma):
    """Exact Gram matrix P_ij = int_{I_i} int_{I_j, s<t} G(t-s) ds dt."""
    h = 1.0 / n
    c1, c2 = 1.0 - gamma, 2.0 - gamma
    Psi = lambda x: x ** c2 / (c1 * c2)
    row = np.zeros(n)
    row[0] = Psi(h)
    d = np.arange(1, n)
    row[1:] = Psi((d + 1) * h) - 2 * Psi(d * h) + Psi((d - 1) * h)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, : i + 1] = row[i - np.arange(i + 1)]
    return P, h


def impact(v, eps=EPS, lam=LAM, vstar=VSTAR):
    a = np.abs(v)
    return np.sign(v) * lam * a * (a / vstar) ** (-eps)


def extremal(n, gamma, n_random=64, seed_u=None, seed=SEED):
    """Maximize K(v) / (||v||_2^{2(1-eps)} ||v||_1^eps) over mean-zero paths.

    The substitution v = sign(u) u^2 makes f(v) = u for eps = 1/2, which
    removes the square-root cusp from the objective.
    """
    rng = np.random.default_rng(seed)
    P, h = propagator_matrix(n, gamma)
    t = np.linspace(0, 1, n, endpoint=False) + 0.5 / n

    def quotient(u):
        v = np.sign(u) * u * u
        n2 = np.sqrt(h * np.sum(v * v))
        n1 = h * np.sum(np.abs(v))
        den = n2 ** (2 * (1 - EPS)) * n1 ** EPS
        return -(v @ (P @ u)) / (den + 1e-300)

    cons = [
        {"type": "eq", "fun": lambda u: np.sum(np.sign(u) * u * u)},
        {"type": "eq", "fun": lambda u: np.sum(u * u) - 1.0},
    ]

    starts = []
    if seed_u is not None:
        ts = np.linspace(0, 1, len(seed_u), endpoint=False) + 0.5 / len(seed_u)
        starts.append(np.interp(t, ts, seed_u))
    for k in range(1, 5):
        starts += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    starts += [
        np.where(t < 0.5, 1.0, -1.0),
        np.where(t < 0.5, -1.0, 1.0),
        np.where(t < 0.05, 3.0, -1.0),
        np.where(t > 0.95, 3.0, -1.0),
    ]
    starts += [rng.standard_normal(n) for _ in range(n_random)]

    best_val, best_u = -np.inf, None
    for u0 in starts:
        u0 = u0 / np.linalg.norm(u0)
        r = minimize(lambda u: -quotient(u), u0, constraints=cons,
                     method="SLSQP", options={"maxiter": 350, "ftol": 1e-12})
        v = np.sign(r.x) * r.x ** 2
        if abs(v.sum()) / max(abs(v).sum(), 1e-12) > 1e-6:
            continue
        val = quotient(r.x)
        if val > best_val:
            best_val, best_u = val, r.x.copy()
    return best_val, np.sign(best_u) * best_u ** 2, P, h


def main(reuse=False):
    n = 128
    P, h = propagator_matrix(n, GAMMA)
    t = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    if reuse:
        t, v = np.loadtxt(Path(__file__).resolve().parent / "extremal_round_trip.csv", delimiter=",",
                          skiprows=1, unpack=True)
        n2 = np.sqrt(h * np.sum(v * v))
        n1 = h * np.sum(np.abs(v))
        Xi = -(v @ (P @ impact(v))) / (n2 ** (2 * (1 - EPS)) * n1 ** EPS)
    else:
        Xi, v, P, h = extremal(n, GAMMA)
    v = v / np.sqrt(h * np.sum(v * v))

    c_G = 1.0 / (1.0 - GAMMA)
    nu_G = 0.5 * np.linalg.eigvalsh((P + P.T) / h).max()
    bound = nu_G ** (1 - EPS) * c_G ** EPS * LAM * VSTAR ** EPS / 2.0 ** (1 - EPS)
    A_op = (P - P.T) / h
    print(f"gamma = {GAMMA}, eps = {EPS}, n = {n}")
    print(f"  c_G, nu_G           = {c_G:.6f}, {nu_G:.6f}")
    print(f"  ||P||_2, ||A||_2    = {np.linalg.norm(P / h, 2):.6f}, "
          f"{np.linalg.norm(A_op, 2):.6f}")
    print(f"  Xi_eps  (numerical lower bound) = {Xi:.6f}")
    print(f"  Xi_eps  (bound)     = {bound:.6f}")
    print(f"  ratio               = {Xi / bound:.6f}")

    Amass = h * np.sum(v * v)
    Bmass = h * np.sum(np.abs(v))
    K = -(v @ (P @ impact(v)))
    s_ray = 0.5 * Xi
    print(f"  critical common slippage along this ray s_ray = {s_ray:.6f}")
    np.savetxt(Path(__file__).resolve().parent / "extremal_round_trip.csv", np.c_[t, v], delimiter=",",
               header="t,v", comments="")

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax[0].axhline(0.0, lw=0.6, color="0.6")
    ax[0].step(t, v, where="mid", lw=1.0, color="#141440")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$v_{\mathrm{num}}(t)$")
    ax[0].set_title("(a)", loc="left", fontsize=10)

    cs = np.linspace(1e-4, 1.2, 800)
    ax[1].axhline(0.0, lw=0.6, color="0.6")
    for fac, style in [(0.9, ":"), (1.0, "-"), (1.1, "--")]:
        s = fac * s_ray
        cost = s * Amass * cs ** 2 + s * Bmass * cs - K * cs ** (2 - EPS)
        ax[1].plot(cs, cost, style, lw=1.2, color="#141440",
                   label=rf"$s/s_{{\mathrm{{ray}}}} = {fac:g}$")
    ax[1].set_ylim(-1.6e-3, 4.0e-3)
    ax[1].set_xlabel(r"$c$")
    ax[1].set_ylabel(r"$C_f^{s,s}[c\,v_{\mathrm{num}}]$")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b)", loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_extremal_round_trip.pdf")


if __name__ == "__main__":
    import sys
    main(reuse="--reuse" in sys.argv)
