"""Extremal round trip for permanent impact (G = 1) and the square-root law.

With Z(t) = int_0^t v and Z(0) = Z(1) = 0,

    K(v) = -<v, P f(v)> = int_0^1 Z(t) f(v(t)) dt,      f(x) = sign(x) sqrt|x|,

and Xi_{1/2} = sup K / (||v||_2 ||v||_1^{1/2}) over round trips.  The
Euler-Lagrange equations of K - mu ||v||_2^2 - nu ||v||_1 have the first
integrals

    buy leg  (v > 0):   Z sqrt(v)/2  + mu v^2 = k,
    sell leg (v < 0):  -Z sqrt|v|/2  + mu v^2 = k,

with the same k on both legs (Erdmann corner condition).  Parametrizing by
w = sqrt(v) makes t, Z and every integral elementary.  With mu = 1 and
k = omega^4 the buy leg runs w from omega down to r*omega and the sell leg
runs u = sqrt|v| from rho*omega down to omega, where rho = rho(r) is fixed
by continuity of Z at the corner and omega by the unit horizon.  This
script evaluates the algebraic global maximum, prints the extremal's
parameters, and checks the value against a fine discretization of the
extremal path. With z = (r*rho)**2, the maximizing z is the unique root
in (0, 1) of 18*z**3 + 39*z**2 + 8*z - 5 = 0.

Output: printed parameters (no figure).
"""

import numpy as np
from scipy.optimize import brentq


def rho_of(r):
    """Corner continuity of Z: (1 - r^4)/r = (rho^4 - 1)/rho."""
    return brentq(lambda p: (p ** 4 - 1) / p - (1 - r ** 4) / r, 1 + 1e-9, 1e3, xtol=1e-15)


def pieces(r):
    p = rho_of(r)
    tau = 6 * (p - r) + (2 / 3) * (r ** -3 - p ** -3)          # horizon / omega
    omega = 1 / tau
    K_hat = 4 * (2 * (1 - r) + (r ** -3 - 1) / 3 - 0.6 * (1 - r ** 5)) \
        - 4 * (0.6 * (p ** 5 - 1) - 2 * (p - 1) - (1 - p ** -3) / 3)
    A_hat = 2 * (0.6 * (1 - r ** 5) + (1 - r)) + 2 * (0.6 * (p ** 5 - 1) + (p - 1))
    B_hat = 4 * (1 - r ** 4) / r
    q = omega * K_hat / np.sqrt(A_hat * B_hat)               # K = omega^5 K_hat etc.
    T_b = omega * (6 * (1 - r) + (2 / 3) * (r ** -3 - 1))
    return dict(rho=p, omega=omega, q=q, T_b=T_b, Q_c=2 * omega ** 3 * (1 - r ** 4) / r)


def extremal_path(r, n):
    """Bucket averages of the extremal rate on n equal buckets."""
    d = pieces(r)
    p, om = d["rho"], d["omega"]
    # buy leg: t(s) for s = w/omega in [r, 1]; sell leg: t(s) for s = u/omega in [1, rho]
    t_buy = lambda s: om * (6 * (1 - s) + (2 / 3) * (s ** -3 - 1))
    t_sell = lambda s: d["T_b"] + om * (6 * (p - s) + (2 / 3) * (s ** -3 - p ** -3))
    edges = np.linspace(0, 1, n + 1)
    v = np.zeros(n)
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        if b <= d["T_b"]:
            sa = brentq(lambda s: t_buy(s) - a, r, 1); sb = brentq(lambda s: t_buy(s) - b, r, 1)
            Qa = 2 * om ** 3 * (1 - sa ** 4) / sa; Qb = 2 * om ** 3 * (1 - sb ** 4) / sb
        elif a >= d["T_b"]:
            sa = brentq(lambda s: t_sell(s) - a, 1, p); sb = brentq(lambda s: t_sell(s) - b, 1, p)
            Qa = 2 * om ** 3 * (sa ** 4 - 1) / sa; Qb = 2 * om ** 3 * (sb ** 4 - 1) / sb
        else:
            sa = brentq(lambda s: t_buy(s) - a, r, 1); sb = brentq(lambda s: t_sell(s) - b, 1, p)
            Qa = 2 * om ** 3 * (1 - sa ** 4) / sa; Qb = 2 * om ** 3 * (sb ** 4 - 1) / sb
        v[i] = (Qb - Qa) * n
    return v


def quotient_discrete(v):
    n = len(v); h = 1 / n
    f = np.sign(v) * np.sqrt(np.abs(v))
    F = np.cumsum(f) * h - f * h
    K = -(np.sum(v * F) * h + np.sum(v * f) * h * h / 2)
    return K / np.sqrt(h * np.sum(v * v) * h * np.sum(np.abs(v)))


if __name__ == "__main__":
    z = brentq(lambda z: 18*z**3 + 39*z**2 + 8*z - 5, 0, 1, xtol=1e-15)
    s = np.sqrt(z)
    r = (np.sqrt(1/s + 3*s) - np.sqrt(1/s - s)) / 2
    d = pieces(r)
    algebraic_constant = np.sqrt((1584*z**2 + 2046*z - 535) / 15360)
    assert abs(d["q"] - algebraic_constant) < 1e-13
    print(f"z*     = {z:.15f}")
    print(f"r*     = {r:.8f}")
    print(f"rho*   = {d['rho']:.8f}")
    print(f"Xi_1/2 = {d['q']:.8f}   (lambda = v_* = 1)")
    print(f"buy leg duration T_b = {d['T_b']:.6f}; rates relative to v(0): buy 1 -> {r**2:.6f}, "
          f"sell {-d['rho']**2:.6f} -> -1")
    for n in (256, 1024, 4096):
        print(f"discretized extremal, n={n}: quotient = {quotient_discrete(extremal_path(r, n)):.8f}")
