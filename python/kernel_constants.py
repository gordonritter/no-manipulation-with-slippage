"""Kernel constants c_G, nu_G, a_G for power-law propagators G(t) = t^{-gamma}.

Discretizes the one-sided operator P on [0, 1] with n equal buckets, using
exact double integrals of the kernel over each pair of buckets, and reports

    nu_G = (1/2) * largest eigenvalue of P + P^*,
    a_G  = (1/2) * operator norm of P - P^*,

together with the L^2 operator norm of P and c_G = 1 / (1 - gamma).
gamma = 0 is the constant kernel, for which nu_G = 1/2, ||P|| = 2/pi and
a_G = 1/pi are exact and serve as a check.
"""

import numpy as np
from scipy.linalg import toeplitz


def constants(gamma, n=1024):
    h = 1.0 / n
    if gamma == 0:
        F = lambda u: u ** 2 / 2  # noqa: E731
        c_G = 1.0
    else:
        F = lambda u: u ** (2 - gamma) / ((1 - gamma) * (2 - gamma))  # noqa: E731
        c_G = 1.0 / (1 - gamma)
    d = np.arange(n)
    col = np.zeros(n)
    col[0] = F(h)
    col[1:] = F((d[1:] + 1) * h) - 2 * F(d[1:] * h) + F((d[1:] - 1) * h)
    P = toeplitz(col, np.zeros(n)) / h  # lower triangular, L^2-normalized
    nu_G = 0.5 * np.linalg.eigvalsh(P + P.T).max()
    a_G = 0.5 * np.linalg.norm(P - P.T, 2)
    normP = np.linalg.norm(P, 2)
    return c_G, nu_G, a_G, normP


if __name__ == "__main__":
    print("gamma    c_G     nu_G    a_G     ||P||")
    for gamma in [0.0, 0.1, 0.25, 0.4]:
        c_G, nu_G, a_G, normP = constants(gamma)
        print(f"{gamma:<6.2f} {c_G:7.4f} {nu_G:7.4f} {a_G:7.4f} {normP:7.4f}")
    print("checks for gamma = 0: 1/2 =", 0.5, " 1/pi =", 1 / np.pi, " 2/pi =", 2 / np.pi)
