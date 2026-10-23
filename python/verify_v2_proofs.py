"""Independent numerical tests of the arguments in no_manipulation_v2.tex.

Run: python3 python/verify_v2_proofs.py
Each test prints PASS or FAIL with the quantities compared. The tests use
step-function trading rates on n equal cells with exact cell integrals of
the kernel, so every inequality is evaluated for an admissible path up to
floating-point rounding. Tolerances are stated beside each check.
"""

import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq

rng = np.random.default_rng(20261022)
FAILURES = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"{tag}: {name} {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- kernels
def cell_matrix(gamma, n):
    """M[i,j] = int_{cell i} int_{cell j, s<t} G(t-s) ds dt for G=t^-gamma."""
    h = 1.0 / n
    a = 1.0 - gamma
    F = lambda x: x ** (1 + a) / (a * (1 + a))
    d = np.arange(n + 1) * h
    Fd = F(d)
    w = np.empty(n)
    w[0] = Fd[1]
    w[1:] = Fd[2:] - 2 * Fd[1:-1] + Fd[:-2]
    idx = np.subtract.outer(np.arange(n), np.arange(n))
    M = np.where(idx >= 0, w[np.abs(idx)], 0.0)
    return M


class Model:
    def __init__(self, gamma, n):
        self.gamma, self.n, self.h = gamma, n, 1.0 / n
        self.M = cell_matrix(gamma, n)
        self.cG = 1.0 / (1.0 - gamma)
        S = self.M + self.M.T
        self.nuG = np.linalg.eigvalsh(S / self.h).max() / 2
        K = self.M - self.M.T
        self.aG = np.linalg.norm(K / self.h, 2) / 2

    def ip_P(self, v, w):
        """<v, P w> for step functions v, w."""
        return v @ self.M @ w

    def cost(self, v, f, eta, sigma):
        h = self.h
        return eta * h * v @ v + sigma * h * np.abs(v).sum() + self.ip_P(v, f(v))

    def random_round_trip(self, smooth=3):
        n = self.n
        t = (np.arange(n) + 0.5) / n
        v = np.zeros(n)
        for k in range(1, smooth + 1):
            v += rng.normal() * np.sin(k * np.pi * t) + rng.normal() * np.cos(k * np.pi * t)
        v -= v.mean()
        return v


def sqrt_law(lam=1.0, vstar=1.0):
    return lambda x: lam * np.sign(x) * np.sqrt(vstar * np.abs(x))


def power_law(eps, lam=1.0, vstar=1.0):
    return lambda x: lam * np.sign(x) * np.abs(x) * (np.abs(x) / vstar + 1e-300) ** (-eps)


# ---------------------------------------------------------------- Theorem 1
def test_theorem1():
    m = Model(0.25, 160)
    f = sqrt_law()
    worst = np.inf
    for _ in range(300):
        v = m.random_round_trip() * rng.uniform(0.1, 5)
        eta, sig, slope = rng.uniform(0.05, 2), rng.uniform(0.05, 2), rng.uniform(0, 3)
        xs = np.linspace(1e-6, np.abs(v).max(), 4000)
        theta = np.max(np.abs(f(xs) - slope * xs) / (eta * xs / m.nuG + sig / m.cG))
        lhs = m.cost(v, f, eta, sig)
        rhs = (1 - theta) * (eta * m.h * v @ v + sig * m.h * np.abs(v).sum())
        worst = min(worst, lhs - rhs)
    report("Theorem 1 cost bound (300 random v, eta, varsigma, m)", worst >= -1e-10,
           f"min(lhs-rhs)={worst:.3e}")


# ---------------------------------------------------------------- Prop quadratic
def test_prop_quadratic():
    m = Model(0.25, 160)
    kappa_star = 0.4
    f = lambda x: x * (1 + kappa_star * np.tanh(np.abs(x)))
    worst = np.inf
    for _ in range(300):
        v = m.random_round_trip() * rng.uniform(0.1, 5)
        eta = rng.uniform(0.05, 2)
        xs = np.linspace(1e-6, np.abs(v).max(), 4000)
        kappa_v = np.max(np.abs(f(xs) - xs) / xs)
        lhs = m.cost(v, f, eta, 0.0)
        rhs = (eta - (m.aG * kappa_v + m.nuG * kappa_v ** 2 / 4)) * m.h * v @ v
        worst = min(worst, lhs - rhs)
    report("Proposition (quadratic bound), lambda=1, kappa*<=0.4", worst >= -1e-10,
           f"min(lhs-rhs)={worst:.3e}")


# ---------------------------------------------------------------- power-law envelope
def test_power_envelope():
    eps = 0.3
    nu, c, lam, vstar = 0.763, 4 / 3, 1.0, 1.0
    f = power_law(eps, lam, vstar)
    xs = np.logspace(-8, 8, 200001)
    ok = True
    for _ in range(5):
        eta, sig = rng.uniform(0.05, 2), rng.uniform(0.05, 2)
        reta, rsig = eta / (nu * lam), sig / (c * lam * vstar)
        Aeps = eps ** eps * ((1 - eps) / 2) ** (1 - eps)
        theta_claim = Aeps / (reta ** (1 - eps) * rsig ** eps)
        m_claim = lam * Aeps * (reta / rsig) ** eps
        theta_of = lambda mm: max(np.max(np.abs(f(xs) - mm * xs) / (eta * xs / nu + sig / c)), mm * nu / eta)
        res = minimize_scalar(theta_of, bounds=(0, 10 * m_claim + 1), method="bounded",
                              options={"xatol": 1e-10})
        ok &= abs(res.fun - theta_claim) < 1e-6 * theta_claim and abs(res.x - m_claim) < 1e-4 * m_claim
        ok &= abs(theta_of(m_claim) - theta_claim) < 1e-6 * theta_claim
    report("Power-law inf_m Theta_m^* and m_eps^* (eps=0.3)", ok)


# ---------------------------------------------------------------- square-root envelope
def test_sqrt_envelope():
    f = sqrt_law()
    xs = np.logspace(-10, 10, 400001)
    ok = True
    for _ in range(8):
        al, be, mm = rng.uniform(0.1, 3), rng.uniform(0.1, 3), rng.uniform(0, 3)
        a = 1.0
        p = mm * math.sqrt(be) / (a * math.sqrt(al))
        claim = a / math.sqrt(al * be) * max(p, (math.sqrt(1 + p * p) - p) / 2)
        num = max(np.max(np.abs(f(xs) - mm * xs) / (al * xs + be)), mm / al)
        ok &= abs(num - claim) < 1e-7 * claim
    # contact point and minimizing slope
    al, be = 0.7, 1.9
    mstar = (1 / (2 * math.sqrt(2))) * math.sqrt(al / be)
    xc = be / (2 * al)
    val = abs(f(xc) - mstar * xc) / (al * xc + be)
    ok &= abs(val - 1 / (2 * math.sqrt(2 * al * be))) < 1e-12
    report("Square-root envelope formula, minimizer p=1/(2 sqrt 2), contact point", ok)


# ---------------------------------------------------------------- ray minimization
def test_exact_power_ray():
    m = Model(0.25, 160)
    ok = True
    for eps in (0.3, 0.5, 0.8):
        f = power_law(eps)
        for _ in range(60):
            v = m.random_round_trip()
            A, B = m.h * v @ v, m.h * np.abs(v).sum()
            K = -m.ip_P(v, f(v))
            eta, sig = rng.uniform(0.01, 1), rng.uniform(0.01, 1)
            phi = lambda lc: m.cost(np.exp(lc) * v, f, eta, sig) / np.exp(lc)
            res = minimize_scalar(phi, bounds=(-40, 40), method="bounded", options={"xatol": 1e-9})
            negative = res.fun < -1e-12
            crit = eps ** eps * (1 - eps) ** (1 - eps) * K
            predicted = eta ** (1 - eps) * sig ** eps * A ** (1 - eps) * B ** eps < crit
            if K > 0:
                c0 = ((1 - eps) * K / (eta * A)) ** (1 / eps)
                ok &= abs(phi(math.log(c0)) - (sig * B - eps / (1 - eps) * eta * A * c0)) < 1e-9
            ok &= negative == predicted
    report("Exact-power ray minimization and manipulation criterion", ok)


# ---------------------------------------------------------------- two-block
def test_two_block():
    gamma, eps = 0.25, 0.5
    n = 400
    m = Model(gamma, n)
    f = power_law(eps)
    ok = True
    for tau in (0.2, 0.5, 0.65, 0.9):
        k = int(round(tau * n))
        tau_g = k / n
        v = np.where(np.arange(n) < k, -1 / tau_g, 1 / (1 - tau_g))
        q = -m.ip_P(v, f(v)) / ((m.h * v @ v) ** (1 - eps) * (m.h * np.abs(v).sum()) ** eps)
        pref = (tau_g * (1 - tau_g)) ** (1 - eps) / (2 ** eps * (1 - gamma) * (2 - gamma))
        br = (1 - tau_g ** (2 - gamma) - (1 - tau_g) ** (2 - gamma)) / (tau_g ** (1 - eps) * (1 - tau_g)) \
            - tau_g ** (eps - gamma) - (1 - tau_g) ** (eps - gamma)
        ok &= abs(q - pref * br) < 1e-10
    report("Two-block lower bound formula vs direct evaluation", ok)


# ---------------------------------------------------------------- kernel constants G=1
def test_kernel_constants_permanent():
    m = Model(0.0, 1500)
    ok = abs(m.nuG - 0.5) < 2e-3 and abs(m.aG - 1 / math.pi) < 2e-3
    report("G=1: nu_G=1/2, a_G=1/pi", ok, f"nu={m.nuG:.5f} a={m.aG:.5f} 1/pi={1/math.pi:.5f}")


# ---------------------------------------------------------------- Lemma mu
def tent(t, p, l):
    return 2 / l * np.maximum(0, 1 - np.abs(2 * (t - p) / l - 1))


def lemma_mu_path(t, h, tau, zeta):
    phi = np.maximum(0, np.minimum(np.minimum(t / h, 1), (tau - t) / h))
    q = 1 + zeta * (tau - h)
    return -tent(t, 0, h) - zeta * phi + q * tent(t, 1 - h, h)


def test_lemma_mu():
    m = Model(0.25, 2000)
    t = (np.arange(m.n) + 0.5) / m.n
    ok = True
    ratios = []
    for h, tau, zeta in [(0.1, 0.8, 2.0), (0.02, 0.95, 0.2), (0.005, 0.99, 0.02)]:
        v = lemma_mu_path(t, h, tau, zeta)
        ratio = -m.ip_P(v, np.sign(v)) / (m.h * np.abs(v).sum())
        ratios.append(ratio)
        ok &= ratio <= m.cG + 1e-9
    ok &= ratios[-1] > 0.9 * m.cG / 2 and ratios == sorted(ratios)
    # random round trips never exceed c_G
    worst = max(-m.ip_P(v, np.sign(v)) / (m.h * np.abs(v).sum()) for v in
                (m.random_round_trip() for _ in range(200)))
    ok &= worst <= m.cG
    report("Lemma (mu_G): c_G/2 <= mu_G <= c_G, construction approaches c_G/2", ok,
           f"ratios={['%.4f' % r for r in ratios]} target c_G/2={m.cG/2:.4f}")


def test_mu_permanent():
    m = Model(0.0, 400)
    worst = 0.0
    for _ in range(400):
        v = m.random_round_trip(smooth=6)
        worst = max(worst, -m.ip_P(v, np.sign(v)) / (m.h * np.abs(v).sum()))
    # direct search on step functions
    n = 24
    mm = Model(0.0, n)
    best = 0.0
    for _ in range(30):
        x0 = rng.normal(size=n)
        obj = lambda x: -(-mm.ip_P(x - x.mean(), np.sign(x - x.mean())) / (mm.h * np.abs(x - x.mean()).sum() + 1e-12))
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-12})
        best = max(best, -res.fun)
    t = (np.arange(2000) + 0.5) / 2000
    m2 = Model(0.0, 2000)
    v = lemma_mu_path(t, 0.005, 0.99, 0.02)
    constr = -m2.ip_P(v, np.sign(v)) / (m2.h * np.abs(v).sum())
    ok = worst <= 0.5 + 1e-9 and best <= 0.5 + 1e-9 and constr > 0.48
    report("mu_G = 1/2 for G=1 (random, optimized, explicit paths all <= 1/2)", ok,
           f"random={worst:.4f} optimized={best:.4f} construction={constr:.4f}")


# ---------------------------------------------------------------- envelope ceiling
def test_envelope_ceiling():
    m = Model(0.25, 2000)
    t = (np.arange(m.n) + 0.5) / m.n
    s_ = math.sqrt(m.nuG * m.cG / 8)
    eta = sig = s_
    al, be = eta / m.nuG, sig / m.cG
    mstar = (1 / (2 * math.sqrt(2))) * math.sqrt(m.cG * eta / (m.nuG * sig))
    theta = 2.5
    v = lemma_mu_path(t, 0.005, 0.99, 0.02)
    witness = -theta * m.ip_P(v, np.sign(v)) - m.cG * m.h * np.abs(v).sum()
    ok = witness > 0
    def ftilde(x, delta):
        g = np.sign(x) * np.minimum(np.abs(x) / delta, 1)
        return theta * be * g + max(mstar - theta * al, 0) * x
    xs = np.logspace(-6, 3, 100000)
    env = np.max(np.abs(ftilde(xs, 0.02) - mstar * xs) / (theta * (al * xs + be)))
    ok &= env <= 1 + 1e-9
    found = False
    for c in (1e-2, 1e-3, 1e-4):
        for delta in (1e-3, 1e-4, 1e-5, 1e-6):
            if m.cost(c * v, lambda x: ftilde(x, delta), eta, sig) < 0:
                found = True
    report("Envelope ceiling: witness round trip and negative-cost f~ (theta=2.5)", ok and found,
           f"witness={witness:.4f} envelope ratio={env:.4f}")


# ---------------------------------------------------------------- transaction-triggered
def caps_path(v, h):
    p, nn = np.maximum(v, 0), np.maximum(-v, 0)
    Q, B = h * p.sum() - h * nn.sum(), h * nn.sum()

    def cap_for(hh, target):
        if target <= 0:
            return 0.0
        if target >= h * hh.sum() * (1 - 1e-13):
            return hh.max()
        return brentq(lambda c: h * np.minimum(hh, c).sum() - target, 0, hh.max() * (1 + 1e-12), xtol=1e-14)
    return p, nn, Q, B, cap_for


def test_ttpm():
    ok = True
    details = []
    for gamma in (0.25, 0.75):
        n = 256
        m = Model(gamma, n)
        RG = 2 ** gamma / (1 - gamma)
        vc, eps = 0.3, 0.5
        Lf = (1 + 1 / vc ** 2) ** (eps / 2)
        fc = lambda x: x * ((x ** 2 + vc ** 2) / (1 + vc ** 2)) ** (-eps / 2)
        for trial in range(25):
            v = m.random_round_trip(smooth=4) + rng.uniform(0.2, 1.5)
            if (v < 0).sum() == 0:
                continue
            p, nn, Q, B, cap_for = caps_path(v, m.h)
            u = np.minimum(p, cap_for(p, Q))
            D = m.h * (v @ v - u @ u)
            H = m.h * np.abs(v - u).sum()
            ok &= abs(H - 2 * B) < 1e-9 and abs(h_ := m.h * (np.abs(v).sum() - np.abs(u).sum()) - H) < 1e-9
            ss = np.linspace(0, B, 2001)
            a = np.array([cap_for(p, Q + B - s) for s in ss])
            b = np.array([cap_for(nn, B - s) for s in ss])
            D_formula = 2 * np.trapezoid(a + b, ss)
            ok &= abs(D - D_formula) < 1e-5 * max(D, 1)   # trapezoid rule on 2001 points
            ok &= np.all(np.diff(a) <= 1e-12) and np.all(np.diff(b) <= 1e-12)
            for eta, sig in ((Lf * RG, 0.0), (Lf * RG, 0.3), (1.5 * Lf * RG, 0.0)):
                gap = m.cost(v, fc, eta, sig) - m.cost(u, fc, eta, sig) - ((eta - Lf * RG) * D + sig * H)
                ok &= gap >= -1e-9
            J = abs(m.ip_P(v, fc(v)) - m.ip_P(u, fc(u)))
            details.append(J / (Lf * RG * D))
    report("Transaction-triggered theorem: caps, D identity, purification bound (gamma=0.25, 0.75)",
           ok, f"max |J[v]-J[u]| / (L_f R_G D) = {max(details):.3f}")


# ---------------------------------------------------------------- Appendix A
def hats(r):
    rho = brentq(lambda x: (1 - r ** 4) / r - (x ** 4 - 1) / x, 1.0, 50.0, xtol=1e-14)
    tau = 6 * (rho - r) + (2 / 3) * (r ** -3 - rho ** -3)
    K = 4 * (2 * (1 - r) + (r ** -3 - 1) / 3 - 0.6 * (1 - r ** 5)) \
        - 4 * (0.6 * (rho ** 5 - 1) - 2 * (rho - 1) - (1 - rho ** -3) / 3)
    A = 2 * (0.6 * (1 - r ** 5) + (1 - r)) + 2 * (0.6 * (rho ** 5 - 1) + (rho - 1))
    B = 4 * (1 - r ** 4) / r
    return rho, tau, K, A, B


def test_candidate_formulas():
    ok = True
    for r in (0.2, 0.33095, 0.6, 0.9):
        rho, tau, Kh, Ah, Bh = hats(r)
        w = 1 / tau
        # quadrature along the two curves (parameter s decreases in time)
        sb = np.linspace(r, 1, 200001)
        tb = w * (6 * (1 - sb) + (2 / 3) * (sb ** -3 - 1))
        Zb = 2 * w ** 3 * (1 - sb ** 4) / sb
        ss = np.linspace(1, rho, 200001)
        ts = tb[0] + w * (6 * (rho - ss) + (2 / 3) * (ss ** -3 - rho ** -3))
        Zs = 2 * w ** 3 * (ss ** 4 - 1) / ss
        # numerical derivative check v = dZ/dt = omega^2 s^2
        vb_num = np.gradient(Zb, tb)
        ok &= np.max(np.abs(vb_num[10:-10] - w ** 2 * sb[10:-10] ** 2)) < 1e-6 * w ** 2
        vs_num = np.gradient(Zs, ts)
        ok &= np.max(np.abs(vs_num[10:-10] + w ** 2 * ss[10:-10] ** 2)) < 1e-6 * w ** 2 * rho ** 2
        # continuity, endpoints
        ok &= abs(tb[-1]) < 1e-12 and abs(ts[0] - 1) < 1e-12 and abs(Zb[0] - Zs[-1]) < 1e-12
        # K, A, B by quadrature in time (sort by t)
        t_all = np.concatenate([tb[::-1], ts[::-1]])
        Z_all = np.concatenate([Zb[::-1], Zs[::-1]])
        v_all = np.concatenate([w ** 2 * sb[::-1] ** 2, -w ** 2 * ss[::-1] ** 2])
        phi = np.sign(v_all) * np.sqrt(np.abs(v_all))
        K = np.trapezoid(Z_all * phi, t_all)
        A = np.trapezoid(v_all ** 2, t_all)
        B = np.trapezoid(np.abs(v_all), t_all)
        ok &= abs(K - w ** 5 * Kh) < 1e-7 * abs(K)
        ok &= abs(A - w ** 5 * Ah) < 1e-7 * A and abs(B - w ** 3 * Bh) < 1e-7 * B
        # stationarity with mu=1, ell=omega^4 along both curves
        wp = w * sb
        ok &= np.max(np.abs(wp ** 4 + Zb * wp / 2 - w ** 4)) < 1e-10
        wm = w * ss
        ok &= np.max(np.abs(wm ** 4 - Zs * wm / 2 - w ** 4)) < 1e-10
    report("Appendix A candidate: rates, continuity, K/A/B formulas, stationarity", ok)


def K_time(rates, durs):
    """K = int Z phi(v) dt for a piecewise constant rate (G=1)."""
    Z = 0.0
    K = 0.0
    for u, d in zip(rates, durs):
        Z1 = Z + u * d
        K += np.sign(u) * math.sqrt(abs(u)) * (Z + Z1) / 2 * d
        Z = Z1
    return K, Z


def signed_area(edges):
    x = y = 0.0
    area = 0.0
    for dx, dy in edges:
        area += (x + dx / 2) * dy  # int Z dF along the edge
        x += dx
        y += dy
    return area


def test_rearrangement():
    ok = True
    for _ in range(300):
        k = rng.integers(3, 9)
        rates = rng.normal(size=k) * rng.uniform(0.2, 3)
        durs = rng.uniform(0.05, 1, size=k)
        rates -= (rates * durs).sum() / durs.sum()      # round trip
        if np.any(np.abs(rates) < 1e-9):
            continue
        K, Z = K_time(rates, durs)
        ok &= abs(Z) < 1e-12
        F1 = sum(np.sign(u) * math.sqrt(abs(u)) * d for u, d in zip(rates, durs))
        if F1 < 0:
            rates = -rates
            K, _ = K_time(rates, durs)
            F1 = -F1
        order = sorted(range(k), key=lambda i: math.atan2(np.sign(rates[i]) * math.sqrt(abs(rates[i])), rates[i]) % (2 * math.pi))
        rs, ds = rates[order], durs[order]
        Ks, _ = K_time(rs, ds)
        ok &= Ks >= K - 1e-12
        ok &= np.all(rs[:np.sum(rs > 0)] > 0) and np.all(np.diff(rs[rs > 0]) <= 1e-12) and np.all(np.diff(rs[rs < 0]) >= -1e-12)
        A1, A2 = (rates ** 2 * durs).sum(), (rs ** 2 * ds).sum()
        B1, B2 = (np.abs(rates) * durs).sum(), (np.abs(rs) * ds).sum()
        ok &= abs(A1 - A2) < 1e-12 and abs(B1 - B2) < 1e-12
        # Fary-Makai on the augmented polygon with random permutations
        edges = [(u * d, np.sign(u) * math.sqrt(abs(u)) * d) for u, d in zip(rates, durs)] + [(0.0, -F1)]
        angles = [math.atan2(e[1], e[0]) % (2 * math.pi) for e in edges]
        sorted_edges = [edges[i] for i in np.argsort(angles)]
        best = signed_area(sorted_edges)
        for _ in range(20):
            perm = rng.permutation(len(edges))
            ok &= signed_area([edges[i] for i in perm]) <= best + 1e-12
        ok &= abs(best - Ks) < 1e-12
    report("Angular rearrangement: K weakly increases, A and B preserved, sorted order is extremal", ok)


def random_buy_sell(kb, ks):
    """Buy-then-sell piecewise constant path with B=2, peak inventory 1, trading time <= 1."""
    xb = np.sort(np.concatenate([[0, 1], rng.uniform(0, 1, kb - 1)]))
    xs = np.sort(np.concatenate([[0, 1], rng.uniform(0, 1, ks - 1)]))
    ub = np.exp(rng.normal(size=kb) * rng.uniform(0.2, 1.5))
    us = np.exp(rng.normal(size=ks) * rng.uniform(0.2, 1.5))
    T = (np.diff(xb) / ub).sum() + (np.diff(xs) / us).sum()
    scale = T / rng.uniform(0.6, 1.0)   # trading time = T/scale <= 1
    ub, us = ub * scale, us * scale
    return xb, ub, xs, us


def path_from_inventory(xb, ub, xs, us):
    # sells run from inventory 1 down to 0, so their time order reverses the inventory order
    rates = np.concatenate([ub, -us[::-1]])
    durs = np.concatenate([np.diff(xb) / ub, (np.diff(xs) / us)[::-1]])
    return rates, durs


def test_inventory_substitution():
    ok = True
    for _ in range(200):
        xb, ub, xs, us = random_buy_sell(rng.integers(1, 6), rng.integers(1, 6))
        rates, durs = path_from_inventory(xb, ub, xs, us)
        K, Z = K_time(rates, durs)
        A = (rates ** 2 * durs).sum()
        B = (np.abs(rates) * durs).sum()
        Ta = durs.sum()
        # inventory formulas
        K_inv = sum((x1 ** 2 - x0 ** 2) / 2 / math.sqrt(u) for x0, x1, u in zip(xb[:-1], xb[1:], ub)) \
            - sum((x1 ** 2 - x0 ** 2) / 2 / math.sqrt(w) for x0, x1, w in zip(xs[:-1], xs[1:], us))
        A_inv = (np.diff(xb) * ub).sum() + (np.diff(xs) * us).sum()
        T_inv = (np.diff(xb) / ub).sum() + (np.diff(xs) / us).sum()
        ok &= abs(K - K_inv) < 1e-12 and abs(A - A_inv) < 1e-12 and abs(B - 2) < 1e-12
        ok &= abs(Ta - T_inv) < 1e-12 and Ta <= 1 + 1e-12 and A * Ta >= 4 - 1e-12
    report("Inventory substitution: K, A, T formulas; A T_a >= 4", ok)


def A_of_z(z):
    return (9 * z ** 2 + 2 * z + 1) * (3 * z ** 2 + 14 * z + 3) / (15 * z ** 2 * (1 + z) ** 2)


def R_of_z(z):
    return z ** 2 * (1 - z) * (9 * z ** 2 + 6 * z + 5) ** 2 / (10 * (1 + z) * (9 * z ** 2 + 2 * z + 1) ** 2 * (3 * z ** 2 + 14 * z + 3))


def candidate_from_z(z):
    s = math.sqrt(z)
    d = math.sqrt(1 / s - s)
    r = (math.sqrt(1 / s + 3 * s) - d) / 2
    rho, tau, Kh, Ah, Bh = hats(r)
    w = 1 / tau
    c = 2 / (w ** 3 * Bh)
    return r, rho, c ** 1.5 * w ** 5 * Kh, c ** 2 * w ** 5 * Ah


def test_pointwise_comparison():
    ok = True
    worst_ratio = 0.0
    for _ in range(400):
        xb, ub, xs, us = random_buy_sell(rng.integers(1, 7), rng.integers(1, 7))
        rates, durs = path_from_inventory(xb, ub, xs, us)
        K, _ = K_time(rates, durs)
        A = (rates ** 2 * durs).sum()
        if A <= 4 + 1e-9:
            continue
        z = brentq(lambda zz: A_of_z(zz) - A, 1e-9, 1 - 1e-12, xtol=1e-14)
        r, rho, Kstar, Astar = candidate_from_z(z)
        ok &= abs(Astar - A) < 1e-8 * A
        ok &= K <= Kstar + 1e-10
        worst_ratio = max(worst_ratio, K / Kstar)
        ok &= K / math.sqrt(2 * A) <= math.sqrt(R_of_z(z)) + 1e-10
    # the candidate itself attains equality, and R is maximal at z_*
    for z in (0.1, 0.258972013165, 0.7):
        r, rho, Kstar, Astar = candidate_from_z(z)
        ok &= abs(Kstar / math.sqrt(2 * Astar) - math.sqrt(R_of_z(z))) < 1e-9
    zs = np.linspace(1e-4, 1 - 1e-4, 100001)
    ok &= abs(zs[np.argmax(R_of_z(zs))] - 0.258972) < 2e-5
    report("Pointwise comparison: every buy-sell path has K <= K(c v_r) at equal A, B=2", ok,
           f"max K/K* = {worst_ratio:.6f}")


def test_global_search_permanent():
    n = 40
    m = Model(0.0, n)
    f = sqrt_law()
    best = 0.0
    def obj(x):
        v = x - x.mean()
        A, B = m.h * v @ v, m.h * np.abs(v).sum()
        if A < 1e-14:
            return 0.0
        return m.ip_P(v, f(v)) / math.sqrt(A * B)
    for _ in range(40):
        x0 = rng.normal(size=n) * rng.uniform(0.3, 3)
        res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 5000})
        best = max(best, -res.fun)
    report("Free-form search (G=1, 40 cells, 40 starts) stays below the algebraic constant", best <= 0.08112574 + 1e-9,
           f"best={best:.6f} bound=0.08112574")


# ---------------------------------------------------------------- Appendix B
def test_appendix_b():
    gamma, ncoarse, nfine = 0.25, 8, 2048
    alpha = 1 - gamma
    Mf = cell_matrix(gamma, nfine) * nfine          # normalized fine P matrix
    ok = True
    for kind in ("nu", "a"):
        H = Mf + Mf.T if kind == "nu" else 1j * (Mf - Mf.T)
        ratio = nfine // ncoarse
        Q = np.kron(np.eye(ncoarse), np.ones((ratio, ratio)) / ratio)
        norm_IQH = np.linalg.norm((np.eye(nfine) - Q) @ H, 2)
        e_n = 4 * ncoarse ** (-alpha) / (alpha * math.sqrt(2 * alpha + 1))
        ok &= norm_IQH <= e_n
        lam_fine = np.linalg.eigvalsh(H).max()
        Mc = cell_matrix(gamma, ncoarse) * ncoarse
        Hc = Mc + Mc.T if kind == "nu" else 1j * (Mc - Mc.T)
        lam_c = np.linalg.eigvalsh(Hc).max()
        ok &= lam_c <= lam_fine + 1e-12 and lam_fine <= lam_c + e_n ** 2 / (lam_c - e_n)
        two_by_two = (lam_c + e_n + math.sqrt((lam_c - e_n) ** 2 + 4 * e_n ** 2)) / 2
        ok &= lam_fine <= two_by_two and two_by_two <= lam_c + e_n ** 2 / (lam_c - e_n) + 1e-12
        print(f"   {kind}: ||(I-Q)H||={norm_IQH:.4f} <= e_n={e_n:.4f};  "
              f"lambda_coarse/2={lam_c/2:.5f} <= lambda_fine/2={lam_fine/2:.5f} <= {(lam_c + e_n**2/(lam_c-e_n))/2:.5f}")
    # the certified interval for nu_G at gamma=1/4 contains the fine estimate
    ok &= 0.76336802 <= np.linalg.eigvalsh(Mf + Mf.T).max() / 2 <= 0.76340853
    report("Appendix B: cell-averaging bound, eigenvalue sandwich, certified interval", ok)


# ---------------------------------------------------------------- horizon conversion
def test_horizon():
    ok = True
    for T in (0.2, 1.0, 3.0):
        n = 300
        gamma = 0.25
        # clock-time model on [0,T] with G_day, f_day, eta_day, varsigma_day
        fday = lambda x: np.sign(x) * np.sqrt(np.abs(x))
        eta_d, sig_d = 0.3, 0.1
        tau = (np.arange(n) + 0.5) * T / n
        nu = np.sin(2 * np.pi * tau / T) + 0.3 * np.cos(4 * np.pi * tau / T)
        nu -= nu.mean()
        # cost in clock time by cell integrals of G_day(tau) = tau^-gamma on cells of width T/n
        Md = cell_matrix(gamma, n) * (T ** (2 - gamma))  # scaling of the double integral
        cost_day = eta_d * (T / n) * nu @ nu + sig_d * (T / n) * np.abs(nu).sum() + nu @ Md @ fday(nu)
        # dimensionless model
        v = T * nu
        f = lambda x: T * fday(x / T)
        G_scale = T ** (-gamma)        # G(u) = G_day(T u)
        M = cell_matrix(gamma, n) * G_scale
        cost = (eta_d / T) * (1 / n) * v @ v + sig_d * (1 / n) * np.abs(v).sum() + v @ M @ f(v)
        ok &= abs(cost - cost_day) < 1e-10 * max(1, abs(cost))
    report("Horizon conversion preserves total cost (T=0.2, 1, 3)", ok)


def test_RG():
    ok = True
    for gamma in (0.25, 0.75, 0.95):
        ts = np.linspace(0, 1, 100001)
        G1 = lambda x: x ** (1 - gamma) / (1 - gamma)
        val = (G1(ts) + G1(1 - ts)).max()
        ok &= abs(val - 2 ** gamma / (1 - gamma)) < 1e-6
    report("R_G = 2^gamma/(1-gamma) for power kernels", ok)


def test_certificate_continuity():
    """|K(u)-K(v)| <= (4/3)(e sqrt B + (sqrt A + e) sqrt(2e)) for the ramped witness."""
    import csv
    from pathlib import Path
    rows = list(csv.DictReader(open(Path(__file__).with_name("extremal_round_trip.csv"))))
    v = np.array([float(r["v"]) for r in rows]); v -= v.mean()
    n = len(v); f = sqrt_law()
    ok = True
    from scipy.linalg import matmul_toeplitz
    for ratio, half in ((64, 1), (64, 4), (256, 8)):
        N = n * ratio
        gamma, h = 0.25, 1.0 / N
        a_ = 1 - gamma
        Fd = (np.arange(N + 1) * h) ** (1 + a_) / (a_ * (1 + a_))
        w = np.empty(N); w[0] = Fd[1]; w[1:] = Fd[2:] - 2 * Fd[1:-1] + Fd[:-2]
        row = np.zeros(N); row[0] = w[0]
        ip_P = lambda x, y: float(x @ matmul_toeplitz((w, row), y))
        step = np.repeat(v, ratio)
        ramp = step.copy()
        t = (np.arange(N) + 0.5) / N
        for i in range(1, n):
            lo, hi = i / n - half / N, i / n + half / N
            sel = (t > lo) & (t < hi)
            ramp[sel] = v[i - 1] + (v[i] - v[i - 1]) * (t[sel] - lo) / (hi - lo)
        ok &= abs(ramp.mean()) < 1e-12
        A, B = h * step @ step, h * np.abs(step).sum()
        K_step, K_ramp = -ip_P(step, f(step)), -ip_P(ramp, f(ramp))
        delta = half / N
        e_true = math.sqrt(h * ((ramp - step) ** 2).sum())
        e = math.sqrt(delta * ((v[1:] - v[:-1]) ** 2).sum() / 6)
        bound = (4 / 3) * (e * math.sqrt(B) + (math.sqrt(A) + e) * math.sqrt(2 * e))
        ok &= e_true <= e * (1 + 1e-9) and abs(K_step - K_ramp) <= bound
        ok &= math.sqrt((h * ramp @ ramp) * (h * np.abs(ramp).sum())) <= (math.sqrt(A) + e) * math.sqrt(B + e)
        print(f"   ramp half-width {delta:.2e}: |dK|={abs(K_step-K_ramp):.3e} <= bound {bound:.3e}; L2 error {e_true:.3e} <= {e:.3e}")
    report("Certificate continuity bound for the ramped witness", ok)


if __name__ == "__main__":
    test_certificate_continuity()
    test_theorem1()
    test_prop_quadratic()
    test_power_envelope()
    test_sqrt_envelope()
    test_exact_power_ray()
    test_two_block()
    test_kernel_constants_permanent()
    test_lemma_mu()
    test_mu_permanent()
    test_envelope_ceiling()
    test_ttpm()
    test_candidate_formulas()
    test_rearrangement()
    test_inventory_substitution()
    test_pointwise_comparison()
    test_global_search_permanent()
    test_appendix_b()
    test_horizon()
    test_RG()
    print()
    print("ALL PASS" if not FAILURES else f"FAILURES: {FAILURES}")
