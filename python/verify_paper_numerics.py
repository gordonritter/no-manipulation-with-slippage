"""Randomized numerical verification of the symbolic content of
no_manipulation_v1.tex.

Trading-rate paths are step functions on n equal buckets of [0, 1]; the
one-sided operator P is discretized with EXACT double integrals of the
kernel over bucket pairs, so <v, P w> is evaluated exactly for step
functions.  Since none of the proofs use continuity of v, every inequality
in the paper must hold exactly (to rounding) in this discrete model.
Suprema over x (Theta, kappa, chi) are computed on fine grids plus scalar
optimization.  Each check is repeated for several random draws of all
parameters.  Tolerances: identities 1e-9, optimization-based formulas 1e-6.
"""
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gamma as Gamma
import mpmath as mp
from scipy.integrate import quad as squad
def fq(fun,a,b):
    return squad(fun,a,b,limit=400)[0]

import sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 12345)
# optional second argument: comma-separated list of sections to run, e.g. "1,2,3"
SECTIONS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
def sec(k):
    return SECTIONS is None or str(k) in SECTIONS
TOL = 1e-9
failures = []
def check(name, cond, detail=""):
    status = "ok " if cond else "FAIL"
    if not cond: failures.append(name)
    print(f"[{status}] {name} {detail}")

# ---------------------------------------------------------------- kernels
class Kernel:
    """Piecewise-constant discretization on n buckets with exact integrals."""
    def __init__(self, kind, n, par):
        self.kind, self.n, self.par = kind, n, par
        h = 1.0 / n; self.h = h
        if kind == "power":
            g = par
            F = lambda u: u ** (2 - g) / ((1 - g) * (2 - g))
            self.G = lambda u: u ** (-g)
            self.cG = 1 / (1 - g)
            self.normG2 = 1 / np.sqrt(1 - 2 * g) if g < 0.5 else np.inf
        elif kind == "exp":
            k = par
            F = lambda u: np.exp(-k * u) / k ** 2
            self.G = lambda u: np.exp(-k * u)
            self.cG = (1 - np.exp(-k)) / k
            self.normG2 = np.sqrt((1 - np.exp(-2 * k)) / (2 * k))
        elif kind == "const":
            F = lambda u: u ** 2 / 2
            self.G = lambda u: np.ones_like(u)
            self.cG = 1.0; self.normG2 = 1.0
        d = np.arange(n); col = np.zeros(n); col[0] = F(h) - F(0) if kind != "power" else F(h)
        if kind == "power":
            col[1:] = F((d[1:] + 1) * h) - 2 * F(d[1:] * h) + F((d[1:] - 1) * h)
        else:
            col[0] = F(h) - F(0) - 0  # int_0^h int_0^t G(t-s) ds dt = F(h)-F(0)-h F'(0)
            Fp0 = -1 / k if kind == "exp" else 0.0  # F'(0)
            col[0] = F(h) - F(0) - h * Fp0
            col[1:] = F((d[1:] + 1) * h) - 2 * F(d[1:] * h) + F((d[1:] - 1) * h)
        # bilinear form <v, P w> = v^T M w for step functions with values v_i, w_j
        self.M = toeplitz(col, np.zeros(n))
        self.Mop = self.M / h            # L^2-normalized operator matrix
        S = self.Mop + self.Mop.T
        self.nu = 0.5 * np.linalg.eigvalsh(S).max()
        self.a = 0.5 * np.linalg.norm(self.Mop - self.Mop.T, 2)
        self.normP = np.linalg.norm(self.Mop, 2)
        # P1 evaluated at bucket right ends: int_0^t G
        self.P1 = np.cumsum(self.M[:, 0]) / h * 0 + (self.M @ np.ones(n)) / h
    # inner products for step functions
    def ip(self, v, w): return self.h * float(v @ w)
    def Pip(self, v, w): return float(v @ self.M @ w)     # <v, P w>
    def Gip(self, v, w): return float(v @ (self.M + self.M.T) @ w)  # <v, G w>
    def Pinf(self, w):  # sup_t |(P w)(t)| approx at bucket midpoints
        return np.abs(self.Mop @ w).max()
    def n1(self, v): return self.h * np.abs(v).sum()
    def n2sq(self, v): return self.h * float(v @ v)

def rand_kernel(n):
    kind = rng.choice(["power", "exp", "const"])
    if kind == "power": return Kernel("power", n, rng.uniform(0.05, 0.45))
    if kind == "exp": return Kernel("exp", n, rng.uniform(0.3, 6.0))
    return Kernel("const", n, None)

def round_trip(n, scale=1.0):
    v = rng.standard_normal(n) * scale
    v = np.cumsum(rng.standard_normal(n)); v = v - v.mean()   # smooth-ish
    return v * scale / np.abs(v).max()

# ------------------------------------------------- 0. kernel sanity checks
n = 400
K1 = Kernel("const", n, None)
check("Volterra: nu_G=1/2", abs(K1.nu - 0.5) < 1e-6, f"{K1.nu:.8f}")
check("Volterra: a_G=1/pi", abs(K1.a - 1 / np.pi) < 1e-5, f"{K1.a:.8f}")
check("Volterra: ||P||=2/pi", abs(K1.normP - 2 / np.pi) < 1e-6, f"{K1.normP:.8f}")
# skew eigenfunctions e^{i(2k+1)pi t}, eigenvalue -2i/((2k+1)pi)
t = (np.arange(n) + 0.5) / n
for k in [0, 1, -2]:
    hfun = np.exp(1j * (2 * k + 1) * np.pi * t)
    lhs = (K1.Mop - K1.Mop.T) @ hfun
    lam = -2j / ((2 * k + 1) * np.pi)
    check(f"skew eigenfunction k={k}", np.abs(lhs - lam * hfun).max() < 5e-3, f"err={np.abs(lhs-lam*hfun).max():.1e} (discretization)")
for _ in range(4):
    K = rand_kernel(n)
    one = np.ones(n)
    lower = fq(lambda u: (1 - u) * float(K.G(np.array(u))), 0, 1)
    check(f"<1,G1>=2 int(1-u)G ({K.kind})", abs(K.Gip(one, one) - 2 * lower) < 1e-6, f"{K.Gip(one,one):.8f} vs {2*lower:.8f}")
    check(f"Lemma constants: int(1-u)G <= nu <= ||P|| <= c_G, a<=||P|| ({K.kind})",
          lower <= K.nu + 1e-9 <= K.normP + 1e-9 <= K.cG + 1e-9 and K.a <= K.normP + 1e-9,
          f"{lower:.4f} {K.nu:.4f} {K.normP:.4f} {K.cG:.4f} a={K.a:.4f}")
    u = rng.standard_normal(n)
    check(f"positive type <u,Gu>>=0 ({K.kind})", K.Gip(u, u) >= -1e-12)
    check(f"<v,Pv> = (1/2)<v,Gv> ({K.kind})", abs(K.Pip(u, u) - 0.5 * K.Gip(u, u)) < TOL)
    check(f"c_G matches quadrature ({K.kind})", abs(K.cG - fq(lambda u: float(K.G(np.array(u))), 0, 1)) < 1e-8)
    if K.normG2 < np.inf:
        check(f"||G||_2 formula ({K.kind})", abs(K.normG2 ** 2 - fq(lambda u: float(K.G(np.array(u))) ** 2, 0, 1)) < 1e-8)

# Fourier transform constant of |t|^{-gamma}
for g in [0.2, 0.5, 0.8]:
    w = 1.7
    C = 2 * Gamma(1 - g) * np.sin(g * np.pi / 2)
    mp.mp.dps = 30
    ft = 2 * (mp.quad(lambda tt: tt ** (-g) * mp.cos(w * tt), [0, 1]) + mp.quadosc(lambda tt: tt ** (-g) * mp.cos(w * tt), [1, mp.inf], omega=w))
    check(f"FT |t|^-g constant g={g}", abs(float(ft) - C * w ** (g - 1)) < 1e-5 * C * w ** (g - 1), f"{float(ft):.10f} vs {C*w**(g-1):.10f}")

# ------------------------------------------- helpers for impact functions
def make_f(kind, lam, vstar, eps=None, vc=None, wobble=None):
    if kind == "power":
        return lambda x: np.sign(x) * lam * np.abs(x) * (np.abs(x) / vstar) ** (-eps)
    if kind == "core":
        return lambda x: lam * x * ((x ** 2 + vc ** 2) / (vstar ** 2 + vc ** 2)) ** (-eps / 2)
    if kind == "wobble":  # continuous odd, near-linear with random nonlinearity
        A, B, C_ = wobble
        fpos = lambda x: lam * x * (1 + A * np.sin(B * x / vstar) + C_ * np.tanh(x / vstar))
        return lambda x: np.sign(x) * fpos(np.abs(x))   # odd extension, as in Definition 2.1

def sup_ratio(num, xmax):
    """sup_{0<x<=xmax} num(x) on a fine log-linear grid with local refinement."""
    xs = np.sort(np.concatenate([np.geomspace(1e-9 * xmax, xmax, 4000), np.linspace(1e-9 * xmax, xmax, 4000)]))
    vals = num(xs); i = np.argmax(vals)
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    if not (hi > lo): return vals[i]
    r = minimize_scalar(lambda x: -num(x), bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    return max(vals[i], -r.fun)

# ----------------------------------------------- 1. Theorem nomanip (3.1)
print("\n--- Theorem 3.1 and its corollaries ---")
for trial in (range(6) if sec(1) else []):
    K = rand_kernel(n)
    lam, vstar = rng.uniform(0.2, 3), rng.uniform(0.5, 5)
    eta, sig = rng.uniform(0, 2), rng.uniform(0, 2)
    if eta + sig == 0: eta = 1.0
    f = make_f("wobble", lam, vstar, wobble=(rng.uniform(-0.3, 0.3), rng.uniform(0.5, 5), rng.uniform(-0.3, 0.3)))
    m = rng.uniform(0, 2 * lam)
    v = round_trip(n, scale=rng.uniform(0.1, 5)) if rng.random() < 0.5 else rng.standard_normal(n) * rng.uniform(0.1, 3)
    cost = eta * K.n2sq(v) + sig * K.n1(v) + K.Pip(v, f(v))
    vinf = np.abs(v).max()
    Theta = sup_ratio(lambda x: np.abs(f(x) - m * x) / (eta * x / K.nu + sig / K.cG), vinf)
    rhs = (1 - Theta) * (eta * K.n2sq(v) + sig * K.n1(v))
    check(f"(nomanip-combined) trial {trial} [{K.kind}]", cost >= rhs - 1e-9, f"cost={cost:.6f} rhs={rhs:.6f} Theta={Theta:.4f}")
    # pointwise estimate (combined-pointwise)
    w = f(v) - m * v
    viol = (np.abs(w) - Theta * (eta * np.abs(v) / K.nu + sig / K.cG)).max()
    check(f"  pointwise |w|<=Theta(eta|v|/nu+sig/c)", viol <= 1e-9 * max(1, Theta), f"max violation={viol:.2e}")
    # <|v|,P|v|> <= nu ||v||^2 and <|v|,P1> <= c_G ||v||_1
    av = np.abs(v)
    check(f"  <|v|,P|v|> <= nu ||v||_2^2", K.Pip(av, av) <= K.nu * K.n2sq(v) + 1e-9)
    check(f"  <|v|,P1> <= c_G ||v||_1", K.Pip(av, np.ones(n)) <= K.cG * K.n1(v) + 1e-9)
    # rate-only corollary (nomanip) with kappa and Theta_{lambda,v} = nu lam kappa / eta
    if eta > 0:
        kappa = sup_ratio(lambda x: np.abs(f(x) - lam * x) / (lam * x), vinf)
        Th_l = sup_ratio(lambda x: np.abs(f(x) - lam * x) / (eta * x / K.nu), vinf)
        check(f"  Theta_lambda,v = nu*lam*kappa/eta", abs(Th_l - K.nu * lam * kappa / eta) < 1e-7 * max(1, Th_l))
        cost0 = eta * K.n2sq(v) + K.Pip(v, f(v))
        check(f"  (nomanip): C >= (eta - lam nu kappa)||v||^2", cost0 >= (eta - lam * K.nu * kappa) * K.n2sq(v) - 1e-9)
        # Proposition quadratic
        cost_q = (eta - lam * (K.a * kappa + K.nu * kappa ** 2 / 4)) * K.n2sq(v)
        check(f"  (quadratic-bound)", cost0 >= cost_q - 1e-9, f"{cost0:.6f} >= {cost_q:.6f}")
        # its ingredients: split identity and Cauchy-Schwarz in G-form
        w = f(v) - lam * v
        Sk = K.M - K.M.T
        split = 0.5 * lam * K.Gip(v, v) + 0.5 * K.Gip(v, w) + 0.5 * float(v @ Sk @ w)
        check(f"  (quadratic-split) identity", abs(split - K.Pip(v, f(v))) < TOL)
        check(f"  CS: |<v,Gw>|<=sqrt(<v,Gv><w,Gw>)", abs(K.Gip(v, w)) <= np.sqrt(max(K.Gip(v, v), 0) * max(K.Gip(w, w), 0)) + 1e-9)
        check(f"  <w,Gw> <= 2 nu ||w||^2", K.Gip(w, w) <= 2 * K.nu * K.n2sq(w) + 1e-9)
        check(f"  |<v,(P-P*)w>| <= 2 a ||v|| ||w||", abs(float(v @ Sk @ w)) <= 2 * K.a * np.sqrt(K.n2sq(v) * K.n2sq(w)) + 1e-9)
        check(f"  ||w||_2 <= lam kappa ||v||_2", np.sqrt(K.n2sq(w)) <= lam * kappa * np.sqrt(K.n2sq(v)) + 1e-9)
    if sig > 0:
        chi = sup_ratio(lambda x: np.abs(f(x) - lam * x), vinf)
        Th_s = sup_ratio(lambda x: np.abs(f(x) - lam * x) / (sig / K.cG), vinf)
        check(f"  Theta_lambda,v = c_G chi/sig", abs(Th_s - K.cG * chi / sig) < 1e-7 * max(1, Th_s))
        costs = sig * K.n1(v) + K.Pip(v, f(v))
        check(f"  (nomanip-spread): C >= (sig - c chi)||v||_1", costs >= (sig - K.cG * chi) * K.n1(v) - 1e-9)

# ------------------------------------------ 2. Power-law section formulas
print("\n--- Power-law impact: m*, inf Theta, square-root formulas ---")
def theta_star(f, m, alpha, beta, slope_inf=0.0):
    """sup_{x>0} |f(x)-mx|/(alpha x+beta); slope_inf = lim f(x)/x supplies the x->infinity limit |slope_inf-m|/alpha."""
    num = lambda x: np.abs(f(x) - m * x) / (alpha * x + beta)
    # sup over (0, inf): grid on log scale then refine; also the limit at infinity (= m/alpha)
    xs = np.geomspace(1e-8, 1e8, 20000); vals = num(xs); i = np.argmax(vals)
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    r = minimize_scalar(lambda x: -num(x), bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    return max(vals[i], -r.fun, abs(slope_inf - m) / alpha)
for trial in (range(5) if sec(2) else []):
    eps = rng.uniform(0.05, 0.95); lam, vstar = rng.uniform(0.2, 3), rng.uniform(0.5, 5)
    eta, sig = rng.uniform(0.1, 3), rng.uniform(0.1, 3); nu, cG = rng.uniform(0.3, 1.2), rng.uniform(0.5, 2)
    f = make_f("power", lam, vstar, eps=eps)
    alpha, beta = eta / nu, sig / cG
    r_eta, r_sig = eta / (nu * lam), sig / (cG * lam * vstar)
    A = eps ** eps * ((1 - eps) / 2) ** (1 - eps)
    m_star = lam * A * (r_eta / r_sig) ** eps
    pred = A / (r_eta ** (1 - eps) * r_sig ** eps)
    th_at_mstar = theta_star(f, m_star, alpha, beta)
    # minimize over m numerically
    res = minimize_scalar(lambda m: theta_star(f, m, alpha, beta), bounds=(0, 5 * m_star + 1), method="bounded", options={"xatol": 1e-10})
    check(f"(theta-epsilon) Theta*_{{m*}} = A/(r_eta^(1-e) r_sig^e) eps={eps:.2f}", abs(th_at_mstar - pred) < 1e-6 * pred, f"{th_at_mstar:.9f} vs {pred:.9f}")
    check(f"(mstar-epsilon) numerical argmin m ~ m*", abs(res.x - m_star) < 1e-4 * m_star and res.fun >= pred - 1e-7, f"argmin={res.x:.6f} m*={m_star:.6f} min={res.fun:.9f}")
    # kappa*, chi* infinite for eps != 0 (grow without bound)
    kap = lambda x: abs(f(x) - lam * x) / (lam * x); chi_ = lambda x: abs(f(x) - lam * x)
    check(f"  kappa(x) grows without bound as x->0, chi(x) as x->inf", kap(1e-200) > kap(1e-100) > kap(1e-50) > 1 and chi_(1e200) > chi_(1e100) > chi_(1e50) > 1)
# square-root weighted formula
for trial in (range(5) if sec(2) else []):
    p = rng.uniform(0, 2)
    g = lambda y: np.abs(y - p * y ** 2) / (1 + y ** 2)
    ys = np.geomspace(1e-6, 1e6, 200001); s = max(g(ys).max(), p)
    check(f"sup|y-py^2|/(1+y^2) = max{{p,(sqrt(1+p^2)-p)/2}} p={p:.3f}", abs(s - max(p, (np.sqrt(1 + p * p) - p) / 2)) < 1e-7)
pp = np.linspace(0, 3, 300001); h = np.maximum(pp, (np.sqrt(1 + pp ** 2) - pp) / 2)
check("min over p at 1/(2sqrt2), value 1/(2sqrt2)", abs(pp[h.argmin()] - 1 / (2 * np.sqrt(2))) < 1e-4 and abs(h.min() - 1 / (2 * np.sqrt(2))) < 1e-4)
for trial in (range(4) if sec(2) else []):
    lam, vstar, eta, sig = rng.uniform(0.2, 3, 4); nu, cG = rng.uniform(0.3, 1.2), rng.uniform(0.5, 2)
    f = make_f("power", lam, vstar, eps=0.5); alpha, beta = eta / nu, sig / cG
    m_star = lam / (2 * np.sqrt(2)) * np.sqrt(cG * eta * vstar / (nu * sig))
    pred = lam * np.sqrt(nu * cG * vstar) / (2 * np.sqrt(2 * eta * sig))
    check(f"(theta-square-root) Theta*_{{m*}}", abs(theta_star(f, m_star, alpha, beta) - pred) < 1e-6 * pred)
    res = minimize_scalar(lambda m: theta_star(f, m, alpha, beta), bounds=(0, 5 * m_star + 1), method="bounded", options={"xatol": 1e-10})
    check(f"  argmin m = m_*", abs(res.x - m_star) < 1e-4 * m_star)
    # (nomanip-square-root-combined) <=> pred <= 1
    check(f"  8 eta sig >= lam^2 v* nu c  <=>  inf Theta <= 1", (8 * eta * sig >= lam ** 2 * vstar * nu * cG) == (pred <= 1 + 1e-12))
    check(f"  A_{{1/2}} = 1/(2sqrt2), e^e(1-e)^(1-e)=1/2", abs(0.5 ** 0.5 * 0.25 ** 0.5 - 1 / (2 * np.sqrt(2))) < 1e-15 and abs(0.5 ** 0.5 * 0.5 ** 0.5 - 0.5) < 1e-15)

# ------------------------------------------ 3. Proposition/Theorem exact-power
print("\n--- Exact power-law characterization (ray argument) ---")
for trial in (range(6) if sec(3) else []):
    K = rand_kernel(n)
    eps = rng.uniform(0.05, 0.95); lam, vstar = rng.uniform(0.2, 3), rng.uniform(0.5, 5)
    eta, sig = rng.uniform(0.1, 3), rng.uniform(0.1, 3)
    f = make_f("power", lam, vstar, eps=eps)
    v = round_trip(n, scale=rng.uniform(0.5, 3))
    c = rng.uniform(0.1, 5)
    check(f"homogeneity f(cx)=c^(1-e)f(x)", np.allclose(f(c * v), c ** (1 - eps) * f(v), rtol=1e-12, atol=1e-12))
    A_, B_, Kk = K.n2sq(v), K.n1(v), -K.Pip(v, f(v))
    cost_cv = eta * K.n2sq(c * v) + sig * K.n1(c * v) + K.Pip(c * v, f(c * v))
    check(f"(ray-cost) trial {trial}", abs(cost_cv - (eta * A_ * c ** 2 + sig * B_ * c - Kk * c ** (2 - eps))) < 1e-9 * max(1, abs(cost_cv)))
    phi = lambda cc: eta * A_ * cc + sig * B_ - Kk * cc ** (1 - eps)
    if Kk > 0:
        c0 = ((1 - eps) * Kk / (eta * A_)) ** (1 / eps)
        dphi = lambda cc: eta * A_ - (1 - eps) * Kk * cc ** (-eps)
        grid = c0 * np.exp(np.linspace(-6, 6, 2001))
        check(f"  c0 is the minimizer of phi", abs(dphi(c0)) < 1e-9 * eta * A_ and phi(grid).min() >= phi(c0) - 1e-12 * abs(phi(c0)) and dphi(c0 * 0.9) < 0 < dphi(c0 * 1.1), f"c0={c0:.3e}")
        check(f"  (ray-min) value", abs(phi(c0) - (sig * B_ - eps / (1 - eps) * eta * A_ * c0)) < 1e-9 * max(1, abs(phi(c0))))
        neg = phi(c0) < 0
        ineq = eta ** (1 - eps) * sig ** eps * A_ ** (1 - eps) * B_ ** eps < eps ** eps * (1 - eps) ** (1 - eps) * Kk
        check(f"  negativity <=> (exact-power) inequality on this ray", neg == ineq)
        ratio = Kk / (A_ ** (1 - eps) * B_ ** eps)
        Lam = K.nu ** (1 - eps) * K.cG ** eps * lam * vstar ** eps / 2 ** (1 - eps)
        check(f"  (xi-bound) ratio <= Lambda", ratio <= Lam + 1e-9, f"{ratio:.5f} <= {Lam:.5f}")
        # Corollary one-sided slippage
        cs = (Kk / (2 * eta * A_)) ** (1 / eps); cl = (2 * sig * B_ / Kk) ** (1 / (1 - eps))
        check(f"  rate-only ray negative for small c", eta * A_ * cs ** 2 - Kk * cs ** (2 - eps) < 0)
        check(f"  spread-only ray negative for large c", sig * B_ * cl - Kk * cl ** (2 - eps) < 0)
    else:
        check(f"  K<=0: ray cost positive", all(phi(cc) > 0 for cc in [1e-3, 1, 1e3]))
    # scale invariance of the Xi quotient
    q = lambda vv: -K.Pip(vv, f(vv)) / (K.n2sq(vv) ** (1 - eps) * K.n1(vv) ** eps)
    check(f"  Xi quotient scale-invariant", abs(q(v) - q(3.7 * v)) < 1e-9 * max(1, abs(q(v))))

# ------------------------------------------- 4. Market calibration algebra
print("\n--- Market calibration ---")
for trial in (range(5) if sec(4) else []):
    S0, vstar, Y0, sd, sBA = rng.uniform(5, 500), rng.uniform(1e5, 1e7), rng.uniform(0.2, 1.5), rng.uniform(0.005, 0.05), rng.uniform(1e-4, 1e-2)
    nu, cG = rng.uniform(0.3, 1.2), rng.uniform(0.5, 2); b_eta = rng.uniform(0.1, 50)
    lam = Y0 * sd * S0 / (cG * vstar)
    sig = sBA * S0 / 2; eta = b_eta * S0 / (100 * vstar)
    lhs_cond = 8 * eta * sig >= lam ** 2 * vstar * nu * cG
    rhs_cond = b_eta >= (nu / cG) * 25 * Y0 ** 2 * sd ** 2 / sBA
    check(f"(square-root-market) equivalence", lhs_cond == rhs_cond)
    # boundary equality
    b_eq = (nu / cG) * 25 * Y0 ** 2 * sd ** 2 / sBA; eta_eq = b_eq * S0 / (100 * vstar)
    check(f"  boundary equality", abs(8 * eta_eq * sig - lam ** 2 * vstar * nu * cG) < 1e-9 * lam ** 2 * vstar)
    # end-of-horizon impact of constant rate x
    K = rand_kernel(n); x = rng.uniform(0.01, 2) * vstar
    imp = K.cG * lam * np.sqrt(x * vstar)
    check(f"  int G(1-s) f(x;1/2) ds = c_G lam v* sqrt(x/v*)", abs(imp - K.cG * lam * vstar * np.sqrt(x / vstar)) < 1e-9 * imp)
    # execution price decomposition
    vv = rng.uniform(-3, 3) * vstar
    lhs = np.sign(vv) * (eta * vv + sig * np.sign(vv)) / S0
    check(f"  sign(v)(S~-S)/S0 = sBA/2 + b_eta/100 |v|/v*", abs(lhs - (sBA / 2 + b_eta / 100 * abs(vv) / vstar)) < 1e-12)

# --------------------------------------------- 5. Envelope ceiling section
print("\n--- Limits of envelope criteria ---")
def tent(tgrid, p, l):
    return 2 / l * np.maximum(0, 1 - np.abs(2 * (tgrid - p) / l - 1))
for trial in (range(4) if sec(5) else []):
    K = rand_kernel(n)
    # mu_G := sup -<v,P sign v>/||v||_1 ; upper bound mu_G <= c_G on random round trips
    ok = True
    for _ in range(50):
        v = round_trip(n, 1.0)
        ok &= -K.Pip(v, np.sign(v)) / K.n1(v) <= K.cG + 1e-9
    check(f"mu_G <= c_G on random round trips ({K.kind})", ok)
    # witness of Lemma mu: sell tent at 0 (mass 1), trickle -eps on [h, tau-h] (trapezoid), buy tent at 1 (mass q)
    N2 = 4000; K2 = Kernel(K.kind, N2, K.par); tg = (np.arange(N2) + 0.5) / N2
    hh, tau, epsl = 0.01, 0.97, 1e-3
    phi = np.clip(np.minimum(tg / hh, (tau - tg) / hh), 0, 1) * (tg <= tau)
    q = 1 + epsl * (tau - hh)
    v = -tent(tg, 0, hh) - epsl * phi + q * tent(tg, 1 - hh, hh)
    Gam = lambda x: fq(lambda u: float(K2.G(np.array(u))), 0, x) if x > 0 else 0.0
    val = -K2.Pip(v, np.sign(v)) / K2.n1(v)
    proof_lb = (-Gam(hh) - epsl * K2.cG * tau + q * (Gam(1) - Gam(1 - tau) - Gam(hh)) ) / (2 * q)  # lemma's lower bound (min over [1-h,1] replaced by its limit value; check separately)
    tt = np.linspace(1 - hh, 1, 41); minval = min(Gam(x) - Gam(x - tau) for x in tt)
    proof_lb = (-Gam(hh) - epsl * K2.cG * tau + q * (minval - Gam(hh))) / (2 * q)
    check(f"  witness: mean zero, ||v||_1 = 2q, sign pattern (-,0,+)",
          abs(K2.ip(v, np.ones(N2))) < 1e-9 and abs(K2.n1(v) - 2 * q) < 1e-9
          and np.all(np.sign(v)[tg < tau] == -1) and np.all(np.sign(v)[(tg > tau) & (tg < 1 - hh)] == 0) and np.all(np.sign(v)[tg > 1 - hh] == 1))
    check(f"  witness ratio >= lemma's lower bound and <= c_G ({K.kind})",
          val >= proof_lb - 1e-6 and val <= K2.cG + 1e-9, f"ratio={val:.4f}  lemma bound={proof_lb:.4f}  c_G/2={K2.cG/2:.4f}")
    # Theorem envelope-ceiling with the POSITIVE law f~ = theta*beta*g_delta + (m - theta*alpha)^+ x
    eta, sig = rng.uniform(0.1, 3), rng.uniform(0.1, 3)
    alpha, beta = eta / K2.nu, sig / K2.cG
    theta = K2.cG / val * rng.uniform(1.05, 1.5)
    for m in [0.0, rng.uniform(0, theta * alpha), theta * alpha * rng.uniform(1.5, 3)]:   # both branches of (m - theta alpha)^+
        rho = beta * (-theta * K2.Pip(v, np.sign(v)) - K2.cG * K2.n1(v))
        check(f"  rho > 0 (theta mu-witness > c_G)", rho > 0)
        c = 0.25 * rho / (eta * K2.n2sq(v) + max(m - theta * alpha, 0) * 0.5 * K2.Gip(v, v))
        delta = 1e-6 * c * np.abs(v[v != 0]).min()
        gd = lambda x: np.sign(x) * np.minimum(np.abs(x) / delta, 1)
        ft = lambda x: theta * beta * gd(x) + max(m - theta * alpha, 0) * x
        xs = np.geomspace(1e-9, 1e6, 20001)
        check(f"  f~ positive, nondecreasing, concave on (0,inf), f~(0+)=0  (m={m:.3f})",
              np.all(ft(xs) > 0) and np.all(np.diff(ft(xs)) >= -1e-12) and np.all(np.diff(np.diff(ft(xs)) / np.diff(xs)) <= 1e-6 * theta * beta / delta) and ft(1e-3 * delta) < 2e-3 * theta * beta)
        check(f"  envelope |f~ - m x| <= theta (alpha x + beta) pointwise",
              np.all(np.abs(ft(xs) - m * xs) <= theta * (alpha * xs + beta) * (1 + 1e-12)))
        Th = theta_star(ft, m, alpha, beta, slope_inf=max(m - theta * alpha, 0))
        check(f"  Theta_m*(f~) <= theta", Th <= theta * (1 + 1e-9), f"{Th:.6f} <= {theta:.6f}")
        cost = eta * K2.n2sq(c * v) + sig * K2.n1(c * v) + K2.Pip(c * v, ft(c * v))
        check(f"  round trip cv has negative cost under f~", cost < 0, f"cost={cost:.6e}")
        rhs = c * (c * (eta * K2.n2sq(v) + max(m - theta * alpha, 0) * 0.5 * K2.Gip(v, v)) + sig * K2.n1(v) + theta * beta * K2.Pip(v, gd(c * v)))
        check(f"  (ceiling-cost) identity", abs(cost - rhs) < 1e-9 * max(1, abs(cost)))
    K = K2  # fine kernel for the corollary check below
    # Corollary: for the square-root law inf_m Theta_m* formula and the threshold equivalence
    lam, vstar = rng.uniform(0.2, 3), rng.uniform(0.5, 5)
    f = make_f("power", lam, vstar, eps=0.5)
    mm = rng.uniform(0, 3)
    Thf = theta_star(f, mm, eta / K.nu, sig / K.cG)
    inf_pred = lam * np.sqrt(K.nu * K.cG * vstar) / (2 * np.sqrt(2 * eta * sig))
    check(f"  Theta_m*(f) >= inf formula", Thf >= inf_pred - 1e-7)
    check(f"  32 eta sig < lam^2 v* nu c  <=>  inf Theta > 2", (32 * eta * sig < lam ** 2 * vstar * K.nu * K.cG) == (inf_pred > 2))
    # Lemma mu asserts mu_G >= c_G/2 for the supremum, via the limit h->0, eps->0, tau->1 of the witness.
    # Test convergence: along a sequence of parameters the witness ratio must increase toward c_G/2 and the
    # lemma's explicit lower bound must approach c_G/2.
    seq = [(0.04, 0.90, 1e-2), (0.02, 0.95, 1e-3), (0.01, 0.98, 1e-4)]
    ratios, gaps = [], []
    for hh_, tau_, eps_ in seq:
        phi_ = np.clip(np.minimum(tg / hh_, (tau_ - tg) / hh_), 0, 1) * (tg <= tau_)
        q_ = 1 + eps_ * (tau_ - hh_)
        v_ = -tent(tg, 0, hh_) - eps_ * phi_ + q_ * tent(tg, 1 - hh_, hh_)
        ratios.append(-K.Pip(v_, np.sign(v_)) / K.n1(v_))
        tt_ = np.linspace(1 - hh_, 1, 41); minv_ = min(Gam(x) - Gam(x - tau_) for x in tt_)
        gaps.append(K.cG / 2 - (-Gam(hh_) - eps_ * K.cG * tau_ + q_ * (minv_ - Gam(hh_))) / (2 * q_))
    check(f"  Lemma mu limit: witness ratios increase toward c_G/2 and the lemma bound's gap to c_G/2 shrinks",
          ratios[0] < ratios[1] < ratios[2] and gaps[0] > gaps[1] > gaps[2] > 0 and gaps[2] < 0.1 * K.cG,
          f"ratios={[round(r,4) for r in ratios]} c_G/2={K.cG/2:.4f} gaps={[round(g,4) for g in gaps]}")

# ---------------------------------------------- 6. TTPM theorem (Lipschitz)
print("\n--- Transaction-triggered manipulation theorem ---")
for trial in (range(6) if sec(6) else []):
    K = rand_kernel(n)
    while K.normG2 == np.inf: K = rand_kernel(n)
    eps, lam, vstar, vc = rng.uniform(0.05, 0.95), rng.uniform(0.2, 3), rng.uniform(0.5, 5), rng.uniform(0.1, 2)
    f = make_f("core", lam, vstar, eps=eps, vc=vc)
    Lf = lam * ((vstar ** 2 + vc ** 2) / vc ** 2) ** (eps / 2)
    # derivative formula, max at 0, decreasing in |x|, f(v*)=lam v*, small-x slope
    xs = np.linspace(-5 * vstar, 5 * vstar, 2001); dx = 1e-6
    fd = (f(xs + dx) - f(xs - dx)) / (2 * dx)
    fp = lam * (vstar ** 2 + vc ** 2) ** (eps / 2) * (vc ** 2 + (1 - eps) * xs ** 2) / (xs ** 2 + vc ** 2) ** (1 + eps / 2)
    check(f"f_c' formula", np.abs(fd - fp).max() < 1e-6 * Lf)
    check(f"  L_f = f_c'(0), derivative decreasing in |x|", abs(fp[1000] - Lf) < 1e-12 and np.all(np.diff(fp[1000:]) <= 1e-12) and fp.max() <= Lf + 1e-12)
    check(f"  f_c(v*) = lam v*", abs(f(vstar) - lam * vstar) < 1e-12)
    check(f"  f_c odd", np.allclose(f(-xs), -f(xs)))
    # sup |f(x)-f(y)|/|x-y| <= L_f on random pairs
    xa, xb = rng.uniform(-10, 10, 2000), rng.uniform(-10, 10, 2000)
    check(f"  global Lipschitz with L_f", np.all(np.abs(f(xa) - f(xb)) <= Lf * np.abs(xa - xb) + 1e-12))
    eta = 2 * K.normG2 * Lf * rng.uniform(1.0, 2.0); sig = rng.uniform(0, 1)
    Q = rng.uniform(0.2, 3)
    v = rng.standard_normal(n).cumsum(); v = v - v.mean() + Q  # integral Q, changes sign typically
    v = v * rng.uniform(0.5, 3)
    v = v - K.ip(v, np.ones(n)) + Q  # exact integral Q
    vp, vm = np.maximum(v, 0), np.maximum(-v, 0)
    B = K.n1(vm); alpha = Q / (Q + B); u = alpha * vp
    check(f"  u in A_Q^one", abs(K.ip(u, np.ones(n)) - Q) < 1e-9 and np.all(u >= 0))
    H = K.n1(v - u); D = K.n2sq(v) - K.n2sq(u)
    check(f"  (ttpm-HD): H=2B, D formula", abs(H - 2 * B) < 1e-9 and abs(D - ((1 - alpha ** 2) * K.n2sq(vp) + K.n2sq(vm))) < 1e-9)
    nv, nu_ = np.sqrt(K.n2sq(v)), np.sqrt(K.n2sq(u)); A_, N_ = np.sqrt(K.n2sq(vp)), np.sqrt(K.n2sq(vm))
    if B > 1e-9:
        check(f"  (ttpm-norm-sum)", (nv + nu_) ** 2 <= ((1 + alpha) * A_ + N_) ** 2 + 1e-9 <= 2 * D / (1 - alpha) + 1e-9)
        check(f"  A>=Q+B, N>=B, (ttpm-D-lower) D>=2B(Q+B)", A_ >= Q + B - 1e-9 and N_ >= B - 1e-9 and D >= 2 * B * (Q + B) - 1e-9)
        check(f"  (ttpm-geometric-bound) H(||v||+||u||)<=2D", H * (nv + nu_) <= 2 * D + 1e-9)
    J = lambda z: K.Pip(z, f(z))
    split = K.Pip(v - u, f(v)) + K.Pip(u, f(v) - f(u))
    check(f"  (ttpm-J-split) identity", abs(J(v) - J(u) - split) < TOL)
    check(f"  ||f(v)||_2<=L||v||_2, ||f(v)-f(u)||_1<=L H", np.sqrt(K.n2sq(f(v))) <= Lf * nv + 1e-9 and K.n1(f(v) - f(u)) <= Lf * H + 1e-9)
    hh = rng.standard_normal(n)
    check(f"  ||P h||_inf <= ||G||_2 ||h||_2 ; same for P*", K.Pinf(hh) <= K.normG2 * np.sqrt(K.n2sq(hh)) + 1e-9 and np.abs(K.Mop.T @ hh).max() <= K.normG2 * np.sqrt(K.n2sq(hh)) + 1e-9)
    check(f"  (ttpm-J-bound) |J[v]-J[u]| <= 2||G||_2 L D", abs(J(v) - J(u)) <= 2 * K.normG2 * Lf * D + 1e-9)
    Cc = lambda z: eta * K.n2sq(z) + sig * K.n1(z) + J(z)
    check(f"  local identity eta D + sig H", abs(eta * (K.n2sq(v) - K.n2sq(u)) + sig * (K.n1(v) - K.n1(u)) - (eta * D + sig * H)) < 1e-9)
    check(f"  (ttpm-purification-bound) C[v]-C[u] >= (eta-2||G||L)D + sig H", Cc(v) - Cc(u) >= (eta - 2 * K.normG2 * Lf) * D + sig * H - 1e-9,
          f"lhs={Cc(v)-Cc(u):.6f} rhs={(eta-2*K.normG2*Lf)*D+sig*H:.6f}")
    check(f"  C[-v] = C[v]", abs(Cc(-v) - Cc(v)) < TOL)
    # Corollary dimensionless form and power-kernel formula
    if K.kind == "power":
        g = K.par; rho = vc / vstar
        lhs = eta / (K.cG * lam) >= 2 * K.normG2 / K.cG * (1 + rho ** -2) ** (eps / 2)
        rhs = eta / (K.cG * lam) >= 2 * (1 - g) / np.sqrt(1 - 2 * g) * (1 + rho ** -2) ** (eps / 2)
        check(f"  (ttpm-linear-core-power-kernel) consistent", lhs == rhs and abs(K.normG2 / K.cG - (1 - g) / np.sqrt(1 - 2 * g)) < 1e-12)
        check(f"  (ttpm-linear-core-condition) <=> dimensionless form", (eta >= 2 * lam * K.normG2 * (1 + rho ** -2) ** (eps / 2)) == lhs)

print("\n==== SUMMARY ====")
print("failures:", failures if failures else "none")
