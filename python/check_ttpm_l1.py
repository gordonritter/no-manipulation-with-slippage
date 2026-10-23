"""Numerical sanity checks for the L1-kernel TTPM theorem.

Run with python3 check_ttpm_l1.py. Uses the standard library only.
Kernel integrals are analytic on constant-rate time cells, including
the diagonal for singular power kernels. These floating-point checks
exercise the theorem; the manuscript's argument supplies its proof.
"""

import math
import random


def cap(values, total):
    """Cap nonnegative cell rates to give the requested sum."""
    values = sorted(values)
    if total <= 0:
        return 0.0
    if total >= math.fsum(values):
        return values[-1]
    prefix = 0.0
    for i, value in enumerate(values):
        level = (total - prefix) / (len(values) - i)
        if level <= value:
            return max(0.0, level)
        prefix += value
    raise AssertionError('Cap not found')


def kernels():
    result = []
    for gamma in (0.0, 0.1, 0.25, 0.49, 0.5, 0.75, 0.9, 0.99):
        def psi(x, g=gamma):
            return x**(2-g) / ((1-g)*(2-g))
        result.append((f'power {gamma:g}', psi, 2**gamma/(1-gamma)))
    for decay in (0.2, 2.0, 20.0, -2.0):
        def psi(x, k=decay):
            return (math.expm1(-k*x)+k*x)/(k*k)
        row_bound = (2*(-math.expm1(-decay/2))/decay if decay > 0
                     else -math.expm1(-decay)/decay)
        result.append((f'exponential {decay:g}', psi, row_bound))
    # Nonmonotone lag kernels: indicator of [left,right]. No positive-type
    # assumption is needed by this theorem.
    for left, right in ((0.2, 0.3), (0.6, 0.9)):
        def psi(x, a=left, b=right):
            return (max(x-a, 0)**2-max(x-b, 0)**2)/2
        def primitive(x):
            return max(x-left, 0)-max(x-right, 0)
        corners = (0, 1, left, right, 1-left, 1-right)
        bound = max(primitive(t)+primitive(1-t) for t in corners)
        result.append((f'lag interval {left:g}:{right:g}', psi, bound))
    result.append(('zero', lambda x: 0.0, 0.0))
    return result


def impacts():
    result = [
        ('linear', lambda x: 1.7*x, 1.7),
        ('saturating', lambda x: 2.3*math.tanh(x/.4), 2.3/.4),
        ('nonmonotone', lambda x: x+2*math.sin(x), 3.0),
        ('kinked', lambda x: math.copysign(
            min(abs(x), .3)+.1*max(abs(x)-.3, 0), x), 1.0),
    ]
    for epsilon, core in ((.5, .01), (.8, .1)):
        def f(x, e=epsilon, c=core):
            return 1.2*x*((x*x+c*c)/(1+c*c))**(-e/2)
        result.append((f'linear core {epsilon:g}', f,
                       1.2*((1+core*core)/(core*core))**(epsilon/2)))
    return result


def weights(n, psi):
    h = 1/n
    return [psi(h)] + [psi((d+1)*h)-2*psi(d*h)+psi((d-1)*h)
                       for d in range(1, n)]


def impact_cost(v, f, row):
    fv = [f(x) for x in v]
    return math.fsum(v[i]*math.fsum(row[i-j]*fv[j] for j in range(i+1))
                     for i in range(len(v)))


def main():
    rng = random.Random(20261021)
    n = 32
    paths = [[1.0]*n, [-1.0]*n,
             [1.0, -0.99999999]*(n//2),
             [30.0]+[-.1]*(n-1), [-30.0]+[1.0]*(n-1),
             [0.0]*(n-2)+[1.0, -0.2],
             [.1+math.sin(2*math.pi*(i+.5)/n) for i in range(n)]]
    for _ in range(36):
        base = [rng.gauss(0, 1) for _ in range(n)]
        offset = rng.uniform(-.8, .8)
        scale = 10**rng.uniform(-4, 4)
        paths.append([scale*(x+offset) for x in base])
    count = intermediate = loss_count = 0
    max_ratio = (0.0, None)
    for name, psi, R in kernels():
        row = weights(n, psi)
        integral = math.fsum((n-d)*row[d] for d in range(n))
        assert abs(integral-psi(1)) <= 1e-9*max(1, abs(psi(1)))
        assert min(row) >= -1e-12*max(1, R)
        for i in range(n):
            row_mass = n*(math.fsum(row[:i+1])+math.fsum(row[:n-i]))
            assert row_mass <= R+1e-9*max(1, R)
        for fname, f, L in impacts():
            for original in paths:
                direction = 1 if math.fsum(original) > 0 else -1
                v = [direction*x for x in original]
                Q = math.fsum(v)/n
                assert Q > 0
                p = [max(x, 0) for x in v]
                negative = [max(-x, 0) for x in v]
                B = math.fsum(negative)/n
                level = cap(p, Q*n)
                u = [min(x, level) for x in p]
                scale = max(abs(x) for x in v)
                tolerance = 1e-9*max(scale, 1e-20)
                assert abs(math.fsum(u)/n-Q) < tolerance
                assert all(x >= 0 for x in u)
                H = math.fsum(abs(x-y) for x, y in zip(v, u))/n
                D = math.fsum((x-y)*(x+y) for x, y in zip(v, u))/n
                assert abs(H-2*B) < tolerance
                assert D >= -tolerance*scale
                Jv, Ju = impact_cost(v, f, row), impact_cost(u, f, row)
                bound = L*R*D
                error = 1e-9*max(L*R*scale*scale, 1e-25)
                assert abs(Jv-Ju) <= bound+error, (name, fname, D, Jv, Ju)
                if bound > error:
                    ratio = abs(Jv-Ju)/bound
                    if ratio > max_ratio[0]:
                        max_ratio = (ratio, (name, fname))
                if Ju-Jv > error:
                    loss_count += 1
                # Cost comparison at the proposed boundary and above it,
                # with zero and positive spread; also check sell-order symmetry.
                assert abs(impact_cost(original, f, row)-Jv) <= error
                for eta, spread in ((L*R, 0), (L*R, .7), (1.5*L*R, .7)):
                    difference = eta*D+spread*H+Jv-Ju
                    lower = (eta-L*R)*D+spread*H
                    assert difference >= lower-error
                count += 1
                # Check the cap path and finite-difference energy estimates.
                if B > tolerance and fname == 'nonmonotone':
                    old_v, old_a, old_b, old_E = v, max(p), max(negative), math.fsum(x*x for x in v)/n
                    for step in range(1, 17):
                        s = B*step/16
                        a = cap(p, (Q+B-s)*n)
                        b = cap(negative, (B-s)*n)
                        w = [min(x, a)-min(y, b) for x, y in zip(p, negative)]
                        E = math.fsum(x*x for x in w)/n
                        ds = B/16
                        e_tol = 1e-9*scale*scale
                        assert abs(math.fsum(w)/n-Q) < tolerance
                        assert abs(math.fsum(abs(x-y) for x, y in zip(w, old_v))/n-2*ds) < tolerance
                        assert 2*(a+b)*ds-e_tol <= old_E-E <= 2*(old_a+old_b)*ds+e_tol
                        old_v, old_a, old_b, old_E = w, a, b, E
                        intermediate += 1
        print(f'PASS {name}')
    print(f'PASS {count} strategy/kernel/impact combinations; {intermediate} cap steps')
    print(f'{loss_count} cases have increased impact cost after capping, covered by the bound')
    print('Largest |J(v)-J(u)|/(L R D):', max_ratio)
    print('Refinement for cell averages of continuous v(t)=0.2+sin(2*pi*t), gamma=0.75:')
    for n in (32, 64, 128, 256):
        gamma = .75
        psi = lambda x: x**(2-gamma)/((1-gamma)*(2-gamma))
        row = weights(n, psi)
        v = [.2+n*(math.cos(2*math.pi*i/n)-math.cos(2*math.pi*(i+1)/n))/(2*math.pi)
             for i in range(n)]
        p = [max(x, 0) for x in v]
        u = [min(x, cap(p, .2*n)) for x in p]
        f = lambda x: math.tanh(x)
        D = math.fsum((x-y)*(x+y) for x, y in zip(v, u))/n
        ratio = abs(impact_cost(v, f, row)-impact_cost(u, f, row))/(2**gamma/(1-gamma)*D)
        assert ratio <= 1+1e-9
        print(f'  n={n:3d}: quotient={ratio:.10f}')


if __name__ == '__main__':
    main()
