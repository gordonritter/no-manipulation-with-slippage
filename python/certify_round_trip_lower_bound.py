"""Certify a fixed witness for G(t)=t^(-1/4), f(v)=sign(v)*sqrt(abs(v)).

Uses only exact Fraction arithmetic and integer square roots. CSV decimals
define exact rational rates; their exact mean is subtracted. No optimization
or floating-point arithmetic enters the certificate. Run with python3.

For h=1/n, Psi(x)=16*x^(7/4)/21, the cell integrals are Psi(h) on
the diagonal and Psi((d+1)h)-2*Psi(dh)+Psi((d-1)h) at lag d>0.

A continuous witness replaces each jump at t=i/n by the straight line
joining the two rates on [i/n-delta,i/n+delta], delta=10^(-32).
Each symmetric ramp preserves the integral exactly. Its squared L2 error
is delta*(jump)^2/6. With e bounding that L2 error, Young's inequality
and |f(x)-f(y)| <= sqrt(2)*sqrt(|x-y|) give
|K(u)-K(v)| <= (4/3)*(e*sqrt(B)+(sqrt(A)+e)*sqrt(2*e)).
Also sqrt(A(u)*B(u)) <= (sqrt(A)+e)*sqrt(B+e).
These bounds certify the explicitly defined continuous witness directly.
"""

import csv
import hashlib
from fractions import Fraction as F
from math import isqrt
from pathlib import Path


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def neg(x):
    return -x[1], -x[0]


def mul(x, y):
    ends = [a * b for a in x for b in y]
    return min(ends), max(ends)


def point(x):
    return x, x


def root(x, degree, digits):
    """Enclose a nonnegative rational root on a decimal rational grid."""
    assert x >= 0 and degree in (2, 4)
    scale = 10 ** digits
    target = x.numerator * scale ** degree // x.denominator
    k = isqrt(target)
    if degree == 4:
        k = isqrt(k)
    lo, hi = F(k, scale), F(k + 1, scale)
    assert lo ** degree <= x < hi ** degree
    return lo, hi


def outward(x, places=24):
    scale = 10 ** places
    lower = x[0].numerator * scale // x[0].denominator
    upper = -((-x[1].numerator * scale) // x[1].denominator)

    def fmt(k):
        sign = '-' if k < 0 else ''
        whole, frac = divmod(abs(k), scale)
        return f'{sign}{whole}.{frac:0{places}d}'

    return f'[{fmt(lower)}, {fmt(upper)}]'


def certify(digits):
    path = Path(__file__).with_name('extremal_round_trip.csv')
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    n = len(rows)
    assert n == 128
    assert all(F(row['t']) == F(2*i+1, 2*n) for i, row in enumerate(rows))
    raw = [F(row['v']) for row in rows]
    mean = sum(raw) / n
    v = [x - mean for x in raw]
    assert sum(v) == 0
    A = sum(x*x for x in v) / n
    B = sum(abs(x) for x in v) / n
    psi = [mul(point(F(16, 21)), root(F(j, n)**7, 4, digits))
           for j in range(n+1)]
    weights = [psi[1]]
    for d in range(1, n):
        weights.append(add(add(psi[d+1], mul(point(F(-2)), psi[d])), psi[d-1]))
    assert all(w[0] > 0 for w in weights)
    f = [root(abs(x), 2, digits) if x >= 0 else neg(root(abs(x), 2, digits))
         for x in v]
    cost = point(F(0))
    for i in range(n):
        for j in range(i+1):
            cost = add(cost, mul(point(v[i]), mul(weights[i-j], f[j])))
    K = neg(cost)
    den = root(A*B, 2, digits)
    q = K[0]/den[1], K[1]/den[0]
    assert q[0] > F('0.042012615472')
    assert q[1] < F('0.042012615473')

    delta = F(1, 10**32)
    assert 2*delta < F(1, n)
    error2 = delta * sum((v[i+1]-v[i])**2 for i in range(n-1)) / 6
    e = root(error2, 2, digits)[1]
    norm_v = root(A, 2, digits)[1]
    loss = F(4, 3)*(e*root(B, 2, digits)[1]
                       + (norm_v+e)*root(2*e, 2, digits)[1])
    den_upper = (norm_v+e)*root(B+e, 2, digits)[1]
    continuous_lower = (K[0]-loss)/den_upper
    assert continuous_lower > F('0.0420125')

    print(f'precision grid: 10^-{digits}; n={n}')
    print('CSV SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
    print('original mean (exact):', mean)
    print('corrected mean (exact):', sum(v)/n)
    for name, bounds in [('A', point(A)), ('B', point(B)), ('K', K), ('q', q)]:
        print(name, outward(bounds))
    print('q interval width < 10^-20:', q[1]-q[0] < F(1, 10**20))
    assert q[1]-q[0] < F(1, 10**20)
    print('continuous ramp half-width (exact):', delta)
    print('L2 error upper bound:', outward(point(e)))
    print('impact profit error upper bound:', outward(point(loss)))
    print('continuous quotient lower bound:', outward(point(continuous_lower)))
    print('CERTIFIED: Xi_(1/2) > 0.0420125 for lambda=v_star=1, gamma=1/4')
    print('CERTIFIED: critical common slippage > 0.02100625')
    print()


if __name__ == '__main__':
    for precision in (30, 40):
        certify(precision)
