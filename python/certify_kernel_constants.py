"""Certify Table 1 using integer cell integrals and rational spectral bounds.

Run: python3 python/certify_kernel_constants.py
NumPy/SciPy propose eigenvectors only. Every reported enclosure is checked
with Python integers and fractions. The continuum argument is in Appendix B
of no_manipulation_v2.tex. No floating-point eigensolver accuracy is assumed.
"""

from fractions import Fraction as F
import json
import math

import numpy as np
from scipy.linalg import matmul_toeplitz, toeplitz
from scipy.sparse.linalg import LinearOperator, eigsh

BITS = 80
SCALE = 1 << BITS
VECTOR_SCALE = 1 << 40
COARSE_N = 128


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def integer_root(x, degree):
    """Return floor(x**(1/degree)), verified by integer powers."""
    if x == 0:
        return 0
    y = 1 << ((x.bit_length() + degree - 1) // degree)
    while True:
        z = ((degree - 1) * y + x // y ** (degree - 1)) // degree
        if z >= y:
            break
        y = z
    require(y ** degree <= x < (y + 1) ** degree, "integer root")
    return y


def cell_column(gamma, n):
    """Integer approximation to the normalized lower-triangular matrix.

    F_d = n*(d/n)**(2-gamma)/((1-gamma)*(2-gamma)).
    Each z_d/SCALE <= F_d <= (z_d+1)/SCALE. Second differences
    consequently have absolute error <= 2/SCALE.
    """
    power = 2 - gamma
    p, q = power.numerator, power.denominator
    numerator_factor = (q * q) ** q * SCALE ** q
    denominator = n ** (p - q) * (p * (p - q)) ** q
    z = [integer_root(d ** p * numerator_factor // denominator, q)
         for d in range(n + 1)]
    return [z[1]] + [z[d + 1] - 2 * z[d] + z[d - 1]
                      for d in range(1, n)]


def signed_convolution(a, b):
    """Exact convolution using integer polynomial multiplication.

    A power-of-two base exceeds twice the maximum possible coefficient.
    Balanced decoding therefore recovers signed coefficients exactly.
    """
    bound = min(len(a), len(b)) * max(map(abs, a)) * max(map(abs, b))
    if bound == 0:
        return [0] * (len(a) + len(b) - 1)
    width = max(1, (bound.bit_length() + 2 + 7) // 8)
    base = 1 << (8 * width)
    require(2 * bound < base, "convolution base")

    def pack(values):
        positive = b''.join(max(x, 0).to_bytes(width, 'little') for x in values)
        negative = b''.join(max(-x, 0).to_bytes(width, 'little') for x in values)
        return int.from_bytes(positive, 'little') - int.from_bytes(negative, 'little')

    length = len(a) + len(b) - 1
    product = pack(a) * pack(b)
    raw = (product % (1 << (8 * width * length))).to_bytes(width * length, 'little')
    result, carry = [], 0
    for i in range(length):
        digit = int.from_bytes(raw[i * width:(i + 1) * width], 'little') + carry
        if digit >= base // 2:
            digit -= base
            carry = 1
        else:
            carry = 0
        result.append(digit)
    require(all(abs(x) <= bound for x in result), "convolution decoding")
    return result


def matrix_column(cell, kind):
    return ([2 * cell[0]] + cell[1:]) if kind == 'nu' else ([0] + cell[1:])


def product(column, vector, kind):
    sign = 1 if kind == 'nu' else -1
    kernel = [sign * x for x in column[:0:-1]] + column
    n = len(column)
    return signed_convolution(kernel, vector)[n - 1:2 * n - 1]


def candidate(column, kind):
    n = len(column)
    c = np.array([float(F(x, SCALE)) for x in column])
    if kind == 'a':
        c = 1j * c
    op = LinearOperator((n, n), dtype=c.dtype,
                        matvec=lambda x: matmul_toeplitz((c, c.conj()), x,
                                                         check_finite=False))
    _, vectors = eigsh(op, k=1, which='LA', tol=1e-12, v0=np.ones(n))
    v = vectors[:, 0]
    real = [int(round(float(x) * VECTOR_SCALE)) for x in v.real]
    imag = [int(round(float(x) * VECTOR_SCALE)) for x in v.imag]
    return real, imag


def rayleigh(column, kind, vector):
    """Exact Rayleigh quotient and squared residual of rational matrix A."""
    x, y = vector
    if kind == 'nu':
        require(not any(y), "real symmetric vector")
        ax, ay = product(column, x, kind), [0] * len(x)
    else:
        # A=iK: A(x+iy)=-Ky+iKx.
        ax = [-z for z in product(column, y, kind)]
        ay = product(column, x, kind)
    norm = sum(a * a + b * b for a, b in zip(x, y))
    require(norm > 0, "nonzero vector")
    dot = sum(a * b + c * d for a, b, c, d in zip(x, ax, y, ay))
    r = F(dot, SCALE * norm)
    second = F(sum(a * a + b * b for a, b in zip(ax, ay)), SCALE ** 2 * norm)
    residual_squared = second - r * r
    require(residual_squared >= 0, "residual variance")
    return r, residual_squared


def fourth_trace(column, kind):
    """Exact trace(A**4); for A=iK this equals sum((K**2)_ij**2)."""
    row = column if kind == 'nu' else [-x for x in column]
    matrix = toeplitz(np.array(column, dtype=object), np.array(row, dtype=object))
    square = matrix @ matrix
    return F(sum(int(x) ** 2 for x in square.flat), SCALE ** 4)


def rational_upper(estimate, predicate, places=9):
    """Floating point proposes a rational; the predicate proves its bound."""
    denominator = 10 ** places
    numerator = math.ceil(estimate * denominator) + 1
    for _ in range(10000):
        value = F(numerator, denominator)
        if predicate(value):
            return value
        numerator += 1
    raise ArithmeticError('could not validate proposed upper bound')


def projection_bound(gamma, n):
    alpha = 1 - gamma
    p, q = alpha.numerator, alpha.denominator
    estimate = 4 * n ** (-float(alpha)) / (float(alpha) * math.sqrt(float(2 * alpha + 1)))
    return rational_upper(estimate, lambda e:
                          (e * e * alpha * alpha * (2 * alpha + 1) / 16) ** q
                          * n ** (2 * p) >= 1)


def coarse_second_bound(gamma, cell, kind):
    column = matrix_column(cell, kind)
    r, _ = rayleigh(column, kind, candidate(column, kind))
    require(r > 0, "positive coarse quotient")
    multiplicity = 1 if kind == 'nu' else 2
    remainder = fourth_trace(column, kind) / multiplicity - r ** 4
    require(remainder > 0, "fourth trace remainder")
    s = rational_upper(float(remainder) ** .25, lambda x: x ** 4 >= remainder)
    # Weyl bound transfers the rational matrix estimate to exact cell integrals.
    s += F(2 * COARSE_N, SCALE)
    e = projection_bound(gamma, COARSE_N)
    estimate = (float(s + e) + math.sqrt(float((s - e) ** 2 + 4 * e * e))) / 2
    upper = rational_upper(estimate, lambda u:
                           u > s and u > e and (u - s) * (u - e) >= e * e)
    return upper


def outward(value, places, upper):
    scale = 10 ** places
    scaled = value * scale
    integer = (-(-scaled.numerator // scaled.denominator) if upper
               else scaled.numerator // scaled.denominator)
    return f'{integer // scale}.{integer % scale:0{places}d}'


def certify(gamma, n, kind, fine_cell, coarse_cell):
    second_bound = coarse_second_bound(gamma, coarse_cell, kind)
    column = matrix_column(fine_cell, kind)
    r, residual_squared = rayleigh(column, kind, candidate(column, kind))
    tau = F(2 * n, SCALE)
    require(r > second_bound + tau, "separation from the remaining spectrum")
    # The second eigenvalue of A is at most second_bound+tau.
    matrix_lower = r - tau
    matrix_upper = r + residual_squared / (r - second_bound - tau) + tau
    require(matrix_upper - matrix_lower < F(1, 10 ** 18), "finite-matrix interval width")
    e = projection_bound(gamma, n)
    require(matrix_lower > e, "continuum spectral separation")
    lower = matrix_lower / 2
    upper = matrix_upper / 2 + e * e / (2 * (matrix_lower - e))
    expected = {F(1, 10): {'nu': 585, 'a': 355},
                F(1, 4): {'nu': 763, 'a': 425},
                F(2, 5): {'nu': 1046, 'a': 518}}[gamma][kind]
    require(F(2 * expected - 1, 2000) < lower <= upper < F(2 * expected + 1, 2000),
            "three-decimal rounding")
    return {'gamma': str(gamma), 'constant': kind, 'cells': n,
            'lower': outward(lower, 8, False), 'upper': outward(upper, 8, True),
            'certified_three_decimals': f'{expected / 1000:.3f}',
            'coarse_second_eigenvalue_upper': outward(second_bound, 8, True),
            'finite_matrix_interval_width_upper': outward(matrix_upper - matrix_lower, 20, True)}


def main():
    # Check balanced convolution against a separate exact quadratic implementation.
    for a, b in [([2, -3, 5], [-7, 0, 11]), ([-9, -2], [3, -4, 1]),
                 ([0, 0, 0], [1, -1])]:
        expected = [sum(a[i] * b[k-i] for i in range(len(a))
                        if 0 <= k-i < len(b)) for k in range(len(a)+len(b)-1)]
        require(signed_convolution(a, b) == expected, "convolution self-check")
    for gamma, n in [(F(1, 10), 2048), (F(1, 4), 2048), (F(2, 5), 16384)]:
        coarse = cell_column(gamma, COARSE_N)
        fine = cell_column(gamma, n)
        for kind in ['nu', 'a']:
            print(json.dumps(certify(gamma, n, kind, fine, coarse)), flush=True)
    print('PASS: all six numerical entries round as reported; gamma=0 uses exact formulas.')


if __name__ == '__main__':
    main()
