"""Certify Appendix A's permanent-impact constant using exact arithmetic.

Run: python3 python/certify_permanent_constant.py
All endpoints, sign tests, square-root bounds, and rounding tests use
integers or fractions.Fraction. No floating-point arithmetic is used.
"""

from fractions import Fraction as Q
from math import isqrt


def polynomial(z):
    return 18*z**3 + 39*z**2 + 8*z - 5


def squared_constant(z):
    return (1584*z**2 + 2046*z - 535) / 15360


def floor_scaled(x, scale):
    return x.numerator * scale // x.denominator


def decimal(k, digits):
    scale = 10**digits
    return f"{k // scale}.{k % scale:0{digits}d}"


def enclose(lo, hi, digits, square_root=False):
    scale = 10**digits
    if square_root:
        left = isqrt(floor_scaled(lo, scale**2))
        right = isqrt(floor_scaled(hi, scale**2)) + 1
        assert Q(left, scale)**2 < lo < hi < Q(right, scale)**2
    else:
        left = floor_scaled(lo, scale)
        right = floor_scaled(hi, scale) + 1
        assert Q(left, scale) < lo < hi < Q(right, scale)
    return Q(left, scale), Q(right, scale), (
        f"({decimal(left, digits)}, {decimal(right, digits)})"
    )


def verify_rounding(lo, hi, rounded, digits):
    target = Q(rounded)
    half_unit = Q(1, 2*10**digits)
    assert target - half_unit < lo < hi < target + half_unit


def main():
    # P is strictly increasing on [0,1], with P(0)<0<P(1).
    # Integer bisection finds adjacent decimal rationals enclosing its root.
    scale = 10**20
    left, right = 0, scale
    while right - left > 1:
        mid = (left + right) // 2
        if polynomial(Q(mid, scale)) < 0:
            left = mid
        else:
            right = mid
    z_lo, z_hi = Q(left, scale), Q(right, scale)
    assert 0 < z_lo < z_hi < 1
    assert polynomial(z_lo) < 0 < polynomial(z_hi)
    print(f"z_* in ({decimal(left, 20)}, {decimal(right, 20)})")
    print(f"P(lower) = {polynomial(z_lo)}")
    print(f"P(upper) = {polynomial(z_hi)}")

    # The reduced expression has derivative (3168*z+2046)/15360>0.
    r_lo, r_hi = squared_constant(z_lo), squared_constant(z_hi)
    assert 0 < r_lo < r_hi
    q_lo, q_hi, interval = enclose(r_lo, r_hi, 12, square_root=True)
    print(f"Xi/(lambda*sqrt(v_*)) in {interval}")
    verify_rounding(q_lo, q_hi, "0.08112574", 8)
    verify_rounding(q_lo, q_hi, "0.081", 3)

    threshold_lo, threshold_hi, interval = enclose(r_lo/4, r_hi/4, 12)
    print(f"Exact product threshold/(lambda^2*v_*) in {interval}")
    verify_rounding(threshold_lo, threshold_hi, "0.00164535", 8)

    ratio_lo, ratio_hi, interval = enclose(1/(4*r_hi), 1/(4*r_lo), 10)
    print(f"Sufficient / exact product threshold in {interval}")
    verify_rounding(ratio_lo, ratio_hi, "38", 0)
    print("All reported decimal roundings verified by rational inequalities.")


if __name__ == "__main__":
    main()
