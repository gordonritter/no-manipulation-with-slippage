"""Verify change 2's comparisons from the existing certified bounds.

Run: python3 python/check_threshold_comparisons.py
The kernel interval is read from kernel_constants_certificate.txt.
This script checks the downstream arithmetic; the operator certificate
itself is supplied by certify_kernel_constants.py and Appendix B.
"""

from decimal import Decimal, localcontext
from fractions import Fraction as Q
import json
from pathlib import Path

from certify_permanent_constant import polynomial, squared_constant


def main():
    records = [json.loads(line) for line in
               Path(__file__).with_name("kernel_constants_certificate.txt")
               .read_text().splitlines() if line.startswith("{")]
    kernel = next(row for row in records
                  if row["gamma"] == "1/4" and row["constant"] == "nu")
    z_lo, z_hi = Q("0.25897201316507985454"), Q("0.25897201316507985455")
    assert polynomial(z_lo) < 0 < polynomial(z_hi)
    p_lo, p_hi = squared_constant(z_lo)/4, squared_constant(z_hi)/4
    envelope_floor = Q(1, 64)
    ratio_lo, ratio_hi = envelope_floor/p_hi, envelope_floor/p_lo
    assert Q("9.45") < ratio_lo < ratio_hi < Q("9.55")

    s_lo, s_hi = Q("0.0420125")/2, Q("0.3567")
    assert s_lo == Q("0.02100625")
    product_upper = Q(kernel["upper"])*Q(4, 3)/8
    assert product_upper == Q("0.127234755") < s_hi**2
    bracket_ratio = s_hi**2/s_lo**2
    assert 285 < bracket_ratio < 295  # 290 to two significant figures.

    with localcontext() as ctx:
        ctx.prec = 30
        def decimal(q):
            return Decimal(q.numerator)/Decimal(q.denominator)
        print("Permanent envelope product floor:", decimal(envelope_floor))
        print("Envelope floor / exact threshold in:",
              decimal(ratio_lo), decimal(ratio_hi))
        print("Transient critical common coefficient in:", decimal(s_lo), decimal(s_hi))
        print("Certified upper product bound:", decimal(product_upper))
        print("Lower product bound:", decimal(s_lo**2))
        print("Ratio of displayed product bounds:", decimal(bracket_ratio))
    print("PASS: all sign, bound, and rounding checks use exact rational arithmetic.")


if __name__ == "__main__":
    main()
