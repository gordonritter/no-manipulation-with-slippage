# Price manipulation theorems for nonlinear transient impact models

Scripts and data for the figures, the table, and the certified constants
in the paper. Scripts live in `python/`; figures are written to `figs/`.

Requirements: Python 3 with NumPy, SciPy, and Matplotlib. The certificate
scripts use only the standard library (`fractions` and integer arithmetic),
except `certify_kernel_constants.py`, which uses NumPy and SciPy to propose
eigenvector candidates that are then checked exactly.

Run every script from the repository root, for example

    python3 python/certify_permanent_constant.py

## Figures and table

| Item in the paper | Script | Output |
| --- | --- | --- |
| Figure 1, sufficient bound on `b_eta` in market units, two panels | `python/fig_b_eta_spread_v2.py` | `figs/fig_b_eta_spread_v2.pdf` |
| Figure 2, square-root law in its envelope, the counterexample impact function, and the round trip of the lemma bounding `mu_G` | `python/fig_envelope.py` | `figs/fig_envelope.pdf` |
| Figure 3, numerical round trip for `G(t) = t^(-1/4)` and cost along its ray | `python/extremal_round_trip_constant.py` | `figs/fig_extremal_round_trip.pdf` and `python/extremal_round_trip.csv` |
| Figure 4, angular ordering of trading edges (Appendix A) | `figs/fig_angular_order.py` | `figs/fig_angular_order.pdf` |
| Table 1, kernel constants `c_G`, `nu_G`, `a_G` for power-law kernels | `python/kernel_constants.py` | printed table |

`python/fig_b_eta_spread.py` is the single-panel version of Figure 1 from
the first submitted version of the paper.

## Certificates

These scripts prove the decimal values printed in the paper using exact
rational arithmetic and integer root bounds. Each writes its result to
standard output; the recorded output files are included.

| Claim in the paper | Script | Recorded output |
| --- | --- | --- |
| Section 3.4: `0.042012615472 < q_num < 0.042012615473` and `Xi_{1/2} > 0.0420125` for `G(t) = t^(-1/4)` | `python/certify_round_trip_lower_bound.py` (reads `python/extremal_round_trip.csv`) | printed |
| Appendix A: bracket for the root `z_*`, `Xi_{1/2}/(lambda sqrt(v_*))` to twelve decimals, the exact product threshold, and the ratio to the sufficient threshold | `python/certify_permanent_constant.py` | `python/permanent_constant_certificate.txt` |
| Appendix B: intervals certifying the three decimal places of every numerical entry in Table 1 | `python/certify_kernel_constants.py` | `python/kernel_constants_certificate.txt` |
| Introduction: the envelope floor, the bracket on the critical common coefficient, and the ratio of product bounds | `python/check_threshold_comparisons.py` (reads the two certificate outputs above) | `python/threshold_comparisons.txt` |

The sharp constant for permanent impact can also be computed by direct
numerical optimization with `python/permanent_impact_extremal.py`.

## Numerical checks of the proofs

| Script | What it checks |
| --- | --- |
| `python/verify_v2_proofs.py` | Theorem 1, the quadratic bound, the envelope formulas, the ray minimization, the two-block bound, the lemma bounding `mu_G`, the envelope counterexample, the rate-capping proof, the Appendix A rearrangement, inventory substitution, and pointwise comparison, the Appendix B error bound, and the continuity bound used in the round-trip certificate |
| `python/check_ttpm_l1.py`, `python/check_ttpm_l1_nonuniform.py` | The cost comparisons in the proof of the transaction-triggered theorem on uniform and nonuniform grids, including singular kernels |
| `python/verify_paper_numerics.py` | Randomized checks of the inequalities in the first submitted version |
