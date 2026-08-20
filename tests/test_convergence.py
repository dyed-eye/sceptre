"""Spec test 4: convergence with vs without Li's factorization.

Partial-height dielectric block (full width, strong contrast eps = 9).  The
permittivity jumps across a y-normal edge, where Ey is discontinuous but
Dy = eps*Ey is continuous; Li's inverse rule along y restores fast uniform
convergence, while the direct ("Laurent") rule converges at a crawling ~1/N and is
still percent-level wrong at N = 56 (L. Li, JOSA A 13, 1870 (1996)).

Measured behaviour on this benchmark (S21 of TE10 at 10 GHz, reference N = 56):

    N          4        6        8        10       12
    Li      1.7e-2   4.3e-3   1.2e-3   2.8e-4   1.7e-5    (monotone, ~3 decades)
    direct  1.4e-1   1.1e-1   8.4e-2   6.6e-2   5.2e-2    (~1/N crawl)

Beyond N ~ 12 the Li error enters an oscillatory ~N^-3 tail (the projected edge
position beats against the truncation) and is no longer pointwise monotone -- the
monotonicity assertion therefore covers the systematic-decay regime above, which is
where the factorization claim lives.
"""

import numpy as np
import pytest

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import Solver

A, B = 0.02286, 0.01016  # WR-90
F_HZ = 10.0e9  # single-mode band
EPS_BLOCK = 9.0
BLOCK = Box(0.0, A, 0.0, 0.45 * B, 0.0, 0.008, EPS_BLOCK)  # partial height, full width

ORDERS = [4, 6, 8, 10, 12]
N_REF = 56


def _s21_te10(factorization: str, n_order: int) -> complex:
    struct = Structure(Waveguide(A, B), [BLOCK])
    solver = Solver(struct, M=1, N=n_order, factorization=factorization)
    res = solver.smatrix(F_HZ)
    return res.coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0))


@pytest.fixture(scope="module")
def errors():
    ref = _s21_te10("li", N_REF)
    err = {}
    for fac in ("li", "direct"):
        err[fac] = np.array([abs(_s21_te10(fac, n) - ref) for n in ORDERS])
    return err, ref


def test_li_converges_monotonically(errors):
    err, _ = errors
    li = err["li"]
    assert np.all(np.diff(li) < 0), f"Li errors not monotone: {li}"


def test_li_markedly_faster_than_direct(errors):
    err, ref = errors
    li, direct = err["li"], err["direct"]
    assert abs(ref) > 0.05  # sanity: transmission is not degenerate at this freq
    assert li[-1] < direct[-1] / 100.0, (
        f"Li error {li[-1]:.3e} not markedly below direct-rule error {direct[-1]:.3e}"
    )
    # Li gains >= 2 decades over the ladder; the direct rule cannot even manage one.
    assert li[-1] < li[0] / 100.0
    assert direct[-1] > direct[0] / 10.0


def test_direct_rule_is_the_slow_one(errors):
    """The unfactorized error at the highest order is worse than Li at the LOWEST."""
    err, _ = errors
    assert err["direct"][-1] > err["li"][0]
