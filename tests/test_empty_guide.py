"""Spec test 1: empty guide.

* Numerical slice eigenproblem reproduces the analytic TE/TM dispersion relation
  beta^2 = eps k0^2 - kc^2 to machine precision.
* The S-matrix of an empty section is a pure diagonal phase factor.
"""

import numpy as np

from sceptre.basis import ModeBasis
from sceptre.fourier import EpsOperators
from sceptre.geometry import Box, Structure, Waveguide
from sceptre.slicesolver import build_fg
from sceptre.solver import C0, Solver

A, B = 0.02286, 0.01016  # WR-90
F_GHZ = 18.0


def _analytic_beta_squared(basis: ModeBasis, k0: float, eps: float = 1.0) -> np.ndarray:
    vals = []
    for m in range(0, basis.M + 1):
        for n in range(0, basis.N + 1):
            if m == 0 and n == 0:
                continue
            kc2 = (m * np.pi / A) ** 2 + (n * np.pi / B) ** 2
            vals.append(eps * k0**2 - kc2)  # TE_mn
            if m >= 1 and n >= 1:
                vals.append(eps * k0**2 - kc2)  # TM_mn (degenerate with TE_mn)
    return np.sort(np.asarray(vals))


def test_empty_guide_dispersion_machine_precision():
    """eig(FG) eigenvalues = -beta^2 must equal the analytic dispersion set."""
    import scipy.linalg as sla

    basis = ModeBasis(A, B, 5, 5)
    k0 = 2 * np.pi * F_GHZ * 1e9 / C0
    ops = EpsOperators(
        exx=np.eye(basis.X.size, dtype=complex),
        eyy=np.eye(basis.Y.size, dtype=complex),
        ezz=np.eye(basis.Z.size, dtype=complex),
    )
    F, G = build_fg(ops, basis, k0)
    lam = sla.eigvals(F @ G)
    beta2_num = np.sort(-lam.real)
    assert np.max(np.abs(lam.imag)) < 1e-6 * k0**2  # spectrum must be real
    beta2_ref = _analytic_beta_squared(basis, k0)
    assert np.allclose(beta2_num, beta2_ref, rtol=0, atol=1e-12 * k0**2)


def test_empty_guide_smatrix_is_pure_phase():
    L = 0.0173
    wg = Waveguide(A, B)
    struct = Structure(wg, [Box(0, A, 0, B, 0.0, L, 1.0)])  # a box of vacuum
    solver = Solver(struct, M=4, N=4)
    res = solver.smatrix(F_GHZ * 1e9)
    s = res.smatrix

    phases = np.exp(1j * res.lead.beta * L)
    assert np.allclose(s.s21, np.diag(phases), atol=1e-13)
    assert np.allclose(s.s12, np.diag(phases), atol=1e-13)
    assert np.max(np.abs(s.s11)) < 1e-13
    assert np.max(np.abs(s.s22)) < 1e-13
