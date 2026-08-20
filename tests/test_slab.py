"""Spec test 2: full-cross-section dielectric slab vs the analytic two-interface solution.

For a slab filling the whole cross-section, modes do not couple and each lead mode
scatters exactly like a 1-D Fresnel two-interface problem with the modal admittances
zeta_TE = beta/k0, zeta_TM = eps k0/beta (v-convention, refs/CONVENTIONS.md):

    r   = (zeta1 - zeta2) / (zeta1 + zeta2)
    S11 = r (1 - X^2) / (1 - r^2 X^2),      X = exp(i beta2 L)
    S21 = (1 - r^2) X / (1 - r^2 X^2)

Required accuracy: |Delta S| < 1e-10 (the implementation is analytic here, so the
error is pure roundoff).
"""

import numpy as np
import pytest

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import C0, Solver

A, B = 0.02286, 0.01016  # WR-90
L = 0.0061


def _analytic_slab(kind, m, n, k0, eps2, L):
    kc2 = (m * np.pi / A) ** 2 + (n * np.pi / B) ** 2
    b1 = np.sqrt(k0**2 - kc2 + 0j)
    b2 = np.sqrt(eps2 * k0**2 - kc2 + 0j)
    if b1.imag < 0:
        b1 = -b1
    if b2.imag < 0:
        b2 = -b2
    z1 = b1 / k0 if kind == "TE" else k0 / b1
    z2 = b2 / k0 if kind == "TE" else eps2 * k0 / b2
    r = (z1 - z2) / (z1 + z2)
    X = np.exp(1j * b2 * L)
    s11 = r * (1 - X**2) / (1 - r**2 * X**2)
    s21 = (1 - r**2) * X / (1 - r**2 * X**2)
    return s11, s21


def _solve_slab(eps2, f_hz):
    struct = Structure(Waveguide(A, B), [Box(0, A, 0, B, 0.0, L, eps2)])
    solver = Solver(struct, M=4, N=4)
    return solver.smatrix(f_hz)


@pytest.mark.parametrize("eps2", [4.0 + 0.0j, 4.0 + 0.5j])
def test_slab_vs_analytic_all_modes(eps2):
    f_hz = 18.0e9  # TE10, TE20, TE01, TE11, TM11 propagating in the empty lead
    k0 = 2 * np.pi * f_hz / C0
    res = _solve_slab(eps2, f_hz)

    for i, (kind, m, n) in enumerate(res.lead.labels):
        s11_ref, s21_ref = _analytic_slab(kind, m, n, k0, eps2, L)
        assert abs(res.smatrix.s11[i, i] - s11_ref) < 1e-10, (kind, m, n)
        assert abs(res.smatrix.s21[i, i] - s21_ref) < 1e-10, (kind, m, n)

    # No inter-mode coupling for a full-cross-section slab:
    for blk in (res.smatrix.s11, res.smatrix.s21):
        off = blk - np.diag(np.diag(blk))
        assert np.max(np.abs(off)) < 1e-12


def test_slab_reciprocity_and_port_symmetry():
    res = _solve_slab(4.0 + 0.0j, 18.0e9)
    s = res.smatrix
    # Symmetric structure: S22 = S11, S12 = S21; reciprocity: S = S^T.
    assert np.allclose(s.s22, s.s11, atol=1e-12)
    assert np.allclose(s.s12, s.s21, atol=1e-12)
    assert np.allclose(s.full(), s.full().T, atol=1e-12)
