"""Unit tests for the analytic lead-mode construction."""

import numpy as np
import pytest

from sceptre.basis import ModeBasis
from sceptre.fourier import EpsOperators
from sceptre.modes import lead_modes
from sceptre.slicesolver import build_fg
from sceptre.solver import C0

A, B = 0.02286, 0.01016  # WR-90


def _k0(f_ghz: float) -> float:
    return 2 * np.pi * f_ghz * 1e9 / C0


@pytest.mark.unit
def test_mode_count_equals_transverse_dimension():
    basis = ModeBasis(A, B, 4, 3)
    lead = lead_modes(basis, _k0(10.0))
    assert lead.W.shape == (basis.size_t, basis.size_t)
    assert len(lead.labels) == basis.size_t


@pytest.mark.unit
def test_te10_dispersion():
    basis = ModeBasis(A, B, 2, 2)
    k0 = _k0(10.0)
    lead = lead_modes(basis, k0)
    i = lead.mode_index("TE", 1, 0)
    beta_ref = np.sqrt(k0**2 - (np.pi / A) ** 2)
    assert lead.beta[i] == pytest.approx(beta_ref, rel=1e-14)
    assert lead.labels[0] == ("TE", 1, 0)  # lowest cutoff first for a > b


@pytest.mark.unit
def test_flux_orthonormality():
    """Pseudo-flux W^T V = I exactly: normalization AND TE/TM flux orthogonality."""
    basis = ModeBasis(A, B, 4, 4)
    lead = lead_modes(basis, _k0(18.0))
    flux = lead.W.T @ lead.V
    assert np.allclose(flux, np.eye(basis.size_t), atol=1e-12)


@pytest.mark.unit
def test_flux_orthonormality_lossy_lead():
    basis = ModeBasis(A, B, 3, 3)
    lead = lead_modes(basis, _k0(12.0), eps=2.0 + 0.3j)
    flux = lead.W.T @ lead.V
    assert np.allclose(flux, np.eye(basis.size_t), atol=1e-12)


@pytest.mark.unit
def test_analytic_modes_satisfy_discrete_maxwell():
    """Cross-validation: analytic modes are exact eigenvectors of the F,G operators.

    F v = i beta e and G e = i beta v must hold for every analytic mode, with the
    operators assembled by the SAME code used for patterned slices (eps = identity ops).
    """
    basis = ModeBasis(A, B, 3, 3)
    k0 = _k0(18.0)
    eps = 2.5 + 0.0j
    ops = EpsOperators(
        exx=eps * np.eye(basis.X.size, dtype=complex),
        eyy=eps * np.eye(basis.Y.size, dtype=complex),
        ezz=eps * np.eye(basis.Z.size, dtype=complex),
    )
    F, G = build_fg(ops, basis, k0)
    lead = lead_modes(basis, k0, eps)
    lhs_e = F @ lead.V
    lhs_v = G @ lead.W
    rhs_e = lead.W * (1j * lead.beta)[None, :]
    rhs_v = lead.V * (1j * lead.beta)[None, :]
    scale = np.abs(1j * k0)
    assert np.allclose(lhs_e, rhs_e, atol=1e-10 * scale)
    assert np.allclose(lhs_v, rhs_v, atol=1e-10 * scale)


@pytest.mark.unit
def test_propagating_mask_wr90_single_mode_band():
    basis = ModeBasis(A, B, 3, 3)
    lead = lead_modes(basis, _k0(10.0))  # only TE10 propagates at 10 GHz in WR-90
    prop = lead.propagating()
    assert prop.sum() == 1
    assert lead.labels[int(np.flatnonzero(prop)[0])] == ("TE", 1, 0)
