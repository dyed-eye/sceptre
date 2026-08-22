"""Unit tests for overlap matrices and permittivity operators (Li vs direct rule)."""

import numpy as np
import pytest
from scipy.integrate import quad

from sceptre.basis import ModeBasis
from sceptre.fourier import (
    EpsOperators,
    build_eps_operators,
    cos_overlap,
    cs_overlap,
    sin_overlap,
)
from sceptre.geometry import Box, Structure, Waveguide
from sceptre.slicesolver import build_fg

A, B = 0.02286, 0.01016  # WR-90


def _sin_fn(m, a):
    return lambda x: np.sqrt(2.0 / a) * np.sin(m * np.pi * x / a)


def _cos_fn(m, a):
    if m == 0:
        return lambda x: np.sqrt(1.0 / a)
    return lambda x: np.sqrt(2.0 / a) * np.cos(m * np.pi * x / a)


@pytest.mark.unit
def test_sin_overlap_matches_quadrature():
    x1, x2 = 0.3 * A, 0.71 * A
    S = sin_overlap(4, A, x1, x2)
    for m in range(1, 5):
        for mp in range(1, 5):
            ref, _ = quad(lambda x: _sin_fn(m, A)(x) * _sin_fn(mp, A)(x), x1, x2)
            assert S[m - 1, mp - 1] == pytest.approx(ref, abs=1e-14)


@pytest.mark.unit
def test_cos_overlap_matches_quadrature():
    x1, x2 = 0.12 * A, 0.55 * A
    C = cos_overlap(4, A, x1, x2)
    for m in range(0, 5):
        for mp in range(0, 5):
            ref, _ = quad(lambda x: _cos_fn(m, A)(x) * _cos_fn(mp, A)(x), x1, x2)
            assert C[m, mp] == pytest.approx(ref, abs=1e-14)


@pytest.mark.unit
def test_full_interval_overlaps_are_identity():
    assert np.allclose(sin_overlap(8, A, 0.0, A), np.eye(8), atol=1e-13)
    assert np.allclose(cos_overlap(8, A, 0.0, A), np.eye(9), atol=1e-13)


def _partial_block_structure(eps=6.0, x_frac=(0.25, 0.75), y_frac=(0.0, 0.5)):
    wg = Waveguide(A, B)
    box = Box(
        x_frac[0] * A, x_frac[1] * A, y_frac[0] * B, y_frac[1] * B, 0.0, 0.005, eps
    )
    return Structure(wg, [box])


@pytest.mark.unit
def test_uniform_layout_gives_scaled_identity():
    wg = Waveguide(A, B)
    struct = Structure(wg, [Box(0, A, 0, B, 0, 0.005, 4.0)])
    layout = struct.segments()[0].cross_section
    basis = ModeBasis(A, B, 3, 3)
    for fac in ("li", "direct"):
        ops = build_eps_operators(layout, basis, fac)
        assert np.allclose(ops.exx, 4.0 * np.eye(basis.X.size))
        assert np.allclose(ops.eyy, 4.0 * np.eye(basis.Y.size))
        assert np.allclose(ops.ezz, 4.0 * np.eye(basis.Z.size))


@pytest.mark.unit
def test_eps_operators_are_symmetric():
    layout = _partial_block_structure().segments()[0].cross_section
    basis = ModeBasis(A, B, 4, 4)
    for fac in ("li", "direct"):
        ops = build_eps_operators(layout, basis, fac)
        for mat in (ops.exx, ops.eyy, ops.ezz):
            assert np.allclose(mat, mat.T, atol=1e-13)


@pytest.mark.unit
def test_li_equals_direct_when_x_uniform():
    """Full-width block (no x-edges): inverse rule along x degenerates to direct rule."""
    layout = _partial_block_structure(x_frac=(0.0, 1.0)).segments()[0].cross_section
    basis = ModeBasis(A, B, 3, 4)
    li = build_eps_operators(layout, basis, "li")
    direct = build_eps_operators(layout, basis, "direct")
    assert np.allclose(li.exx, direct.exx, atol=1e-12)
    # ... but the inverse rule along y must differ (that is the whole point):
    assert not np.allclose(li.eyy, direct.eyy, atol=1e-6)


@pytest.mark.unit
def test_cs_overlap_matches_quadrature():
    x1, x2 = 0.2 * A, 0.83 * A
    CS = cs_overlap(4, A, x1, x2)
    for m in range(0, 5):
        for mp in range(1, 5):
            ref, _ = quad(lambda x: _cos_fn(m, A)(x) * _sin_fn(mp, A)(x), x1, x2)
            assert CS[m, mp - 1] == pytest.approx(ref, abs=1e-14)


@pytest.mark.unit
def test_cs_overlap_full_interval_parity_pattern():
    """Over [0, a] the cos-sin overlap vanishes iff m + m' is even.

    The exact odd-parity VALUES are subsumed by
    test_cs_overlap_matches_quadrature (1e-14 abs vs scipy quad on a
    sub-interval); this test pins the structural selection rule."""
    CS = cs_overlap(6, A, 0.0, A)
    for m in range(0, 7):
        for mp in range(1, 7):
            if (m + mp) % 2 == 0:
                assert abs(CS[m, mp - 1]) < 1e-14
            else:
                assert abs(CS[m, mp - 1]) > 1e-3


@pytest.mark.unit
def test_build_fg_exy_coupling_placement_and_symmetry():
    basis = ModeBasis(A, B, 4, 4)
    layout = _partial_block_structure().segments()[0].cross_section
    ops0 = build_eps_operators(layout, basis, "li")
    rng = np.random.default_rng(7)
    exy = rng.normal(size=(basis.X.size, basis.Y.size)) + 1j * rng.normal(
        size=(basis.X.size, basis.Y.size)
    )
    ops = EpsOperators(exx=ops0.exx, eyy=ops0.eyy, ezz=ops0.ezz, exy=exy)
    k0 = 2 * np.pi * 12e9 / 299792458.0
    F0, G0 = build_fg(ops0, basis, k0)
    F, G = build_fg(ops, basis, k0)
    assert np.allclose(F, F0)  # F never sees the transverse eps coupling
    nx = basis.X.size
    delta = (G - G0) / (1j * k0)
    assert np.allclose(delta[:nx, nx:], exy, atol=1e-13)
    assert np.allclose(delta[nx:, :nx], exy.T, atol=1e-13)
    assert np.allclose(delta[:nx, :nx], 0.0, atol=1e-13)
    # complex-symmetric G is structural reciprocity — must survive the coupling
    scale = np.max(np.abs(G))
    assert np.max(np.abs(G - G.T)) < 1e-13 * scale


@pytest.mark.unit
def test_build_fg_rejects_exy_with_asr_metric_operators():
    basis = ModeBasis(A, B, 3, 3)
    eye_w = np.eye((basis.M + 1) * (basis.N + 1), dtype=complex)
    ops = EpsOperators(
        exx=np.eye(basis.X.size, dtype=complex),
        eyy=np.eye(basis.Y.size, dtype=complex),
        ezz=np.eye(basis.Z.size, dtype=complex),
        mzz=eye_w,
        exy=np.zeros((basis.X.size, basis.Y.size), dtype=complex),
    )
    with pytest.raises(ValueError, match="exy"):
        build_fg(ops, basis, 500.0)


@pytest.mark.unit
def test_direct_rule_eps_zz_matches_hand_sum():
    """eps_zz for a two-strip layout against an explicit Fourier-coefficient formula."""
    layout = (
        _partial_block_structure(eps=5.0, x_frac=(0.0, 1.0), y_frac=(0.0, 0.5))
        .segments()[0]
        .cross_section
    )
    basis = ModeBasis(A, B, 2, 2)
    ops = build_eps_operators(layout, basis, "direct")
    # eps(y) = 5 on [0, b/2), 1 on (b/2, b]; matrix element between s_m s_n and s_m' s_n':
    # delta_mm' * [ 5*I_{nn'}(0,b/2) + 1*I_{nn'}(b/2,b) ].
    Sy_lo = sin_overlap(2, B, 0.0, 0.5 * B)
    Sy_hi = sin_overlap(2, B, 0.5 * B, B)
    expected_y = 5.0 * Sy_lo + 1.0 * Sy_hi
    ref = np.kron(np.eye(2), expected_y)
    assert np.allclose(ops.ezz, ref, atol=1e-13)
