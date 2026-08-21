"""ASR (adaptive spatial resolution) tests: map, operators, leads, convergence.

Measured context for the convergence assertions (eps = 80 block, S21 of TE10 at
10 GHz, reference = ASR eta=0.3 at N = 72, self-consistent to 5e-5):

    N        16       24       32       56
    plain   8.8e-2   5.0e-2   2.4e-2   5.2e-3     (plain li, ~N^-1 pre-asymptotic)
    ASR     1.2e-2   2.6e-3   7.4e-4   7.1e-5     (eta = 0.3)

i.e. ASR at N = 24 already beats plain li at N = 80 (error 1.9e-3).
"""

import warnings

import numpy as np
import pytest

from sceptre import AsrConfig, Box, Structure, Solver, Waveguide
from sceptre.asr import AsrMap1D, build_maps, gauss_gram
from sceptre.fourier import cos_overlap, sin_overlap
from sceptre.solver import C0

A, B = 0.02286, 0.01016  # WR-90
EDGE = 0.45 * B


def _block_structure(eps):
    return Structure(Waveguide(A, B), [Box(0.0, A, 0.0, EDGE, 0.0, 0.008, eps)])


def _s21(struct, n, eta=None, m=1, f=10e9):
    asr = AsrConfig(eta=eta) if eta is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = Solver(struct, M=m, N=n, asr=asr)
    return solver.smatrix(f).coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0))


# ---------------------------------------------------------------- map & grams


@pytest.mark.unit
def test_map_fixes_edges_and_compresses():
    amap = AsrMap1D(B, [EDGE], eta=0.25)
    assert amap.x(np.array([0.0, EDGE, B])) == pytest.approx([0.0, EDGE, B])
    assert amap.dx(np.array([0.0, EDGE, B])) == pytest.approx([0.25, 0.25, 0.25])
    u = np.linspace(0, B, 2001)
    x = amap.x(u)
    assert np.all(np.diff(x) > 0)  # bijective
    assert np.max(amap.dx(u)) < 2.0  # bounded stretch


@pytest.mark.unit
def test_identity_map_when_no_interior_edges():
    amap = AsrMap1D(A, [], eta=0.2)
    assert amap.identity
    u = np.linspace(0, A, 7)
    assert np.allclose(amap.x(u), u)
    assert np.allclose(amap.dx(u), 1.0)


@pytest.mark.unit
def test_gauss_gram_matches_analytic_overlaps():
    s = gauss_gram("sin", 6, B, [(0.0, EDGE, 1.0), (EDGE, B, 1.0)])
    assert np.allclose(s, sin_overlap(6, B, 0, B), atol=1e-13)
    c = gauss_gram("cos", 6, B, [(0.0, 0.3 * B, 1.0)])
    assert np.allclose(c, cos_overlap(6, B, 0, 0.3 * B), atol=1e-13)


# ------------------------------------------------------------- exactness


def test_identity_eta_reproduces_plain_solver():
    struct = _block_structure(9.0)
    plain = _s21(struct, 10)
    ident = _s21(struct, 10, eta=1.0)
    assert abs(plain - ident) < 1e-10


def test_asr_lead_matches_analytic_dispersion():
    struct = _block_structure(9.0)
    solver = Solver(struct, M=1, N=16, asr=AsrConfig())
    k0 = 2 * np.pi * 10e9 / C0
    lead = solver._lead(k0)
    assert lead.labels[0] == ("TE", 1, 0)
    beta_ref = np.sqrt(k0**2 - (np.pi / A) ** 2)
    assert abs(lead.beta[0] - beta_ref) / beta_ref < 1e-8
    gram = lead.W.T @ lead.V
    assert np.max(np.abs(gram - np.eye(len(gram)))) < 1e-9


def test_asr_slab_multimode_vs_analytic():
    """Full-width slab at 18 GHz: degenerate TE11/TM11 pairs propagate, so this
    exercises the cluster alignment; S21 must match the analytic two-interface
    formula to truncation accuracy."""
    L, eps2 = 0.0061, 4.0
    struct = Structure(Waveguide(A, B), [Box(0, A, 0, B, 0.0, L, eps2)])
    # force a nontrivial map with a fake interior edge structure? No: a uniform
    # slab has no interior edges, both maps degrade to identity and ASR must
    # reproduce the analytic path bit-for-bit at the physics level.
    res = Solver(struct, M=3, N=3, asr=AsrConfig()).smatrix(18e9)
    k0 = 2 * np.pi * 18e9 / C0
    for kind, m, n in [("TE", 1, 0), ("TE", 1, 1), ("TM", 1, 1)]:
        kc2 = (m * np.pi / A) ** 2 + (n * np.pi / B) ** 2
        b1 = np.sqrt(k0**2 - kc2 + 0j)
        b2 = np.sqrt(eps2 * k0**2 - kc2 + 0j)
        z1 = b1 / k0 if kind == "TE" else k0 / b1
        z2 = b2 / k0 if kind == "TE" else eps2 * k0 / b2
        r = (z1 - z2) / (z1 + z2)
        X = np.exp(1j * b2 * L)
        s21_ref = (1 - r**2) * X / (1 - r**2 * X**2)
        got = res.coeff(2, (kind, m, n), 1, (kind, m, n))
        assert abs(got - s21_ref) < 1e-8, (kind, m, n)


# ------------------------------------------------------- the point of ASR


@pytest.fixture(scope="module")
def ceramic():
    struct = _block_structure(80.0)
    ref = _s21(struct, 64, eta=0.3)
    return struct, ref


def test_asr_beats_plain_li_at_high_contrast(ceramic):
    struct, ref = ceramic
    err_plain_56 = abs(_s21(struct, 56) - ref)
    err_asr_24 = abs(_s21(struct, 24, eta=0.3) - ref)
    err_asr_40 = abs(_s21(struct, 40, eta=0.3) - ref)
    assert err_asr_24 < err_plain_56  # 3.3x smaller basis, better accuracy
    assert err_asr_40 < err_plain_56 / 5.0


def test_asr_structural_identities_high_contrast(ceramic):
    """Unitarity and reciprocity must survive the metric operators."""
    struct, _ = ceramic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Solver(struct, M=1, N=24, asr=AsrConfig()).smatrix(10e9)
    sp = res.port_smatrix()
    assert np.max(np.abs(sp.conj().T @ sp - np.eye(len(sp)))) < 1e-8
    s = res.smatrix.full()
    assert np.max(np.abs(s - s.T)) < 1e-8 * np.max(np.abs(s))


# ------------------------------------------------------------ recommendation


@pytest.mark.unit
def test_high_contrast_without_asr_warns():
    with pytest.warns(UserWarning, match="contrast"):
        Solver(_block_structure(80.0), M=1, N=8)


@pytest.mark.unit
def test_no_warning_with_asr_or_low_contrast():
    with warnings.catch_warnings():
        # only the contrast recommendation is under test; at very small N the
        # (legitimate) ASR edge-thinning warning may fire independently
        warnings.filterwarnings("error", message=".*contrast.*")
        Solver(_block_structure(80.0), M=1, N=8, asr=AsrConfig())
        Solver(_block_structure(9.0), M=1, N=8)


@pytest.mark.unit
def test_asr_requires_li():
    with pytest.raises(ValueError, match="li"):
        Solver(_block_structure(9.0), M=1, N=8, factorization="direct", asr=AsrConfig())


@pytest.mark.unit
def test_build_maps_collects_edges():
    xmap, ymap = build_maps(_block_structure(9.0), eta=0.3)
    assert xmap.identity  # full-width block: no interior x-edges
    assert not ymap.identity
    assert np.any(np.isclose(ymap.breaks, EDGE, rtol=1e-12))


# ------------------------------------------- dense-edge (staircase) robustness
#
# A staircased curved shape puts interior edges every ~pixel; the map then
# oscillates eta <-> 2-eta between every pair, pushing metric content beyond the
# basis bandwidth (aliasing) and destabilizing the numerical lead-mode stage
# (observed: port-column energies up to 3.5 on a staircased eps=80 disk).
# Edges closer than the basis can resolve must not become compression points.


def _staircase_disk(
    a: float, r: float, h: float, eps: complex, k: int, cy_off: float = 0.0
):
    """K x-strip staircase of a cylinder (axis || z) in an a x a guide; cy_off
    shifts the centre in y (a wall-touching disk creates near-coincident edges,
    the worst case for the ASR map)."""
    boxes = []
    cy = a / 2 + cy_off
    xs = np.linspace(-r, r, k + 1)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        half = r * r - (0.5 * (x1 + x2)) ** 2
        if half <= 0:
            continue
        half = np.sqrt(half)
        boxes.append(
            Box(
                a / 2 + x1,
                a / 2 + x2,
                max(cy - half, 0.0),
                min(cy + half, a),
                0.0,
                h,
                eps,
            )
        )
    return Structure(Waveguide(a, a), boxes)


@pytest.mark.unit
def test_map_min_interval_thins_dense_edges():
    dense = list(np.arange(0.05, 1.0, 0.02))
    amap = AsrMap1D(1.0, dense, eta=0.3, min_interval=0.1)
    gaps = np.diff(amap.breaks)
    assert gaps.min() >= 0.1 - 1e-12
    assert amap.breaks[0] == 0.0 and amap.breaks[-1] == 1.0
    # back-compat: min_interval=0 keeps every edge
    full = AsrMap1D(1.0, dense, eta=0.3)
    assert len(full.breaks) == len(dense) + 2


@pytest.mark.unit
def test_map_all_edges_dropped_degrades_to_identity():
    amap = AsrMap1D(1.0, [0.3, 0.4, 0.5], eta=0.3, min_interval=0.9)
    assert amap.identity
    assert amap.dropped == 3
    u = np.linspace(0.0, 1.0, 11)
    assert np.allclose(amap.x(u), u)
    assert np.allclose(amap.dx(u), 1.0)


@pytest.mark.unit
def test_build_maps_thins_and_warns_on_staircase():
    struct = _staircase_disk(0.032, 15e-3, 5e-3, 80.0, 32)
    with pytest.warns(UserWarning, match="thinned"):
        xmap, ymap = build_maps(struct, eta=0.3, min_x=0.032 / 8, min_y=0.032 / 8)
    assert np.diff(xmap.breaks).min() >= 0.032 / 8 - 1e-12
    assert np.diff(ymap.breaks).min() >= 0.032 / 8 - 1e-12
    # sparse-edge structures are untouched and silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, ymap_block = build_maps(
            _block_structure(80.0), eta=0.3, min_x=2 * A, min_y=B / 12
        )
    assert np.any(np.isclose(ymap_block.breaks, EDGE, rtol=1e-12))


def test_asr_staircase_disk_stays_unitary():
    """Regression: eps=80 staircased wall-touching disk, the exact config that
    produced port-column energies of 2.5-3.5 at 5.700 GHz before edge thinning."""
    struct = _staircase_disk(0.032, 15e-3, 5e-3, 80.0, 32, cy_off=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Solver(struct, M=16, N=16, asr=AsrConfig(eta=0.3)).smatrix(5.700e9)
    sp = res.port_smatrix()
    energies = np.sum(np.abs(sp) ** 2, axis=0)
    assert np.max(np.abs(energies - 1.0)) < 1e-3, energies
