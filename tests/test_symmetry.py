"""x-mirror symmetry sectorization (Solver(symmetry="x")).

For a structure mirror-symmetric about x = a/2, the transverse problem block-
diagonalizes by parity of the x-index m (derivatives preserve m; the eps
operators of a symmetric layout never couple odd and even m).  Solving the two
half-size sectors independently must reproduce the full solver's S-matrix to
numerical precision -- that is the contract tested here.
"""

import numpy as np
import pytest

from sceptre.asr import AsrConfig
from sceptre.basis import ModeBasis
from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import Solver
from sceptre.symmetry import lead_columns, require_x_symmetric, x_sectors

A, B = 0.02286, 0.01016  # WR-90
F0 = 18.0e9  # several propagating modes


def symmetric_structure() -> Structure:
    """Two z-segments, both x-symmetric, asymmetric in y (full TE/TM coupling)."""
    centered = Box(0.3 * A, 0.7 * A, 0.0, 0.55 * B, 0.0, 0.004, 6.0)
    pair_l = Box(0.15 * A, 0.35 * A, 0.2 * B, 0.9 * B, 0.004, 0.007, 3.0)
    pair_r = Box(0.65 * A, 0.85 * A, 0.2 * B, 0.9 * B, 0.004, 0.007, 3.0)
    return Structure(Waveguide(A, B), [centered, pair_l, pair_r])


@pytest.fixture(scope="module")
def solvers():
    struct = symmetric_structure()
    return (
        Solver(struct, M=6, N=6),
        Solver(struct, M=6, N=6, symmetry="x"),
    )


def test_x_sectors_partition_every_space():
    basis = ModeBasis(A, B, 5, 4)
    odd, even = x_sectors(basis)
    for space in "XYZW":
        size = getattr(basis, space).size
        both = np.concatenate([getattr(odd, space), getattr(even, space)])
        assert np.array_equal(np.sort(both), np.arange(size))
    both_t = np.concatenate([odd.t, even.t])
    assert np.array_equal(np.sort(both_t), np.arange(basis.size_t))


def test_lead_columns_match_sector_row_counts():
    basis = ModeBasis(A, B, 5, 4)
    from sceptre.modes import lead_modes

    lead = lead_modes(basis, 2000.0)
    for sec in x_sectors(basis):
        cols = lead_columns(lead.labels, sec)
        assert len(cols) == len(sec.t)


def test_te10_and_te01_in_opposite_sectors():
    basis = ModeBasis(A, B, 4, 4)
    from sceptre.modes import lead_modes

    lead = lead_modes(basis, 2000.0)
    odd, even = x_sectors(basis)
    i10 = lead.mode_index("TE", 1, 0)
    i01 = lead.mode_index("TE", 0, 1)
    assert i10 in lead_columns(lead.labels, odd)
    assert i01 in lead_columns(lead.labels, even)


def test_sectored_matches_full_smatrix(solvers):
    plain, sectored = solvers
    s_full = plain.smatrix(F0).smatrix.full()
    s_sect = sectored.smatrix(F0).smatrix.full()
    scale = np.max(np.abs(s_full))
    assert np.max(np.abs(s_full - s_sect)) < 1e-9 * scale


def test_sectored_matches_full_direct_factorization():
    struct = symmetric_structure()
    s_full = Solver(struct, M=5, N=5, factorization="direct").smatrix(F0)
    s_sect = Solver(struct, M=5, N=5, factorization="direct", symmetry="x").smatrix(F0)
    a, b = s_full.smatrix.full(), s_sect.smatrix.full()
    assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(a))


def test_sectored_unitary_on_propagating_block(solvers):
    _, sectored = solvers
    sp = sectored.smatrix(F0).port_smatrix()
    gram = sp.conj().T @ sp
    assert np.max(np.abs(gram - np.eye(sp.shape[0]))) < 1e-10


def test_sectored_complex_frequency_matches(solvers):
    plain, sectored = solvers
    fc = F0 + 2.0e8j
    idx = plain.smatrix(F0).propagating_indices()
    d_full = plain.det_port_s(fc, idx)
    d_sect = sectored.det_port_s(fc, idx)
    assert abs(d_full - d_sect) < 1e-8 * abs(d_full)


def test_asymmetric_structure_raises():
    offcenter = Box(0.1 * A, 0.5 * A, 0.0, 0.5 * B, 0.0, 0.004, 4.0)
    struct = Structure(Waveguide(A, B), [offcenter])
    with pytest.raises(ValueError, match="symmetric"):
        Solver(struct, M=4, N=4, symmetry="x")


def test_symmetry_with_asr_raises():
    struct = symmetric_structure()
    with pytest.raises(ValueError, match="ASR"):
        Solver(struct, M=4, N=4, symmetry="x", asr=AsrConfig())


def test_unknown_symmetry_raises():
    struct = symmetric_structure()
    with pytest.raises(ValueError, match="symmetry"):
        Solver(struct, M=4, N=4, symmetry="y")


def test_nearly_symmetric_eps_rejected():
    """A 1e-6-relative eps asymmetry must raise, not silently sector-project."""
    left = Box(0.2 * A, 0.5 * A, 0.0, 0.5 * B, 0.0, 0.004, 4.0)
    right = Box(0.5 * A, 0.8 * A, 0.0, 0.5 * B, 0.0, 0.004, 4.0 * (1 + 1e-6))
    struct = Structure(Waveguide(A, B), [left, right])
    with pytest.raises(ValueError, match="symmetric"):
        Solver(struct, M=4, N=4, symmetry="x")


def test_sectored_m1_empty_z_sector_matches():
    """M=1 leaves one parity class with an empty Z-space (0x0 inv path)."""
    centered = Box(0.25 * A, 0.75 * A, 0.0, 0.6 * B, 0.0, 0.004, 4.0)
    struct = Structure(Waveguide(A, B), [centered])
    s_full = Solver(struct, M=1, N=4).smatrix(F0).smatrix.full()
    s_sect = Solver(struct, M=1, N=4, symmetry="x").smatrix(F0).smatrix.full()
    assert np.max(np.abs(s_full - s_sect)) < 1e-9 * np.max(np.abs(s_full))


def test_sectored_build_fg_slices_exy_blocks():
    """A mirror-commuting exy couples only equal m-parity classes; the sectored
    G must equal the full G restricted to the sector, cross blocks zero."""
    from sceptre.fourier import EpsOperators, build_eps_operators
    from sceptre.slicesolver import build_fg

    basis = ModeBasis(A, B, 5, 4)
    centered = Box(0.25 * A, 0.75 * A, 0.0, 0.6 * B, 0.0, 0.004, 4.0)
    layout = Structure(Waveguide(A, B), [centered]).segments()[0].cross_section
    ops0 = build_eps_operators(layout, basis, "li")

    m_x, _ = basis.X.mn()
    m_y, _ = basis.Y.mn()
    rng = np.random.default_rng(3)
    exy = rng.normal(size=(basis.X.size, basis.Y.size)) * (
        (m_x[:, None] % 2) == (m_y[None, :] % 2)
    )
    ops = EpsOperators(exx=ops0.exx, eyy=ops0.eyy, ezz=ops0.ezz, exy=exy + 0j)
    k0 = 2 * np.pi * 15e9 / 299792458.0
    _F, G = build_fg(ops, basis, k0)
    odd, even = x_sectors(basis)
    for sec in (odd, even):
        _Fs, Gs = build_fg(ops, basis, k0, sec)
        ix = np.ix_(sec.t, sec.t)
        assert np.allclose(Gs, G[ix], atol=1e-13 * np.max(np.abs(G)))
    cross = G[np.ix_(odd.t, even.t)]
    assert np.max(np.abs(cross)) < 1e-13 * np.max(np.abs(G))


def test_require_x_symmetric_accepts_staircase_roundoff():
    """Mirrored staircase edges built by linspace differ by ulps, not physics."""
    xs = np.linspace(-0.3 * A, 0.3 * A, 9) + 0.5 * A
    edges = np.concatenate([[0.0], xs, [A]])
    eps = np.ones(len(edges) - 1, dtype=complex)
    eps[1:-1] = 4.0
    from sceptre.geometry import CrossSection

    layout = CrossSection(edges, np.array([0.0, B]), eps[:, None])
    require_x_symmetric(layout, A)  # must not raise
