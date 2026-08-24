"""Validation of directly-constructed CrossSections.

`CrossSection` used to be reached only through `Structure`, which guarantees
its invariants by construction. It is now also a documented public entry point
for explicitly graded permittivity maps (docs/inverse-design.md), so the same
invariants have to be enforced where a caller can violate them. Every case here
previously produced a silent wrong answer rather than an error.
"""

import numpy as np
import pytest

from sceptre.geometry import CrossSection, Segment, Structure, Waveguide
from sceptre.solver import Solver

A, B = 0.032, 0.032


def _edges(n=8):
    return np.linspace(0.0, A, n + 1), np.linspace(0.0, B, n + 1)


class _GridStructure:
    """The adapter documented in docs/inverse-design.md."""

    def __init__(self, waveguide, segments, background=1.0 + 0.0j):
        self.waveguide = waveguide
        self.background = background
        self.boxes = ()
        self.shapes = ()
        self._segments = list(segments)

    def segments(self):
        return list(self._segments)


def test_cross_section_rejects_eps_cells_shape_mismatch():
    xe, ye = _edges()
    with pytest.raises(ValueError, match="eps_cells"):
        CrossSection(xe, ye, np.full((7, 8), 4.0 + 0j))


def test_cross_section_rejects_non_increasing_edges():
    xe, ye = _edges()
    bad = xe.copy()
    bad[3], bad[4] = bad[4], bad[3]
    with pytest.raises(ValueError, match="increasing"):
        CrossSection(bad, ye, np.full((8, 8), 4.0 + 0j))


def test_cross_section_rejects_zero_permittivity():
    """Structure.background and Shape.eps both reject zero; a cell must too."""
    xe, ye = _edges()
    eps = np.full((8, 8), 4.0 + 0j)
    eps[2, 3] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        CrossSection(xe, ye, eps)


def test_cross_section_normalises_real_eps_to_complex():
    xe, ye = _edges()
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0))
    assert cs.eps_cells.dtype == complex
    assert cs.uniform_eps == 4.0 + 0j


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cross_section_rejects_non_finite_permittivity(bad):
    """NaN compares False against everything, so it passed the ordering and
    zero tests; a UNIFORM layout of non-finite eps then takes the analytic fast
    path (no scipy check_finite) and returned S = NaN with no error at all."""
    xe, ye = _edges()
    eps = np.full((8, 8), 4.0 + 0j)
    eps[1, 2] = bad
    with pytest.raises(ValueError, match="finite"):
        CrossSection(xe, ye, eps)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_cross_section_rejects_non_finite_edges(bad):
    xe, ye = _edges()
    broken = xe.copy()
    broken[3] = bad
    with pytest.raises(ValueError, match="finite"):
        CrossSection(broken, ye, np.full((8, 8), 4.0 + 0j))
    broken = ye.copy()
    broken[3] = bad
    with pytest.raises(ValueError, match="finite"):
        CrossSection(xe, broken, np.full((8, 8), 4.0 + 0j))


def test_cross_section_rejects_complex_edges():
    """numpy would only raise a default-suppressed ComplexWarning and drop the
    imaginary part."""
    xe, ye = _edges()
    with pytest.raises(ValueError, match="real"):
        CrossSection(xe.astype(complex), ye, np.full((8, 8), 4.0 + 0j))


def test_cross_section_copies_its_inputs():
    """np.asarray returns the CALLER's array when the dtype already matches, so
    a validated layout could be corrupted afterwards by ordinary in-place
    mutation -- silently changing its ops-cache key too."""
    xe, ye = _edges()
    eps = np.full((8, 8), 4.0 + 0j)
    cs = CrossSection(xe, ye, eps)
    key_before = cs.key()
    eps[0, 0] = 0.0  # would violate the nonzero invariant
    xe[0] = -1.0  # would violate both spanning and monotonicity
    assert cs.eps_cells[0, 0] == 4.0 + 0j
    assert cs.x_edges[0] == 0.0
    assert cs.key() == key_before


def test_cross_section_arrays_are_read_only():
    xe, ye = _edges()
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0 + 0j))
    with pytest.raises(ValueError):
        cs.eps_cells[0, 0] = 0.0
    with pytest.raises(ValueError):
        cs.x_edges[0] = -1.0


def test_error_messages_localise_the_offending_entry():
    """A 96x96 optimiser-searched grid is not diffable by hand."""
    xe, ye = _edges()
    eps = np.full((8, 8), 4.0 + 0j)
    eps[5, 6] = 0.0
    with pytest.raises(ValueError, match=r"eps_cells\[5, 6\]"):
        CrossSection(xe, ye, eps)


def test_box_structure_with_non_finite_eps_is_caught_by_the_same_funnel():
    """Structure._layout routes every box through CrossSection, so the box path
    inherits the finiteness guarantee without a second check."""
    from sceptre.geometry import Box

    with pytest.raises(ValueError, match="finite"):
        Structure(Waveguide(A, B), [Box(0, A, 0, B, 0.0, 0.01, np.inf)]).segments()


def test_cross_section_accepts_a_valid_graded_map():
    xe, ye = _edges(16)
    xm = 0.5 * (xe[:-1] + xe[1:])
    ym = 0.5 * (ye[:-1] + ye[1:])
    x, y = np.meshgrid(xm, ym, indexing="ij")
    eps = 1.0 + 6.0 * (
        0.5 + 0.4 * np.cos(2 * np.pi * x / A) * np.cos(2 * np.pi * y / B)
    )
    cs = CrossSection(xe, ye, eps.astype(complex))
    assert cs.eps_cells.shape == (16, 16)
    assert not cs.is_uniform


def test_solver_rejects_cross_section_not_spanning_the_guide():
    """A map covering half the guide silently left the rest undefined."""
    xe = np.linspace(0.0, A / 2, 9)
    ye = np.linspace(0.0, B, 9)
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0 + 0j))
    struct = _GridStructure(Waveguide(A, B), [Segment(0.0, 0.01, cs)])
    with pytest.raises(ValueError, match="span"):
        Solver(struct, M=4, N=4)


def test_solver_rejects_cross_section_not_starting_at_the_wall():
    xe = np.linspace(0.001, A + 0.001, 9)
    ye = np.linspace(0.0, B, 9)
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0 + 0j))
    struct = _GridStructure(Waveguide(A, B), [Segment(0.0, 0.01, cs)])
    with pytest.raises(ValueError, match="span"):
        Solver(struct, M=4, N=4)


def test_solver_rejects_cross_section_not_spanning_the_guide_in_y():
    """Mirror of the x cases: the y branches of the span check were untested,
    so an x/y or a/b typo in them would not have been caught."""
    xe = np.linspace(0.0, A, 9)
    ye = np.linspace(0.0, B / 2, 9)
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0 + 0j))
    struct = _GridStructure(Waveguide(A, B), [Segment(0.0, 0.01, cs)])
    with pytest.raises(ValueError, match="span"):
        Solver(struct, M=4, N=4)


def test_solver_rejects_cross_section_not_starting_at_the_wall_in_y():
    xe = np.linspace(0.0, A, 9)
    ye = np.linspace(0.001, B + 0.001, 9)
    cs = CrossSection(xe, ye, np.full((8, 8), 4.0 + 0j))
    struct = _GridStructure(Waveguide(A, B), [Segment(0.0, 0.01, cs)])
    with pytest.raises(ValueError, match="span"):
        Solver(struct, M=4, N=4)


def test_solver_accepts_a_spanning_graded_structure():
    xe, ye = _edges(16)
    cs = CrossSection(xe, ye, np.full((16, 16), 4.0 + 0j))
    struct = _GridStructure(Waveguide(A, B), [Segment(0.0, 0.01, cs)])
    res = Solver(struct, M=4, N=4).smatrix(5.85e9)
    assert np.isfinite(res.smatrix.s11).all()


def test_box_structures_still_build():
    """The internal path seeds [0, a] x [0, b], so it must be unaffected."""
    from sceptre.geometry import Box

    struct = Structure(Waveguide(A, B), [Box(0.004, 0.02, 0.004, 0.02, 0.0, 0.01, 9.0)])
    for seg in struct.segments():
        cs = seg.cross_section
        assert cs.x_edges[0] == 0.0 and cs.x_edges[-1] == pytest.approx(A)
        assert cs.eps_cells.shape == (len(cs.x_edges) - 1, len(cs.y_edges) - 1)
