"""KFJ subpixel-smoothing factorization: structural correctness only.

KFJ is REFUTED for high-contrast accuracy (LEDGER.md H1) and ships as a
comparison / low-contrast tool per explicit user decision — these tests
verify the assembly is energy-conserving and reduces to Li in the
grid-aligned limit, NOT that it is accurate at eps=80."""

import numpy as np
import pytest

from sceptre.basis import ModeBasis
from sceptre.fourier import build_eps_operators
from sceptre.geometry import Structure, Waveguide
from sceptre.kfj import KfjConfig, build_kfj_operators, kfj_cells
from sceptre.modes import lead_modes
from sceptre.shapes import Cylinder, Shape
from sceptre.slicesolver import solve_slice
from sceptre.smatrix import cascade, interface_smatrix, propagation_smatrix

A_WG = 0.032
WG = Waveguide(A_WG, A_WG)
CX, CY, R, H = A_WG / 2, A_WG / 2 + 1e-3, 15e-3, 5e-3


class _AlignedRect(Shape):
    """Axis-aligned rectangle whose edges coincide with the KFJ grid."""

    def __init__(self, x1, x2, y1, y2, eps, z1=0.0, z2=H):
        super().__init__(z1, z2, eps)
        self._b = (x1, x2, y1, y2)

    def level_set(self, x, y):
        x1, x2, y1, y2 = self._b
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.maximum.reduce([x1 - x, x - x2, y1 - y, y - y2])

    @property
    def bbox(self):
        return self._b


@pytest.mark.unit
def test_kfj_docstring_states_the_refutation():
    import sceptre.kfj as kfj_mod

    doc = (kfj_mod.__doc__ or "").lower()
    assert "refuted" in doc and "high-contrast" in doc.replace(
        "high contrast", "high-contrast"
    )


@pytest.mark.unit
def test_kfj_fill_fractions_recover_disk_area():
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 + 0j)
    xe, ye, exx, eyy, exy, ezz = kfj_cells((cyl,), WG, KfjConfig(cells=96))
    areas = np.outer(np.diff(xe), np.diff(ye))
    fills = (ezz.real - 1.0) / (80.0 - 1.0)
    disk_area = float(np.sum(fills * areas))
    assert disk_area == pytest.approx(np.pi * R * R, rel=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("background", [1.0 + 0j, 2.25 + 0j])
def test_kfj_grid_aligned_rectangle_equals_li(background):
    """Grid-aligned rectangle: KFJ must reduce exactly to Li — INCLUDING in a
    non-vacuum host (regression: KFJ once hardcoded the vacuum background)."""
    n_cells = 32
    rect = _AlignedRect(0.25 * A_WG, 0.75 * A_WG, 0.25 * A_WG, 0.75 * A_WG, 6.0 + 0j)
    struct = Structure(WG, shapes=[rect], background=background)
    layout = struct.segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 5, 5)
    ops_kfj = build_kfj_operators(
        (rect,), layout, WG, basis, KfjConfig(cells=n_cells), background=background
    )
    ops_li = build_eps_operators(layout, basis, "li")
    assert np.allclose(ops_kfj.exx, ops_li.exx, atol=1e-10)
    assert np.allclose(ops_kfj.eyy, ops_li.eyy, atol=1e-10)
    assert np.allclose(ops_kfj.ezz, ops_li.ezz, atol=1e-10)
    assert ops_kfj.exy is not None
    assert np.max(np.abs(ops_kfj.exy)) < 1e-12  # no boundary ring, no coupling


@pytest.mark.unit
def test_kfj_rejects_cell_shared_by_two_shapes():
    c1 = Cylinder(0.014, 0.016, 4e-3, 0.0, H, 10.0 + 0j)
    c2 = Cylinder(0.0181, 0.016, 4e-3, 0.0, H, 12.0 + 0j)  # overlapping
    with pytest.raises(ValueError, match="KFJ grid"):
        kfj_cells((c1, c2), WG, KfjConfig(cells=32))


@pytest.mark.integration
def test_kfj_disk_unitary_and_reciprocal():
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 + 0j)
    layout = Structure(WG, shapes=[cyl]).segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 12, 12)
    k0 = 2 * np.pi * 5.44e9 / 299792458.0
    ops = build_kfj_operators((cyl,), layout, WG, basis, KfjConfig())
    modes = solve_slice(layout, basis, k0, ops=ops)
    lead = lead_modes(basis, k0, 1.0)
    s = cascade(
        [
            interface_smatrix(lead.W, lead.V, modes.W, modes.V),
            propagation_smatrix(modes.beta, H),
            interface_smatrix(modes.W, modes.V, lead.W, lead.V),
        ]
    )
    idx = np.array([lead.mode_index("TE", 1, 0), lead.mode_index("TE", 0, 1)])
    ix = np.ix_(idx, idx)
    s4 = np.block([[s.s11[ix], s.s12[ix]], [s.s21[ix], s.s22[ix]]])
    cols = np.sum(np.abs(s4) ** 2, axis=0)
    assert np.max(np.abs(cols - 1.0)) < 1e-10
    full = s.full()
    assert np.max(np.abs(full - full.T)) < 1e-12 * np.max(np.abs(full))
