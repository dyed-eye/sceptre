"""NVF factorization: sharp eps, projection-resolved rules along a normal field.

Structural contracts tested here (Task 3): the w=1 forced-x limit reproduces
Li's lamellar inverse rule; the symmetrized operator keeps S = S^T and
unitarity at machine precision; window validation guards the level-set
singularity; overlapping windows are rejected.  Accuracy (the golden test)
lives with the Solver plumbing in this file too but is added by Task 5.
"""

import numpy as np
import pytest

from sceptre.basis import ModeBasis
from sceptre.fourier import build_eps_operators
from sceptre.geometry import Structure, Waveguide
from sceptre.modes import lead_modes
from sceptre.nvf import NvfConfig, build_nvf_operators
from sceptre.shapes import Cylinder, Shape
from sceptre.slicesolver import solve_slice
from sceptre.smatrix import cascade, interface_smatrix, propagation_smatrix

A_WG = 0.032
WG = Waveguide(A_WG, A_WG)
CX, CY, R, H = A_WG / 2, A_WG / 2 + 1e-3, 15e-3, 5e-3


class _XSlab(Shape):
    """x-lamellar slab: level set depends on x only; normal = ±x-hat."""

    def __init__(self, x1, x2, eps, z1=0.0, z2=5e-3):
        super().__init__(z1, z2, eps)
        self._x1, self._x2 = x1, x2

    def level_set(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.maximum(self._x1 - x, x - self._x2) * np.ones_like(
            np.asarray(y, dtype=float)
        )

    def normal(self, x, y):
        # exact x-hat everywhere (the FD gradient degenerates on the ridge)
        shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        return np.ones(shape), np.zeros(shape)

    @property
    def bbox(self):
        return (self._x1, self._x2, 0.0, A_WG)


class _YSlab(Shape):
    """y-lamellar slab: level set depends on y only; normal = ±y-hat."""

    def __init__(self, y1, y2, eps, z1=0.0, z2=5e-3):
        super().__init__(z1, z2, eps)
        self._y1, self._y2 = y1, y2

    def level_set(self, x, y):
        y = np.asarray(y, dtype=float)
        return np.maximum(self._y1 - y, y - self._y2) * np.ones_like(
            np.asarray(x, dtype=float)
        )

    def normal(self, x, y):
        shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        return np.zeros(shape), np.ones(shape)

    @property
    def bbox(self):
        return (0.0, A_WG, self._y1, self._y2)


def _port_s4(layout, shapes, basis, k0, config):
    """3-part cascade (lead | NVF slice | lead) -> 4x4 propagating-port block."""
    ops = build_nvf_operators(shapes, layout, WG, basis, config)
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
    full = np.block([[s.s11[ix], s.s12[ix]], [s.s21[ix], s.s22[ix]]])
    return full, s


@pytest.mark.unit
def test_forced_x_limit_reproduces_li_lamellar():
    """w = 1 (window=inf) + x-only normals on a y-uniform lamellar layout
    must give exactly Li's rules: exx = inverse-x, eyy = direct."""
    slab = _XSlab(0.3 * A_WG, 0.7 * A_WG, 6.0 + 0j)
    struct = Structure(WG, shapes=[slab])
    layout = struct.segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 5, 4)
    ops_nvf = build_nvf_operators((slab,), layout, WG, basis, NvfConfig(window=np.inf))
    ops_li = build_eps_operators(layout, basis, "li")
    assert np.allclose(ops_nvf.exx, ops_li.exx, atol=1e-10)
    assert np.allclose(ops_nvf.ezz, ops_li.ezz, atol=1e-12)
    # direct rule along y for a y-uniform layout == eps-weighted identity mix
    ops_direct = build_eps_operators(layout, basis, "direct")
    assert np.allclose(ops_nvf.eyy, ops_direct.eyy, atol=1e-10)
    assert ops_nvf.exy is not None
    assert np.max(np.abs(ops_nvf.exy)) < 1e-12  # x-normals never couple X to Y


@pytest.mark.unit
def test_forced_y_limit_reproduces_li_lamellar():
    """Mirror of the x-limit: y-normals -> eyy inverse rule, exx direct."""
    slab = _YSlab(0.3 * A_WG, 0.7 * A_WG, 6.0 + 0j)
    struct = Structure(WG, shapes=[slab])
    layout = struct.segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 4, 5)
    ops_nvf = build_nvf_operators((slab,), layout, WG, basis, NvfConfig(window=np.inf))
    ops_li = build_eps_operators(layout, basis, "li")
    ops_direct = build_eps_operators(layout, basis, "direct")
    assert np.allclose(ops_nvf.eyy, ops_li.eyy, atol=1e-10)
    assert np.allclose(ops_nvf.exx, ops_direct.exx, atol=1e-10)
    assert np.allclose(ops_nvf.ezz, ops_li.ezz, atol=1e-12)
    assert ops_nvf.exy is not None
    assert np.max(np.abs(ops_nvf.exy)) < 1e-12


@pytest.mark.integration
def test_nvf_disk_unitary_and_reciprocal():
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 + 0j, k=32)
    layout = Structure(WG, shapes=[cyl]).segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 12, 12)
    k0 = 2 * np.pi * 5.44e9 / 299792458.0
    s4, s_full = _port_s4(layout, (cyl,), basis, k0, NvfConfig())
    cols = np.sum(np.abs(s4) ** 2, axis=0)
    assert np.max(np.abs(cols - 1.0)) < 1e-10
    full = s_full.full()
    assert np.max(np.abs(full - full.T)) < 1e-12 * np.max(np.abs(full))


@pytest.mark.unit
def test_window_larger_than_cylinder_radius_raises():
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 + 0j)
    layout = Structure(WG, shapes=[cyl]).segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 4, 4)
    with pytest.raises(ValueError, match="window"):
        build_nvf_operators((cyl,), layout, WG, basis, NvfConfig(window=2 * R))


@pytest.mark.unit
def test_overlapping_windows_raise():
    c1 = Cylinder(0.012, 0.016, 4e-3, 0.0, H, 10.0 + 0j)
    c2 = Cylinder(0.020, 0.016, 4e-3, 0.0, H, 10.0 + 0j)  # 8 mm apart, r=4
    struct = Structure(WG, shapes=[c1, c2])
    layout = struct.segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 4, 4)
    with pytest.raises(ValueError, match="overlap"):
        build_nvf_operators((c1, c2), layout, WG, basis, NvfConfig(window=3e-3))


@pytest.mark.integration
def test_golden_disk_line_accuracy_nvf_vs_li():
    """THE acceptance test: eps=80 benchmark disk, factorization='nvf',
    N=20, one solve per frequency — pol-B line within the CALIBRATED ±5 MHz
    of COMSOL 5.4357 GHz (calibration measured −0.8 MHz at the default
    window).  Plain Li at the same N has no line within ±50 MHz (its line
    sits ~+110 MHz up and Richardson-extrapolation over an N-ladder is
    required to do what NVF does in one solve).  Line identification:
    nearest strong peak to the reference — NEVER a global argmax (real
    below-window neighbor modes exist at 5.32-5.41)."""
    import warnings

    from sceptre.solver import Solver

    f_ref = 5.4357e9
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 + 0j, k=64)
    struct = Structure(WG, shapes=[cyl])

    def line(solver, f_lo, f_hi, step, floor):
        fs = np.arange(f_lo, f_hi + 1, step)
        ts = []
        for f in fs:
            res = solver.smatrix(f)
            i01 = res.lead.mode_index("TE", 0, 1)
            ts.append(abs(res.smatrix.s21[i01, i01]))
        ts = np.array(ts)
        best = None
        for k in range(1, len(fs) - 1):
            if ts[k] > ts[k - 1] and ts[k] >= ts[k + 1] and ts[k] >= floor:
                den = ts[k - 1] - 2 * ts[k] + ts[k + 1]
                d = 0.5 * (ts[k - 1] - ts[k + 1]) / den if den else 0.0
                fpk = fs[k] + d * step
                if best is None or abs(fpk - f_ref) < abs(best - f_ref):
                    best = fpk
        return best

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s_nvf = Solver(struct, 20, 20, factorization="nvf", symmetry="x")
        line_nvf = line(s_nvf, f_ref - 20e6, f_ref + 20e6, 2e6, floor=0.5)
        assert line_nvf is not None
        assert abs(line_nvf - f_ref) < 5e6  # calibrated tolerance

        s_li = Solver(struct, 20, 20, symmetry="x")
        line_li = line(s_li, f_ref - 50e6, f_ref + 50e6, 4e6, floor=0.5)
        assert line_li is None  # Li's line is far above at this truncation


@pytest.mark.integration
def test_nvf_two_separated_cylinders_success_path():
    """Multi-shape accumulation path: two well-separated cylinders solve,
    conserve energy, stay reciprocal, and BOTH contribute to S (dropping one
    changes the result far beyond tolerance)."""
    import warnings

    from sceptre.solver import Solver

    c1 = Cylinder(0.010, 0.016, 3e-3, 0.0, H, 10.0 + 0j, k=24)
    c2 = Cylinder(0.022, 0.016, 3e-3, 0.0, H, 10.0 + 0j, k=24)
    f0 = 5.6e9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s_pair = Solver(
            Structure(WG, shapes=[c1, c2]), 10, 10, factorization="nvf"
        ).smatrix(f0)
        s_single = Solver(
            Structure(WG, shapes=[c1]), 10, 10, factorization="nvf"
        ).smatrix(f0)
    sp = s_pair.port_smatrix()
    gram = sp.conj().T @ sp
    assert np.max(np.abs(gram - np.eye(sp.shape[0]))) < 1e-10
    full = s_pair.smatrix.full()
    assert np.max(np.abs(full - full.T)) < 1e-12 * np.max(np.abs(full))
    # the second shape's contribution is present (accumulation, not overwrite)
    d = np.max(np.abs(s_pair.port_smatrix() - s_single.port_smatrix()))
    assert d > 1e-3


@pytest.mark.unit
def test_lossy_eps_assembly_keeps_complex_dtype():
    cyl = Cylinder(CX, CY, R, 0.0, H, 80.0 * (1 + 0.007j))
    layout = Structure(WG, shapes=[cyl]).segments()[0].cross_section
    basis = ModeBasis(A_WG, A_WG, 6, 6)
    ops = build_nvf_operators((cyl,), layout, WG, basis, NvfConfig())
    assert np.max(np.abs(np.imag(ops.ezz))) > 0
    assert np.max(np.abs(np.imag(ops.exx))) > 0
