"""Two-layer curved geometry: Shape level-set base + Cylinder primitive.

Contract: a Cylinder's exact-interval staircase reproduces the hand-written
disk_boxes recipe; the generic level-set bisection fallback agrees with it;
Structure folds shape staircases into segments and carries the shapes on
each Segment (the routing data for tensor factorizations).
"""

import numpy as np
import pytest

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.shapes import Cylinder, Shape

A_WG = 0.032
CX, CY, R, H = A_WG / 2, A_WG / 2 + 1e-3, 15e-3, 5e-3
EPS = 80.0 + 0j
WG = Waveguide(A_WG, A_WG)


def disk_boxes_reference(r, h, eps, k, cx=CX, cy=CY):
    """The campaign recipe (reports/comsol_parity/runs/parity_common.py)."""
    boxes = []
    xs = np.linspace(-r, r, k + 1)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        xm = 0.5 * (x1 + x2)
        half = r * r - xm * xm
        if half <= 0:
            continue
        half = np.sqrt(half)
        y1 = max(cy - half, 0.0)
        y2 = min(cy + half, A_WG)
        if y2 <= y1:
            continue
        boxes.append(Box(cx + x1, cx + x2, y1, y2, 0.0, h, eps))
    return boxes


@pytest.mark.unit
def test_cylinder_staircase_matches_disk_boxes():
    cyl = Cylinder(CX, CY, R, 0.0, H, EPS)
    got = cyl.staircase(WG, 32)
    ref = disk_boxes_reference(R, H, EPS, 32)
    assert len(got) == len(ref)
    for g, r_ in zip(got, ref):
        for attr in ("x1", "x2", "y1", "y2", "z1", "z2", "eps"):
            assert getattr(g, attr) == pytest.approx(getattr(r_, attr), abs=1e-15)


@pytest.mark.unit
def test_cylinder_level_set_and_normal():
    cyl = Cylinder(CX, CY, R, 0.0, H, EPS)
    th = np.linspace(0.1, 2 * np.pi, 17)
    xb, yb = CX + R * np.cos(th), CY + R * np.sin(th)
    assert np.allclose(cyl.level_set(xb, yb), 0.0, atol=1e-15)
    assert cyl.level_set(np.array([CX]), np.array([CY]))[0] == pytest.approx(-R)
    nx, ny = cyl.normal(xb, yb)
    assert np.allclose(nx, np.cos(th), atol=1e-12)
    assert np.allclose(ny, np.sin(th), atol=1e-12)
    assert np.allclose(np.hypot(nx, ny), 1.0, atol=1e-12)


@pytest.mark.unit
def test_generic_level_set_staircase_agrees_with_exact():
    class GenericCircle(Shape):
        def __init__(self):
            super().__init__(z1=0.0, z2=H, eps=EPS)

        def level_set(self, x, y):
            return np.hypot(np.asarray(x) - CX, np.asarray(y) - CY) - R

        @property
        def bbox(self):
            return (CX - R, CX + R, CY - R, CY + R)

    got = GenericCircle().staircase(WG, 16)
    ref = Cylinder(CX, CY, R, 0.0, H, EPS).staircase(WG, 16)
    assert len(got) == len(ref)
    for g, r_ in zip(got, ref):
        assert g.y1 == pytest.approx(r_.y1, abs=1e-9 * A_WG)
        assert g.y2 == pytest.approx(r_.y2, abs=1e-9 * A_WG)


@pytest.mark.unit
def test_default_finite_difference_normal_matches_analytic():
    """The FD-gradient default normal is the path every user Shape without an
    override hits — compare against Cylinder's analytic radial normal."""

    class GenericCircle(Shape):
        def __init__(self):
            super().__init__(z1=0.0, z2=H, eps=EPS)

        def level_set(self, x, y):
            return np.hypot(np.asarray(x) - CX, np.asarray(y) - CY) - R

        @property
        def bbox(self):
            return (CX - R, CX + R, CY - R, CY + R)

    gen = GenericCircle()
    th = np.linspace(0.05, 2 * np.pi, 23)
    for rho in (R, 0.8 * R, 1.1 * R):  # on and near the boundary
        x, y = CX + rho * np.cos(th), CY + rho * np.sin(th)
        nx, ny = gen.normal(x, y)
        assert np.allclose(nx, np.cos(th), atol=1e-6)
        assert np.allclose(ny, np.sin(th), atol=1e-6)
        assert np.allclose(np.hypot(nx, ny), 1.0, atol=1e-9)
    # the zero-gradient guard must return finite values at the centre
    nx0, ny0 = gen.normal(np.array([CX]), np.array([CY]))
    assert np.isfinite(nx0).all() and np.isfinite(ny0).all()


@pytest.mark.unit
def test_generic_staircase_rejects_nonconvex_strips():
    class TwoLobes(Shape):
        """Two disjoint y-intervals per strip — must raise, never bridge."""

        def __init__(self):
            super().__init__(z1=0.0, z2=H, eps=4.0 + 0j)

        def level_set(self, x, y):
            y = np.asarray(y, dtype=float)
            lobe1 = np.abs(y - 0.010) - 0.002
            lobe2 = np.abs(y - 0.022) - 0.002
            return np.minimum(lobe1, lobe2) * np.ones_like(np.asarray(x, float))

        @property
        def bbox(self):
            return (0.010, 0.022, 0.008, 0.024)

    with pytest.raises(ValueError, match="convex"):
        TwoLobes().staircase(WG, 8)


@pytest.mark.unit
def test_structure_with_shape_equals_structure_with_boxes():
    s_shape = Structure(WG, shapes=[Cylinder(CX, CY, R, 0.0, H, EPS, k=32)])
    s_boxes = Structure(WG, disk_boxes_reference(R, H, EPS, 32))
    seg_a, seg_b = s_shape.segments(), s_boxes.segments()
    assert len(seg_a) == len(seg_b) == 1
    la, lb = seg_a[0].cross_section, seg_b[0].cross_section
    assert np.allclose(la.x_edges, lb.x_edges)
    assert np.allclose(la.y_edges, lb.y_edges)
    assert np.allclose(la.eps_cells, lb.eps_cells)
    assert seg_a[0].shapes and seg_a[0].shapes[0] is s_shape.shapes[0]
    assert seg_b[0].shapes == ()


@pytest.mark.unit
def test_wall_touching_cylinder_staircases_clamped():
    # touches y = a exactly (cy + r = 0.017 + 0.015 = 0.032)
    cyl = Cylinder(CX, CY, R, 0.0, H, EPS)
    boxes = cyl.staircase(WG, 32)
    assert max(b.y2 for b in boxes) <= WG.b + 1e-15
    Structure(WG, shapes=[cyl])  # must not raise bounds validation


@pytest.mark.unit
def test_shapes_only_z_span():
    s = Structure(WG, shapes=[Cylinder(CX, CY, R, 1e-3, 6e-3, EPS)])
    assert s.z_span == (1e-3, 6e-3)


@pytest.mark.unit
def test_off_guide_shape_raises():
    with pytest.raises(ValueError, match="waveguide"):
        Structure(WG, shapes=[Cylinder(A_WG + 0.02, CY, R, 0.0, H, EPS)])


@pytest.mark.unit
def test_shapes_is_keyword_only():
    with pytest.raises(TypeError):
        # four positionals: shapes must never bind positionally
        Structure(WG, [], 1.0, [Cylinder(CX, CY, R, 0.0, H, EPS)])  # type: ignore[misc]
