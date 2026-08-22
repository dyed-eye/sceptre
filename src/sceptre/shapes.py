"""Curved cross-section shapes: level-set base class + exact primitives.

Two layers (both usable directly):

* ``Shape`` — subclass with a ``level_set`` (signed distance, negative
  inside) and a ``bbox``; a generic bisection staircase and a finite-
  difference normal come free.  One material interval per x-strip is
  assumed (convex-in-y cross-sections).
* ``Cylinder`` — analytic level set, radial normal, and an EXACT-interval
  x-strip staircase (the chord formula), reproducing the validated
  campaign recipe bit-for-bit.

Shapes carry what a staircase of boxes cannot: the boundary normal field
that tensor factorizations (``factorization="nvf"``/``"kfj"``) require.
Staircases clamp themselves to the waveguide, so wall-touching shapes are
legal.
"""

from __future__ import annotations

import numpy as np

from .geometry import Box, Waveguide

_DEFAULT_K = 64  # strips across the shape's x-extent; ≤1.1 MHz geometry
# error measured at 0.94 mm strips on the r = 15 mm benchmark disk


class Shape:
    """Level-set-defined z-uniform obstacle; subclass and provide level_set/bbox.

    Treat instances as immutable once handed to a Structure: Solver captures
    staircases at construction and caches operators per shape identity —
    mutating attributes afterwards silently serves stale geometry."""

    def __init__(self, z1: float, z2: float, eps: complex, k: int = _DEFAULT_K):
        if not z1 < z2:
            raise ValueError("shape must have positive z-extent")
        if eps == 0:
            raise ValueError("shape permittivity must be nonzero (vacuum is 1)")
        if k < 4:
            raise ValueError("staircase needs k >= 4 strips")
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.eps = complex(eps)
        self.k = int(k)

    def level_set(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance-like function: negative inside, zero on the boundary."""
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(x1, x2, y1, y2) bounding box of the material region."""
        raise NotImplementedError

    @property
    def scale(self) -> float:
        """Characteristic in-plane half-size; NVF window defaults/validation
        scale with it (window must stay below it to avoid the level-set
        medial-axis / center singularity of the normal field)."""
        x1, x2, y1, y2 = self.bbox
        return 0.5 * min(x2 - x1, y2 - y1)

    def normal(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normal from the level-set gradient (central differences)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x1, x2, y1, y2 = self.bbox
        h = 1e-7 * max(x2 - x1, y2 - y1)
        gx = (self.level_set(x + h, y) - self.level_set(x - h, y)) / (2 * h)
        gy = (self.level_set(x, y + h) - self.level_set(x, y - h)) / (2 * h)
        norm = np.hypot(gx, gy)
        norm = np.where(norm > 0, norm, 1.0)
        return gx / norm, gy / norm

    def _y_interval(self, xm: float, y_lo: float, y_hi: float) -> tuple | None:
        """Material y-interval at strip midpoint xm by scan + bisection."""
        ys = np.linspace(y_lo, y_hi, 513)
        phi = self.level_set(np.full_like(ys, xm), ys)
        inside = np.flatnonzero(phi < 0)
        if len(inside) == 0:
            return None
        i0, i1 = int(inside[0]), int(inside[-1])
        if np.any(phi[i0 : i1 + 1] >= 0):
            raise ValueError(
                f"level set is not convex in y at x = {xm:.4g} (multiple "
                "material intervals per strip are unsupported by the generic "
                "staircase; override Shape.staircase for such sections)"
            )
        lo = (
            float(ys[i0])
            if i0 == 0
            else self._bisect(xm, float(ys[i0 - 1]), float(ys[i0]))
        )
        hi = (
            float(ys[i1])
            if i1 == len(ys) - 1
            else self._bisect(xm, float(ys[i1 + 1]), float(ys[i1]))
        )
        return (lo, hi) if hi > lo else None

    def _bisect(self, xm: float, y_out: float, y_in: float) -> float:
        for _ in range(60):
            mid = 0.5 * (y_out + y_in)
            if self.level_set(np.array([xm]), np.array([mid]))[0] < 0:
                y_in = mid
            else:
                y_out = mid
        return 0.5 * (y_out + y_in)

    def staircase(self, waveguide: Waveguide, k: int | None = None) -> list[Box]:
        """K x-strip staircase, y-intervals found on the level set, guide-clamped."""
        k = self.k if k is None else k
        bx1, bx2, by1, by2 = self.bbox
        y_lo, y_hi = max(by1, 0.0), min(by2, waveguide.b)
        boxes = []
        xs = np.linspace(bx1, bx2, k + 1)
        for x1, x2 in zip(xs[:-1], xs[1:]):
            xm = 0.5 * (x1 + x2)
            interval = self._y_interval(xm, y_lo, y_hi)
            if interval is None:
                continue
            y1, y2 = max(interval[0], 0.0), min(interval[1], waveguide.b)
            cx1, cx2 = max(x1, 0.0), min(x2, waveguide.a)
            if y2 <= y1 or cx2 <= cx1:
                continue
            boxes.append(Box(cx1, cx2, y1, y2, self.z1, self.z2, self.eps))
        return boxes


class Cylinder(Shape):
    """Circular cylinder (axis ∥ z): exact chords, analytic radial normal."""

    def __init__(
        self,
        cx: float,
        cy: float,
        r: float,
        z1: float,
        z2: float,
        eps: complex,
        k: int = _DEFAULT_K,
    ):
        super().__init__(z1, z2, eps, k)
        if r <= 0:
            raise ValueError("cylinder radius must be positive")
        self.cx = float(cx)
        self.cy = float(cy)
        self.r = float(r)

    def level_set(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.hypot(np.asarray(x) - self.cx, np.asarray(y) - self.cy) - self.r

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.cx - self.r, self.cx + self.r, self.cy - self.r, self.cy + self.r)

    @property
    def scale(self) -> float:
        return self.r

    def normal(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = np.asarray(x, dtype=float) - self.cx
        dy = np.asarray(y, dtype=float) - self.cy
        rho = np.hypot(dx, dy)
        rho = np.where(rho > 0, rho, 1.0)
        return dx / rho, dy / rho

    def staircase(self, waveguide: Waveguide, k: int | None = None) -> list[Box]:
        """Exact y-intervals at strip midpoints (the validated campaign recipe)."""
        k = self.k if k is None else k
        boxes = []
        xs = np.linspace(-self.r, self.r, k + 1)
        for x1, x2 in zip(xs[:-1], xs[1:]):
            xm = 0.5 * (x1 + x2)
            half = self.r * self.r - xm * xm
            if half <= 0:
                continue
            half = np.sqrt(half)
            y1 = max(self.cy - half, 0.0)
            y2 = min(self.cy + half, waveguide.b)
            cx1, cx2 = max(self.cx + x1, 0.0), min(self.cx + x2, waveguide.a)
            if y2 <= y1 or cx2 <= cx1:
                continue
            boxes.append(Box(cx1, cx2, y1, y2, self.z1, self.z2, self.eps))
        return boxes
