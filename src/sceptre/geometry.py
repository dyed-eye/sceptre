"""Geometry description: rectangular PEC waveguide with axis-aligned dielectric boxes.

The structure is staircase-sliced along z into segments on which the permittivity
layout epsilon(x, y) is z-uniform.  Only piecewise-constant (box) obstacles are
supported; a smoothly varying epsilon(z) must be staircased by the caller into boxes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_GEOM_TOL = 1e-12  # relative tolerance for merging coincident breakpoints


@dataclass(frozen=True)
class Waveguide:
    """Rectangular waveguide cross-section, PEC walls at x=0,a and y=0,b."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("waveguide dimensions must be positive")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box [x1,x2]x[y1,y2]x[z1,z2] of relative permittivity eps.

    Later boxes in a Structure override earlier ones where they overlap.
    """

    x1: float
    x2: float
    y1: float
    y2: float
    z1: float
    z2: float
    eps: complex

    def __post_init__(self) -> None:
        if not (self.x1 < self.x2 and self.y1 < self.y2 and self.z1 < self.z2):
            raise ValueError("box must have positive extent in every direction")
        if self.eps == 0:
            raise ValueError(
                "box permittivity must be nonzero (vacuum is eps=1); eps=0 would "
                "NaN-poison the inverse-rule factorization"
            )


@dataclass(frozen=True)
class CrossSection:
    """Piecewise-constant permittivity layout on a rectilinear grid.

    eps_cells[i, j] is the permittivity on [x_edges[i], x_edges[i+1]] x
    [y_edges[j], y_edges[j+1]].
    """

    x_edges: np.ndarray  # shape (nx+1,), x_edges[0] = 0, x_edges[-1] = a
    y_edges: np.ndarray  # shape (ny+1,)
    eps_cells: np.ndarray  # complex, shape (nx, ny)

    @property
    def is_uniform(self) -> bool:
        return bool(np.all(self.eps_cells == self.eps_cells.flat[0]))

    @property
    def uniform_eps(self) -> complex:
        if not self.is_uniform:
            raise ValueError("cross-section is not uniform")
        return complex(self.eps_cells.flat[0])

    def key(self) -> tuple[object, object, object]:
        """Hashable identity for caching per-layout operators."""
        return (
            tuple(np.round(self.x_edges, 15)),
            tuple(np.round(self.y_edges, 15)),
            tuple(self.eps_cells.ravel()),
        )


@dataclass(frozen=True)
class Segment:
    """One z-uniform slice of the structure."""

    z1: float
    z2: float
    cross_section: CrossSection

    @property
    def length(self) -> float:
        return self.z2 - self.z1


@dataclass(frozen=True)
class Structure:
    """A waveguide containing dielectric boxes, sliceable into z-uniform segments.

    Immutable: Solver captures segments() at construction, so a mutable Structure
    would silently leave solvers on stale geometry.  Build a new Structure instead.
    """

    waveguide: Waveguide
    boxes: Sequence[Box] = ()
    background: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", tuple(self.boxes))
        if self.background == 0:
            raise ValueError("background permittivity must be nonzero (vacuum is 1)")
        a, b = self.waveguide.a, self.waveguide.b
        for box in self.boxes:
            if box.x1 < -_GEOM_TOL * a or box.x2 > a * (1 + _GEOM_TOL):
                raise ValueError(f"box exceeds waveguide in x: {box}")
            if box.y1 < -_GEOM_TOL * b or box.y2 > b * (1 + _GEOM_TOL):
                raise ValueError(f"box exceeds waveguide in y: {box}")

    @property
    def z_span(self) -> tuple[float, float]:
        """z-extent of the obstacle region (structure faces = port reference planes)."""
        if not self.boxes:
            raise ValueError("structure has no boxes; z-span undefined")
        return (min(b.z1 for b in self.boxes), max(b.z2 for b in self.boxes))

    def segments(self) -> list[Segment]:
        """Slice the structure into z-uniform segments between consecutive z-breakpoints."""
        z_lo, z_hi = self.z_span
        breaks = _merge_breakpoints(
            [z_lo, z_hi] + [b.z1 for b in self.boxes] + [b.z2 for b in self.boxes],
            scale=max(abs(z_hi - z_lo), 1.0),
        )
        segments = []
        for z1, z2 in zip(breaks[:-1], breaks[1:]):
            z_mid = 0.5 * (z1 + z2)
            active = [b for b in self.boxes if b.z1 <= z_mid <= b.z2]
            segments.append(Segment(z1, z2, self._layout(active)))
        return segments

    def _layout(self, active_boxes: Sequence[Box]) -> CrossSection:
        a, b = self.waveguide.a, self.waveguide.b
        x_edges = _merge_breakpoints(
            [0.0, a] + [v for box in active_boxes for v in (box.x1, box.x2)], scale=a
        )
        y_edges = _merge_breakpoints(
            [0.0, b] + [v for box in active_boxes for v in (box.y1, box.y2)], scale=b
        )
        nx, ny = len(x_edges) - 1, len(y_edges) - 1
        eps = np.full((nx, ny), self.background, dtype=complex)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
        for box in active_boxes:  # later boxes override earlier ones
            in_x = (x_mid > box.x1) & (x_mid < box.x2)
            in_y = (y_mid > box.y1) & (y_mid < box.y2)
            eps[np.ix_(in_x, in_y)] = box.eps
        return CrossSection(x_edges, y_edges, eps)


def _merge_breakpoints(values: list[float], scale: float) -> np.ndarray:
    """Sort values and merge points closer than _GEOM_TOL * scale."""
    vals = np.sort(np.asarray(values, dtype=float))
    merged = [vals[0]]
    for v in vals[1:]:
        if v - merged[-1] > _GEOM_TOL * scale:
            merged.append(v)
    return np.asarray(merged)
