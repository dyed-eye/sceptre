"""Normal-vector-field (NVF) factorization: sharp eps, rotation-covariant rules.

The scalar Li rules apply the inverse rule along x or y — correct only for
axis-aligned interfaces.  For a curved boundary the correct rule follows the
LOCAL normal: inverse rule for the normal E-component (D_perp continuous),
direct (Laurent) rule tangentially (E_par continuous).  Discretized with
Galerkin (exact-overlap) matrices this is

    eps_hat = E + (1/2) (B P + P B),   E = diag(<<eps>>_X, <<eps>>_Y),
    B = diag(<<1/eps>>_X^-1, <<1/eps>>_Y^-1) - E,
    P = [[<<w nx nx>>, <<w nx ny>>], [<<w nx ny>>^T, <<w ny ny>>]],

with w(d) = cos^2(pi d / 2W) a compactly supported window of the level-set
distance d (w -> 0 away from the boundary, where eps is uniform and every
rule is exact).  The SYMMETRIZED product keeps eps_hat = eps_hat^T, hence
G = G^T — structural reciprocity/unitarity survive at any truncation.
Validated on the eps=80 benchmark disk: the line lands at the converged
position at N ~ 16-20 in ONE solve where plain Li needs an N-ladder plus
Richardson extrapolation (see docs/factorizations.md and LEDGER.md H3).

Ingredients are assembled from EXACT per-cell overlaps: eps and 1/eps from
the shape's exact-interval staircase layout (no rasterization moire), the
smooth windowed projection fields from centroid-sampled quadrature cells
(second-order accurate).  ``window=math.inf`` forces w = 1 everywhere — the
lamellar/Li limit used by tests; single-shape only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis
from .fourier import EpsOperators, cos_overlap, cs_overlap, sin_overlap
from .geometry import CrossSection, Waveguide
from .shapes import Shape

# Default window as a fraction of the shape scale, fixed by the Task-3
# calibration sweep on the eps=80 benchmark disk (docs/plans/
# 2026-08-22-productionize-nvf-kfj.md): fraction 0.20 gave the best line
# accuracy at both N=16 (+4.3 MHz) and N=20 (-0.8 MHz vs COMSOL 5.4357).
# Provenance: cm-scale X-band geometry; the fraction (not an absolute
# length) is what transfers across bands.
WINDOW_FRACTION = 0.20

_W_EPS = 1e-6  # window support threshold for quadrature and overlap checks


@dataclass(frozen=True)
class NvfConfig:
    """NVF knobs.  window: absolute half-width [m] of the boundary window
    (None = WINDOW_FRACTION * shape.scale per shape; math.inf = w == 1, the
    Li-limit/testing mode).  quad_cells: quadrature strips across a shape's
    window region for the smooth projection fields."""

    window: float | None = None
    quad_cells: int = 192

    def __post_init__(self) -> None:
        if self.window is not None and self.window <= 0:
            raise ValueError("NvfConfig.window must be positive (or None)")
        if self.quad_cells < 16:
            raise ValueError("NvfConfig.quad_cells must be >= 16")


def _galerkin(
    xe: np.ndarray,
    ye: np.ndarray,
    cells: np.ndarray,
    basis: ModeBasis,
    kind: str,
) -> np.ndarray:
    """Galerkin matrix of a cell-wise-constant function in one component space.

    Strip-factored: one kron per x-interval (y-cells summed first)."""
    M, N = basis.M, basis.N
    a, b = basis.a, basis.b
    if kind == "XX":
        ox = [cos_overlap(M, a, xe[i], xe[i + 1]) for i in range(len(xe) - 1)]
        oy = [sin_overlap(N, b, ye[j], ye[j + 1]) for j in range(len(ye) - 1)]
        size = (basis.X.size, basis.X.size)
    elif kind == "YY":
        ox = [sin_overlap(M, a, xe[i], xe[i + 1]) for i in range(len(xe) - 1)]
        oy = [cos_overlap(N, b, ye[j], ye[j + 1]) for j in range(len(ye) - 1)]
        size = (basis.Y.size, basis.Y.size)
    elif kind == "ZZ":
        ox = [sin_overlap(M, a, xe[i], xe[i + 1]) for i in range(len(xe) - 1)]
        oy = [sin_overlap(N, b, ye[j], ye[j + 1]) for j in range(len(ye) - 1)]
        size = (basis.Z.size, basis.Z.size)
    elif kind == "XY":
        ox = [cs_overlap(M, a, xe[i], xe[i + 1]) for i in range(len(xe) - 1)]
        oy = [cs_overlap(N, b, ye[j], ye[j + 1]).T for j in range(len(ye) - 1)]
        size = (basis.X.size, basis.Y.size)
    else:  # pragma: no cover - internal
        raise ValueError(kind)
    out = np.zeros(size, dtype=complex)
    for i in range(len(xe) - 1):
        col = cells[i]
        if not np.any(col != 0):
            continue
        ysum = np.zeros_like(oy[0], dtype=complex)
        for j in range(len(ye) - 1):
            if col[j] != 0:
                ysum += col[j] * oy[j]
        out += np.kron(ox[i], ysum)
    return out


def _shape_window(shape: Shape, config: NvfConfig) -> float:
    w = config.window if config.window is not None else WINDOW_FRACTION * shape.scale
    if math.isfinite(w) and w >= shape.scale:
        raise ValueError(
            f"NVF window {w:.4g} m must stay below the shape scale "
            f"{shape.scale:.4g} m: beyond it the normal field hits the "
            "level-set medial axis (silent wrong physics)"
        )
    return w


def _projection_cells(shape: Shape, wdw: float, waveguide: Waveguide, n_cells: int):
    """Quadrature grid + centroid-sampled (w nx nx, w nx ny, w ny ny) cells."""
    x1, x2, y1, y2 = shape.bbox
    if math.isinf(wdw):
        gx1, gx2, gy1, gy2 = 0.0, waveguide.a, 0.0, waveguide.b
    else:
        gx1, gx2 = max(x1 - wdw, 0.0), min(x2 + wdw, waveguide.a)
        gy1, gy2 = max(y1 - wdw, 0.0), min(y2 + wdw, waveguide.b)
    xe = np.linspace(gx1, gx2, n_cells + 1)
    ye = np.linspace(gy1, gy2, n_cells + 1)
    xm = 0.5 * (xe[:-1] + xe[1:])
    ym = 0.5 * (ye[:-1] + ye[1:])
    xg, yg = np.meshgrid(xm, ym, indexing="ij")
    if math.isinf(wdw):
        w = np.ones_like(xg)
    else:
        d = shape.level_set(xg, yg)
        w = np.where(np.abs(d) < wdw, np.cos(np.pi * d / (2 * wdw)) ** 2, 0.0)
    nx, ny = shape.normal(xg, yg)
    return xe, ye, w, w * nx * nx, w * nx * ny, w * ny * ny, xg, yg


def build_nvf_operators(
    shapes: tuple[Shape, ...],
    layout: CrossSection,
    waveguide: Waveguide,
    basis: ModeBasis,
    config: NvfConfig | None = None,
) -> EpsOperators:
    """Assemble the NVF EpsOperators for one z-uniform segment."""
    config = config or NvfConfig()
    if not shapes:
        raise ValueError("NVF needs at least one Shape (boxes carry no normals)")
    if math.isinf(config.window or 1.0) and len(shapes) > 1:
        raise ValueError("window=inf (Li-limit mode) supports a single shape only")

    # -- sharp eps / 1/eps from the exact staircase layout ------------------
    xe, ye = layout.x_edges, layout.y_edges
    eps = layout.eps_cells
    inv_eps = 1.0 / eps
    ce_x = _galerkin(xe, ye, eps, basis, "XX")
    ce_y = _galerkin(xe, ye, eps, basis, "YY")
    ezz = _galerkin(xe, ye, eps, basis, "ZZ")
    ci_x = np.linalg.inv(_galerkin(xe, ye, inv_eps, basis, "XX"))
    ci_y = np.linalg.inv(_galerkin(xe, ye, inv_eps, basis, "YY"))

    # -- windowed projection fields, one grid per shape ---------------------
    nx_, ny_ = basis.X.size, basis.Y.size
    p_xx = np.zeros((nx_, nx_), dtype=complex)
    p_yy = np.zeros((ny_, ny_), dtype=complex)
    p_xy = np.zeros((nx_, ny_), dtype=complex)
    supports = []
    for shape in shapes:
        wdw = _shape_window(shape, config)
        qxe, qye, w, pxx, pxy, pyy, xg, yg = _projection_cells(
            shape, wdw, waveguide, config.quad_cells
        )
        mask = w > _W_EPS
        supports.append((shape, xg[mask], yg[mask]))
        p_xx += _galerkin(qxe, qye, np.where(mask, pxx, 0.0), basis, "XX")
        p_yy += _galerkin(qxe, qye, np.where(mask, pyy, 0.0), basis, "YY")
        p_xy += _galerkin(qxe, qye, np.where(mask, pxy, 0.0), basis, "XY")

    # -- reject overlapping windows (blended normals are unverified) --------
    for i, (si, xi, yi) in enumerate(supports):
        for sj, _xj, _yj in supports[i + 1 :]:
            wj = _shape_window(sj, config)
            if math.isinf(wj):
                continue
            dj = sj.level_set(xi, yi)
            if np.any(np.abs(dj) < wj):
                raise ValueError(
                    f"NVF windows of {si} and {sj} overlap; blended normal "
                    "fields are unsupported — reduce the window or separate "
                    "the shapes"
                )

    # -- symmetrized projection-resolved operator ---------------------------
    e_blk = np.block([[ce_x, np.zeros((nx_, ny_))], [np.zeros((ny_, nx_)), ce_y]])
    b_blk = np.block(
        [[ci_x - ce_x, np.zeros((nx_, ny_))], [np.zeros((ny_, nx_)), ci_y - ce_y]]
    )
    p_blk = np.block([[p_xx, p_xy], [p_xy.T, p_yy]])
    eps_t = e_blk + 0.5 * (b_blk @ p_blk + p_blk @ b_blk)
    return EpsOperators(
        exx=eps_t[:nx_, :nx_],
        eyy=eps_t[nx_:, nx_:],
        ezz=ezz,
        exy=eps_t[:nx_, nx_:],
    )
