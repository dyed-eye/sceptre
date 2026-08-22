"""KFJ anisotropic subpixel smoothing — a comparison tool, NOT an accuracy fix.

Boundary cells of a fine rectilinear grid carry the Kottke-Farjadpour-Johnson
effective tensor (harmonic <1/eps>^-1 along the boundary normal, arithmetic
<eps> tangentially; eps_zz arithmetic), assembled with Li's rules on the
diagonal and the direct rule for the eps_xy coupling.

**Measured limitation — REFUTED for high-contrast accuracy (LEDGER.md H1):**
subpixel smoothing cancels the real-space grid error of FDTD/planewave
methods, but the FMM has no grid — the spectral basis resolves the smoothing
layer as a REAL graded ring, adding a first-order layer shift (∝ 1/cells)
and spurious anisotropic-ring modes at eps ≈ 80.  Use NVF for high-contrast
curved boundaries; KFJ remains available for cross-method comparison and for
low-contrast work (see docs/factorizations.md for the measured numbers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis
from .fourier import EpsOperators, cos_overlap, sin_overlap
from .geometry import CrossSection, Waveguide
from .nvf import _galerkin
from .shapes import Shape


@dataclass(frozen=True)
class KfjConfig:
    """cells: grid cells across each shape's bbox span; supersample: points
    per cell axis for fill fractions."""

    cells: int = 96
    supersample: int = 16

    def __post_init__(self) -> None:
        if self.cells < 8:
            raise ValueError("KfjConfig.cells must be >= 8")
        if self.supersample < 2:
            raise ValueError("KfjConfig.supersample must be >= 2")


def _grid_edges(shapes: tuple[Shape, ...], waveguide: Waveguide, cells: int):
    """Union of per-shape fine grids plus guide bounds (rectilinear)."""
    xs = [np.array([0.0, waveguide.a])]
    ys = [np.array([0.0, waveguide.b])]
    for s in shapes:
        x1, x2, y1, y2 = s.bbox
        xs.append(np.linspace(max(x1, 0.0), min(x2, waveguide.a), cells + 1))
        ys.append(np.linspace(max(y1, 0.0), min(y2, waveguide.b), cells + 1))
    xe = np.unique(np.concatenate(xs))
    ye = np.unique(np.concatenate(ys))
    return xe, ye


def kfj_cells(
    shapes: tuple[Shape, ...],
    waveguide: Waveguide,
    config: KfjConfig | None = None,
    background: complex = 1.0 + 0.0j,
):
    """(xe, ye, exx, eyy, exy, ezz) tensor cell arrays for the smoothed grid.

    background is the host permittivity the fill fractions mix against
    (Structure.background — vacuum only by default)."""
    config = config or KfjConfig()
    bg = complex(background)
    xe, ye = _grid_edges(shapes, waveguide, config.cells)
    nx, ny = len(xe) - 1, len(ye) - 1
    exx = np.full((nx, ny), bg, dtype=complex)
    eyy = np.full((nx, ny), bg, dtype=complex)
    exy = np.zeros((nx, ny), dtype=complex)
    ezz = np.full((nx, ny), bg, dtype=complex)
    ss = config.supersample
    frac = (np.arange(ss) + 0.5) / ss
    for i in range(nx):
        xs = xe[i] + frac * (xe[i + 1] - xe[i])
        xm = 0.5 * (xe[i] + xe[i + 1])
        for j in range(ny):
            ys = ye[j] + frac * (ye[j + 1] - ye[j])
            ym = 0.5 * (ye[j] + ye[j + 1])
            xg, yg = np.meshgrid(xs, ys, indexing="ij")
            f, owner = 0.0, None
            for s in shapes:
                fill = float(np.mean(s.level_set(xg, yg) < 0))
                if fill > 0.0:
                    if owner is not None:
                        raise ValueError(
                            "shapes too close for the KFJ grid (one cell is "
                            "covered by two shapes) — increase KfjConfig.cells "
                            "or separate the shapes"
                        )
                    f, owner = fill, s
            if owner is None:
                continue
            eps = owner.eps
            ez = f * eps + (1 - f) * bg
            ezz[i, j] = ez
            if f >= 1.0:
                exx[i, j] = eyy[i, j] = ez
                continue
            nx_, ny_ = owner.normal(np.array([xm]), np.array([ym]))
            nxv, nyv = float(nx_[0]), float(ny_[0])
            e_par = ez
            e_perp = 1.0 / (f / eps + (1 - f) / bg)
            exx[i, j] = e_perp * nxv**2 + e_par * nyv**2
            eyy[i, j] = e_perp * nyv**2 + e_par * nxv**2
            exy[i, j] = (e_perp - e_par) * nxv * nyv
    return xe, ye, exx, eyy, exy, ezz


def build_kfj_operators(
    shapes: tuple[Shape, ...],
    layout: CrossSection,
    waveguide: Waveguide,
    basis: ModeBasis,
    config: KfjConfig | None = None,
    background: complex = 1.0 + 0.0j,
) -> EpsOperators:
    """Assemble KFJ EpsOperators.  `layout` (the sharp staircase) is unused —
    the smoothed grid REPLACES the staircase geometry by construction."""
    del layout
    if not shapes:
        raise ValueError("KFJ needs at least one Shape (boxes carry no normals)")
    xe, ye, exx_c, eyy_c, exy_c, ezz_c = kfj_cells(
        shapes, waveguide, config, background
    )
    M, N = basis.M, basis.N
    a, b = basis.a, basis.b
    nx, ny = exx_c.shape

    ezz = _galerkin(xe, ye, ezz_c, basis, "ZZ")
    exy = _galerkin(xe, ye, exy_c, basis, "XY")

    # Li rules on the tensor diagonal (inverse along the axis, direct across).
    cx = [cos_overlap(M, a, xe[i], xe[i + 1]) for i in range(nx)]
    sy = [sin_overlap(N, b, ye[j], ye[j + 1]) for j in range(ny)]
    exx = np.zeros((basis.X.size, basis.X.size), dtype=complex)
    for j in range(ny):
        inv_eps_x = np.zeros((M + 1, M + 1), dtype=complex)
        for i in range(nx):
            inv_eps_x += (1.0 / exx_c[i, j]) * cx[i]
        exx += np.kron(np.linalg.inv(inv_eps_x), sy[j])

    sx = [sin_overlap(M, a, xe[i], xe[i + 1]) for i in range(nx)]
    cy = [cos_overlap(N, b, ye[j], ye[j + 1]) for j in range(ny)]
    eyy = np.zeros((basis.Y.size, basis.Y.size), dtype=complex)
    for i in range(nx):
        inv_eps_y = np.zeros((N + 1, N + 1), dtype=complex)
        for j in range(ny):
            inv_eps_y += (1.0 / eyy_c[i, j]) * cy[j]
        eyy += np.kron(sx[i], np.linalg.inv(inv_eps_y))

    return EpsOperators(exx=exx, eyy=eyy, ezz=ezz, exy=exy)
