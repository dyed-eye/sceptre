"""Adaptive spatial resolution (ASR) for high-contrast obstacles.

Granet's ASR, reformulated through transformation optics: a smooth coordinate
map x = X(u), y = Y(v) that compresses spatial resolution near the dielectric
edges is EXACTLY equivalent to solving Maxwell's equations on the undeformed
(u, v, z) box for the transformed fields E~ = (X' E_x, Y' E_y, E_z) in the
diagonal anisotropic materials (Ward-Pendry, J = diag(1/X', 1/Y', 1)):

    eps~ = eps * diag( Y'/X',  X'/Y',  X'Y' ),
    mu~  =       diag( Y'/X',  X'/Y',  X'Y' ).

Consequently the whole FMM machinery -- the symmetric F/G operators, Li's
factorization rules, the stable S-cascade -- carries over verbatim; only the
multiplication operators change.  They are assembled by per-interval
Gauss-Legendre quadrature: the metric factors are smooth and every
discontinuity of eps sits ON an interval boundary, so the quadrature is
spectrally accurate.

The map (per direction, breakpoints u_j = x_j at every dielectric edge and
both walls; on each interval, t = (u - u_j)/Du):

    X(u) = x_j + Dx [ t - (1 - eta) sin(2 pi t) / (2 pi) ],
    X'(u) = 1 - (1 - eta) cos(2 pi t),

so X' = eta at every edge (compression factor), X' is C^1 globally (X'' = 0 at
interval ends), the map is a bijection for any eta in (0, 1], and eta = 1 is
the identity map.

Lead modes under ASR are numerical eigenvectors of the transformed uniform
guide, labeled against projected analytic TE/TM patterns and normalized to
unit plain-coefficient pseudo-flux.  Because dx dy = X'Y' du dv and the
transformed transverse fields carry exactly one metric factor each, the plain
coefficient flux e^T v EQUALS the physical pseudo-flux -- the ASR S-matrix is
the physical S-matrix with no normalization gauge at all.

References:
* G. Granet, "Reformulation of the lamellar grating problem through the
  concept of adaptive spatial resolution," JOSA A 16, 2510 (1999).
* T. Vallius, M. Honkanen, "Reformulation of the Fourier modal method with
  adaptive spatial resolution," Opt. Express 10, 24 (2002).
* A. J. Ward, J. B. Pendry, J. Mod. Opt. 43, 773 (1996) -- coordinate
  transforms as equivalent materials.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis
from .fourier import EpsOperators
from .geometry import CrossSection, Structure

_EDGE_MERGE_TOL = 1e-9  # relative tolerance for merging coincident edges


@dataclass(frozen=True)
class AsrConfig:
    """User-facing ASR switch.  eta = X' at the edges (1 = identity map).

    Default 0.3 measured optimal on the eps = 80 benchmark: much smaller eta
    over-compresses and makes the SMOOTH lead fields expensive to represent;
    eta -> 1 loses the edge resolution gain.
    """

    eta: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("AsrConfig.eta must be in (0, 1]")


class AsrMap1D:
    """Piecewise-smooth bijection X: [0, L] -> [0, L] with X(u_j) = u_j at edges.

    min_interval > 0 thins dense edge sets: an interior edge becomes a map
    breakpoint only if it is at least min_interval away from the previously
    kept breakpoint and from the far wall.  On each interval X' swings
    eta -> 2-eta -> eta, so intervals shorter than the basis resolution put
    metric content beyond the representable bandwidth (aliased operators) and
    destabilize the numerical lead-mode stage; such edges must not compress.
    Dropped edges lose only the ASR resolution boost -- quadrature stays exact
    because build_asr_operators integrates on the union of cell edges and map
    breakpoints regardless.
    """

    def __init__(self, length: float, interior_edges, eta: float,
                 min_interval: float = 0.0):
        interior = sorted(
            float(e)
            for e in interior_edges
            if _EDGE_MERGE_TOL * length < e < length * (1 - _EDGE_MERGE_TOL)
        )
        self.length = float(length)
        self.dropped = 0  # edges rejected by min_interval (coincidence merges
        edges = [0.0]  # by the rounding below stay silent, as before)
        for e in interior:
            if e - edges[-1] >= min_interval and length - e >= min_interval:
                edges.append(e)
            else:
                self.dropped += 1
        edges.append(float(length))
        self.breaks = np.unique(np.round(np.asarray(edges), 15))
        self.eta = float(eta)
        self.identity = self.eta == 1.0 or len(self.breaks) == 2
        # With no interior edges compression would only waste resolution at the
        # PEC walls (fields are smooth there) -- degrade to the identity map.

    def x(self, u):
        u = np.asarray(u, dtype=float)
        if self.identity:
            return u.copy()
        out = np.empty_like(u)
        idx = np.clip(
            np.searchsorted(self.breaks, u, side="right") - 1, 0, len(self.breaks) - 2
        )
        u0 = self.breaks[idx]
        du = self.breaks[idx + 1] - u0
        t = (u - u0) / du
        out = u0 + du * (t - (1.0 - self.eta) * np.sin(2 * np.pi * t) / (2 * np.pi))
        return out

    def dx(self, u):
        u = np.asarray(u, dtype=float)
        if self.identity:
            return np.ones_like(u)
        idx = np.clip(
            np.searchsorted(self.breaks, u, side="right") - 1, 0, len(self.breaks) - 2
        )
        u0 = self.breaks[idx]
        du = self.breaks[idx + 1] - u0
        t = (u - u0) / du
        return 1.0 - (1.0 - self.eta) * np.cos(2 * np.pi * t)


def build_maps(
    structure: Structure, eta: float, min_x: float = 0.0, min_y: float = 0.0
) -> tuple[AsrMap1D, AsrMap1D]:
    """One shared map per direction from the union of edges over ALL segments
    (all slices must live in the same u, v coordinates for trivial interface
    matching in the S-cascade).

    min_x / min_y thin edges denser than the caller's basis can resolve (see
    AsrMap1D); a warning reports how many compression points were dropped so a
    staircase-heavy structure degrades transparently rather than aliasing.
    """
    a, b = structure.waveguide.a, structure.waveguide.b
    xs: list[float] = []
    ys: list[float] = []
    for seg in structure.segments():
        xs.extend(seg.cross_section.x_edges[1:-1])
        ys.extend(seg.cross_section.y_edges[1:-1])
    xmap = AsrMap1D(a, xs, eta, min_interval=min_x)
    ymap = AsrMap1D(b, ys, eta, min_interval=min_y)
    # report per axis, and only for axes where thinning was actually requested
    parts = [
        f"{m.dropped} {ax}-edge(s)"
        for ax, m, lim in (("x", xmap, min_x), ("y", ymap, min_y))
        if lim > 0.0 and m.dropped
    ]
    if parts:
        warnings.warn(
            f"ASR map thinned: {' and '.join(parts)} closer than the basis "
            "resolution were not made compression points (dense staircases "
            "alias the metric operators). The dropped edges keep exact "
            "quadrature but no ASR resolution boost; for heavily staircased "
            "shapes consider plain Li with a larger N.",
            UserWarning,
            stacklevel=2,
        )
    return xmap, ymap


# ---------------------------------------------------------------------------
# Quadrature Gram matrices
# ---------------------------------------------------------------------------


def _basis_rows(kind: str, mmax: int, length: float, u: np.ndarray) -> np.ndarray:
    """Orthonormal basis functions evaluated at nodes: rows = functions."""
    arg = np.outer(np.arange(0 if kind == "cos" else 1, mmax + 1), np.pi * u / length)
    if kind == "cos":
        rows = np.sqrt(2.0 / length) * np.cos(arg)
        rows[0] = np.sqrt(1.0 / length)
        return rows
    return np.sqrt(2.0 / length) * np.sin(arg)


def gauss_gram(kind: str, mmax: int, length: float, pieces) -> np.ndarray:
    """Gram matrix  G_{mm'} = sum over pieces of  int_{u1}^{u2} w(u) f_m f_m' du.

    pieces: iterable of (u1, u2, w) with w a callable or scalar, smooth on the
    open interval.  Node count scales with the trig bandwidth so the result is
    accurate to roundoff.
    """
    size = mmax + 1 if kind == "cos" else mmax
    out = np.zeros((size, size), dtype=complex)
    for u1, u2, w in pieces:
        if u2 - u1 <= 0:
            continue
        nq = int(3.2 * mmax * (u2 - u1) / length) + 24
        nodes, weights = np.polynomial.legendre.leggauss(nq)
        u = 0.5 * (u2 - u1) * nodes + 0.5 * (u1 + u2)
        wq = 0.5 * (u2 - u1) * weights
        wval = w(u) if callable(w) else np.full(u.shape, complex(w), dtype=complex)
        rows = _basis_rows(kind, mmax, length, u)
        out += (rows * (wq * wval)) @ rows.T
    return out


def _refined_pieces(cell_edges: np.ndarray, map_breaks: np.ndarray):
    """Partition refined by both the eps-cell edges and the map breakpoints
    (the metric has derivative kinks at map breakpoints), tagged by cell index."""
    merged = np.unique(np.round(np.concatenate([cell_edges, map_breaks]), 15))
    mids = 0.5 * (merged[:-1] + merged[1:])
    cell_of = np.clip(
        np.searchsorted(cell_edges, mids, side="right") - 1, 0, len(cell_edges) - 2
    )
    return [(merged[k], merged[k + 1], int(cell_of[k])) for k in range(len(merged) - 1)]


# ---------------------------------------------------------------------------
# Material operators of the transformed problem
# ---------------------------------------------------------------------------


def build_asr_operators(
    layout: CrossSection, basis: ModeBasis, xmap: AsrMap1D, ymap: AsrMap1D
) -> EpsOperators:
    """eps~ and mu~ multiplication operators (Li rules along edge normals).

    Ward-Pendry tensors (J = Lambda^-1 = diag(1/X', 1/Y', 1)):
        eps~ = eps * diag(Y'/X', X'/Y', X'Y'),  mu~ = diag(Y'/X', X'/Y', X'Y'),
    acting on the TRANSFORMED field components E~_u = X' E_x, E~_v = Y' E_y.
    Sanity check of the orientation: an E_z/H_v wave running along u carries
    k_u^2 = omega^2 eps~_z mu~_v = eps k0^2 X'^2, i.e. k_u = k_x X' -- FEWER
    oscillations where the map compresses, which is the whole point of ASR.

    Separable strip forms (v-strip j has x-profile eps_j(u)):
      1/eps~_u = (1/eps)(X'/Y'); the inverse rule along u at fixed v gives
      (floor(1/eps~_u)_u)^-1 = Y'(v) * B_j^-1 with B_j = Gram_cc[(1/eps_j) X'],
      then the direct rule along v:
          eps_xx = sum_j  B_j^-1  (x)  Gram_ss[Y' chi_j].
    Symmetrically for eps~_v; eps~_z and every mu~ component are smooth (or
    multiply tangential-continuous fields) and take the plain direct rule.
    """
    M, N = basis.M, basis.N
    a, b = basis.a, basis.b
    xe, ye = layout.x_edges, layout.y_edges
    eps = layout.eps_cells
    nx, ny = eps.shape

    inv_xp = lambda u: 1.0 / xmap.dx(u)  # noqa: E731
    inv_yp = lambda v: 1.0 / ymap.dx(v)  # noqa: E731

    px = _refined_pieces(xe, xmap.breaks)  # (u1, u2, cell i)
    py = _refined_pieces(ye, ymap.breaks)  # (v1, v2, cell j)

    # -- eps~_u: inverse rule along u, direct along v -------------------------
    exx = np.zeros((basis.X.size, basis.X.size), dtype=complex)
    for j in range(ny):
        b_j = gauss_gram(
            "cos",
            M,
            a,
            [
                (u1, u2, lambda u, i=i: (1.0 / eps[i, j]) * xmap.dx(u))
                for u1, u2, i in px
            ],
        )
        s_j = gauss_gram(
            "sin",
            N,
            b,
            [(v1, v2, ymap.dx) for v1, v2, jj in py if jj == j],
        )
        exx += np.kron(np.linalg.inv(b_j), s_j)

    # -- eps~_v: inverse rule along v, direct along u -------------------------
    eyy = np.zeros((basis.Y.size, basis.Y.size), dtype=complex)
    for i in range(nx):
        d_i = gauss_gram(
            "cos",
            N,
            b,
            [
                (v1, v2, lambda v, j=j: (1.0 / eps[i, j]) * ymap.dx(v))
                for v1, v2, j in py
            ],
        )
        s_i = gauss_gram(
            "sin",
            M,
            a,
            [(u1, u2, xmap.dx) for u1, u2, ii in px if ii == i],
        )
        eyy += np.kron(s_i, np.linalg.inv(d_i))

    # -- eps~_z = eps X'Y': direct rule (E_z tangential everywhere) -----------
    ezz = np.zeros((basis.Z.size, basis.Z.size), dtype=complex)
    sx_c = [gauss_gram("sin", M, a, [(u1, u2, xmap.dx)]) for u1, u2, _ in px]
    sy_c = [gauss_gram("sin", N, b, [(v1, v2, ymap.dx)]) for v1, v2, _ in py]
    for k, (u1, u2, i) in enumerate(px):
        for l, (v1, v2, j) in enumerate(py):  # noqa: E741
            ezz += eps[i, j] * np.kron(sx_c[k], sy_c[l])

    # -- mu~ components: smooth metric-only factors, direct rule --------------
    # mu~_v = X'/Y' on X-space (H~y); mu~_u = Y'/X' on Y-space (H~x);
    # mu~_z = X'Y' on W-space (H~z, inverted in build_fg).
    myy = np.kron(
        gauss_gram("cos", M, a, [(u1, u2, xmap.dx) for u1, u2, _ in px]),
        gauss_gram("sin", N, b, [(v1, v2, inv_yp) for v1, v2, _ in py]),
    )
    mxx = np.kron(
        gauss_gram("sin", M, a, [(u1, u2, inv_xp) for u1, u2, _ in px]),
        gauss_gram("cos", N, b, [(v1, v2, ymap.dx) for v1, v2, _ in py]),
    )
    mzz = np.kron(
        gauss_gram("cos", M, a, [(u1, u2, xmap.dx) for u1, u2, _ in px]),
        gauss_gram("cos", N, b, [(v1, v2, ymap.dx) for v1, v2, _ in py]),
    )
    return EpsOperators(exx=exx, eyy=eyy, ezz=ezz, mxx=mxx, myy=myy, mzz=mzz)
