"""Permittivity operators in the sin/cos modal basis, with correct Fourier factorization.

Background
----------
Products of a discontinuous permittivity with a discontinuous field component must be
factorized with Li's rules, otherwise the modal expansion converges slowly (and
non-uniformly) at dielectric edges:

* L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures,"
  J. Opt. Soc. Am. A 13, 1870 (1996) — direct ("Laurent") vs inverse rule.
* L. Li, "New formulation of the Fourier modal method for crossed surface-relief
  gratings," J. Opt. Soc. Am. A 14, 2758 (1997) — mixed rules for 2-D patterns.

Adaptation to the closed PEC guide: the guide cross-section is the quarter period of an
even mirror extension, so the exponential-basis Toeplitz algebra becomes products of
overlap (Gram) matrices of the orthonormal sin/cos functions.  For piecewise-constant
axis-aligned layouts every 1-D "Fourier multiplication matrix" is an exact analytic
overlap sum over the layout cells, and Li's crossed-grating rule
    eps_xx = ceil( floor(1/eps)_x ^ -1 )_y
(inverse rule along x, direct rule along y, applied strip-wise) becomes

    eps_xx = sum_j  inv( sum_i (1/eps_ij) * Cx_i )  (x)  Sy_j        (Kronecker)

over y-strips j, and symmetrically for eps_yy.  The Ez elimination uses the plain
direct-rule matrix eps_zz (inverted later), because Ez is continuous across all
lateral edges (see refs/CONVENTIONS.md).

The direct-rule ("Laurent") variants are kept for the convergence benchmark
(tests/test_convergence.py) that demonstrates why the factorization matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import inv

from .basis import ModeBasis
from .geometry import CrossSection

try:  # Numba accelerates the overlap-matrix assembly loops.  numba is a declared
    # dependency, so the fallback below is only reached when numba cannot import
    # on an unsupported interpreter -- the pure-Python fills stay correct, just slow.
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _sin_overlap_fill(Mmax: int, a: float, x1: float, x2: float) -> np.ndarray:
    """Overlap matrix of orthonormal s_m = sqrt(2/a) sin(m pi x/a) over [x1, x2].

    S[m-1, mp-1] = integral_{x1}^{x2} s_m(x) s_mp(x) dx,  m, mp = 1..Mmax  (exact).
    """
    out = np.empty((Mmax, Mmax))
    pia = np.pi / a
    for m in range(1, Mmax + 1):
        for mp in range(m, Mmax + 1):
            if m == mp:
                k = m * pia
                val = 0.5 * (x2 - x1) - (np.sin(2 * k * x2) - np.sin(2 * k * x1)) / (
                    4 * k
                )
            else:
                dk = (m - mp) * pia
                sk = (m + mp) * pia
                val = 0.5 * (
                    (np.sin(dk * x2) - np.sin(dk * x1)) / dk
                    - (np.sin(sk * x2) - np.sin(sk * x1)) / sk
                )
            out[m - 1, mp - 1] = (2.0 / a) * val
            out[mp - 1, m - 1] = out[m - 1, mp - 1]
    return out


@njit(cache=True)
def _cos_overlap_fill(Mmax: int, a: float, x1: float, x2: float) -> np.ndarray:
    """Overlap matrix of orthonormal c_m over [x1, x2], m = 0..Mmax (exact).

    c_0 = 1/sqrt(a), c_m = sqrt(2/a) cos(m pi x/a).
    """
    out = np.empty((Mmax + 1, Mmax + 1))
    pia = np.pi / a
    for m in range(0, Mmax + 1):
        for mp in range(m, Mmax + 1):
            if m == 0 and mp == 0:
                raw = x2 - x1
            elif m == mp:
                k = m * pia
                raw = 0.5 * (x2 - x1) + (np.sin(2 * k * x2) - np.sin(2 * k * x1)) / (
                    4 * k
                )
            else:
                dk = (m - mp) * pia
                sk = (m + mp) * pia
                raw = 0.5 * (
                    (np.sin(dk * x2) - np.sin(dk * x1)) / dk
                    + (np.sin(sk * x2) - np.sin(sk * x1)) / sk
                )
            nm = (1.0 / a) if m == 0 else (2.0 / a)
            nmp = (1.0 / a) if mp == 0 else (2.0 / a)
            out[m, mp] = np.sqrt(nm * nmp) * raw
            out[mp, m] = out[m, mp]
    return out


def sin_overlap(Mmax: int, a: float, x1: float, x2: float) -> np.ndarray:
    return _sin_overlap_fill(Mmax, a, float(x1), float(x2))


def cos_overlap(Mmax: int, a: float, x1: float, x2: float) -> np.ndarray:
    return _cos_overlap_fill(Mmax, a, float(x1), float(x2))


@dataclass(frozen=True)
class EpsOperators:
    """Multiplication-by-eps operators on the component spaces of a ModeBasis.

    exx : acts on X-space (Ex);   Li: inverse rule in x, direct in y.
    eyy : acts on Y-space (Ey);   Li: inverse rule in y, direct in x.
    ezz : acts on Z-space (Ez);   direct rule (invert this matrix for the Ez elimination).
    """

    exx: np.ndarray
    eyy: np.ndarray
    ezz: np.ndarray


def build_eps_operators(
    layout: CrossSection, basis: ModeBasis, factorization: str = "li"
) -> EpsOperators:
    if factorization not in ("li", "direct"):
        raise ValueError(f"unknown factorization {factorization!r}")

    if layout.is_uniform:
        eps = layout.uniform_eps
        return EpsOperators(
            exx=eps * np.eye(basis.X.size, dtype=complex),
            eyy=eps * np.eye(basis.Y.size, dtype=complex),
            ezz=eps * np.eye(basis.Z.size, dtype=complex),
        )

    M, N = basis.M, basis.N
    a, b = basis.a, basis.b
    xe, ye = layout.x_edges, layout.y_edges
    eps = layout.eps_cells  # (nx, ny)
    nx, ny = eps.shape

    # 1-D overlap matrices per layout interval (exact analytic integrals).
    Sx = [sin_overlap(M, a, xe[i], xe[i + 1]) for i in range(nx)]
    Cx = [cos_overlap(M, a, xe[i], xe[i + 1]) for i in range(nx)]
    Sy = [sin_overlap(N, b, ye[j], ye[j + 1]) for j in range(ny)]
    Cy = [cos_overlap(N, b, ye[j], ye[j + 1]) for j in range(ny)]

    ezz = np.zeros((basis.Z.size, basis.Z.size), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            ezz += eps[i, j] * np.kron(Sx[i], Sy[j])

    if factorization == "direct":
        exx = np.zeros((basis.X.size, basis.X.size), dtype=complex)
        eyy = np.zeros((basis.Y.size, basis.Y.size), dtype=complex)
        for i in range(nx):
            for j in range(ny):
                exx += eps[i, j] * np.kron(Cx[i], Sy[j])
                eyy += eps[i, j] * np.kron(Sx[i], Cy[j])
        return EpsOperators(exx=exx, eyy=eyy, ezz=ezz)

    # -- Li factorization (inverse rule along the discontinuity normal) --
    # eps_xx: for each y-strip, invert the cos-basis matrix of 1/eps(., y).
    exx = np.zeros((basis.X.size, basis.X.size), dtype=complex)
    for j in range(ny):
        inv_eps_x = np.zeros((M + 1, M + 1), dtype=complex)
        for i in range(nx):
            inv_eps_x += (1.0 / eps[i, j]) * Cx[i]
        exx += np.kron(inv(inv_eps_x), Sy[j])

    # eps_yy: for each x-strip, invert the cos-basis matrix of 1/eps(x, .).
    eyy = np.zeros((basis.Y.size, basis.Y.size), dtype=complex)
    for i in range(nx):
        inv_eps_y = np.zeros((N + 1, N + 1), dtype=complex)
        for j in range(ny):
            inv_eps_y += (1.0 / eps[i, j]) * Cy[j]
        eyy += np.kron(Sx[i], inv(inv_eps_y))

    return EpsOperators(exx=exx, eyy=eyy, ezz=ezz)
