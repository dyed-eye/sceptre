"""Per-slice modal eigenproblem of the Fourier modal method.

For a z-uniform slice, eliminating Ez and H~z from Maxwell's equations gives the
first-order system (v-convention of modes.py: e = [Ex; Ey], v = [H~y; -H~x]):

    d e / dz = F v,        d v / dz = G e,

with the symmetric operators (all derivative matrices carry their signs, see basis.py):

    F = i [ k0 + Dx_ZX Ezz^-1 Dx_XZ / k0   ,  Dx_ZX Ezz^-1 Dy_YZ / k0
            Dy_ZY Ezz^-1 Dx_XZ / k0        ,  k0 + Dy_ZY Ezz^-1 Dy_YZ / k0 ]

    G = i [ k0 exx + Dy_WX Dy_XW / k0      ,  -Dy_WX Dx_YW / k0
            -Dx_WY Dy_XW / k0              ,  k0 eyy + Dx_WY Dx_YW / k0 ]

F = F^T and G = G^T hold at ANY truncation (the derivative operators are mutual
negative transposes and the eps operators are symmetric); this is the discrete
counterpart of Lorentz reciprocity and is what makes the final S-matrix satisfy
S = S^T and, for lossless media, unitarity to machine precision.

Modal ansatz e, v ~ exp(i beta z) gives the dense eigenproblem

    (F G) w = -beta^2 w,       v-eigenvector u = G w / (i beta),

solved with scipy.linalg.eig (LAPACK zgeev).  Uniform slices bypass the eigensolver
and use the analytic TE/TM modes (machine-exact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .basis import ModeBasis
from .fourier import EpsOperators, build_eps_operators
from .geometry import CrossSection
from .modes import lead_modes


@dataclass(frozen=True)
class SliceModes:
    """Modal basis of one z-uniform slice: E-columns W, H-columns V, constants beta."""

    W: np.ndarray
    V: np.ndarray
    beta: np.ndarray


def build_fg(
    ops: EpsOperators, basis: ModeBasis, k0: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the symmetric F and G operators for one slice.

    With ops.m** = None this is the physical mu = 1 problem.  Under ASR the
    mu~ operators carry the smooth metric of the coordinate map (asr.py); they
    are symmetric Gram matrices, so F = F^T and G = G^T are preserved and with
    them the structural reciprocity/unitarity of the cascade.
    """
    ezz_inv = np.linalg.inv(ops.ezz)
    Ix = np.eye(basis.X.size)
    Iy = np.eye(basis.Y.size)
    myy = Ix * k0 if ops.myy is None else k0 * ops.myy
    mxx = Iy * k0 if ops.mxx is None else k0 * ops.mxx

    a11 = myy + basis.dx_ZX @ ezz_inv @ basis.dx_XZ / k0
    a12 = basis.dx_ZX @ ezz_inv @ basis.dy_YZ / k0
    a21 = basis.dy_ZY @ ezz_inv @ basis.dx_XZ / k0
    a22 = mxx + basis.dy_ZY @ ezz_inv @ basis.dy_YZ / k0
    F = 1j * np.block([[a11, a12], [a21, a22]])

    if ops.mzz is None:
        g11 = k0 * ops.exx + basis.dy_WX @ basis.dy_XW / k0
        g12 = -basis.dy_WX @ basis.dx_YW / k0
        g21 = -basis.dx_WY @ basis.dy_XW / k0
        g22 = k0 * ops.eyy + basis.dx_WY @ basis.dx_YW / k0
    else:
        mzz_inv = np.linalg.inv(ops.mzz)
        g11 = k0 * ops.exx + basis.dy_WX @ mzz_inv @ basis.dy_XW / k0
        g12 = -basis.dy_WX @ mzz_inv @ basis.dx_YW / k0
        g21 = -basis.dx_WY @ mzz_inv @ basis.dy_XW / k0
        g22 = k0 * ops.eyy + basis.dx_WY @ mzz_inv @ basis.dx_YW / k0
    G = 1j * np.block([[g11, g12], [g21, g22]])
    return F, G


def _forward_branch(beta: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    """Select the forward branch: Im beta >= 0 (decay in +z), Re beta > 0 for real beta.

    Roundoff-level negative imaginary parts are tolerated (flipping them would flip
    the propagation phase of an essentially-propagating mode for no stability gain;
    the cascaded S-matrix is invariant under the labeling either way).
    """
    scale = np.max(np.abs(beta))
    flip = np.imag(beta) < -rel_tol * scale
    out = np.where(flip, -beta, beta)
    return out


def solve_slice(
    layout: CrossSection,
    basis: ModeBasis,
    k0: complex,
    factorization: str = "li",
    ops: EpsOperators | None = None,
) -> SliceModes:
    """Modal decomposition of one z-uniform slice at (complex) wavenumber k0.

    A caller-provided `ops` always takes the numerical path -- under ASR even
    a uniform layout carries nontrivial metric operators, so the analytic
    shortcut is only valid when we assemble the operators ourselves.
    """
    if ops is None:
        if layout.is_uniform:
            lead = lead_modes(basis, k0, layout.uniform_eps)
            return SliceModes(lead.W, lead.V, lead.beta)
        ops = build_eps_operators(layout, basis, factorization)
    F, G = build_fg(ops, basis, k0)
    lam, W = sla.eig(F @ G)  # LAPACK zgeev; keep outside any jitted code
    beta = _forward_branch(np.sqrt(-lam + 0j))
    if np.any(np.abs(beta) < 1e-14 * np.max(np.abs(beta))):
        raise ArithmeticError(
            "slice mode exactly at cutoff (beta = 0); perturb frequency or geometry"
        )
    V = (G @ W) / (1j * beta)[None, :]
    return SliceModes(W, V, beta)
