"""x-mirror symmetry sectorization of the modal problem.

Under the mirror x -> a - x the 1-D basis functions transform as
s_m -> (-1)^(m+1) s_m and c_m -> (-1)^m c_m, and the field components pick up
the vector/pseudovector signs (Ex, Hy~, Hz~ flip; Ey, Ez, Hx~ keep).  The net
coefficient parity is (-1)^(m+1) in EVERY component space, so the mirror
eigensectors are simply {m odd} and {m even}.  All derivative operators
preserve m, and the eps operators of an x-symmetric layout never couple the
two classes -- the full per-frequency pipeline (slice eigenproblems,
interfaces, cascade) closes sector-by-sector at half size, ~4x cheaper.

TE10 lives in the odd-m sector, TE01 in the even-m sector: each port
polarization is solved in its own half-size problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis, Space
from .fourier import EpsOperators
from .geometry import CrossSection

# Mirrored staircase edges from floating-point constructions (linspace) match
# only to ulps; anything beyond this relative tolerance is a genuinely
# asymmetric layout, not roundoff.
_SYM_TOL = 1e-9


@dataclass(frozen=True)
class Sector:
    """Index sets of one mirror eigensector (m_odd selects the class)."""

    m_odd: bool
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    W: np.ndarray
    t: np.ndarray  # rows of the transverse vector [Ex; Ey]


def _space_indices(space: Space, m_odd: bool) -> np.ndarray:
    m, _n = space.mn()
    return np.flatnonzero((m % 2 == 1) == m_odd)


def x_sectors(basis: ModeBasis) -> tuple[Sector, Sector]:
    """The (odd-m, even-m) mirror sectors of a basis."""
    sectors = []
    for m_odd in (True, False):
        xi = _space_indices(basis.X, m_odd)
        yi = _space_indices(basis.Y, m_odd)
        t = np.concatenate([xi, basis.X.size + yi])
        sectors.append(
            Sector(
                m_odd,
                xi,
                yi,
                _space_indices(basis.Z, m_odd),
                _space_indices(basis.W, m_odd),
                t,
            )
        )
    return sectors[0], sectors[1]


def lead_columns(labels: list[tuple[str, int, int]], sector: Sector) -> np.ndarray:
    """Lead-mode columns whose x-index m belongs to the sector."""
    return np.flatnonzero([(m % 2 == 1) == sector.m_odd for _kind, m, _n in labels])


def slice_eps_ops(ops: EpsOperators, sector: Sector) -> EpsOperators:
    """Restrict the eps operators to one sector (mu~ operators unsupported: ASR).

    exy (X rows, Y cols) slices rectangularly; a mirror-commuting exy couples
    only equal m-parity classes, so the discarded cross blocks are zero.
    """
    if ops.mxx is not None or ops.myy is not None or ops.mzz is not None:
        raise ValueError("sector slicing does not support ASR metric operators")
    return EpsOperators(
        exx=ops.exx[np.ix_(sector.X, sector.X)],
        eyy=ops.eyy[np.ix_(sector.Y, sector.Y)],
        ezz=ops.ezz[np.ix_(sector.Z, sector.Z)],
        exy=None if ops.exy is None else ops.exy[np.ix_(sector.X, sector.Y)],
    )


def require_x_symmetric(layout: CrossSection, a: float) -> None:
    """Raise ValueError unless the layout is mirror-symmetric about x = a/2."""
    mirrored_edges = a - layout.x_edges[::-1]
    if not np.allclose(layout.x_edges, mirrored_edges, atol=_SYM_TOL * a, rtol=0.0):
        raise ValueError(
            "cross-section x-edges are not mirror-symmetric about a/2; "
            "symmetry='x' requires an x-symmetric structure"
        )
    eps_scale = float(np.max(np.abs(layout.eps_cells)))
    if not np.allclose(
        layout.eps_cells,
        layout.eps_cells[::-1, :],
        atol=_SYM_TOL * eps_scale,
        rtol=0.0,
    ):
        raise ValueError(
            "cross-section permittivity is not mirror-symmetric in x; "
            "symmetry='x' requires an x-symmetric structure"
        )
