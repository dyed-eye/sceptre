"""Modal basis bookkeeping for the closed PEC rectangular waveguide.

Component parities (see refs/CONVENTIONS.md):

    space X (Ex, Hy~):  cos_m(x) s_n(y),  m in 0..M, n in 1..N
    space Y (Ey, Hx~):  s_m(x) cos_n(y),  m in 1..M, n in 0..N
    space Z (Ez):       s_m(x) s_n(y),    m in 1..M, n in 1..N
    space W (Hz~):      cos_m(x) cos_n(y),m in 0..M, n in 0..N

with orthonormal 1-D functions c_0 = 1/sqrt(a), c_m = sqrt(2/a) cos(m pi x / a),
s_m = sqrt(2/a) sin(m pi x / a).  Indexing within each space is m-major.

The derivative operators below are exact in this basis and carry their signs:
    d/dx s_m = +(m pi/a) c_m        d/dx c_m = -(m pi/a) s_m   (and c_0' = 0)
(normalization factors match for m >= 1, and m = 0 never mixes, so the operators
are one-entry-per-row rectangular matrices).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Space:
    """Index bookkeeping for one component space (m-major ordering)."""

    m0: int
    m1: int
    n0: int
    n1: int

    @property
    def n_count(self) -> int:
        return self.n1 - self.n0 + 1

    @property
    def m_count(self) -> int:
        return self.m1 - self.m0 + 1

    @property
    def size(self) -> int:
        return self.m_count * self.n_count

    def index(self, m: int, n: int) -> int:
        return (m - self.m0) * self.n_count + (n - self.n0)

    def mn(self) -> tuple[np.ndarray, np.ndarray]:
        """Arrays of (m, n) for every basis function, in index order."""
        m = np.repeat(np.arange(self.m0, self.m1 + 1), self.n_count)
        n = np.tile(np.arange(self.n0, self.n1 + 1), self.m_count)
        return m, n


class ModeBasis:
    """Truncated sin/cos basis for cross-section a x b with orders M (x) and N (y)."""

    def __init__(self, a: float, b: float, M: int, N: int):
        if M < 1 or N < 1:
            raise ValueError("need M >= 1 and N >= 1")
        self.a, self.b, self.M, self.N = a, b, M, N
        self.X = Space(0, M, 1, N)  # Ex, Hy~
        self.Y = Space(1, M, 0, N)  # Ey, Hx~
        self.Z = Space(1, M, 1, N)  # Ez
        self.W = Space(0, M, 0, N)  # Hz~
        self._sealed = True  # identity is now immutable (see __setattr__)

    def __setattr__(self, name: str, value) -> None:
        # The derivative operators below are cached_property matrices derived from
        # (a, b, M, N, spaces); mutating those after a cache fill would silently
        # desynchronize them.  Seal the identity fields after __init__.
        if getattr(self, "_sealed", False) and name in (
            "a",
            "b",
            "M",
            "N",
            "X",
            "Y",
            "Z",
            "W",
            "_sealed",
        ):
            raise AttributeError(
                f"ModeBasis is immutable ({name!r} cannot be set); create a new instance"
            )
        super().__setattr__(name, value)

    @property
    def size_t(self) -> int:
        """Dimension of the transverse field vector [Ex; Ey]."""
        return self.X.size + self.Y.size

    # -- derivative operators (rectangular, one entry per matched (m, n) pair) --

    def _dmat(self, src: Space, dst: Space, kvals: np.ndarray) -> np.ndarray:
        """Matrix taking src-space coefficients to dst-space, entry kvals per (m,n)."""
        out = np.zeros((dst.size, src.size))
        m_lo, m_hi = max(src.m0, dst.m0), min(src.m1, dst.m1)
        n_lo, n_hi = max(src.n0, dst.n0), min(src.n1, dst.n1)
        for m in range(m_lo, m_hi + 1):
            for n in range(n_lo, n_hi + 1):
                out[dst.index(m, n), src.index(m, n)] = kvals[m]
        return out

    def _kx(self) -> np.ndarray:
        return np.arange(self.M + 1) * np.pi / self.a

    def _ky(self) -> np.ndarray:
        return np.arange(self.N + 1) * np.pi / self.b

    @cached_property
    def dx_XZ(self) -> np.ndarray:
        """d/dx : X (cos_m s_n) -> Z (s_m s_n), value -(m pi/a)."""
        return self._dmat(self.X, self.Z, -self._kx())

    @cached_property
    def dy_YZ(self) -> np.ndarray:
        """d/dy : Y (s_m cos_n) -> Z, value -(n pi/b)."""
        ky = -self._ky()
        out = np.zeros((self.Z.size, self.Y.size))
        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                out[self.Z.index(m, n), self.Y.index(m, n)] = ky[n]
        return out

    @cached_property
    def dx_ZX(self) -> np.ndarray:
        """d/dx : Z (s_m s_n) -> X (cos_m s_n), value +(m pi/a)."""
        return self._dmat(self.Z, self.X, self._kx())

    @cached_property
    def dy_ZY(self) -> np.ndarray:
        """d/dy : Z -> Y, value +(n pi/b)."""
        ky = self._ky()
        out = np.zeros((self.Y.size, self.Z.size))
        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                out[self.Y.index(m, n), self.Z.index(m, n)] = ky[n]
        return out

    @cached_property
    def dx_YW(self) -> np.ndarray:
        """d/dx : Y (s_m cos_n) -> W (cos_m cos_n), value +(m pi/a)."""
        return self._dmat(self.Y, self.W, self._kx())

    @cached_property
    def dy_XW(self) -> np.ndarray:
        """d/dy : X (cos_m s_n) -> W, value +(n pi/b)."""
        ky = self._ky()
        out = np.zeros((self.W.size, self.X.size))
        for m in range(0, self.M + 1):
            for n in range(1, self.N + 1):
                out[self.W.index(m, n), self.X.index(m, n)] = ky[n]
        return out

    @cached_property
    def dx_WY(self) -> np.ndarray:
        """d/dx : W (cos_m cos_n) -> Y (s_m cos_n), value -(m pi/a)."""
        return self._dmat(self.W, self.Y, -self._kx())

    @cached_property
    def dy_WX(self) -> np.ndarray:
        """d/dy : W -> X (cos_m s_n), value -(n pi/b)."""
        ky = -self._ky()
        out = np.zeros((self.X.size, self.W.size))
        for m in range(0, self.M + 1):
            for n in range(1, self.N + 1):
                out[self.X.index(m, n), self.W.index(m, n)] = ky[n]
        return out
