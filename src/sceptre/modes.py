"""Analytic TE/TM eigenmodes of the uniform (empty or homogeneously filled) guide.

These serve two purposes:
  * exact port modes for the scattering matrix (phase reference at the structure faces),
  * exact modal solution of any z-uniform segment with uniform permittivity
    (bypasses the numerical eigensolver, giving machine-precision empty-guide and
    uniform-slab results).

Conventions: see refs/CONVENTIONS.md.  Fields are stored as coefficient vectors in the
orthonormal sin/cos basis of a ModeBasis.  The transverse E-vector is e = [Ex; Ey]
(X-space block then Y-space block); the transverse H-vector is stored as

    v = [H~y; -H~x]

so that its blocks live in the SAME spaces as e (H~y shares Ex's cos*sin space, H~x
shares Ey's sin*cos space) and the pseudo-flux is a plain dot product:

    integral (e x h~) . z dA = e_x . h~_y - e_y . h~_x = e^T v.

Forward modes are normalized to e^T v = 1 (no conjugation), which makes reciprocity
read S = S^T.  For a uniform section the impedance relation collapses to v = zeta * e
with zeta = beta/k0 (TE) or eps*k0/beta (TM).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis


@dataclass(frozen=True)
class LeadModes:
    """Complete forward-mode set of a uniform guide section at wavenumber k0."""

    basis: ModeBasis
    eps: complex
    k0: complex
    W: np.ndarray  # (T, T) transverse-E coefficient columns e = [ex; ey]
    V: np.ndarray  # (T, T) transverse-H~ coefficient columns v = [hy; -hx]
    beta: np.ndarray  # (T,) propagation constants, Im beta >= 0 branch
    labels: list[tuple[str, int, int]]  # ("TE"|"TM", m, n) per column
    kc2: np.ndarray  # (T,) cutoff wavenumbers squared

    def mode_index(self, kind: str, m: int, n: int) -> int:
        return self.labels.index((kind, m, n))

    def propagating(self, rel_tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of propagating modes (real beta, up to rel_tol)."""
        return np.abs(np.imag(self.beta)) <= rel_tol * np.abs(self.beta)


def lead_modes(basis: ModeBasis, k0: complex, eps: complex = 1.0 + 0.0j) -> LeadModes:
    a, b, M, N = basis.a, basis.b, basis.M, basis.N
    T = basis.size_t

    # 1-D normalization factors of the orthonormal basis functions.
    def nc(m: int, length: float) -> float:
        return np.sqrt((1.0 if m == 0 else 2.0) / length)

    def ns(m: int, length: float) -> float:
        return np.sqrt(2.0 / length)

    entries = []  # (kc2, sort_type, kind, m, n)
    for m in range(0, M + 1):
        for n in range(0, N + 1):
            if m == 0 and n == 0:
                continue
            entries.append(((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2, 0, "TE", m, n))
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            entries.append(((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2, 1, "TM", m, n))
    entries.sort(key=lambda e: (e[0], e[1], e[3], e[4]))
    if len(entries) != T:
        raise AssertionError("mode count must equal transverse basis dimension")

    W = np.zeros((T, T), dtype=complex)
    V = np.zeros((T, T), dtype=complex)
    beta = np.zeros(T, dtype=complex)
    kc2 = np.zeros(T)
    labels = []

    for col, (kc2_val, _, kind, m, n) in enumerate(entries):
        kx, ky = m * np.pi / a, n * np.pi / b
        ex = np.zeros(basis.X.size, dtype=complex)
        ey = np.zeros(basis.Y.size, dtype=complex)
        if kind == "TE":
            # Hz ~ cos(kx x) cos(ky y);  Et ~ (-dHz/dy, +dHz/dx)
            if n >= 1:
                ex[basis.X.index(m, n)] = ky / (nc(m, a) * ns(n, b))
            if m >= 1:
                ey[basis.Y.index(m, n)] = -kx / (ns(m, a) * nc(n, b))
        else:
            # Ez ~ sin(kx x) sin(ky y);  Et ~ (dEz/dx, dEz/dy)
            ex[basis.X.index(m, n)] = kx / (nc(m, a) * ns(n, b))
            ey[basis.Y.index(m, n)] = ky / (ns(m, a) * nc(n, b))

        # Branch choice per channel (defines the resonance sheet, refs/CONVENTIONS.md):
        #  * OPEN channels (Re disc >= 0): principal sqrt(disc).  Analytic across the
        #    real-omega axis into Im omega < 0, where it acquires Im beta < 0 --
        #    the outgoing/growing continuation a resonance pole requires.  Never
        #    "fix" the sign: forcing Im beta >= 0 at Im omega < 0 silently switches
        #    the sheet and mirrors every pole into the upper half plane.
        #  * CLOSED channels (Re disc < 0): i*sqrt(-disc).  The argument -disc has
        #    positive real part, far from the principal cut, so the mode stays on
        #    its decaying branch (Im beta > 0) under continuation -- with a plain
        #    principal sqrt(disc) the argument would hug the cut along the negative
        #    real axis and jump to the exponentially growing side for Im omega < 0.
        # Regions containing a lead cutoff (disc ~ 0) are genuine branch points of
        # S and must be excluded from contour searches regardless.
        disc = eps * k0**2 - kc2_val + 0j
        bta = np.sqrt(disc) if disc.real >= 0 else 1j * np.sqrt(-disc)
        # Dimensionless admittance zeta: v = zeta * e  (see module docstring).
        zeta = bta / k0 if kind == "TE" else eps * k0 / bta
        # Pseudo-flux e^T v = zeta * (ex.ex + ey.ey); scale so it equals 1.
        scale = 1.0 / np.sqrt(zeta * (ex @ ex + ey @ ey))
        ex, ey = scale * ex, scale * ey

        W[: basis.X.size, col] = ex
        W[basis.X.size :, col] = ey
        V[:, col] = zeta * W[:, col]
        beta[col] = bta
        kc2[col] = kc2_val
        labels.append((kind, m, n))

    return LeadModes(basis, complex(eps), complex(k0), W, V, beta, labels, kc2)
