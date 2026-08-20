"""Numerically stable S-matrix cascading (never T-matrices).

Implements the scattering-matrix recursion of

* L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling
  layered diffraction gratings," J. Opt. Soc. Am. A 13, 1024 (1996),

in Redheffer star-product form.  Stability: every propagation factor that appears is
X = exp(i beta d) with Im beta >= 0, so |X| <= 1 -- growing exponentials never enter,
unlike the T-matrix recursion which is exponentially unstable for evanescent modes.

Block convention: b = S a with a = [a1 (incident +z at the left); a2 (incident -z at
the right)], b = [b1 (outgoing -z left); b2 (outgoing +z right)]:

    b1 = s11 a1 + s12 a2
    b2 = s21 a1 + s22 a2
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np


@dataclass(frozen=True)
class SMatrix:
    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray

    @property
    def n(self) -> int:
        return self.s11.shape[0]

    def full(self) -> np.ndarray:
        """Assembled 2n x 2n matrix [[s11, s12], [s21, s22]]."""
        return np.block([[self.s11, self.s12], [self.s21, self.s22]])


def identity_smatrix(n: int) -> SMatrix:
    z = np.zeros((n, n), dtype=complex)
    i = np.eye(n, dtype=complex)
    return SMatrix(z, i, i, z.copy())


def interface_smatrix(
    W1: np.ndarray, V1: np.ndarray, W2: np.ndarray, V2: np.ndarray
) -> SMatrix:
    """S-matrix of the interface between modal bases (W1, V1) and (W2, V2).

    Continuity of e and v across the interface:
        W1 (c1+ + c1-) = W2 (c2+ + c2-),   V1 (c1+ - c1-) = V2 (c2+ - c2-).
    With M = W1^-1 W2 and N = V1^-1 V2, eliminating gives the standard result.
    """
    M = np.linalg.solve(W1, W2)
    N = np.linalg.solve(V1, V2)
    MpN = M + N
    MmN = M - N
    inv_MpN = np.linalg.inv(MpN)
    s11 = MmN @ inv_MpN
    s21 = 2.0 * inv_MpN
    s12 = 0.5 * (MpN - MmN @ inv_MpN @ MmN)
    s22 = -inv_MpN @ MmN
    return SMatrix(s11, s12, s21, s22)


def propagation_smatrix(beta: np.ndarray, d: float) -> SMatrix:
    """S-matrix of homogeneous propagation over length d.

    On the real-frequency axis every |entry| <= 1 (Im beta >= 0).  Under
    complex-frequency continuation the open channels legitimately grow
    (Im beta < 0 on the resonance sheet); guard against runaway continuation.
    """
    growth = float(np.max(-np.imag(beta) * d, initial=0.0))
    if growth > 200.0:
        raise ArithmeticError(
            f"propagation factor exp({growth:.0f}) overflows: the complex-frequency "
            "continuation is too deep below the real axis for this structure"
        )
    x = np.exp(1j * beta * d)
    n = len(beta)
    z = np.zeros((n, n), dtype=complex)
    return SMatrix(z, np.diag(x), np.diag(x), z.copy())


def redheffer(A: SMatrix, B: SMatrix) -> SMatrix:
    """Star product: S of (A followed by B)."""
    n = A.n
    eye = np.eye(n, dtype=complex)
    inv1 = np.linalg.inv(eye - B.s11 @ A.s22)
    inv2 = np.linalg.inv(eye - A.s22 @ B.s11)
    s11 = A.s11 + A.s12 @ inv1 @ B.s11 @ A.s21
    s12 = A.s12 @ inv1 @ B.s12
    s21 = B.s21 @ inv2 @ A.s21
    s22 = B.s22 + B.s21 @ inv2 @ A.s22 @ B.s12
    return SMatrix(s11, s12, s21, s22)


def cascade(smatrices: list[SMatrix]) -> SMatrix:
    if not smatrices:
        raise ValueError("nothing to cascade")
    return reduce(redheffer, smatrices)
