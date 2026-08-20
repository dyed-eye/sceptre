"""Numerical lead (port) modes of the ASR-transformed uniform guide.

Under ASR the uniform lead is filled with the metric-anisotropic materials of
asr.py, so its modes are no longer single sin/cos harmonics.  They are obtained
from the same eig(FG) problem as any slice and then

1. matched to the analytic mode set by FIELD-PATTERN OVERLAP against the
   quadrature-projected analytic modes (eigenvalue proximity is unreliable in
   the deep-evanescent tail, where truncation shifts exceed cluster spacing),
2. within each exactly degenerate cluster (TE_mn/TM_mn pairs) aligned to the
   projected patterns by least squares inside the numerical eigenspace (so the
   aligned vectors remain true eigenvectors but acquire TE/TM identity),
3. flux-orthonormalized in the symmetric bilinear form B(w1, w2) =
   w1^T G w2 / (i beta) and scaled to unit pseudo-flux, which keeps S = S^T
   and lossless unitarity exact.

Normalization note: the unknowns are the transformed fields E~_u = X' E_x,
E~_v = Y' E_y (asr.py), for which the plain coefficient flux e^T v equals the
physical pseudo-flux (dx dy = X'Y' du dv absorbs exactly the two metric
factors) -- the ASR S-matrix is the physical S-matrix, no gauge involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

from .asr import AsrMap1D
from .basis import ModeBasis
from .fourier import EpsOperators
from .modes import LeadModes, mode_table
from .slicesolver import _forward_branch, build_fg

_OVERLAP_MIN = 0.7  # propagating modes must match their analytic pattern this well


@dataclass(frozen=True)
class LeadPatterns:
    """k0-independent analytic mode patterns projected onto the u,v basis."""

    labels: list[tuple[str, int, int]]
    kc2: np.ndarray
    P: np.ndarray  # (T, T) unnormalized [ex; ey] pattern columns


def _projection(
    kind: str, mmax: int, length: float, amap: AsrMap1D, with_metric: bool
) -> np.ndarray:
    """P[m_phys, k] = integral of [X'(u)] trig(m_phys pi X(u)/L) basisfun_k(u) du.

    with_metric multiplies the physical pattern by X'(u): the transformed
    transverse components are E~_u = X' E_x and E~_v = Y' E_y, so each pattern
    carries the metric factor of ITS OWN direction only.
    """
    m_lo = 0 if kind == "cos" else 1
    m_phys = np.arange(m_lo, mmax + 1)
    size = len(m_phys)
    out = np.zeros((size, size))
    for u1, u2 in zip(amap.breaks[:-1], amap.breaks[1:]):
        nq = int(3.2 * mmax * (u2 - u1) / length) + 24
        nodes, weights = np.polynomial.legendre.leggauss(nq)
        u = 0.5 * (u2 - u1) * nodes + 0.5 * (u1 + u2)
        wq = 0.5 * (u2 - u1) * weights
        xu = amap.x(u)
        arg = np.outer(m_phys, np.pi * xu / length)
        phys = np.cos(arg) if kind == "cos" else np.sin(arg)
        if with_metric:
            phys = phys * amap.dx(u)
        basis_rows = np.sqrt(2.0 / length) * (
            np.cos(np.outer(m_phys, np.pi * u / length))
            if kind == "cos"
            else np.sin(np.outer(m_phys, np.pi * u / length))
        )
        if kind == "cos":
            basis_rows[0] = np.sqrt(1.0 / length)
        out += (phys * wq) @ basis_rows.T
    return out


def lead_patterns(basis: ModeBasis, xmap: AsrMap1D, ymap: AsrMap1D) -> LeadPatterns:
    a, b = basis.a, basis.b
    # E~_u patterns: cos-type in u WITH the X' metric, sin-type in v without;
    # E~_v patterns: sin-type in u without, cos-type in v WITH the Y' metric.
    pc_x = _projection("cos", basis.M, a, xmap, with_metric=True)
    ps_x = _projection("sin", basis.M, a, xmap, with_metric=False)
    pc_y = _projection("cos", basis.N, b, ymap, with_metric=True)
    ps_y = _projection("sin", basis.N, b, ymap, with_metric=False)

    table = mode_table(basis)
    T = basis.size_t
    P = np.zeros((T, T), dtype=complex)
    labels = []
    kc2 = np.zeros(T)
    for col, (kc2_val, kind, m, n) in enumerate(table):
        kx, ky = m * np.pi / a, n * np.pi / b
        ex = np.zeros((basis.M + 1, basis.N))  # X-grid (m_u 0..M, n_v 1..N)
        ey = np.zeros((basis.M, basis.N + 1))  # Y-grid (m_u 1..M, n_v 0..N)
        if kind == "TE":
            if n >= 1:
                ex = ky * np.outer(pc_x[m], ps_y[n - 1])
            if m >= 1:
                ey = -kx * np.outer(ps_x[m - 1], pc_y[n])
        else:
            ex = kx * np.outer(pc_x[m], ps_y[n - 1])
            ey = ky * np.outer(ps_x[m - 1], pc_y[n])
        P[: basis.X.size, col] = ex.ravel()
        P[basis.X.size :, col] = ey.ravel()
        labels.append((kind, m, n))
        kc2[col] = kc2_val
    return LeadPatterns(labels, kc2, P)


def asr_lead_modes(
    basis: ModeBasis,
    k0: complex,
    eps: complex,
    ops: EpsOperators,
    patterns: LeadPatterns,
) -> LeadModes:
    """Numerical modes of the transformed uniform lead, labeled and normalized."""
    T = basis.size_t
    F, G = build_fg(ops, basis, k0)
    lam, Wn = sla.eig(F @ G)

    # Match numerical eigenvectors to analytic modes by PATTERN OVERLAP, not by
    # eigenvalue proximity: in the deep-evanescent tail the truncation shift of
    # an eigenvalue routinely exceeds the spacing between clusters, so nearest
    # eigenvalue steals columns from neighbors, while the field-shape overlap
    # still ranks the right column first.  (Degenerate TE/TM pairs may come out
    # as arbitrary mixes here -- the cluster alignment below untangles them.)
    p_hat = patterns.P / np.linalg.norm(patterns.P, axis=0, keepdims=True)
    w_hat = Wn / np.linalg.norm(Wn, axis=0, keepdims=True)
    quality = np.abs(p_hat.conj().T @ w_hat)  # (analytic, numerical)
    # Globally optimal bipartite assignment (Hungarian); greedy claiming can in
    # principle mislabel an evanescent mode whose best candidate was taken by a
    # higher-quality competing pair.
    a_idx, n_idx = linear_sum_assignment(-quality)
    match = np.full(T, -1)
    match[a_idx] = n_idx

    # Exactly degenerate clusters (identical kc2: the TE_mn/TM_mn pairs).
    clusters: dict[float, list[int]] = {}
    for ai, kc2_val in enumerate(patterns.kc2):
        clusters.setdefault(float(kc2_val), []).append(ai)

    W = np.zeros((T, T), dtype=complex)
    V = np.zeros((T, T), dtype=complex)
    beta = np.zeros(T, dtype=complex)
    for members in clusters.values():
        cols = [match[ai] for ai in members]
        lam_c = np.mean(lam[cols])
        beta_c = complex(_forward_branch(np.sqrt(np.atleast_1d(-lam_c + 0j)))[0])
        sub = Wn[:, cols]
        if len(members) == 1:
            aligned = [sub[:, 0]]
        else:
            # Align to analytic patterns INSIDE the numerical eigenspace: the
            # least-squares combination of degenerate eigenvectors stays an
            # eigenvector but acquires the analytic TE/TM identity.
            aligned = []
            for ai in members:
                coef, *_ = np.linalg.lstsq(sub, patterns.P[:, ai], rcond=None)
                aligned.append(sub @ coef)
        # Flux Gram-Schmidt in the symmetric form B(w1, w2) = w1^T G w2 / (i beta):
        # G = G^T makes B symmetric, so one-sided orthogonalization suffices.
        done: list[tuple[np.ndarray, np.ndarray, complex]] = []
        for ai, w in zip(members, aligned):
            gw = G @ w
            for w_prev, gw_prev, b_prev in done:
                w = w - w_prev * (w_prev @ gw) / b_prev
                gw = G @ w
            flux = (w @ gw) / (1j * beta_c)
            # G/beta is O(1) in k0 units, so flux/|w|^2 is an O(1) dimensionless
            # quality measure; vanishing means the cluster basis is defective.
            if abs(flux) < 1e-12 * float(np.linalg.norm(w)) ** 2:
                raise RuntimeError(
                    "degenerate ASR lead cluster with vanishing pseudo-flux; "
                    "increase the truncation order"
                )
            done.append((w, gw, complex(w @ gw)))
            scale = 1.0 / np.sqrt(flux)
            W[:, ai] = scale * w
            V[:, ai] = scale * gw / (1j * beta_c)
            beta[ai] = beta_c

    _check_propagating_overlap(basis, k0, eps, patterns, W, beta)
    return LeadModes(
        basis,
        complex(eps),
        complex(k0),
        W,
        V,
        beta,
        list(patterns.labels),
        patterns.kc2.copy(),
    )


def _check_propagating_overlap(basis, k0, eps, patterns, W, beta) -> None:
    """Propagating port modes must clearly match their analytic pattern."""
    prop = np.abs(beta.imag) <= 1e-6 * np.abs(beta)
    for i in np.flatnonzero(prop):
        p = patterns.P[:, i]
        w = W[:, i]
        quality = abs(np.vdot(p, w)) / (np.linalg.norm(p) * np.linalg.norm(w))
        if quality < _OVERLAP_MIN:
            raise RuntimeError(
                f"ASR lead mode {patterns.labels[i]} matches its analytic pattern "
                f"with overlap {quality:.2f} < {_OVERLAP_MIN}; increase M/N"
            )
