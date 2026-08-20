"""Exceptional-point (EP) search: drive two S-matrix poles to coalescence.

An EP of a two-resonance system is a point in parameter space where two poles of
S(omega) (zeros of h = 1/det S) merge into one DOUBLE zero.  Resolving an
ever-shrinking pole splitting is numerically hopeless near coalescence, so the
search never separates the pair.  Instead it solves the double-root conditions

    h(p1, p2, omega) = 0   and   dh/domega (p1, p2, omega) = 0

with a NESTED Newton iteration that exploits the local factorization
h ~ c (omega - omega_1)(omega - omega_2):

  inner:  for fixed p, Newton on h'(omega) = 0 converges to the saddle
          omega_c = (omega_1 + omega_2)/2 -- exactly, since h' is locally linear;
  outer:  g(p) = h(p, omega_c(p)) = -c (delta/2)^2 vanishes iff the splitting
          delta does; Newton on the 2x2 real system g(p) = 0 in (p1, p2).

The saddle value also yields the splitting estimate  |delta| = 2 sqrt(2 |g / h''|),
which is the convergence measure (no pole pair is ever resolved explicitly).

Confirmation is the Puiseux (square-root branch) behaviour: perturbing the
parameters by t along a generic direction splits the double pole as

    omega_pm(t) = omega_EP +- c sqrt(t) + O(t),

so log|omega_+ - omega_-| vs log t has slope 1/2.  (Same structure as the
eigenvalue perturbation theory of defective matrices used by e.g. EasterEig.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .poles import find_zeros_poles


@dataclass(frozen=True)
class EPResult:
    p: tuple[float, float]  # (p1, p2) at the EP
    omega: complex  # degenerate pole position (saddle omega_c at the final p)
    converged: bool
    iterations: int
    splitting: float  # final |omega_+ - omega_-| estimate from the quadratic model
    history: list[tuple[float, float, complex, float]]  # (p1, p2, omega_c, splitting)


def _saddle(
    h_w: Callable[[complex], complex],
    w0: complex,
    dw: float,
    tol: float,
    wander: float,
    max_iter: int = 60,
):
    """Inner Newton: solve h'(omega) = 0 from w0.  Returns (omega_c, h_c, h'', ok).

    Derivatives by central finite differences with step dw.  For the locally
    quadratic h of a near-degenerate pair this converges in a couple of steps
    (h' is linear, h'' is constant).  h has many other critical points (saddles
    between distant zeros, structure near its own poles); the wander limit keeps
    the iteration attached to the tracked pair -- exceeding it reports failure
    instead of silently converging to an unrelated feature.
    """
    w = complex(w0)
    hpp = None
    for _ in range(max_iter):
        h0 = h_w(w)
        h_plus, h_minus = h_w(w + dw), h_w(w - dw)
        hp_val = (h_plus - h_minus) / (2 * dw)
        hpp = (h_plus - 2 * h0 + h_minus) / dw**2
        if hpp == 0:
            return w, h0, hpp, False
        step = hp_val / hpp
        if abs(step) > 0.5 * wander:
            step *= 0.5 * wander / abs(step)
        w = w - step
        if abs(w - w0) > wander:
            return w, h_w(w), hpp, False
        if abs(step) < tol:
            break
    return w, h_w(w), hpp, True


def find_ep(
    h: Callable[[float, float, complex], complex],
    p0: tuple[float, float],
    omega0: complex,
    scales: tuple[float, float, float],
    *,
    tol_split_rel: float = 1e-7,
    max_iter: int = 40,
    fd_rel: float = 1e-4,
    max_step: float = 0.25,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    wander_rel: float = 0.15,
) -> EPResult:
    """Nested-Newton EP search for h(p1, p2, omega) (see module docstring).

    scales = (s_p1, s_p2, s_omega): characteristic magnitudes normalizing the
    unknowns.  Convergence: splitting estimate < tol_split_rel * s_omega.
    fd_rel: relative step of the outer finite-difference Jacobian; max_step caps
    each normalized parameter step; bounds = ((lo1, hi1), (lo2, hi2)) clips the
    physical parameters (e.g. keep the loss positive); wander_rel * s_omega is
    how far the inner saddle iteration may stray from its warm start.
    """
    s1, s2, sw = scales
    dw = 1e-5 * sw  # inner FD step in omega
    tol_w = 1e-9 * sw
    tol_split = tol_split_rel * sw
    wander = wander_rel * sw

    def clip(u_vec: np.ndarray) -> np.ndarray:
        if bounds is None:
            return u_vec
        lo = np.array([bounds[0][0] / s1, bounds[1][0] / s2])
        hi = np.array([bounds[0][1] / s1, bounds[1][1] / s2])
        return np.clip(u_vec, lo, hi)

    def eval_g(p1: float, p2: float, w_start: complex):
        w_c, h_c, hpp, ok = _saddle(lambda w: h(p1, p2, w), w_start, dw, tol_w, wander)
        split = 2.0 * np.sqrt(2.0 * abs(h_c / hpp)) if (ok and hpp) else np.inf
        return w_c, h_c, split, ok

    u = clip(np.array([p0[0] / s1, p0[1] / s2]))
    history: list[tuple[float, float, complex, float]] = []
    converged = False
    it = 0
    w_c, g_val, split, ok = eval_g(u[0] * s1, u[1] * s2, complex(omega0))
    if not ok:
        raise RuntimeError(
            "inner saddle iteration failed at the starting point; "
            "provide a closer (p0, omega0)"
        )
    for it in range(1, max_iter + 1):
        if split < tol_split:
            converged = True
            break
        # Outer FD Jacobian of (Re g, Im g) w.r.t. normalized (p1, p2).
        J = np.zeros((2, 2))
        jac_ok = True
        for j in range(2):
            du = np.zeros(2)
            du[j] = fd_rel
            u_p, u_m = u + du, u - du
            _, gp, _, ok_p = eval_g(u_p[0] * s1, u_p[1] * s2, w_c)
            _, gm, _, ok_m = eval_g(u_m[0] * s1, u_m[1] * s2, w_c)
            jac_ok = jac_ok and ok_p and ok_m
            J[0, j] = (gp.real - gm.real) / (2 * fd_rel)
            J[1, j] = (gp.imag - gm.imag) / (2 * fd_rel)
        if not jac_ok:
            break
        try:
            step = np.linalg.solve(J, -np.array([g_val.real, g_val.imag]))
        except np.linalg.LinAlgError:
            break
        norm = float(np.linalg.norm(step))
        if norm > max_step:
            step *= max_step / norm
        # Backtracking on |g|; a step that cannot decrease |g| ends the search.
        improved = False
        for _ in range(8):
            u_try = clip(u + step)
            w_try, g_try, split_try, ok_try = eval_g(u_try[0] * s1, u_try[1] * s2, w_c)
            if ok_try and abs(g_try) < abs(g_val):
                u, w_c, g_val, split = u_try, w_try, g_try, split_try
                improved = True
                break
            step *= 0.5
        history.append((u[0] * s1, u[1] * s2, w_c, split))
        if not improved:
            converged = split < tol_split
            break

    return EPResult(
        p=(u[0] * s1, u[1] * s2),
        omega=w_c,
        converged=converged or split < tol_split,
        iterations=it,
        splitting=float(split),
        history=history,
    )


def pole_pair(
    f: Callable[[complex], complex],
    center: complex,
    half_width: float,
    rel_tol: float = 1e-12,
    **finder_kwargs,
) -> tuple[complex, complex]:
    """The two poles of f inside a box around `center` (for Puiseux scans).

    f is det S (poles of f are the resonances).  Raises if not exactly two poles
    are resolved -- callers should choose the box so it isolates the pair.
    Extra keyword arguments are forwarded to find_zeros_poles (e.g. a coarse
    min_cell: the cluster moments + Newton polish resolve the pair, the contour
    subdivision does not need to).
    """
    finder_kwargs.setdefault("cluster_max", 2)
    res = find_zeros_poles(f, center, half_width, rel_tol=rel_tol, **finder_kwargs)
    if len(res.poles) != 2:
        raise RuntimeError(
            f"expected an isolated pole pair near {center:.6g}, found "
            f"{len(res.poles)} poles (splitting below resolution or box too small)"
        )
    return res.poles[0], res.poles[1]


@dataclass(frozen=True)
class PuiseuxFit:
    exponent: float  # fitted slope of log|split| vs log t  (0.5 at a clean EP)
    coefficient: complex  # |c| in omega_pm = omega_EP +- c sqrt(t)
    t_values: np.ndarray
    splittings: np.ndarray  # complex omega_+ - omega_- for each t


def puiseux_fit(t_values, splittings) -> PuiseuxFit:
    """Fit |split| ~ 2|c| t^alpha; a clean EP gives alpha = 1/2.

    The sign of each splitting is gauge (root-labeling) dependent; only magnitudes
    enter the exponent fit.  The coefficient uses the median of |split|/(2 sqrt t).
    """
    t = np.asarray(t_values, dtype=float)
    s = np.asarray(splittings, dtype=complex)
    alpha, _ = np.polyfit(np.log(t), np.log(np.abs(s)), 1)
    c = 0.5 * np.median(np.abs(s) / np.sqrt(t))
    return PuiseuxFit(float(alpha), complex(c), t, s)
