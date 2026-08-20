"""Pole and zero location for analytic functions via Cauchy contour integration.

Strategy (following the contour-integral recipes of Bykov & Doskolovich for
scattering-matrix poles, combined with Delves-Lyness moments):

1. Argument principle on a rectangle: the winding number of f along the boundary
   counts zeros minus poles inside, N = Z - P.  Each edge is sampled adaptively so
   the phase increment per sub-interval stays below pi/2 (and the magnitude jump
   below a factor ~e), which makes the branch tracking of log f unambiguous.
2. Adaptive quadrilateral subdivision until each cell contains at most a small
   cluster of same-type points (|winding| <= cluster_max).  A cell with winding 0 is
   discarded only when its first moment is also negligible: a hidden zero-pole PAIR
   has winding 0 but first moment z_zero - z_pole != 0 (exact residue identity), so
   it triggers subdivision instead of silent loss.
3. Delves-Lyness moments  m_k = (1/2 pi i) oint z^k f'(z)/f(z) dz  give the power
   sums of the enclosed locations; Newton's identities turn them into a polynomial
   whose roots seed step 4.
4. Newton refinement on f (zeros) or 1/f (poles) with a central finite-difference
   derivative, to ~rel_tol * scale accuracy.

References:
* D. A. Bykov, L. L. Doskolovich, "Numerical methods for calculating poles of the
  scattering matrix with applications in grating theory," J. Lightwave Technol. 31,
  793 (2013); arXiv:1206.3388.
* L. M. Delves, J. N. Lyness, "A numerical method for locating the zeros of an
  analytic function," Math. Comp. 21, 543 (1967).
* W. R. Sweeney, C. W. Hsu, A. D. Stone, "Theory of reflectionless scattering modes,"
  Phys. Rev. A 102, 063511 (2020); arXiv:1909.04017 -- S-matrix zeros as physical
  targets (f = det S carries both: its poles are resonances, its zeros are S zeros).

The integration loops walk a user-supplied Python callable, so they are plain
NumPy/Python by design; Numba is reserved for the matrix-assembly loops (fourier.py)
where it actually pays off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

_TWO_PI_I = 2j * np.pi


class ContourError(RuntimeError):
    pass


class _EdgeOnRoot(Exception):
    """A contour node landed numerically on a zero/pole; jitter the box and retry."""


@dataclass
class PoleZeroResult:
    zeros: list[complex] = field(default_factory=list)
    poles: list[complex] = field(default_factory=list)

    def summary(self) -> str:
        z = ", ".join(f"{v:.6g}" for v in self.zeros)
        p = ", ".join(f"{v:.6g}" for v in self.poles)
        return f"zeros: [{z}]  poles: [{p}]"


class _Budget:
    def __init__(self, max_evals: int):
        self.left = max_evals

    def spend(self, n: int = 1) -> None:
        self.left -= n
        if self.left < 0:
            raise ContourError("contour search exceeded its function-evaluation budget")


def _edge_moments(f, z0, z1, n_moments, budget, n_seed, cache, max_bisect=48):
    """Adaptive integration of d(log f) and z^k d(log f) along edge z0 -> z1.

    `cache` maps t in [0, 1] to f(z0 + t (z1 - z0)) and is shared across sampling
    escalations, so recomputing the edge at doubled n_seed only pays for the new
    points.  Sub-intervals are bisected until the endpoint phase step is below
    pi/4 and the magnitude step below a factor e^0.7 -- but NO finite criterion
    can prove the absence of a 2*pi wrap between two samples, which is why
    _contour_moments escalates n_seed until the winding number stabilizes.
    """

    def evaluate(t: float) -> complex:
        if t in cache:
            return cache[t]
        budget.spend()
        try:
            with np.errstate(all="ignore"):  # a node exactly at a pole is handled below
                v = f(z0 + t * (z1 - z0))
        except (ArithmeticError, ValueError) as exc:  # includes np.linalg.LinAlgError
            raise _EdgeOnRoot() from exc
        if v == 0 or not np.isfinite(v):
            raise _EdgeOnRoot()
        cache[t] = v
        return v

    ts = [i / n_seed for i in range(n_seed + 1)]
    for t in ts:
        evaluate(t)
    stack = [(ts[i], ts[i + 1], 0) for i in range(n_seed)]
    total_dlog = 0.0 + 0.0j
    moments = np.zeros(n_moments, dtype=complex)
    while stack:
        t0, t1, depth = stack.pop()
        f0, f1 = cache[t0], cache[t1]
        ratio = f1 / f0
        dphase = float(np.angle(ratio))
        dmag = abs(np.log(abs(ratio)))
        if (abs(dphase) > 0.25 * np.pi or dmag > 0.7) and depth < max_bisect:
            tm = 0.5 * (t0 + t1)
            evaluate(tm)
            stack.append((t0, tm, depth + 1))
            stack.append((tm, t1, depth + 1))
            continue
        dlog = np.log(abs(ratio)) + 1j * dphase
        total_dlog += dlog
        z_mid = z0 + 0.5 * (t0 + t1) * (z1 - z0)
        zk = z_mid
        for k in range(n_moments):
            moments[k] += zk * dlog
            zk *= z_mid
    return total_dlog, moments


@dataclass(frozen=True)
class _Rect:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def corners(self):
        return (
            complex(self.x0, self.y0),
            complex(self.x1, self.y0),
            complex(self.x1, self.y1),
            complex(self.x0, self.y1),
        )

    @property
    def diameter(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    def split(self):
        # Deliberately OFF-center split ratios: roots often sit at "nice" (dyadic)
        # coordinates, and exact-midpoint splitting would then park cell corners
        # exactly on a root, poisoning the boundary integrals of every neighbor.
        xm = self.x0 + 0.5137 * (self.x1 - self.x0)
        ym = self.y0 + 0.4863 * (self.y1 - self.y0)
        return (
            _Rect(self.x0, xm, self.y0, ym),
            _Rect(xm, self.x1, self.y0, ym),
            _Rect(self.x0, xm, ym, self.y1),
            _Rect(xm, self.x1, ym, self.y1),
        )


def _contour_moments(f, rect, n_moments, budget, max_jitter=8):
    """Winding number and Delves-Lyness moments over a rectangle boundary.

    Sampling escalation: the boundary is integrated at edge seedings 8, 16, 32, ...
    (evaluations are cached, so each escalation only pays for the new points) until
    two consecutive levels agree on the integer winding number.  This is the
    defense against phase aliasing -- a feature of f narrower than the sampling
    (e.g. a resonance peak between two samples) can wrap the phase by a full 2*pi
    that any single-level bisection criterion silently misses.
    """
    r = rect
    for attempt in range(max_jitter):
        try:
            caches = [dict() for _ in range(4)]
            prev_w = None
            for n_seed in (8, 16, 32, 64):
                total = 0.0 + 0.0j
                moments = np.zeros(n_moments, dtype=complex)
                cs = r.corners
                for cache, (za, zb) in zip(caches, zip(cs, cs[1:] + cs[:1])):
                    dlog, mom = _edge_moments(
                        f, za, zb, n_moments, budget, n_seed, cache
                    )
                    total += dlog
                    moments += mom
                winding = total.imag / (2 * np.pi)
                w = int(round(winding))
                if abs(winding - w) <= 0.25 and w == prev_w:
                    return w, moments / _TWO_PI_I, r
                prev_w = w if abs(winding - w) <= 0.25 else None
            raise _EdgeOnRoot()  # winding never stabilized: jitter and retry
        except _EdgeOnRoot:
            # Asymmetric padding so a root sitting exactly on an edge or corner
            # ends up strictly inside or outside the retried box.
            pad = (1.3e-3 * (attempt + 1)) * max(r.x1 - r.x0, r.y1 - r.y0)
            r = _Rect(r.x0 - pad, r.x1 + 0.71 * pad, r.y0 - 0.53 * pad, r.y1 + pad)
    raise ContourError("contour keeps hitting zeros/poles after repeated jitter")


def _power_sums_to_locations(power_sums: np.ndarray) -> np.ndarray:
    """Newton's identities: power sums s_1..s_k of k points -> the k points."""
    k = len(power_sums)
    e = np.zeros(k + 1, dtype=complex)  # elementary symmetric polynomials
    e[0] = 1.0
    for i in range(1, k + 1):
        acc = 0.0 + 0.0j
        for j in range(1, i + 1):
            acc += (-1) ** (j - 1) * e[i - j] * power_sums[j - 1]
        e[i] = acc / i
    coeffs = [(-1) ** i * e[i] for i in range(k + 1)]
    return np.roots(coeffs)


def _finite(v: complex) -> bool:
    return bool(np.isfinite(v.real) and np.isfinite(v.imag))


def refine_zero(
    f: Callable[[complex], complex],
    guess: complex,
    tol: float,
    max_iter: int = 80,
    max_step: float | None = None,
) -> complex:
    """Newton iteration with central-difference derivative; returns the refined zero.

    Robustness rules:
    * Non-finite evaluations terminate at the current point: when the target is a
      zero of 1/f (a pole of f), quadratic convergence eventually lands EXACTLY on
      the pole, where complex-inf arithmetic would otherwise produce NaN.
    * Evaluation exceptions (e.g. solver overflow guards fired by a wild step) are
      treated the same way.
    * Steps are clamped to max_step (default: ~the guess magnitude): a noisy
      finite-difference derivative must not catapult the iterate out of the region
      where f is even defined.
    """
    z = complex(guess)
    scale = max(abs(z), tol * 1e3)
    if max_step is None:
        max_step = 0.5 * scale

    def safe(zz: complex) -> complex | None:
        try:
            with np.errstate(all="ignore"):  # exact pole hits divide by zero benignly
                v = f(zz)
        except (ArithmeticError, ValueError):
            return None
        return v if _finite(v) else None

    for _ in range(max_iter):
        f0 = safe(z)
        if f0 is None or f0 == 0:
            break  # sitting exactly on the root (or a singular point): done
        h = 1e-7 * max(abs(z), 1e-3 * scale)
        fp, fm = safe(z + h), safe(z - h)
        if fp is None or fm is None:
            break
        d = (fp - fm) / (2 * h)
        if d == 0:
            break
        step = f0 / d
        if not _finite(step):
            break
        if abs(step) > max_step:
            step *= max_step / abs(step)
        z = z - step
        if abs(step) < tol:
            break
    return z


def refine_pole(
    f: Callable[[complex], complex],
    guess: complex,
    tol: float,
    max_iter: int = 80,
    max_step: float | None = None,
) -> complex:
    return refine_zero(lambda z: 1.0 / f(z), guess, tol, max_iter, max_step)


def _dedupe(points: list[complex], tol: float) -> list[complex]:
    out: list[complex] = []
    for p in sorted(points, key=lambda z: (z.real, z.imag)):
        if not any(abs(p - q) < tol for q in out):
            out.append(p)
    return out


def find_zeros_poles(
    f: Callable[[complex], complex],
    center: complex,
    half_width: float,
    half_height: float | None = None,
    *,
    cluster_max: int = 3,
    min_cell: float | None = None,
    rel_tol: float = 1e-12,
    max_depth: int = 24,
    max_evals: int = 400_000,
) -> PoleZeroResult:
    """Locate all zeros and poles of analytic f inside a rectangle around `center`.

    Refinement target ~ rel_tol * scale with scale = max(|center|, box size).
    Cells smaller than min_cell holding |winding| > 1 are resolved through moments
    as same-type clusters (multiple roots / near-coalescent pairs) instead of being
    subdivided forever.
    """
    if half_height is None:
        half_height = half_width
    scale = max(abs(center), half_width, half_height)
    tol = rel_tol * scale
    if min_cell is None:
        min_cell = 1e-5 * max(half_width, half_height)
    budget = _Budget(max_evals)

    root_rect = _Rect(
        center.real - half_width,
        center.real + half_width,
        center.imag - half_height,
        center.imag + half_height,
    )
    result = PoleZeroResult()
    stack = [(root_rect, 0)]
    while stack:
        rect, depth = stack.pop()
        w, moments, used_rect = _contour_moments(f, rect, cluster_max, budget)
        # Hidden zero-pole pair: winding 0 but first moment (= sum z - sum p) large.
        # The 2% threshold sits above the moment quadrature noise of root-free cells
        # (which scales with cell size); a pair separated by less than ~2% of the
        # first enclosing winding-0 cell is not detectable this way -- callers who
        # care should shrink the search box around the suspect region.
        hidden_pair = w == 0 and abs(moments[0]) > 0.02 * used_rect.diameter
        if w == 0 and not hidden_pair:
            continue
        needs_split = (
            abs(w) > cluster_max
            or hidden_pair
            or (abs(w) > 1 and used_rect.diameter > min_cell)
        )
        if needs_split and depth < max_depth and used_rect.diameter > min_cell:
            for child in used_rect.split():
                stack.append((child, depth + 1))
            continue
        if w == 0:
            continue  # unresolved suspicion at max depth; nothing safe to report
        k = abs(w)
        if k > cluster_max:
            # A min_cell-sized cell with more same-type roots than we have moments
            # for cannot be resolved -- fail loudly rather than under-report.
            raise ContourError(
                f"{k} coincident roots within min_cell={min_cell:g} exceed "
                f"cluster_max={cluster_max}; raise cluster_max or shrink min_cell"
            )
        sign = 1.0 if w > 0 else -1.0
        seeds = _power_sums_to_locations(sign * moments[:k])
        refine = refine_zero if w > 0 else refine_pole
        target = result.zeros if w > 0 else result.poles
        step_cap = max(used_rect.diameter, 1e3 * tol)
        refined = [refine(f, complex(seed), tol, max_step=step_cap) for seed in seeds]
        # Accept only roots that stayed inside (a slightly padded) cell: a seed
        # poisoned by a root just outside the boundary makes Newton escape there,
        # which would double-count it and silently drop a root of THIS cell.
        pad = 0.02 * used_rect.diameter + 100 * tol
        inside = [
            r
            for r in refined
            if used_rect.x0 - pad <= r.real <= used_rect.x1 + pad
            and used_rect.y0 - pad <= r.imag <= used_rect.y1 + pad
        ]
        distinct = _dedupe(inside, 10 * tol)
        if len(distinct) < k and depth < max_depth and used_rect.diameter > min_cell:
            # Lost or merged seeds: resolve by splitting (the off-center split
            # lines move, so the pathological configuration does not recur).
            for child in used_rect.split():
                stack.append((child, depth + 1))
            continue
        target.extend(distinct)

    result.zeros = _dedupe(result.zeros, 50 * tol)
    result.poles = _dedupe(result.poles, 50 * tol)
    return result
