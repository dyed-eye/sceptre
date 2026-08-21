# Usage guide — sweeps, staircases, high contrast, comparisons

Field notes from validating SCEPTRE against a real COMSOL research campaign
(chiral cavity mirrors at ε = 80, twisted-bar molecules at ε = 6.4, measured VNA
data). The performance essentials and sanity checks live in the README; this file
covers the rest:

1. [Parallel frequency sweeps](#1-parallel-frequency-sweeps)
2. [Shapes that aren't boxes](#2-shapes-that-arent-boxes-staircase-them-and-dont-over-refine)
3. [High contrast: ASR vs plain Li](#3-high-contrast-ε--25-asr-helps-blocks-plain-li-handles-staircases)
4. [Trusting a truncation: the N-ladder](#4-trust-n-only-after-a-ladder--and-expect-mode-dependent-rates)
5. [Comparing against another solver or a measurement](#5-comparing-against-another-solver-or-a-measurement)
6. [Resonances: poles, not peak fitting](#6-resonances-complex-frequency-poles-not-peak-fitting)
7. [Mirror-symmetric structures: `symmetry="x"`](#7-mirror-symmetric-structures-symmetryx)

The audit that produced these recipes (with runnable scripts for each) lives in
`reports/comsol_parity/` in the author's working tree — session artifacts, not
part of the repo. Everything needed to apply the recipes is inlined below.

## 1. Parallel frequency sweeps

Frequency points are independent, and operators are cached inside a `Solver`, so
sweeps pay only eigensolves. The pattern that works:

- **One `Solver` per worker, contiguous frequency chunks.** Each process builds
  its own `Solver` once and loops over a slice of the frequency list, which
  amortizes construction and keeps the operator cache warm.
- **`workers × blas_threads ≤ cores`.** Oversubscription is not a mild penalty
  here — LAPACK thread-thrashing can cost 10× (see the README's thread advice).
- **Budget ~1 GB of RAM per worker** for a modal dimension around T ≈ 1200
  (T = 2MN + M + N, the size of the cascade matrices; M = N = 24 gives T = 1200).
  On a 16-core / 20 GB machine, 4 workers × 4 threads is the sweet spot; more
  workers OOM-killed our pools.

Skeleton (the worker must live in an importable module so `spawn` can pickle it,
and the thread env must be set before numpy loads in each process):

```python
# sweep.py
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # BEFORE numpy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sceptre import Solver, Structure, Waveguide

def solve_chunk(job):
    solver = Solver(Structure(Waveguide(*job["ab"]), job["boxes"]),
                    M=job["N"], N=job["N"])
    return [(f, solver.smatrix(f)) for f in job["freqs"]]

def sweep(boxes, ab, n, freqs, workers=4):
    jobs = [dict(boxes=boxes, ab=ab, N=n, freqs=chunk)
            for chunk in np.array_split(freqs, workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [item for chunk in pool.map(solve_chunk, jobs) for item in chunk]
```

## 2. Shapes that aren't boxes: staircase them, and don't over-refine

Any convex cross-section (disk, notched disk, rotated bar) staircases into one
`Box` per x-strip, with the exact y-interval evaluated at the strip midpoint:

```python
import numpy as np
from sceptre import Box

def strip_boxes(y_interval, x_lo, x_hi, k, z1, z2, eps):
    """One Box per x-strip; y_interval(x_mid) returns (y1, y2) or None."""
    xs = np.linspace(x_lo, x_hi, k + 1)
    boxes = []
    for x1, x2 in zip(xs[:-1], xs[1:]):
        iv = y_interval(0.5 * (x1 + x2))
        if iv is not None and iv[1] - iv[0] > 1e-9:
            boxes.append(Box(x1, x2, iv[0], iv[1], z1, z2, eps))
    return boxes

# a disk of radius r centred at (cx, cy), thickness h along z:
def disk_interval(xm):
    d2 = r**2 - (xm - cx) ** 2
    if d2 <= 0:
        return None
    half = np.sqrt(d2)
    return (cy - half, cy + half)

disk = strip_boxes(disk_interval, cx - r, cx + r, k=32, z1=0.0, z2=h, eps=80.0)
```

For a rotated rectangle, the same pattern applies — a vertical line through a
convex shape is always a single interval, computable exactly in the bar frame.

- **K ≈ 32 strips is already geometry-converged** for wavelength-scale shapes:
  doubling to K = 64 changed resonance positions by *zero* and |S| by ~3e-4 in
  our tests. Remaining error is modal truncation (N), not the staircase.
  Refine N, not K.
- **For tolerance / Monte-Carlo studies, fix the pixel grid to the guide**, not
  to the part. Re-fitting strips to each perturbed shape makes the
  discretization jump discontinuously between samples and masquerade as fake
  sensitivity. Acceptance test: re-solve the nominal with the grid shifted half
  a pixel — the change must be far below the smallest sensitivity you claim.

## 3. High contrast (ε ≳ 25): ASR helps blocks, plain Li handles staircases

ASR's coordinate map compresses resolution at dielectric edges — great for a few
isolated edges (the validated ε = 80 block: N = 24 with ASR ≈ N = 80 without).
On a dense staircase the map used to oscillate beyond the basis bandwidth and
produced *unphysical* S (energy columns up to 3.5×). The solver now thins edges
closer than the basis can resolve and warns when it does.

**Read that warning as: "ASR is running safely but with few compression points
here — expect roughly plain-Li convergence, and consider plain Li with a larger
N instead."** Plain Li is structurally unitary at any truncation, so it can
never lie about energy, which makes it the safe default for heavily staircased
ε = 80 shapes.

## 4. Trust N only after a ladder — and expect mode-dependent rates

Never quote high-contrast numbers from a single truncation. Run N ∈ {24, 32, 40}
and watch the *observable you care about*. Two behaviors we measured on an
ε = 80 notched disk: the resonance line converges smoothly (+49 → +29 → +24 MHz
vs the reference; Richardson extrapolation over the last two rungs landed within
4 MHz), while a second mode whose field hugs the sharp corners converged visibly
slower — different modes of the same structure have different N requirements.
Two ladder rungs plus Richardson extrapolation of a line position is often
cheaper and more accurate than one brute-force huge-N solve.

## 5. Comparing against another solver or a measurement

Every mismatch we ever chased came down to conventions or comparison protocol,
never the physics. In order of likelihood:

1. **Time convention.** SCEPTRE is e^{−iωt}; COMSOL, VNAs, and most engineering
   tools are e^{+jωt}. Conjugate their S-data (this also swaps the handedness
   labels: RCP ↔ LCP, right-/left-circular polarization).
2. **Reference planes.** SCEPTRE references S at the obstacle faces with zero
   lead length. De-embed external data through its leads (÷ e^{2iβL}) before
   comparing phases. Magnitudes need none of this.
3. **Per-mode sign gauge.** A relative sign on a mode (e.g. TE01) between two
   codes is normalization, not error. It flips cross-terms and swaps enantiomer
   labels (T_RR ↔ T_LL, the co-polarized circular transmissions). Determine it
   once per dataset by trying both and keeping the match; diagonal elements and
   |S| are gauge-invariant.
4. **Never compare single frequencies near a resonance.** A converged-but-
   shifted line (see §4) makes fixed-frequency S-elements disagree wildly while
   the physics agrees. Real case: COMSOL |S31|/|S41| = 0.68/0.64 vs SCEPTRE
   0.89/0.39 at the same frequency looked like a broken setup — evaluated
   18 MHz up the flank (the measured line offset), SCEPTRE read 0.68/0.63.
   **Score spectrally, at each code's own resonance line**; treat the common
   line shift as a separate, single number that converges away with N.
   Extrapolating a fixed-frequency flank value over N conflates the two and
   wildly overestimates the N you need.

## 6. Resonances: complex-frequency poles, not peak fitting

Q factors and linewidths from `det_port_s` + `find_zeros_poles` beat FWHM fits
of |S(f)|: no frequency grid fine enough to resolve a narrow line is ever
needed, and there are no interpolation systematics. When resonances overlap,
peak fitting can also report mode *spacing* as mode *splitting*; that mistake
forced the campaign we validated against to retract its extracted couplings.
Keep contours away from lead-mode cutoffs (branch points of S(ω)); seed from a
cheap real-axis scan.

## 7. Mirror-symmetric structures: `symmetry="x"`

If every cross-section is mirror-symmetric about x = a/2 (a centred disk or
staircase qualifies; a rotated bar does not), pass `Solver(..., symmetry="x")`.
The modal problem block-diagonalizes by parity of the x Fourier index, and the
solver runs the two half-size sectors independently — measured ~3× per point at
N = 24–32 on an ε = 80 staircased disk, identical S to the plain path at 1e-13.
TE10 and TE01 excitations live in opposite sectors, so both port polarizations
are still returned.

- The solver *checks* the discretized layout and raises on anything asymmetric
  beyond roundoff — an off-centre box is a hard error, not a silent projection.
- Works for real and complex frequencies (pole hunts included) and both
  factorizations. Not supported together with ASR.
- The staircase helpers in §2 produce exactly mirror-symmetric layouts when the
  shape is centred; linspace-level roundoff in the edges is tolerated.
