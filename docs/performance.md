# Performance: threads, sweeps, symmetry, N-ladders

## BLAS threads — the 10× lever

The cascade algebra is dominated by LAPACK calls, and OpenBLAS at its
default thread count **thrashes** on the matrix sizes SCEPTRE produces
(measured on a 16-core laptop: a 312×312 inversion takes 217 ms at 16
threads, 10 ms at 4; n = 1200: 517 vs 196 ms).

Two ways to set the thread count, by situation:

**1. At runtime (works anytime — notebooks, after numpy is loaded):**

```python
import sceptre
sceptre.set_blas_threads(sceptre.recommended_blas_threads())   # process-wide
# or scoped per solver:
solver = sceptre.Solver(struct, 20, 20, blas_threads=4)
```

Built on threadpoolctl, which talks to the loaded BLAS directly.
`recommended_blas_threads()` returns min(4, cpu_count) — the measured knee
on 16-core OpenBLAS at n ≈ 300–2000. Other BLAS builds (MKL, Accelerate)
may knee elsewhere: sweep once on your machine (time one `smatrix` call at
1/2/4/8 threads) and trust your numbers over the default.

**Not thread-safe:** the limits are process-global; Python threads running
solves concurrently with different limits race. Process-based parallelism
(the sweep pattern below) is unaffected.

**2. Environment variables (zero-dependency; ONLY works before numpy loads):**

```python
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # FIRST, before numpy
import numpy as np
```

Setting these after numpy imports is a **silent no-op** — the classic
footgun the runtime API exists to remove. Inside sweep worker processes,
either pattern works (set the env at the top of the worker module, or call
`set_blas_threads` in the worker init).

## Parallel frequency sweeps

Frequency points are independent and per-layout operators are cached inside
a `Solver`, so sweeps pay only eigensolves. The pattern that works:

- **One `Solver` per worker process, contiguous frequency chunks** — each
  worker builds its Solver once, amortizing operator assembly.
- **workers × blas_threads ≤ cores.** Oversubscription is not a mild
  penalty here (see above).
- **Budget RAM per worker:** ~1 GB at T ≈ 1200 (M = N = 24), ~2.1 GB at
  M = N = 40, ~3.5 GB at M = N = 56 (T = 2MN + M + N is the cascade matrix
  dimension). On a 16-core/20 GB machine: 4 workers × 4 threads at N ≤ 32,
  3 workers at N = 40, 2–3 at N = 56.

```python
# sweep.py — worker must live in an importable module (spawn pickling)
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
        return [x for chunk in pool.map(solve_chunk, jobs) for x in chunk]
```

## `symmetry="x"` — free 3× for mirror-symmetric structures

If every cross-section is mirror-symmetric about x = a/2 (centred disks and
staircases qualify), pass `Solver(..., symmetry="x")`: the modal problem
splits into two independent half-size parity sectors (TE10 excitation in
one, TE01 in the other). Measured ~2.7–3.2× per point at N = 24–32,
approaching 4× as N grows; S identical to the plain path at 1e-13. The
solver *checks* the layout and raises on anything asymmetric beyond
roundoff. Works at complex frequency and with every factorization except
ASR; linspace-level roundoff in staircase edges is tolerated.

## Per-point costs (measured, 16-core laptop, 4 BLAS threads)

| case | truncation | s/point |
|---|---|---|
| ε = 6.4 twisted-bar pair (3 segments) | M = N = 12 | 0.9 |
| ε = 80 staircased disk, `li` | M = N = 24 | 10.6 (3.9 with `symmetry="x"`) |
| ε = 80 staircased disk, `li` | M = N = 32 | 50 (15 with symmetry) |
| ε = 80 staircased disk, `li` | M = N = 40 | ~200 (74 with symmetry) |
| ε = 80 disk, `nvf` | M = N = 20 | ~3 (and no ladder needed) |

## Trust N only after a ladder — and expect mode-dependent rates

Never quote high-contrast `li` numbers from a single truncation. Run
N ∈ {24, 32, 40} and watch the observable you care about; extrapolate with
**two-point p = 1 Richardson on the two highest rungs**
(f∞ = f₂ − (f₁ − f₂)·(1/n₂)/(1/n₁ − 1/n₂)). Measured behaviors worth
knowing:

- Line positions converge from ABOVE (~1/N): benchmark disk
  +214/+145/+112/+92 MHz at N = 12/16/20/24; Richardson over N = 32/40
  landed within +1.8 MHz of the reference.
- Fitting the exponent p from three rungs amplifies grid noise into
  tens-of-MHz errors — don't; two-point p = 1 is the protocol.
- Different modes of the same structure converge at different rates
  (corner-hugging modes are slowest), and outside the asymptotic regime
  Richardson silently fails: check that rung-to-rung ratios are consistent
  with 1/N before extrapolating.
- With `factorization="nvf"` on smooth shapes, the ladder collapses: the
  line is N-stable by N ≈ 20 ([factorizations.md](factorizations.md)).
