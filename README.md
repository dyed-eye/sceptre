# SCEPTRE

**S**-matrix **C**ascading **E**igenmode **P**ropagation **T**hrough **R**ectangular
**E**nclosures — a Python implementation of the **Fourier modal method (FMM) / mode
matching** for closed rectangular waveguides with PEC walls containing finite
dielectric obstacles, with numerically stable **S-matrix cascading**, complex-frequency
**pole/zero location**, and **exceptional-point (EP) search**.

> Naming note: "FMM" here means *Fourier modal method* (a.k.a. rigorous
> coupled-wave analysis in its periodic incarnation) — **not** the fast multipole
> method.

## Method summary

- **Geometry.** Rectangular guide, cross-section a x b, PEC walls, axis z. Obstacles
  are axis-aligned boxes of piecewise-constant complex permittivity ε (μ = 1, loss
  allowed as Im ε > 0), staircase-sliced into z-uniform segments — plus a level-set
  **Shape layer** (`Cylinder`, subclassable) that carries exact staircases AND the
  boundary normal fields needed by the tensor factorizations.
- **Basis.** Vector modes of the empty guide built from sin/cos functions that satisfy
  the PEC conditions exactly (the image/mirror extension of the Fourier basis). The
  formulation is fully vectorial: TE and TM families couple through the obstacle.
- **Per slice.** The transverse Maxwell operator is assembled in the mode basis with
  the **correct Fourier factorization**: Li's inverse rule applied along the normal of
  each dielectric edge, adapted to the sin/cos basis (strip-wise crossed-grating rule;
  see `src/sceptre/fourier.py`). The dense complex eigenproblem is solved with LAPACK
  `zgeev` (`scipy.linalg.eig`), giving the propagation constants and modal profiles of
  the loaded slice. Uniform slices bypass the eigensolver and use analytic modes.
- **Interfaces.** Modal continuity yields interface S-matrices; segments are combined
  with the numerically **stable S-matrix recursion** (Li 1996 / Redheffer star
  product) — never T-matrices, so evanescent exponentials never grow.
- **Outputs.** Full modal S(ω) between guided-mode ports; analytic continuation to
  complex ω; **Cauchy contour-integral pole/zero counting** (argument principle +
  Delves–Lyness moments) with **Newton refinement**; 2-D parameter-space **EP search**
  that solves the double-root system h = ∂h/∂ω = 0 for h = 1/det S, confirmed by the
  **Puiseux** square-root splitting fit.
- **ASR for high contrast.** `Solver(..., asr=AsrConfig())` enables Granet-style
  adaptive spatial resolution, implemented through its transformation-optics
  equivalent (smooth coordinate map = anisotropic eps~/mu~ on the undeformed box,
  so the symmetric operator structure and Li's rules carry over untouched).
  Measured on an eps = 80 ceramic block: ASR at N = 24 beats plain Li at N = 80
  (~37x cheaper eigensolve at equal accuracy); FEM-verified to 0.13%.  The solver
  emits a recommendation warning when the permittivity contrast exceeds ~25 and
  ASR is off.
- **NVF for high-contrast curved shapes.** `Solver(..., factorization="nvf")`
  applies the inverse rule along a windowed boundary NORMAL field (Popov–Nevière
  fast Fourier factorization, symmetrized so S = Sᵀ and unitarity stay structural).
  Measured on the eps = 80 benchmark disk: the resonance line lands within
  **±5 MHz of the FEM/VNA reference at N = 20 in one solve**, where plain Li is
  +92 MHz at N = 24 and needs an N-ladder + Richardson extrapolation (~40× more
  compute) for the same accuracy. A KFJ subpixel-smoothing factorization
  (`"kfj"`) ships alongside for cross-method comparison, with its measured
  high-contrast limitation documented.

Conventions (time convention, normalization, port definitions): `refs/CONVENTIONS.md`.
Validation status: analytic cases at machine precision; unitarity/reciprocity at
1e-10..1e-14; live COMSOL 6.1 FEM cross-checks at 0.6–0.7% (ε = 9, plain Li) and
0.13% (ε = 80, ASR). The full validation log (`VALIDATION.md`) is a local artifact,
deliberately kept out of version control.

## Install / run

```bash
uv sync                     # install (Python >= 3.11, numpy/scipy/numba)
uv run pytest -q            # full test suite
uv run python examples/01_empty_guide.py
uv run python examples/02_slab.py
uv run python examples/03_ep_hunt.py
```

## Quick start

```python
import numpy as np
from sceptre import Box, Structure, Solver, Waveguide

wg = Waveguide(a=0.02286, b=0.01016)          # WR-90, dimensions in meters
block = Box(x1=0, x2=wg.a, y1=0, y2=0.45*wg.b,
            z1=0, z2=0.008, eps=9.0)          # partial-height dielectric block
solver = Solver(Structure(wg, [block]), M=1, N=24,  # M, N: modal truncation
                factorization="li")                 # orders along x and y

res = solver.smatrix(10e9)                     # S at 10 GHz
s21 = res.coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0))
print(abs(s21))

# complex-frequency resonance hunt:
from sceptre.poles import find_zeros_poles
det = lambda f: solver.det_port_s(f, np.array([0]))     # TE10 port block
found = find_zeros_poles(det, center=10e9 - 0.3e9j, half_width=2e9)
print(found.summary())                         # zeros & poles of det S
```

## Performance essentials

The single biggest lever, worth ~10× on a typical desktop: **cap the BLAS
thread count** (~4 on OpenBLAS at these matrix sizes — measured: a 312×312
inversion takes 217 ms at 16 threads, 10 ms at 4). Env vars only work BEFORE
numpy first loads (a silent no-op afterwards); the runtime API works anytime:

```python
import sceptre
sceptre.set_blas_threads(sceptre.recommended_blas_threads())   # anytime
# or per solver:  Solver(..., blas_threads=4)
# or, zero-dependency, BEFORE numpy loads (e.g. in sweep workers):
#   import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
```

Measured per-frequency costs after this fix (same 16-core laptop, 4 threads):

| case | truncation | s/point |
|---|---|---|
| ε = 6.4 twisted-bar pair (3 z-segments) | M = N = 12 | 0.9 |
| ε = 80 staircased disk | M = N = 24 | 10.6 |
| ε = 80 staircased disk | M = N = 32 | 50 |
| ε = 80 staircased disk | M = N = 40 | ~200 |

If solves feel 10× slower than this, check the thread setting before blaming the
method.

Second lever, worth another ~3× when it applies: structures mirror-symmetric
about x = a/2 (centred disks and staircases) can pass
`Solver(..., symmetry="x")` — the modal problem splits into two independent
half-size parity sectors, one per port polarization, with bit-identical S
(1e-13). The ε = 80 disk above drops to 3.9 s/pt at N = 24 and 15 s/pt at
N = 32. Third lever, when the shape is smooth and the contrast high:
`factorization="nvf"` replaces the whole N-ladder with one N ≈ 20 solve.

## Sanity checks that catch real bugs

- **Column energy**: for a lossless structure every column of the propagating
  `port_smatrix()` must sum to 1 to ~1e-6. This single check caught a real ASR
  bug — it is the difference between "converging" and "wrong".
- **Reciprocity is structural** (S = Sᵀ to ~1e-13); if it's off, you mislabeled
  ports or modes, not physics.
- **Symmetry controls**: an untwisted bar pair must give exactly zero cross-pol;
  a mirrored geometry must equal the enantiomer transform M·S·M, M = diag(1,−1).
  These cost one solve each and validate the whole geometry/mode plumbing.

## Documentation

Full docs (absolute links so they work from PyPI):

| | |
|---|---|
| [Method](https://github.com/dyed-eye/sceptre/blob/main/docs/method.md) | basis, eigenproblem, factorization, cascading, conventions, poles |
| [Geometry](https://github.com/dyed-eye/sceptre/blob/main/docs/geometry.md) | boxes, staircases, the Shape/Cylinder level-set layer |
| [Factorizations](https://github.com/dyed-eye/sceptre/blob/main/docs/factorizations.md) | li / direct / ASR / NVF / KFJ + the measured use-case matrix |
| [Performance](https://github.com/dyed-eye/sceptre/blob/main/docs/performance.md) | BLAS threads, parallel sweeps, symmetry, N-ladders, memory |
| [Comparisons](https://github.com/dyed-eye/sceptre/blob/main/docs/comparisons.md) | cross-solver/measurement protocol, pole-based resonance extraction |
| [API](https://github.com/dyed-eye/sceptre/blob/main/docs/api.md) | every public symbol |

Start at [docs/index.md](https://github.com/dyed-eye/sceptre/blob/main/docs/index.md).

## Package layout

```
src/sceptre/
  geometry.py     boxes, shapes-aware z-slicing, cross-section layouts
  shapes.py       level-set Shape base + Cylinder (staircase + normal field)
  basis.py        sin/cos component spaces, exact derivative operators
  fourier.py      overlap matrices; Li vs direct (Laurent) eps operators
  nvf.py          normal-vector-field factorization (high-contrast curves)
  kfj.py          KFJ subpixel smoothing (comparison tool; see docs)
  modes.py        analytic TE/TM lead modes (ports), flux normalization
  slicesolver.py  per-slice symmetric F/G operators (incl. eps_xy), zgeev
  symmetry.py     x-mirror parity sectorization (symmetry="x")
  smatrix.py      interface/propagation S-matrices, Redheffer star cascade
  solver.py       top-level Solver / SResult API, factorization dispatch
  threads.py      runtime BLAS thread control (threadpoolctl)
  poles.py        contour pole/zero finder (argument principle + moments + Newton)
  ep.py           EP Newton (double-root system) + Puiseux confirmation
  asr.py          adaptive spatial resolution: maps, quadrature Grams, eps~/mu~ ops
  asr_modes.py    numerical port modes of the ASR-transformed leads
  comsol/         COMSOL detection, MPh driver, Java model generator, comparison
```

## References

1. L. Li, *Use of Fourier series in the analysis of discontinuous periodic
   structures*, JOSA A **13**, 1870 (1996) — factorization rules.
2. L. Li, *Formulation and comparison of two recursive matrix algorithms for modeling
   layered diffraction gratings*, JOSA A **13**, 1024 (1996) — stable S-recursion.
3. L. Li, *New formulation of the Fourier modal method for crossed surface-relief
   gratings*, JOSA A **14**, 2758 (1997) — 2-D (crossed) factorization.
4. E. Silberstein, P. Lalanne, J.-P. Hugonin, Q. Cao, *Use of grating theories in
   integrated optics*, JOSA A **18**, 2865 (2001) — FMM for aperiodic waveguides.
5. M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, JOSA A **12**, 1068
   (1995) — baseline RCWA eigenproblem structure.
6. D. A. Bykov, L. L. Doskolovich, J. Lightwave Technol. **31**, 793 (2013),
   arXiv:1206.3388 — S-matrix pole calculation recipes.
7. W. R. Sweeney, C. W. Hsu, A. D. Stone, PRA **102**, 063511 (2020),
   arXiv:1909.04017 — S-matrix zeros and RSM operators.
8. G. Granet, JOSA A **16**, 2510 (1999); T. Vallius & M. Honkanen, Opt. Express
   **10**, 24 (2002) — adaptive spatial resolution; A. J. Ward & J. B. Pendry,
   J. Mod. Opt. **43**, 773 (1996) — coordinate transforms as materials.
9. E. Popov & M. Nevière, JOSA A **17**, 1773 (2000); T. Schuster et al.,
   JOSA A **24**, 2880 (2007) — fast Fourier factorization along normal
   vector fields (the NVF factorization).
10. A. F. Oskooi, C. Kottke, S. G. Johnson, Opt. Lett. **34**, 2778 (2009);
    C. Kottke, A. Farjadpour, S. G. Johnson, PRE **77**, 036611 (2008) —
    anisotropic subpixel smoothing (the KFJ factorization; see
    docs/factorizations.md for why it does not transfer to spectral bases).
11. T. Itoh (ed.), *Numerical Techniques for Microwave and Millimeter-Wave
    Passive Structures*, Wiley (1989) — closed-guide mode matching,
    overlap/normalization conventions.

## License

MIT (see [LICENSE](https://github.com/dyed-eye/sceptre/blob/main/LICENSE)).
Clean-room implementation: structural inspiration
(not code) from grcwa/torcwa; EP methodology cross-checked against the EasterEig
approach; no code ported from GPL-encumbered packages (S4, RETICOLO).
