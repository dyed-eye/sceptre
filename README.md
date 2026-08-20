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
  allowed as Im ε > 0), staircase-sliced into z-uniform segments.
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

Conventions (time convention, normalization, port definitions): `refs/CONVENTIONS.md`.
Validation report incl. the COMSOL cross-check: `VALIDATION.md`.

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
solver = Solver(Structure(wg, [block]), M=1, N=24, factorization="li")

res = solver.smatrix(10e9)                     # S at 10 GHz
s21 = res.coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0))
print(abs(s21))

# complex-frequency resonance hunt:
from sceptre.poles import find_zeros_poles
det = lambda f: solver.det_port_s(f, np.array([0]))     # TE10 port block
found = find_zeros_poles(det, center=10e9 - 0.3e9j, half_width=2e9)
print(found.summary())                         # zeros & poles of det S
```

## Package layout

```
src/sceptre/
  geometry.py     boxes, z-slicing, cross-section layouts
  basis.py        sin/cos component spaces, exact derivative operators
  fourier.py      overlap matrices; Li vs direct (Laurent) eps operators
  modes.py        analytic TE/TM lead modes (ports), flux normalization
  slicesolver.py  per-slice symmetric F/G operators, zgeev eigenproblem
  smatrix.py      interface/propagation S-matrices, Redheffer star cascade
  solver.py       top-level Solver / SResult API
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
9. T. Itoh (ed.), *Numerical Techniques for Microwave and Millimeter-Wave Passive
   Structures*, Wiley (1989) — closed-guide mode matching, overlap/normalization
   conventions.

## License

MIT (see [LICENSE](LICENSE)). Clean-room implementation: structural inspiration
(not code) from grcwa/torcwa; EP methodology cross-checked against the EasterEig
approach; no code ported from GPL-encumbered packages (S4, RETICOLO).
