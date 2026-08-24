# SCEPTRE documentation

**S-matrix Cascading Eigenmode Propagation Through Rectangular Enclosures** —
a Fourier modal method (FMM / mode matching) solver for closed rectangular
PEC waveguides containing dielectric obstacles, with numerically stable
S-matrix cascading and complex-frequency pole/zero location.

| Page | What it covers |
|---|---|
| [method.md](method.md) | The numerics: modal basis, eigenproblem, Li factorization, cascading, conventions, poles |
| [geometry.md](geometry.md) | Boxes, staircasing curved shapes, the `Shape`/`Cylinder` level-set layer |
| [factorizations.md](factorizations.md) | `li` / `direct` / ASR / NVF / KFJ — what each does and **when to use which** (the use-case matrix) |
| [performance.md](performance.md) | BLAS threads (the 10× lever), parallel sweeps, `symmetry="x"`, N-ladders, memory budgets |
| [inverse-design.md](inverse-design.md) | Graded (continuously varying) ε: `Structure.from_segments`, `CrossSection`, why smooth maps converge at low N, per-candidate cost |
| [comparisons.md](comparisons.md) | Comparing against other solvers/measurements without fooling yourself; pole-based resonance extraction |
| [api.md](api.md) | Every public symbol with signatures and examples |

## Quickstart

```python
import sceptre

# A high-permittivity ceramic disk in a 32 mm square guide, touching a wall:
wg = sceptre.Waveguide(0.032, 0.032)
disk = sceptre.Cylinder(cx=0.016, cy=0.017, r=0.015, z1=0.0, z2=0.005,
                        eps=80.0 + 0j)
struct = sceptre.Structure(wg, shapes=[disk])

# High contrast + curved boundary -> NVF factorization (see factorizations.md)
solver = sceptre.Solver(struct, M=20, N=20, factorization="nvf",
                        symmetry="x",          # disk is x-centred: ~3x faster
                        blas_threads=sceptre.recommended_blas_threads())

res = solver.smatrix(5.44e9)
t_te01 = res.coeff(2, ("TE", 0, 1), 1, ("TE", 0, 1))   # TE01 transmission
```

Measured on the ε=80 benchmark disk: this configuration reads the resonance
line within **±5 MHz of the reference at N=20 in one solve**, where the
scalar `li` factorization needs an N=24/32/40 ladder plus Richardson
extrapolation for the same accuracy (details and all numbers in
[factorizations.md](factorizations.md)).

For box (rectangular) obstacles the default `factorization="li"` is exact-
rule and structurally unitary — start there:

```python
box = sceptre.Box(0.008, 0.024, 0.0, 0.016, 0.0, 0.006, eps=6.0)
solver = sceptre.Solver(sceptre.Structure(wg, [box]), M=12, N=12)
```

## Three habits that save days

1. **Set BLAS threads.** OpenBLAS at default thread counts is 10–20× slower
   on these matrix sizes. `blas_threads=sceptre.recommended_blas_threads()`
   or the env-var pattern for sweep workers — [performance.md](performance.md).
2. **Never quote a high-contrast number from a single truncation.** Run an
   N-ladder or use NVF; [performance.md](performance.md) shows the protocol.
3. **Extract resonances from complex-frequency poles, not peak fits** —
   [comparisons.md](comparisons.md).
