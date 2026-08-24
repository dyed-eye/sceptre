# Graded permittivity: explicit ε grids

Boxes and shapes describe piecewise-constant media. A continuously varying ε
(printed infill, porous or composite fill) is given to the solver as the grid
itself. Smooth maps also converge at much lower truncation than staircased ones.

## `Structure.from_segments()`

```python
import numpy as np
from sceptre import CrossSection, Segment, Solver, Structure, Waveguide

a = b = 0.032
xe = ye = np.linspace(0, a, 97)
xm, ym = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
X, Y = np.meshgrid(xm, ym, indexing="ij")
eps = 1.0 + 6.0 * (0.5 + 0.4 * np.cos(2 * np.pi * X / a) * np.cos(2 * np.pi * Y / b))

struct = Structure.from_segments(
    Waveguide(a, b), [Segment(0.0, 0.015, CrossSection(xe, ye, eps.astype(complex)))]
)
s = Solver(struct, M=8, N=8).smatrix(5.85e9)
```

Returns an ordinary `Structure`. Stack slices along z with more segments: they
must be finite, of positive extent, and contiguous in increasing z, since the
cascade walks them in order using only each length — a gap would be dropped and
an overlap counted twice. All four cases raise.

## `CrossSection`

`eps_cells[i, j]` on `[x_edges[i], x_edges[i+1]] × [y_edges[j], y_edges[j+1]]`.
Validated on construction (shape against edge counts, real finite increasing
edges, finite nonzero cells), error names the offending index; arrays are copied
and read-only. `Solver` requires each cross-section to span `[0, a] × [0, b]`.

## Convergence

Truncation error follows the Fourier decay of ε: 1/n for a staircased edge,
faster for a smooth map. Measured on one footprint (peak ε = 7, 35° tilt,
96 × 96 cells, 32 mm guide) with only the transition width varying, against each
structure's own N = 20 result:

| ε profile | N = 8 | N = 12 | N = 16 |
|---|---|---|---|
| edge 7.0 mm | 3.7e-4 | 7.3e-5 | 1.7e-5 |
| edge 2.4 mm | 4.4e-3 | 8.5e-4 | 1.8e-4 |
| edge 0.84 mm | 1.6e-2 | 4.9e-3 | 1.6e-3 |
| sharp (staircased boxes) | 3.2e-2 | 1.3e-2 | 5.9e-3 |

Convergence tracks the *realised* edge width, not the intended harmonic cutoff:
a band-limited map through a steep logistic fell back to 1.5× over a staircase.

## Cost

Rebuilding the ε operators dominates (0.32 s of a 0.32 s cold solve, 96 × 96
cells, N = 8) and scales with cell count. At equal N a graded map is slower than
an equivalent box layout (1.15 s vs 0.39 s, 64 × 64 cells vs 2063 boxes); the
saving comes only from needing lower N. Resolve the transition, no finer.

A harmonic cutoff KH sets the minimum feature at a/(2·KH) — 5.3 mm at KH = 3 in
a 32 mm guide — so printability and convergence share one parameter.

## Other options

`li` (or `direct` to benchmark). `nvf`/`kfj` need `Shape` geometry and are
rejected here; they serve the opposite case, bodies that must stay sharp —
[factorizations.md](factorizations.md). `symmetry="x"` validates every
cross-section for mirror symmetry, so build symmetry into the parameterisation.
ASR composes as usual.

## Caveats

- Fabrication must realise graded ε; a design only buildable by thresholding is
  not one of these.
- Homogenisation: unit cell (nozzle, pore, pitch) far below both wavelength and
  smallest feature. Mixing-law exponents need calibration against a sample.
- Loss tracks the design variables. Driven metrics carry a T/(T+A)² factor and
  can change sign between tan δ = 0 and 10⁻⁴; splittings are far more tolerant.
  Optimise with loss in the objective.
- Score the worst member of a fixed manufacturing ensemble, not the nominal
  point.
- Re-check the winner on an N-ladder, 8/12/16/20.
- Optimise the loaded observable; ratio objectives can be maximised by degrading
  the rest of the system.

N-ladder protocol: [performance.md](performance.md). Scoring against an external
reference or measurement: [comparisons.md](comparisons.md).
