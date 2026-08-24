# Inverse design with band-limited permittivity

## The idea

Instead of optimising *geometry* parameters — positions, widths, tilt angles of
sharp dielectric bodies — make the **Fourier coefficients of ε(x, y) the design
variables**. The permittivity map is then smooth by construction, and the device
is realised as a graded medium (printed infill density, a porous or composite
fill) rather than as a machined solid.

For FMM this is not cosmetic. Truncation error is governed by how fast the
Fourier coefficients of ε decay: a sharp or staircased edge decays like 1/n, so
error dies slowly in N; a smooth map decays fast, so a low truncation is already
converged. Band-limiting ε **removes the dominant error source instead of paying
truncation to resolve it** — which is what makes a search over thousands of
candidates affordable.

## Measured: the payoff tracks the realised edge width

Same footprint, same peak ε = 7, same 35° tilt, ε on a 96 × 96 cell grid in a
32 mm guide; only the transition width of the profile varies. Error is each
structure's distance from **its own** N = 20 answer, so resonance drift cannot
contaminate the comparison.

| ε profile | N = 8 | N = 12 | N = 16 |
|---|---|---|---|
| edge 7.0 mm | 3.7e-4 | 7.3e-5 | 1.7e-5 |
| edge 2.4 mm | 4.4e-3 | 8.5e-4 | 1.8e-4 |
| edge 0.84 mm | 1.6e-2 | 4.9e-3 | 1.6e-3 |
| sharp (staircased boxes) | 3.2e-2 | 1.3e-2 | 5.9e-3 |

Monotonic in edge width, exactly as Fourier decay predicts. **A 7 mm-edge design
at N = 8 is ~90× more accurate than the sharp one at N = 8, and still beats it at
N = 16.** In a loop running 10²–10³ candidates that difference is the entire
compute budget.

Note what the table is really saying: the gain is set by the **edge width the ε
map actually has**, not by the nominal harmonic cutoff you designed with. A
band-limited potential pushed through a steep nonlinearity (a high-gain logistic,
or any thresholding step) is no longer band-limited — measured on a random
KH = 3 map squashed by a steep logistic, the advantage over a staircase fell to
~1.5×. If you use a positivity map, keep its gain low enough that the transition
stays several cells wide.

## This is not "smoothing a sharp structure"

Worth stating explicitly, because it looks superficially like a refuted idea.
Artificially smoothing a sharp boundary to help convergence **does not work in
FMM** — the spectral basis resolves the smoothing layer as genuine structure, and
both anisotropic (KFJ) and scalar variants were measured and rejected here
(`LEDGER.md`, [factorizations.md](factorizations.md)).

Harmonic-space design is different in kind: you change the *device*, not the
model of the device. The fabricated object really is graded, so nothing is being
approximated away and there is no layer to mis-resolve. The two live at opposite
ends of the same axis — `nvf` exists for bodies that must stay sharp; this page
is for devices that are legitimately smooth.

## Expressing a graded ε map

`Structure` derives its segments from boxes and shapes. To supply an explicit ε
grid, build `CrossSection`/`Segment` directly and pass them through a small
adapter — `Solver` only requires `.waveguide`, `.background`, `.segments()`, plus
`.boxes`/`.shapes` (read only by the tensor factorizations and the symmetry
check):

```python
import numpy as np
from sceptre import CrossSection, Segment, Solver, Waveguide

class GridStructure:
    """Minimal structure adapter for an explicit permittivity grid."""
    def __init__(self, waveguide, segments, background=1.0 + 0.0j):
        self.waveguide, self.background = waveguide, background
        self.boxes, self.shapes = (), ()
        self._segments = list(segments)

    def segments(self):
        return list(self._segments)

a = b = 0.032
xe, ye = np.linspace(0, a, 97), np.linspace(0, b, 97)
xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
X, Y = np.meshgrid(xm, ym, indexing="ij")

fill = 0.5 + 0.4 * np.cos(2 * np.pi * X / a) * np.cos(2 * np.pi * Y / b)  # band-limited
eps = (1.0 + (7.0 - 1.0) * fill).astype(complex)

struct = GridStructure(Waveguide(a, b), [Segment(0.0, 0.015, CrossSection(xe, ye, eps))])
s = Solver(struct, M=8, N=8, factorization="li").smatrix(5.85e9)
```

*(A first-class constructor for explicit-segment structures would be a natural
addition; today the adapter above is the supported route, and its contract is the
four attributes listed.)*

## Constraints

1. **The fabrication must actually deliver graded ε.** Printed infill, porous or
   composite fills qualify; a machined solid does not. Optimising a graded map
   you can only realise by thresholding throws away both the convergence benefit
   and the design — you are optimising an object you cannot build.
2. **Keep the band limit through the parameterisation** (see above). Check the
   realised edge width of the winning map, not the harmonic cutoff you intended.
3. **The ε grid is itself a staircase, and it is not free.** `CrossSection` is
   piecewise-constant per cell, so cells must resolve the transition, and cost
   scales with cell count. At equal N a graded map is usually *slower* than an
   equivalent box layout (measured: 1.15 s vs 0.39 s at N = 8, for 64 × 64 cells
   against 2063 boxes). **The win comes entirely from needing a lower N**, so
   only claim it after the truncation check below.
4. **The band limit doubles as the minimum-feature constraint.** A cutoff at
   harmonic index KH gives a half-period of a/(2·KH) — 5.3 mm at KH = 3 in a
   32 mm guide. Manufacturability and fast convergence are bought with the same
   knob, which is the main practical elegance of the method. Lower KH also makes
   a design likelier to be single-mode, if the objective needs that.
5. **Homogenisation must hold, and the mixing law is an approximation.** The
   composite unit cell (nozzle, pore, lattice pitch) must sit far below both the
   guide wavelength and the smallest design feature. Mapping ε to a fill fraction
   (e.g. Lichtenecker, f = ln ε / ln ε_solid) has an exponent that depends on
   lattice geometry and print orientation — calibrate it against a witness sample
   before trusting absolute numbers.
6. **Loss is fill-coupled: do not optimise lossless.** High-ε regions are the
   high-material, high-absorption regions, so loss is a function of the design
   variables rather than a uniform background. Driven, steady-state figures of
   merit carry a T/(T+A)² factor and can collapse — or change sign — between
   tan δ = 0 and 10⁻⁴, while mode quantities such as level splittings are far
   more loss-tolerant. A lossless optimum is not merely optimistic; it can be a
   *different* optimum. Loss belongs in the objective, not in a post-hoc check.
7. **Optimise the ensemble, not the nominal point.** Nominal optima of graded
   designs are frequently traps. Score each candidate on a fixed (deterministic,
   not resampled) manufacturing ensemble — material permittivity spread, frozen
   ε-map perturbations, loss — and take the worst member. Two failure modes to
   avoid: refining on a cheap sub-ensemble that omits the binding perturbation
   (that is not worst-case optimisation), and scoring unscoreable members as
   zero, which flattens the objective to zero gradient everywhere. Make coverage
   the integer part of the merit and worst-case quality the fractional part.
8. **Verify the cheap truncation on the winner.** Low N is the point of the
   method, and optimising a biased surrogate is the matching risk. Re-evaluate
   the winning design at N = 8/12/16/20 and require the objective to be flat
   (fractions of a percent). If it is still drifting, the search ran below
   convergence — see the N-ladder protocol in [performance.md](performance.md).
9. **Save designs, and re-evaluate what you saved.** Write the top-N design
   vectors on every improvement, not at the end, and re-run each saved design at
   the settings it was found at, requiring it to reproduce its reported
   objective. Reported numbers whose stored design cannot reproduce them are a
   recurring and cheap-to-catch failure.
10. **Factorization and symmetry interactions.** Graded maps use `li` (or
    `direct` for benchmarking); `nvf`/`kfj` require `Shape` geometry and are
    unnecessary here, since smooth ε is precisely the case Li already handles
    well. `symmetry="x"` works — it validates every segment's cross-section for
    mirror symmetry — but the ε map must be symmetric to machine precision, so
    build symmetry into the parameterisation rather than hoping the optimiser
    finds it.

## Where the objective belongs

One structural warning, independent of SCEPTRE: optimise the quantity the
experiment measures, in the configuration it will be measured in. A bare-device
figure of merit can anti-correlate with the loaded-system performance you care
about, and a contrast-style objective can be maximised by *degrading* the rest of
the system — making two channels small makes their ratio large. Whenever a figure
of merit is a ratio or a difference, pair it with an absolute-scale term, and
check what the empty or reference system already achieves before crediting the
design with the result.

*Provenance: the method was demonstrated at campaign scale in
`C:\emae\phys\chiral_cavities\harmonic_optimization_report.md` (10³ candidates,
COMSOL cross-check to 2.0 %); the convergence and cost numbers on this page were
measured independently on this codebase.*
