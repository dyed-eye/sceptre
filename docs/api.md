# API reference

Everything importable from `sceptre`. Conventions throughout: SI units
(metres, Hz), e^{−iωt} time dependence, Im ε > 0 lossy.

## Geometry

### `Waveguide(a, b)`
Rectangular PEC guide cross-section, walls at x = 0, a and y = 0, b.

### `Box(x1, x2, y1, y2, z1, z2, eps)`
Axis-aligned dielectric brick. Later boxes in a Structure override earlier
ones where they overlap. `eps` may be complex (`80*(1+0.007j)` = lossy).

### `Shape(z1, z2, eps, k=64)`
Level-set base class for curved z-uniform obstacles. Subclass and provide
`level_set(x, y)` (signed distance, negative inside) and `bbox`; you
inherit a bisection `staircase(waveguide, k)`, a finite-difference
`normal(x, y)`, and `scale`. Shapes carry the boundary normal field that
tensor factorizations need. See [geometry.md](geometry.md).

### `Cylinder(cx, cy, r, z1, z2, eps, k=64)`
Circular cylinder (axis ∥ z): analytic level set and radial normal,
exact-interval staircase.

### `Structure(waveguide, boxes=(), background=1.0, *, shapes=())`
Immutable scene description; `segments()` slices it into z-uniform
`Segment`s (each carrying its covering `shapes`). `z_span` gives the port
reference planes. `shapes` is keyword-only.

### `CrossSection` / `Segment`
The rectilinear ε layout of one z-uniform slice, and the slice itself
(`z1`, `z2`, `cross_section`, `shapes`). Produced by `Structure.segments()`.

## Solving

### `Solver(structure, M, N, factorization="li", lead_eps=None, asr=None, symmetry=None, blas_threads=None, nvf=None, kfj=None)`
FMM solver at fixed truncation (M, N).

- `factorization`: `"li"` (default, exact rules on boxes/staircases),
  `"direct"` (benchmark only), `"nvf"` / `"kfj"` (tensor rules; require
  shapes-only structures) — selection guide in
  [factorizations.md](factorizations.md).
- `asr=AsrConfig(...)`: adaptive spatial resolution for high-contrast
  blocks (exclusive with tensor factorizations).
- `symmetry="x"`: solve the two mirror-parity sectors independently for
  x-symmetric structures (~3× faster, bit-identical S).
- `blas_threads`: scoped BLAS thread limit around each solve.
- `nvf=NvfConfig(...)` / `kfj=KfjConfig(...)`: tensor-factorization knobs.

Key methods:
- `smatrix(freq) -> SResult` — total modal S-matrix; complex `freq` allowed
  (resonance-sheet continuation). Per-layout operators are cached, so
  sweeps and pole hunts pay only eigensolves.
- `det_port_s(freq, indices=None) -> complex` — det of the port S-matrix,
  the analytic scalar for pole hunts. Pass one fixed `indices` set for
  sweeps over a complex region.
- `sweep(freqs) -> list[SResult]`.

### `SResult`
`freq`, `smatrix` (full modal `SMatrix`), `lead` (`LeadModes`);
`port_smatrix(indices=None)` assembles the 2p×2p propagating-port block;
`coeff(out_port, out_mode, in_port, in_mode)` reads one element, e.g.
`res.coeff(2, ("TE", 0, 1), 1, ("TE", 0, 1))`.

### `AsrConfig(eta=0.3)` · `NvfConfig(window=None, quad_cells=192)` · `KfjConfig(cells=96, supersample=16)`
Frozen config dataclasses. `NvfConfig.window=None` → 0.20 × shape scale
(calibrated); `math.inf` → the single-shape Li-limit testing mode.
`KfjConfig` documents its measured high-contrast limitation in
[factorizations.md](factorizations.md).

## Modal layer

### `ModeBasis(a, b, M, N)`
Index bookkeeping and exact derivative operators of the sin/cos basis.

### `lead_modes(basis, k0, eps=1.0) -> LeadModes`
Analytic TE/TM modes of the uniform guide: columns `W` (E), `V` (H̃),
`beta`, `labels`, `kc2`; `mode_index("TE", m, n)`, `propagating()`.

### `LeadModes`
The container above; forward modes normalized to eᵀv = 1 (reciprocity
gauge).

## S-matrix algebra

### `SMatrix(s11, s12, s21, s22)`
Two-sided scattering block container; `full()` assembles the 2n×2n matrix.

### `redheffer(A, B) -> SMatrix` / `cascade(list) -> SMatrix`
Star product of two (or many) S-matrices — numerically stable composition
(never T-matrices).

### `interface_smatrix(W1, V1, W2, V2) -> SMatrix`
Interface between two modal bases from field continuity.

### `propagation_smatrix(beta, d) -> SMatrix`
Homogeneous propagation over length d; guards against runaway
complex-frequency continuation.

## Threads

### `recommended_blas_threads() -> int`
min(4, cpu_count) — the measured OpenBLAS knee for SCEPTRE-sized LAPACK
calls (provenance in the docstring; sweep once on your machine).

### `set_blas_threads(n)` / `blas_thread_limit(n)`
Process-wide setter and scoped context manager (threadpoolctl-based; work
AFTER numpy import; not safe across concurrent Python threads — see
[performance.md](performance.md)).

## Constants

### `C0`
Vacuum speed of light, m/s.

## Beyond the top-level namespace

`sceptre.poles.find_zeros_poles` (winding-number contour search for zeros
and poles of an analytic function) and `sceptre.ep` (exceptional-point
location) are importable from their modules; see
[comparisons.md](comparisons.md) and [method.md](method.md).
