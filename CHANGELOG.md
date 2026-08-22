# Changelog

## 0.2.0 — 2026-08-22

### Added

- **`symmetry="x"`**: x-mirror parity sectorization — structures symmetric
  about x = a/2 solve as two independent half-size problems, ~3× faster per
  point with bit-identical S (1e-13). Works at complex frequency and with
  every factorization except ASR.
- **Shape geometry layer** (`Shape`, `Cylinder`): level-set-defined curved
  obstacles carrying exact-interval staircases AND boundary normal fields;
  `Structure(..., shapes=[...])` (keyword-only), shapes-aware `Segment`s.
- **`factorization="nvf"`** (+ `NvfConfig`): normal-vector-field
  factorization for high-contrast curved shapes — symmetrized
  projection-resolved rules along a windowed normal field, S = Sᵀ and
  unitarity structural. Measured on the ε = 80 benchmark disk: resonance
  line within ±5 MHz of the FEM/VNA reference at N = 20 in one solve
  (plain Li: +92 MHz at N = 24, needs an N-ladder + Richardson). Golden
  regression test included.
- **`factorization="kfj"`** (+ `KfjConfig`): Kottke–Farjadpour–Johnson
  anisotropic subpixel smoothing, shipped as a cross-method comparison tool
  with its measured high-contrast limitation documented (the spectral basis
  resolves the smoothing layer as real structure).
- **Runtime BLAS thread control** (`set_blas_threads`,
  `blas_thread_limit`, `recommended_blas_threads`,
  `Solver(blas_threads=...)`): threadpoolctl-based, works after numpy
  import — removes the env-var import-order footgun.
- **ε_xy transverse coupling** through the operator/eigenproblem pipeline
  (`EpsOperators.exy`, `build_fg`), preserving G = Gᵀ and the mirror
  sectorization.
- **Documentation tree** (`docs/`): method, geometry, factorizations with a
  measured use-case matrix, performance/threads, comparison protocol, API
  reference.

### Changed

- ASR edge thinning on dense staircases (`AsrMap1D` minimum interval;
  per-axis warnings): dense-edge maps no longer alias the operator basis —
  wall-touching ε = 80 staircase sweeps now conserve energy to 4e-7.
- The high-contrast recommendation warning now points at
  `factorization="nvf"` when curved Shape geometry is present.
- New dependency: `threadpoolctl>=3.0`.

### Fixed

- ASR dense-edge instability (port-column energies up to 3.5 on staircased
  high-contrast shapes) via basis-bandwidth-aware edge thinning.

## 0.1.0 — 2026-08-20

Initial release: vectorial FMM for closed PEC rectangular waveguides with
box obstacles; exact-overlap Li/direct factorizations; analytic lead modes;
stable Redheffer S-matrix cascading; complex-frequency continuation with
contour pole/zero location (argument principle + Delves–Lyness moments) and
exceptional-point search with Puiseux confirmation; Granet-style adaptive
spatial resolution (ASR) via its transformation-optics equivalent; COMSOL
cross-verification tooling.
