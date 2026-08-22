# Productionize NVF + KFJ Factorizations Implementation Plan

Created: 2026-08-22
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature
Execution: Main

> **Status Lifecycle:** PENDING → COMPLETE → VERIFIED
> **Approval Gate:** Implementation CANNOT proceed until `Approved: Yes`
> **Execution: Main** — 8 tasks, but Tasks 1→3→5 form a tightly-coupled chain
> through fourier/slicesolver/solver with deep physics context from the
> prototypes (LEDGER.md); fresh-context executors would re-derive it. User may
> override to Subagent at approval.

## Summary

**Goal:** Ship `factorization="nvf"` (validated: one N=20 solve ≈ the Li
N-ladder + Richardson at ε=80) and `factorization="kfj"` (comparison tool,
documented limitations) in the sceptre library, backed by a two-layer curved
geometry API (level-set `Shape` base + `Cylinder` primitive), plus runtime
BLAS-thread management and a complete Markdown docs tree with release-ready
packaging.

**Architecture:** New `shapes.py` provides the geometry layer (level-set +
normal field + exact-interval staircase); new `nvf.py`/`kfj.py` assemble the
operators into the existing `EpsOperators` container extended with an `exy`
coupling block; `slicesolver.build_fg` gains the coupling (keeping G = Gᵀ and
x-mirror sectorization); `solver.py` plumbs the new factorization values with
AsrConfig-style config dataclasses. Threads via threadpoolctl (runtime, no
import-order footgun). Docs absorb USAGE.md into a structured `docs/` tree.

**Segment→shape routing (the dispatch rule):** `Segment` gains a
`shapes: tuple[Shape, ...]` field (shapes whose [z1, z2] cover the segment's
z-interval), populated by `Structure.segments()`; `Solver._segment_ops`
receives the Segment. Under nvf/kfj, segments WITH shapes route to the
nvf/kfj builder; segments without shapes are uniform background (analytic
path). Because nvf/kfj Structures are shapes-only (mixed boxes rejected),
every ε discontinuity lies inside some shape's window — away from windows
the cross-section is uniform, where direct and inverse rules coincide
exactly. This is precisely WHY the shapes-only restriction exists: it makes
the NVF background-region behavior exact rather than silently degraded.

**Tech Stack:** numpy/scipy/numba (existing), threadpoolctl (new dependency),
uv_build packaging, Markdown docs.

## Scope

### In Scope

- `src/sceptre/shapes.py`: `Shape` level-set base + `Cylinder` primitive
- `src/sceptre/geometry.py`: `Structure` accepts shapes alongside boxes
- `src/sceptre/fourier.py`: mixed cos·sin overlaps, `EpsOperators.exy` field
- `src/sceptre/nvf.py`, `src/sceptre/kfj.py`: operator assembly + configs
- `src/sceptre/slicesolver.py`: εxy coupling in `build_fg` incl. sector path
- `src/sceptre/solver.py`: `factorization="nvf"|"kfj"`, validation, caching
- `src/sceptre/threads.py`: threadpoolctl-based BLAS thread control
- `docs/` tree: index, method, geometry, factorizations (use-case matrix),
  performance (threads/machine sizing), API reference; USAGE.md absorbed
- Release packaging: pyproject metadata, CHANGELOG.md, version 0.2.0
- Golden accuracy regression test (ε=80 disk line, NVF)

### Out of Scope

- Corner-adapted normal fields (crescent horn tips) — future work; NVF ships
  validated on smooth boundaries, docs state the corner caveat
- MIXED boxes + shapes under nvf/kfj: REJECTED with a clear ValueError in
  this release (outside the NVF window the operator degrades to the direct
  rule, which mishandles high-contrast box edges that Li's inverse rule
  covers — mixing would silently change box accuracy). Box regions need a
  future `Rectangle` shape (axis normals recover Li exactly via the NVF
  limit) — deferred
- ASR + NVF combination (orthogonal mechanisms; research)
- z-mirror sectorization, LU reuse, GPU (backlog items 2)
- Actually publishing to PyPI (user runs `uv publish` manually)
- Migrating reports/comsol_parity scripts to the new API

## Prerequisites

- Prototypes validated (scripts/nvf_prototype.py, scripts/kfj_prototype.py;
  evidence in LEDGER.md — H3 SUPPORTED, H1 refuted-but-shipped-anyway per
  user decision)
- Baseline: the CURRENT working tree (including the uncommitted ASR fix,
  symmetry feature, and doc edits — deliberate unreleased 0.2.0 work the
  user commits manually) is the baseline: 70 tests green, ruff +
  basedpyright clean. Implementation Step 0: re-run `uv run pytest -q` and
  record the count before Task 1 — every later "no regressions" gate
  measures against THIS baseline, not HEAD.

## Context for Implementer

- **Patterns to follow:** `AsrConfig` (frozen dataclass + `__post_init__`
  validation, `src/sceptre/asr.py:59`); operator assembly loops
  (`src/sceptre/fourier.py:142-199` — exact per-cell overlaps, kron sums, Li
  strip-wise inversions); Solver validation order (`src/sceptre/solver.py`
  `__init__` — fail fast before state is built); sector index bookkeeping
  (`src/sceptre/symmetry.py`).
- **Conventions:** e^{−iωt}, Im ε > 0 lossy; S referenced at structure faces;
  G = Gᵀ is load-bearing (structural reciprocity/unitarity — never break it:
  the NVF operator MUST use the symmetrized product ½(BP̂ + P̂B)); files ≤300
  lines (hard 500); one-line docstrings unless complex.
- **Key files to read first:** `scripts/nvf_prototype.py` (working NVF math:
  `cell_fields`, `galerkin`, the symmetrized operator, `cs_overlap`),
  `scripts/kfj_prototype.py` (KFJ tensors + mixed overlaps), `LEDGER.md`
  (what was measured, what failed and why), `src/sceptre/fourier.py`,
  `src/sceptre/slicesolver.py`, `src/sceptre/symmetry.py`.
- **Gotchas:**
  - Rasterization moiré: the prototype's centroid-sampled 2-D cells wobble
    the line ±3 MHz with quadrature grid; production ε/1/ε ingredient
    matrices MUST use the shape's exact-interval x-strip staircase (like
    `disk_boxes` — exact y-intervals at strip midpoints), only the SMOOTH
    window/normal fields may be centroid-sampled (second-order accurate).
  - The spurious 5.36–5.41 pol-B cluster moves with window width W — it is a
    window artifact. Task 3 tunes the window (compact support, boundary-
    distance based) against a measurable criterion; if not fully killable,
    the residual amplitude bound goes into the docs as a known limitation.
  - εxy sector slicing: the XY block connects EQUAL m-parity classes
    (analytically checked; `LEDGER.md` Notes) — slice as
    `exy[np.ix_(sec.X, sec.Y)]`.
  - `EpsOperators` is frozen; `exy: np.ndarray | None = None` keeps every
    existing construction site valid.
  - Thread limits set via env AFTER numpy import are silently void — that is
    exactly why threads.py uses threadpoolctl, which works at runtime.
  - KFJ is REFUTED for high-contrast accuracy (LEDGER H1) and ships anyway
    per explicit user decision — as a comparison/low-contrast tool. Its docs
    section must state the measured failure mechanism plainly.
- **Domain context:** modal truncation error at ε≈80 converges ~1/N with Li
  on staircases (line blueshift +92 MHz at N=24 on the reference disk); NVF
  keeps ε sharp and changes the factorization rule along a windowed normal
  field, giving N-stable lines at the true position by N≈20. Reference
  numbers: COMSOL disk pol-B line 5.4357 GHz, campaign Richardson 5.4375.

## Progress Tracking

**MANDATORY: Update this checklist as tasks complete. Change `[ ]` to `[x]`.**

- [x] Task 1: Mixed overlaps + εxy through the eigenproblem pipeline
- [x] Task 2: shapes.py — Shape base, Cylinder, Structure integration
- [x] Task 3: nvf.py — NVF operator assembly + NvfConfig + window design
- [x] Task 4: kfj.py — KFJ tensor assembly + KfjConfig
- [x] Task 5: Solver plumbing + golden accuracy validation
- [x] Task 6: threads.py — runtime BLAS thread management
- [x] Task 7: Docs tree (absorb USAGE.md) + README restructure
- [x] Task 8: Release packaging (pyproject, CHANGELOG, version)
      (uv build ✓; py.typed in wheel ✓; version 0.2.0 consistent ✓;
      suite 105-green on Python 3.11 / 3.12 / 3.13 — classifiers verified)

**Total Tasks:** 8 | **Completed:** 8 | **Remaining:** 0

> Golden test measured 2026-08-22: PASSES at N=20, NVF line −0.8 MHz vs
> COMSOL 5.4357 within the ±5 MHz calibrated tolerance; Li shows no line
> within ±50 MHz at the same N. Runtime 155 s → stays `integration` marker.
> Suite: 105 passed; ruff + basedpyright clean.

## Implementation Tasks

### Task 1: Mixed overlaps + εxy through the eigenproblem pipeline

**Objective:** Add exact cos·sin overlap matrices and an optional `exy`
coupling block flowing from `EpsOperators` through `build_fg` (full and
sectored paths) while preserving G = Gᵀ.

**Dependencies:** None
**Wave:** 1

**Files:**

- Modify: `src/sceptre/fourier.py` (add `cs_overlap`, `EpsOperators.exy`)
- Modify: `src/sceptre/slicesolver.py` (`build_fg`: `g12 += k0·exy`,
  `g21 += k0·exyᵀ`; sector branch slices `exy[np.ix_(X, Y)]`)
- Modify: `src/sceptre/symmetry.py` (`slice_eps_ops` handles `exy`)
- Test: `tests/test_fourier.py` (extend), `tests/test_symmetry.py` (extend)

**Key Decisions / Notes:**

- Port `cs_overlap` from `scripts/kfj_prototype.py:44` (already exact; add
  the m=0 row normalization test). Keep in fourier.py next to sin/cos fills.
- `exy` defaults to `None`; `build_fg` adds coupling only when present —
  zero behavioral change for li/direct/ASR.
- G symmetry: `g21` uses `exy.T` (not conj) — complex-symmetric like the
  rest of the operator algebra.
- Defense-in-depth: `build_fg` asserts `exy is None or ops.mzz is None` —
  Solver-level validation already forbids nvf/kfj + ASR, but build_fg is
  importable and must not silently combine exy with ASR metric operators.

**Definition of Done:**

- [ ] `cs_overlap(M, a, 0, a)` matches analytic full-range values
      (0 for m+m′ even, 2/π·normalized·2m′/(m′²−m²) pattern for odd) to 1e-12
- [ ] `cs_overlap` over sub-intervals matches numpy quadrature to 1e-9
- [ ] `build_fg` with a random symmetric-compatible exy returns G with
      max|G − Gᵀ| < 1e-13·|G|
- [ ] Sectored build_fg with an x-mirror-odd exy matches the full solve on a
      symmetric test structure to 1e-9
- [ ] All existing 70 tests still pass

**Verify:**

- `uv run pytest tests/test_fourier.py tests/test_symmetry.py -q`
- `uv run pytest -q` (no regressions)

### Task 2: shapes.py — Shape base, Cylinder, Structure integration

**Objective:** Two-layer curved geometry: a `Shape` level-set base class
(signed distance, unit normal, bbox, z-extent, ε, exact-interval staircase)
and a `Cylinder` primitive; `Structure` accepts shapes alongside boxes.

**Dependencies:** None
**Wave:** 1

**Files:**

- Create: `src/sceptre/shapes.py`
- Modify: `src/sceptre/geometry.py` (`Structure(waveguide, boxes=(),
  shapes=(), background=1)`; `segments()` folds `shape.staircase()` boxes in;
  shapes kept on the Structure for solver access)
- Modify: `src/sceptre/__init__.py` (export `Shape`, `Cylinder`)
- Test: `tests/test_shapes.py` (new)

**Key Decisions / Notes:**

- `Shape` abstract: `level_set(x, y) -> ndarray` (signed distance, <0
  inside), `normal(x, y) -> (nx, ny)` (default: normalized level-set
  gradient by central differences; primitives override analytically),
  `bbox`, `z1`, `z2`, `eps`, `staircase(waveguide, k) -> list[Box]` —
  takes the Waveguide so it can CLAMP intervals to [0, a]×[0, b] itself
  (wall-touching shapes must not trip Structure's bounds validation);
  default generic bisection on the level set per x-strip; `Cylinder`
  overrides with the exact chord formula à la `disk_boxes`
  (`reports/comsol_parity/runs/parity_common.py:39`).
- `Cylinder(cx, cy, r, z1, z2, eps, k=64)`: analytic level set ρ−r, radial
  normal, exact clamped y-intervals.
- Default staircase k=64 (measured: ≤1.1 MHz geometry error at 0.94 mm
  strips for r=15 mm; 64 gives margin at no visible cost). Absolute-count
  caveat documented: k spans THIS shape's x-extent, so very large or
  multi-scale shapes should raise it (docs note, geometry.md).
- Structure: `shapes` is KEYWORD-ONLY (`Structure(wg, boxes, *, shapes=(),
  background=1)`) so a future positional `background` can't silently bind
  to it. Frozen as before; shapes normalized to a tuple in `__post_init__`
  with bounds validation against the guide bbox.
- `Structure.z_span` includes shape z-extents (a shapes-only Structure must
  not raise); `Segment` gains `shapes: tuple[Shape, ...]` — the shapes
  whose [z1, z2] cover that segment — populated by `segments()` (this is
  the routing data `Solver._segment_ops` dispatches on).

**Definition of Done:**

- [ ] `Cylinder.staircase(wg, 64)` reproduces `disk_boxes(...)` boxes
      exactly (same intervals) for the campaign disk parameters
- [ ] `Cylinder.level_set`/`normal` exact on sampled points (|∇φ|=1 on the
      boundary ring, n radial to 1e-12)
- [ ] `Structure(shapes=[cyl]).segments()` equals
      `Structure(boxes=disk_boxes(...)).segments()` layout-for-layout, and
      each segment's `shapes` tuple contains the cylinder
- [ ] A WALL-TOUCHING cylinder (r=15 mm, cy=a/2+1 mm — touches y=a exactly)
      staircases without raising; intervals clamped to [0, b]
- [ ] Shapes-only `Structure.z_span` works (no "has no boxes" error)
- [ ] Generic `Shape.staircase` (bisection path) agrees with the exact
      Cylinder staircase to 1e-9 in interval endpoints
- [ ] Off-guide shape (fully outside) raises ValueError like boxes do

**Verify:**

- `uv run pytest tests/test_shapes.py -q`
- `uv run pytest tests/test_solver_api.py -q` (Structure API unbroken)

### Task 3: nvf.py — NVF operator assembly + NvfConfig + window design

**Objective:** Production NVF: symmetrized projection-resolved factorization
with exact-staircase ε ingredients and a principled compact window; returns
`EpsOperators` (with `exy`) consumable by `build_fg`.

**Dependencies:** Task 1, Task 2
**Wave:** 2

**Files:**

- Create: `src/sceptre/nvf.py` (`NvfConfig`, `build_nvf_operators(shapes,
  layout, basis, config)`)
- Test: `tests/test_nvf.py` (new)

**Key Decisions / Notes:**

- Port the working math from `scripts/nvf_prototype.py` with two fixes:
  (1) ε and 1/ε Galerkin matrices from the shape's EXACT-interval staircase
  cells (targets the moiré — the prototype's centroid rasterization is the
  suspected ±3 MHz wobble source; the CALIBRATION below verifies rather
  than assumes this); (2) window redesign: compact support
  w(d) = cos²(πd/2W) for |d| < W (d = level-set distance) — a NEW window
  family, never measured with the Gaussian prototype, hence:
- **CALIBRATION RUN (this task's core deliverable, drives the golden):**
  on the ε=80 campaign disk with the production assembly (exact ε cells +
  cos² window), jointly measure LINE POSITION and CLUSTER AMPLITUDE over a
  W sweep (≥5 values spanning 0.08r–0.27r) at N=16 and N=20. Record the
  table in this plan under Task 3 on completion. Choose the default W from
  it (joint objective: cluster amplitude minimal, line stable); derive the
  golden test's N and tolerance FROM the measured line scatter (Task 5
  consumes these numbers — they are NOT pre-committed).
- Symmetrized operator only: ε̂ = ⟦ε⟧ + ½(BP̂ + P̂B), B = ⟦1/ε⟧⁻¹ − ⟦ε⟧
  (G = Gᵀ is non-negotiable). The plain product stays in the prototype.
- Multiple shapes: supported only while windows DO NOT overlap — assembly
  raises ValueError if any quadrature cell carries w > 1e-6 from two shapes
  (blended-normal physics is unverified; partition-of-unity blending goes
  to Deferred Ideas). Two well-separated shapes get an equality test.
- ezz stays direct-rule from the staircase layout (Ez tangential in-plane).
- `NvfConfig(window: float | None = None, quad_cells: int = 192)`:
  `window=None` resolves PER SHAPE as `f_w · (shape scale)` (Cylinder scale
  = r; fraction f_w fixed by the calibration) — an absolute default in
  metres would be wrong at any other band by construction. Explicit
  `window` values are absolute and validated: Cylinder raises if W ≥ r
  (normal-field singularity at the centre would be sampled with nonzero
  weight — silent wrong physics otherwise); generic Shapes document the
  medial-axis requirement.
- Frequency-independent → cacheable per layout in Solver's ops cache. Cache
  key: `(layout.key(), factorization, config, tuple(id(s) for s in
  segment.shapes))` — id() is safe because Solver docs already fix the
  Structure for the object's lifetime, and it imposes no hashability
  requirement on user Shape subclasses. `CrossSection.key()` alone is NOT
  sufficient (two windows on one staircase must not share ops).

**Definition of Done:**

- [ ] Forced-normal x̂/ŷ limit: NVF operator entries match the strip-wise Li
      operators on a lamellar (box) layout to 1e-10
- [ ] Lossless disk at N=12: propagating port S unitary to 1e-10 and
      S = Sᵀ to 1e-12 (structural, from the symmetrized form)
- [x] Calibration table (MEASURED 2026-08-22, scripts/nvf_calibration.py,
      production assembly, err vs COMSOL 5.4357):

      | frac | W mm | N=16 err | N=16 clmax | N=20 err | N=20 clmax |
      |---|---|---|---|---|---|
      | 0.08 | 1.20 | −20.8 | 0.87 | −14.6 | 0.91 |
      | 0.12 | 1.80 | −8.5 | 0.91 | −5.9 | 0.99 |
      | 0.15 | 2.25 | −23.6 | 1.00 | −2.7 | 0.86 |
      | 0.20 | 3.00 | **+4.3** | 1.00 | **−0.8** | 0.98 |
      | 0.27 | 4.05 | −30.8 | 1.00 | −0.1 | 1.00 |

      Chosen: WINDOW_FRACTION = 0.20 (best joint N=16/N=20 behavior; the
      N=20 series is monotone −14.6→−0.8; some N=16 outliers are neighbor-
      mode grabs by the scan tracker — see below). Golden: N=20, tol ±5 MHz.
- [x] Spurious-cluster criterion — **VOIDED WITH EVIDENCE, criterion
      retired**: the 5.30–5.42 peaks are REAL disk modes below the COMSOL
      window (whose grid started at 5.4 GHz). Proof (scripts/
      disk_low_ladder.py): plain-Li N=24/32/40 ladder on 5.30–5.50
      Richardson-extrapolates pol-B modes to 5.3592 / 5.3166 / ≈5.406 —
      matching NVF's directly computed positions (5.364 / 5.397–5.404 at
      N=16–24) to ~5 MHz, and the rung data cross-checks the main line
      (5.4883@N40 vs the campaign ladder's 5.4892). NVF renders REAL
      below-window physics near-converged at N=16; nothing to suppress.
      Docs must present this as a capability with the mode-identification
      caveat (spectra are denser than the reference window suggested).
- [x] Two well-separated cylinders (AMENDED at verification: the literal
      "1e-6 |S| equality vs single-cylinder" is not physically attainable —
      real inter-cylinder coupling exceeds 1e-6 at any finite separation;
      implemented as test_nvf_two_separated_cylinders_success_path: pair
      solve succeeds, unitary 1e-10, reciprocal 1e-12, and both shapes
      demonstrably contribute to the accumulated operator)
- [ ] Overlapping windows raise ValueError; Cylinder with W ≥ r raises
- [ ] Complex ε (tanδ=0.007) assembly works (no dtype loss)

**Verify:**

- `uv run pytest tests/test_nvf.py -q`
- `uv run pytest -q`

### Task 4: kfj.py — KFJ tensor assembly + KfjConfig

**Objective:** KFJ anisotropic subpixel smoothing as a first-class (flagged,
documented-limitation) factorization: tensor cells from shape fill fractions
and normals, Li-diagonal + direct-εxy assembly.

**Dependencies:** Task 1, Task 2
**Wave:** 2

**Files:**

- Create: `src/sceptre/kfj.py` (`KfjConfig(cells: int = 96)`,
  `build_kfj_operators(shapes, layout, basis, config)`)
- Test: `tests/test_kfj.py` (new)

**Key Decisions / Notes:**

- Port from `scripts/kfj_prototype.py` (`kfj_cells`, `build_ops_tensor`)
  using Shape.level_set for fill fractions (supersampling 16×16/cell) and
  Shape.normal at cell centroids.
- This method is REFUTED for high-contrast accuracy (LEDGER H1): ships per
  user decision as a comparison/low-contrast tool. The module docstring and
  docs must carry the measured mechanism (layer shift ∝ 1/cells + spurious
  anisotropic-ring modes at ε≈80).
- No window design work here — the formulation is what it is.
- **LOW-CONTRAST LADDER (evidence for the docs matrix — MEASURED 2026-08-22,
  scripts/lowcontrast_ladder.py):** ε=4 disk, fixed 5.6 GHz (no in-band
  resonances at ε=4, so fixed-f error vs converged li N=28 reference is the
  honest metric), max|ΔS4|:

  | arm | N=8 | N=12 | N=16 |
  |---|---|---|---|
  | li | 3.8e-3 | 2.0e-3 | 1.1e-3 |
  | nvf | 5.1e-3 | 1.7e-3 | 6.5e-4 |
  | kfj | 4.4e-3 | 2.7e-3 | 2.1e-3 |

  (RE-MEASURED at verification with the shipped WINDOW_FRACTION=0.20 —
  the first run predated the window calibration and understated nvf;
  caught by the compliance reviewer re-running scripts/lowcontrast_ladder
  .py; log now saved as scripts/lowcontrast_ladder.log.)
  Verdict for the matrix: all arms sub-percent by N=12; li best at N=8 and
  simplest (recommended on those grounds); nvf marginally faster from
  N=12 up; kfj weakest at every N.

**Definition of Done:**

- [ ] Lossless disk at N=12: port S unitary to 1e-10, S = Sᵀ to 1e-12
- [ ] Fill-fraction sanity: total dielectric area from cells within 0.1% of
      πr² for the test cylinder
- [ ] Full-cell limit: a Box-aligned rectangle shape yields operators equal
      to plain Li on the same layout to 1e-10
- [ ] Module docstring states the measured high-contrast failure mechanism
- [ ] Low-contrast ladder table (ε=4, li/nvf/kfj at N=8/12/16) recorded in
      this plan for the docs matrix

**Verify:**

- `uv run pytest tests/test_kfj.py -q`

### Task 5: Solver plumbing + golden accuracy validation

**Objective:** `Solver(structure, M, N, factorization="nvf"|"kfj",
nvf=NvfConfig(...), kfj=KfjConfig(...))` end-to-end, with validation,
caching, symmetry="x" compatibility, and the golden ε=80 accuracy test.

**Dependencies:** Task 3, Task 4
**Wave:** 3

**Files:**

- Modify: `src/sceptre/solver.py` (accept new factorization values + config
  args; `_segment_ops` dispatches to nvf/kfj builders; cache key includes
  config; validation: nvf/kfj require `structure.shapes`, mutually exclusive
  with `asr`; extend the high-contrast warning to RECOMMEND nvf when curved
  shapes + contrast ≥ 25 + factorization="li")
- Modify: `src/sceptre/__init__.py` (export `NvfConfig`, `KfjConfig`)
- Test: `tests/test_solver_api.py` (extend), `tests/test_nvf.py` (golden)

**Key Decisions / Notes:**

- Golden test (integration marker): ε=80 campaign disk via
  `Cylinder(r=15mm, h=5mm, cy=a/2+1mm)`, `factorization="nvf"`,
  `symmetry="x"`. The golden's N AND tolerance come from Task 3's
  calibration table (not pre-committed — the cos² window + exact-cell
  configuration has never been measured; prototype evidence: −0.9 MHz at
  N=20 with the Gaussian window). Reference value: COMSOL 5.4357 GHz
  (external truth; the campaign's own Richardson 5.4375 is NOT the anchor —
  one reference, stated in the test docstring). Tolerance = max(2× the
  calibration's measured line scatter over W and N, 5 MHz). Line
  identification: nearest peak to the reference within ±20 MHz with
  amplitude ≥ 0.5 — NEVER global argmax (that tracker failed on multi-line
  spectra during the research phase). Runtime: measured once during this
  task; the marker (integration vs a new `slow`) chosen from the
  measurement, not asserted.
- symmetry="x" + nvf: `require_x_symmetric` continues to check the
  discretized layout; add a shape-level check (cx == a/2 within tol) so the
  error message points at the shape.
- Boxes-only + factorization="nvf" → ValueError with a message naming
  Cylinder/Shape. MIXED boxes + shapes + nvf/kfj → ValueError too (see Out
  of Scope: box edges would silently lose Li's inverse rule outside the
  window; message names the deferred Rectangle-shape path).

**Definition of Done:**

- [ ] Golden: NVF disk pol-B line within the CALIBRATED tolerance of
      5.4357 GHz at the calibrated N (numbers recorded here by Task 3)
- [ ] Same golden structure with plain Li at the same N is > 50 MHz off
      (documents the win inside the test suite)
- [ ] `symmetry="x"` + nvf matches full nvf solve to 1e-9 on the disk
- [ ] `symmetry="x"` + kfj matches full kfj solve to 1e-9 on the disk
      (the kfj εxy sector path gets its own equality test, not just nvf's)
- [ ] nvf+asr, kfj+asr, nvf-without-shapes, and MIXED boxes+shapes with
      nvf/kfj each raise ValueError
- [ ] High-contrast + curved shapes + li emits the recommendation warning
- [ ] `uv run pytest -q` fully green

**Verify:**

- `uv run pytest tests/test_nvf.py tests/test_kfj.py tests/test_solver_api.py -q`
- `uv run pytest -q -m integration`

### Task 6: threads.py — runtime BLAS thread management

**Objective:** Machine-adaptive BLAS thread control that works AFTER numpy
import: `sceptre.set_blas_threads(n)`, `sceptre.recommended_blas_threads()`,
and `Solver(..., blas_threads=n)` scoping limits around solves.

**Dependencies:** None
**Wave:** 1

**Files:**

- Create: `src/sceptre/threads.py`
- Modify: `src/sceptre/solver.py` (optional `blas_threads` arg; wrap
  `smatrix` body in `threadpool_limits` when set)
- Modify: `pyproject.toml` (add `threadpoolctl>=3.0` dependency)
- Modify: `src/sceptre/__init__.py` (exports)
- Test: `tests/test_threads.py` (new)

**Key Decisions / Notes:**

- `recommended_blas_threads()`: measured pathology is thrashing ABOVE ~4
  threads at sceptre's matrix sizes → `min(4, os.cpu_count() or 1)`; the
  docstring states the provenance ("measured on 16-core OpenBLAS at
  n≈300–2000; other BLAS builds may knee elsewhere — sweep once on your
  machine") so the number reads as a measured default, not a law. Docs
  explain per-machine tuning + the workers×threads ≤ cores sweep rule.
- threadpoolctl works at runtime (unlike env vars) — this removes the
  before-numpy-import footgun entirely; docs keep the env-var pattern as
  the recommended approach INSIDE ProcessPool workers.
- **NOT THREAD-SAFE**: threadpool_limits mutates process-global BLAS state;
  concurrent smatrix calls from Python threads with different blas_threads
  race (last-exit-wins restore). Stated prominently in threads.py docstring
  AND docs/performance.md; process-based parallelism (the library's sweep
  pattern) is unaffected.
- `blas_threads=None` (default) = do nothing (current behavior).

**Definition of Done:**

- [ ] `set_blas_threads(4)` + a solve completes; threadpoolctl reports the
      limit active inside (assert via `threadpool_info()` in-test)
- [ ] `Solver(..., blas_threads=2)`: limit active during `smatrix`, restored
      after (scoped, not global)
- [ ] `recommended_blas_threads()` returns min(4, cpu_count) and is
      monkeypatch-testable
- [ ] Unit tests mock nothing heavier than cpu_count (fast)

**Verify:**

- `uv run pytest tests/test_threads.py -q`

### Task 7: Docs tree (absorb USAGE.md) + README restructure

**Objective:** Complete Markdown documentation under `docs/`: method, geometry,
factorization guide with use-case matrix, performance, API reference; USAGE.md
absorbed; README becomes the PyPI landing page linking into the tree.

**Dependencies:** Task 5, Task 6
**Wave:** 4

**Files:**

- Create: `docs/index.md` (map of the docs + quickstart)
- Create: `docs/method.md` (FMM for closed PEC guides, matched sin/cos
  harmonics, S-cascading, conventions, poles/EP capabilities)
- Create: `docs/geometry.md` (boxes, staircasing — from USAGE §2; shapes,
  level sets, Cylinder, when staircase K matters)
- Create: `docs/factorizations.md` (li, direct, ASR, NVF, KFJ: what each
  does, when to use, the use-case matrix, measured evidence per claim)
- Create: `docs/performance.md` (threads: threadpoolctl API + env-var
  pattern + machine sizing table, workers×RAM budgets, symmetry="x",
  N-ladders — from USAGE §1/§4, sweep skeleton)
- Create: `docs/comparisons.md` (cross-code/measurement protocol + poles —
  USAGE §5/§6)
- Create: `docs/api.md` (every public symbol, signatures, one-line + example)
- Modify: `README.md` (tighten to landing page; link tree; PyPI-renderable)
- Delete: `docs/USAGE.md` (contents absorbed; README/VALIDATION links updated)

**Key Decisions / Notes:**

- The use-case matrix is the centerpiece (rows: contrast low ε≲10 / high
  ε≳25 × boundary type box-aligned / smooth curved / sharp-cornered; cells:
  recommended factorization + expected convergence + evidence pointer).
  Cells with measurements cite them (incl. Task 4's ε=4 ladder for the
  low-contrast column). The sharp-cornered column, which has only Li
  crescent evidence, is EXPLICITLY labeled "untested with NVF —
  extrapolation; corner-adapted normal fields are future work" — labeled
  extrapolation is permitted ONLY there; every other cell needs a number.
- Every performance/accuracy claim carries its measured number (this repo's
  discipline: no vibes). KFJ page states the refutation mechanism.
- README links to docs/ MUST be absolute GitHub URLs (docs/ is not in the
  wheel; PyPI readers have no relative path). Repo visibility must be
  confirmed by the user — see Open Questions.
- docs/method.md acceptance: covers (at minimum) the modal basis + parities,
  the F/G eigenproblem, Li's rules and why factorization matters, S-matrix
  cascading stability, conventions (time sign, references, branch sheets),
  and the pole/EP capability — each section pointing to the module that
  implements it.
- USAGE.md absorption map (ALL seven sections, checked in DoD):
  §1 parallel sweeps → performance.md; §2 staircasing → geometry.md;
  §3 high contrast ASR vs plain Li → factorizations.md (ASR section);
  §4 N-ladders → performance.md; §5 comparison protocol → comparisons.md;
  §6 poles → comparisons.md; §7 symmetry="x" → performance.md.
- Grep for `USAGE.md` references (README, VALIDATION.md) and update links.

**Definition of Done:**

- [ ] All seven docs files exist with real content (no TODO/stub sections)
- [ ] Every USAGE.md section (1-7) verifiably absorbed per the map above —
      each section's key content findable in its target page before the
      USAGE.md delete lands
- [ ] Use-case matrix covers all 6 cells with a recommendation + evidence
- [ ] Thread guidance includes: threadpoolctl API, env-var worker pattern,
      the min(4, cores) rule, and the workers×threads≤cores sweep rule
- [ ] No dangling references to docs/USAGE.md anywhere in the repo
- [ ] Every `sceptre.__all__` symbol appears in docs/api.md
- [ ] README renders standalone (no relative links that break on PyPI)

**Verify:**

- `grep -rn "USAGE.md" --include="*.md" .` → only historical mentions in
  reports/ and plans/
- `uv run python - <<'EOF'` script asserting every `__all__` name is in
  docs/api.md

### Task 8: Release packaging (pyproject, CHANGELOG, version)

**Objective:** Release-ready package metadata: classifiers, urls, keywords,
CHANGELOG.md, version 0.2.0, clean build.

**Dependencies:** Task 6 (dependency list final), Task 7 (README final)
**Wave:** 4

**Files:**

- Modify: `pyproject.toml` (classifiers: Python 3.11-3.13, Science/Physics,
  MIT, Typed; `[project.urls]`; keywords; threadpoolctl dep from Task 6)
- Create: `CHANGELOG.md` (0.1.0 baseline summary + 0.2.0: symmetry="x", NVF,
  KFJ, shapes, threads, docs)
- Modify: `src/sceptre/__init__.py` + `pyproject.toml` (version 0.2.0)
- Test: build check (not a pytest test)

**Definition of Done:**

- [ ] `uv build` succeeds; wheel + sdist produced
- [ ] Wheel contents include `sceptre/py.typed` (the Typed classifier is a
      lie without it) — checked by listing the wheel
- [ ] Version single-sourced or consistency-asserted: `__init__.__version__`
      == pyproject version (one-line check in the build verify)
- [ ] `uv run python -c "import sceptre; print(sceptre.__version__)"` → 0.2.0
- [ ] CHANGELOG covers both versions with dates
- [ ] classifiers/urls/keywords present; python-version classifiers limited
      to versions the suite actually ran on (`uv run --python 3.13 pytest -q`
      once; trim classifiers if it fails rather than claiming it)

**Verify:**

- `uv build 2>&1 | tail -2` and `python -m zipfile -l dist/*.whl | grep py.typed`
- `uv run --python 3.13 pytest -q` (or trim the classifier)
- `uv run pytest -q` (final full suite)

## Testing Strategy

- **Unit:** overlap-matrix identities vs quadrature, level-set/staircase
  geometry exactness, config validation, thread-limit scoping (mock
  cpu_count), operator symmetry (G = Gᵀ) at any truncation.
- **Integration:** unitarity/reciprocity of full solves per factorization;
  the golden ε=80 disk-line accuracy test (NVF ±5 MHz at N=16, Li >100 MHz
  documents the delta); symmetry="x" equality per factorization.
- **Manual verification (spec-verify phase):** run the prototype ladder once
  through the NEW API (`Cylinder` + `factorization="nvf"`) and confirm the
  N=16/20 line positions reproduce the prototype's; `uv build`; render
  README/docs and eyeball the matrix table.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| cos² window family behaves differently from the measured Gaussian (line AND cluster both move with W — objectives can conflict) | Med | High | Task 3's calibration measures line + cluster JOINTLY over the W sweep before anything is committed; the golden's N and tolerance are derived from that table, and the W default is the joint optimum. If no W meets both objectives, the measured trade-off is recorded and the golden anchors at the stable-line W |
| Exact-staircase ε ingredients don't fully remove the moiré (the kq-probe was an inference, not a measurement of this configuration) | Med | Med | The calibration table IS the measurement of the production configuration; tolerance is set from its observed scatter, not from the prototype's |
| εxy plumbing breaks existing factorizations | Low | High | `exy=None` default + full-suite regression gate in Task 1 DoD before anything else builds on it |
| Symmetry sectoring of exy has an index-bookkeeping bug | Med | Med | Dedicated sector-vs-full equality test with synthetic x-odd exy (Task 1) and the real NVF disk (Task 5) |
| threadpoolctl interacts badly with worker processes (nested limits) | Low | Med | Scoped context (restore on exit) + docs keep the env-var pattern as the recommended approach INSIDE ProcessPool workers |
| Docs drift from code (API reference by hand) | Med | Low | Task 7 verify script asserts every `__all__` symbol is documented; re-run in spec-verify |
| KFJ users misread it as an accuracy feature | Med | Med | Module docstring + docs page lead with the measured refutation; the high-contrast warning recommends NVF, never KFJ |

## Goal Verification

### Truths (what must be TRUE for the goal to be achieved)

- A user can solve the ε=80 experiment disk with
  `Cylinder + factorization="nvf"` at N=16 and read the pol-B line within
  ±5 MHz of the measured/COMSOL value — no ladder, no Richardson
- A user can select `factorization="kfj"` and gets documented,
  energy-conserving results (with its limitation stated where they look)
- A user on any machine can set BLAS threads AFTER importing numpy via
  `sceptre.set_blas_threads` / `Solver(blas_threads=...)`
- A user landing on the repo/PyPI can navigate docs/ to decide which
  factorization fits their contrast + boundary shape in one table
- `uv build` produces a publishable 0.2.0 wheel

### Artifacts (what must EXIST to support those truths)

- `src/sceptre/shapes.py` (Shape + Cylinder, exact staircase + normals)
- `src/sceptre/nvf.py`, `src/sceptre/kfj.py` (operator builders + configs)
- `src/sceptre/threads.py` (threadpoolctl integration)
- `tests/test_shapes.py`, `test_nvf.py` (incl. golden), `test_kfj.py`,
  `test_threads.py`
- `docs/{index,method,geometry,factorizations,performance,comparisons,api}.md`
- `CHANGELOG.md`, updated `pyproject.toml`

### Key Links (critical connections that must be WIRED)

- `Structure.shapes` → `Solver._segment_ops` → `build_nvf_operators` /
  `build_kfj_operators` → `EpsOperators.exy` → `build_fg` g12/g21 blocks
- `Solver(symmetry="x")` sector slicing → `exy[np.ix_(sec.X, sec.Y)]`
- `Solver(blas_threads=n)` → threadpool_limits scope around `smatrix`
- High-contrast warning → names NVF when shapes present
- README → docs/index.md → per-topic pages (no broken links)

## Open Questions

- Exact default for `NvfConfig.window` (as a fraction of shape scale) —
  decided by Task 3's calibration, recorded in the plan on completion.
- **For the user at approval:** will the GitHub repo be public at release
  time? README→docs links must be absolute URLs (docs/ is not in the
  wheel); a private repo leaves PyPI readers with dead links — the
  alternative is shipping docs/ in the sdist and linking relatively.

### Deferred Ideas

- Corner-adapted normal fields (crescent horns) — next research iteration
- ASR×NVF composition study
- Sphinx/mkdocs site build on top of the Markdown tree
- Auto-selection heuristic (`factorization="auto"`) once NVF has more
  field time
