# Factorizations: li, direct, ASR, NVF, KFJ

The factorization is HOW the discontinuous permittivity is turned into
modal-space operators. It is the dominant accuracy lever at high contrast.
All numbers below are measured on this codebase (benchmark: ε = 80 ceramic
disk, r = 15 mm, h = 5 mm, touching the wall of a 32 mm square guide;
reference line 5.4357 GHz from an independent FEM solution confirmed by VNA
measurement to ±1 MHz).

## The use-case matrix

| | **Box-aligned** | **Smooth curved** (disks, fillets) | **Sharp-cornered** (notches, bare bar corners) |
|---|---|---|---|
| **Low contrast (ε ≲ 10)** | `li` (exact rules, exact geometry) | `li` on a staircase — simplest; all arms sub-percent by N = 12, `nvf` marginally faster (table below) | `li` on a staircase |
| **High contrast (ε ≳ 25)** | `li`; ASR if few isolated edges (block benchmark: ASR at N = 24 ≈ plain at N ≈ 80) | **`nvf`** — one N = 20 solve lands −0.8 MHz from the reference; `li` needs an N = 24/32/40 ladder + Richardson for the same | **`nvf`** — measured on the ε = 80 sharp-horned crescent: the N = 20 single solve lands 3 MHz from the `li` 3-rung Richardson anchor and N = 28 lands on it exactly (5.779 GHz); `li` raw at N = 24 was still +45 MHz. A generic level-set FD normal field suffices; no corner adaptation needed. |

Every cell in this table is backed by a measurement (the sharp-corner/NVF
cell: `reports/comsol_parity/09_nvf_closure.md`, P-C).

Away from the calibration band, expect a residual high-side bias that
shrinks super-linearly with N (measured +4…+10 MHz at N = 28 in the
5.7-5.85 GHz band vs pole-continued references). Because convergence is
faster than 1/N, two-point p = 1 Richardson OVERSHOOTS on `nvf` series —
quote single-N values with the rung drift as the uncertainty instead.

## `li` (default) — Li's exact rules on exact cells

Inverse rule along x/y per strip, direct rule across, assembled from exact
analytic overlap integrals. **Structurally unitary and reciprocal at any
truncation** — it can never lie about energy, which makes it the safe
default everywhere and the reference arm of every comparison. Its weakness:
at curved high-contrast boundaries line positions converge only ~1/N
(measured, benchmark disk: +214/+145/+112/+92 MHz at N = 12/16/20/24,
p ≈ 1.2). The cure is the N-ladder + Richardson protocol
([performance.md](performance.md)) or NVF.

## `direct` — Laurent rule (benchmark arm)

The naive factorization; kept because the convergence gap between `direct`
and `li` (tests/test_convergence.py) is the classic demonstration of why
factorization matters. Not for production use.

## ASR — adaptive spatial resolution (`asr=AsrConfig()`)

A smooth coordinate map compresses resolution at dielectric edges. Great
for a FEW isolated edges: on the validated ε = 80 rectangular block, ASR at
N = 24 matches plain Li at N ≈ 80. On dense staircases the solver thins
edges the basis cannot resolve and warns — expect roughly plain-Li
convergence there and prefer plain `li` with larger N, or NVF for curved
shapes. ASR is exclusive with the tensor factorizations.

## `nvf` — normal-vector-field factorization (high-contrast curved shapes)

Keeps ε sharp and changes the RULE: inverse rule along a windowed boundary
normal field, direct rule tangentially, symmetrized so S = Sᵀ and unitarity
hold structurally (measured 1e-15..1e-14). Requires `Shape` geometry
(normals); shapes-only structures.

Measured on the benchmark disk (window sweep, production assembly, error of
the pol-B line vs the 5.4357 GHz reference):

| window / scale | N=16 | N=20 |
|---|---|---|
| 0.08 | −20.8 MHz | −14.6 MHz |
| 0.12 | −8.5 | −5.9 |
| 0.15 | −23.6 | −2.7 |
| **0.20 (default)** | **+4.3** | **−0.8** |
| 0.27 | −30.8 | −0.1 |

- The default `NvfConfig(window=None)` resolves to **0.20 × shape scale**
  (a fraction, not a length — it transfers across bands; provenance:
  cm-scale X-band calibration). The N = 16 outliers at other windows are
  partly neighbor-mode grabs by scan trackers — see the caveat below.
- **One N = 20 solve replaces the whole Li ladder** (~40× less compute for
  ladder-grade accuracy). The golden regression test pins this:
  ±5 MHz at N = 20.
- NVF also resolves REAL modes the reference never scanned: below the
  reference window (5.30–5.42 GHz) it places three disk modes within
  ~5 MHz of independent Li-ladder Richardson values (5.359/5.317/≈5.406)
  already at N = 16.
- **Caveat — spectra are dense.** Because everything is near-converged at
  once, neighboring real modes appear at their true spacing. Track lines by
  *nearest peak to a reference with an amplitude floor*, never by global
  argmax.
- Validity guards: the window must stay below the shape scale (level-set
  medial axis / center singularity — enforced), overlapping windows of
  multiple shapes are rejected (blended normals are unverified), and
  `window=math.inf` is a single-shape Li-limit testing mode.

## `kfj` — subpixel smoothing (comparison tool; measured limitation)

Kottke–Farjadpour–Johnson anisotropic effective tensors on the boundary
cells of a fine grid — the standard FDTD/planewave trick, included for
cross-method comparison. **It does not fix high-contrast accuracy in an
FMM, and we measured why:** subpixel smoothing cancels *real-space grid*
error, but a spectral method has no grid — the basis resolves the smoothing
layer as a REAL graded ring, adding a first-order layer shift (the line
marches with cell count: 5.5198/5.5400/5.5585 at 48/96/192 cells across the
diameter, N = 16) plus spurious anisotropic-ring modes at ε ≈ 80. At ε = 4
it converges but is the weakest arm (table below). Prefer `li` at low
contrast and `nvf` at high contrast; use `kfj` when you specifically want
the smoothed-geometry model.

## Low-contrast evidence (ε = 4 disk, fixed 5.6 GHz, max |ΔS₄| vs a converged li N = 28 reference; scripts/lowcontrast_ladder.log)

| arm | N=8 | N=12 | N=16 |
|---|---|---|---|
| li | **3.8e-3** | 2.0e-3 | 1.1e-3 |
| nvf | 5.1e-3 | **1.7e-3** | **6.5e-4** |
| kfj | 4.4e-3 | 2.7e-3 | 2.1e-3 |

At low contrast every arm converges to sub-percent by N = 12. `li` is best
at the coarsest truncation and needs no window machinery; `nvf` (with the
calibrated default window) converges slightly faster from N = 12 up; `kfj`
is the weakest arm at every N. Recommendation stays `li` at low contrast —
on simplicity, not on a claimed accuracy edge it doesn't uniformly have.

## Selection logic in one paragraph

Boxes → `li`. Curved shapes at low contrast → `li` on the staircase.
Curved shapes at high contrast — smooth OR sharp-cornered — → `nvf` (the
solver's high-contrast warning will point you there when it applies; the
generic level-set normal field handles corners, measured on the sharp
crescent). ASR → high-contrast *blocks* with few edges. `direct` → never
(benchmarks only). `kfj` → cross-method comparison studies only.

A device whose permittivity is *physically* graded (printed infill, porous or
composite fill) is a different situation entirely, and the one case where smooth
ε is a genuine advantage rather than a modelling error — see
[inverse-design.md](inverse-design.md).
