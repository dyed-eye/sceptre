# The method

SCEPTRE solves Maxwell's equations in a closed rectangular PEC waveguide
containing z-piecewise-uniform dielectric obstacles, by modal expansion in
the guide's exact transverse basis and stable S-matrix cascading along z.
This page maps the physics to the modules that implement it.

## Modal basis and parities — `basis.py`

The PEC boundary conditions on the guide walls (x ∈ [0, a], y ∈ [0, b])
force each field component into a definite sin/cos product basis:

| Space | Components | Basis | Indices |
|---|---|---|---|
| X | Ex, H̃y | cos_m(x)·s_n(y) | m = 0..M, n = 1..N |
| Y | Ey, H̃x | s_m(x)·cos_n(y) | m = 1..M, n = 0..N |
| Z | Ez | s_m(x)·s_n(y) | m,n ≥ 1 |
| W | H̃z | cos_m(x)·cos_n(y) | m,n ≥ 0 |

with orthonormal 1-D functions (c_0 = 1/√a, c_m = √(2/a)·cos(mπx/a), s_m =
√(2/a)·sin(mπx/a)). Derivative operators in this basis are exact one-entry-
per-row matrices (`ModeBasis.dx_XZ` etc.). The transverse truncation is
(M, N); the transverse vector dimension is T = 2MN + M + N.

## The per-slice eigenproblem — `slicesolver.py`

Eliminating Ez and H̃z from Maxwell's equations on a z-uniform slice gives
the first-order system de/dz = F v, dv/dz = G e for the transverse fields
e = [Ex; Ey], v = [H̃y; −H̃x]. Both F and G are complex-symmetric at ANY
truncation — the discrete form of Lorentz reciprocity — which is what makes
the final S-matrix satisfy S = Sᵀ and, for lossless media, unitarity to
machine precision. Tensor factorizations (NVF/KFJ) add a symmetric off-
diagonal ε_xy coupling into G and preserve this structure.

The modal ansatz e ~ exp(iβz) yields the dense eigenproblem
(FG)w = −β²w, solved with LAPACK `zgeev` (the dominant cost, O(T³)).
Uniform slices bypass the eigensolver: their TE/TM modes are analytic
(`modes.py`), machine-exact.

## Why factorization matters — `fourier.py`

Products of a discontinuous ε with a discontinuous field component must be
factorized with Li's rules (inverse rule along the discontinuity normal,
direct rule tangentially), or the modal expansion converges slowly and
non-uniformly at dielectric edges. For axis-aligned box layouts SCEPTRE
implements Li's crossed-grating rules with **exact analytic overlap
integrals** — no FFTs, no Gibbs error in the operators themselves. For
curved boundaries the correct rule follows the local normal: that is the
NVF factorization (`nvf.py`). The choice of rule is the single biggest
accuracy lever at high permittivity contrast — measured numbers in
[factorizations.md](factorizations.md).

## Stable cascading — `smatrix.py`

Slices are joined with interface S-matrices from modal continuity and
combined with the Redheffer star product (Li 1996). Every propagation
factor that appears is exp(iβd) with Im β ≥ 0, so growing exponentials
never enter — unlike the T-matrix recursion, which is exponentially
unstable for evanescent modes. Ports are referenced at the structure faces
with zero lead length; forward modes are normalized to eᵀv = 1 (no
conjugation), which makes reciprocity read S = Sᵀ.

## Conventions (load-bearing)

- Time convention **e^{−iωt}**; lossy media have **Im ε > 0**.
- Forward propagation e^{+iβz}, branch Im β ≥ 0 on the real axis.
- Under complex-frequency continuation, OPEN channels keep the principal
  sqrt (acquiring Im β < 0 below the real axis — the outgoing/growing
  continuation a resonance pole requires); CLOSED channels use i·sqrt(−disc)
  and stay decaying. Never "fix" branch signs after the fact: it silently
  switches the Riemann sheet (`modes.py` documents this per channel).
- Regions containing a lead-mode cutoff are genuine branch points of S(ω)
  and must be excluded from contour searches.

## Poles, zeros, exceptional points — `poles.py`, `ep.py`, `solver.py`

`Solver.smatrix(freq)` accepts complex frequency; per-layout permittivity
operators are frequency-independent and cached, so pole hunts only pay for
eigensolves. `Solver.det_port_s` gives the analytic scalar fed to
`poles.find_zeros_poles`, an adaptive winding-number contour search that
separates zeros from poles and resolves near-coalescent clusters through
moments. Resonance Q's and linewidths come from pole locations
(Q = Re f / (2|Im f|)) — no frequency grid fine enough to resolve a narrow
line is ever needed ([comparisons.md](comparisons.md) shows why this beats
FWHM fitting). `ep.py` locates exceptional points where eigenvalues of the
port S-matrix coalesce.
