# SCEPTRE Conventions

Every reference below uses a different convention. This file is the single source of truth
for SCEPTRE; every module and test adheres to it. If a formula from a paper disagrees with
code, the code follows THIS file and the comment says how the paper's convention maps.

## Time and frequency

- Time convention: **e^(−iωt)** (physics convention).
  - Consequence: a lossy medium has **Im ε > 0**.
  - Consequence: resonance poles of S(ω) lie in the **lower half** complex-ω plane
    (Im ω_pole < 0), decaying quasi-normal modes ~ e^(−iω_p t).
  - Papers using e^(+jωt) (most microwave literature, e.g. Itoh) map via j = −i,
    i.e. conjugate all their formulas.

## Geometry and axes

- Waveguide cross-section: 0 ≤ x ≤ a, 0 ≤ y ≤ b; walls are PEC. Propagation axis z.
- Obstacles: axis-aligned boxes of piecewise-constant relative permittivity ε (complex),
  μ = 1 everywhere.

## Propagation and branch choice

- Forward (+z) dependence: **e^(+iβz)**; on the real-frequency axis Im β ≥ 0 (decay
  in +z) for every mode.
- Branch choice per channel, with disc = ε k0² − kc² (this defines the resonance
  sheet under complex-frequency continuation; see modes.py):
  * **open** channels (Re disc ≥ 0): β = principal sqrt(disc) — analytic across the
    real axis, acquiring Im β < 0 for Im ω < 0 (the outgoing/growing continuation a
    resonance pole requires);
  * **closed** channels (Re disc < 0): β = i·sqrt(−disc) — stays on the decaying
    branch (a plain principal sqrt would hug its cut along ℝ⁻ and jump to the
    exponentially growing side below the axis).
  Regions containing a lead cutoff (disc ≈ 0) are genuine branch points of S(ω) and
  must be excluded from contour searches.

## Fields and normalization

- Magnetic field is stored rescaled: **H̃ = Z0 · H**, Z0 = sqrt(μ0/ε0), so that
  Maxwell reads ∇×E = i k0 H̃, ∇×H̃ = −i k0 ε E with k0 = ω/c. All modal impedances
  are then dimensionless: Z_TE = k0/β, Z_TM = β/(k0 ε).
- Mode normalization: **reciprocity (pseudo-flux) normalization**
      ∫∫ (e_i × h̃_j) · ẑ dA = δ_ij      (NO complex conjugate)
  for forward modes. This is the normalization in which reciprocity reads **S = Sᵀ**.
  For a propagating mode of a lossless guide at real ω, mode fields are real and this
  coincides with fixed power P = Z0⁻¹·(1/2)·∫Re(E×H*)·ẑ dA per unit amplitude
  (up to the constant 1/2 Z0): |amplitude|² is proportional to carried power, so
  Σ_i |S_ij|² = 1 per column expresses energy conservation over propagating ports.
- Backward mode of the same index: same e, flipped h̃ (h̃ → −h̃), amplitude c⁻ with
  z-dependence e^(−iβz).
- **Internal storage (code convention):** the transverse H-vector is stored as
  **v = [H̃y; −H̃x]** so its blocks live in the SAME component spaces as e = [Ex; Ey]
  (H̃y shares Ex's cos·sin space, H̃x shares Ey's sin·cos space).  Consequences:
  pseudo-flux = plain dot product eᵀv; uniform-guide impedance relation collapses to
  v = ζ e (ζ = β/k0 for TE, εk0/β for TM); and both slice operators F, G of
  slicesolver.py become symmetric — the discrete form of Lorentz reciprocity that
  makes S = Sᵀ and lossless unitarity hold at any truncation.

## Ports and S-matrix

- Port 1: input face at z = z_min of the structure (waves incident in +z).
- Port 2: output face at z = z_max (waves incident in −z).
- Phase reference planes are exactly the structure faces (no lead line length included).
- b = S a with a = [a1; a2], b = [b1; b2] stacked per port, each block ordered by the
  lead-mode ordering (sorted by cutoff; TE before TM at equal cutoff; TE10 first for a > b).
- S11 = reflection at port 1; S21 = transmission port 1 → port 2.

## Basis bookkeeping (truncation M, N)

Orthonormal 1-D functions on [0, a] (same with b for y):
  c_0(x) = 1/√a,  c_m(x) = √(2/a) cos(mπx/a) (m ≥ 1),  s_m(x) = √(2/a) sin(mπx/a).

Component parities enforced by the PEC walls (image/mirror extension of the Fourier basis):

| component | x-basis | y-basis | m-range | n-range | space |
|-----------|---------|---------|---------|---------|-------|
| Ex, H̃y   | cos     | sin     | 0..M    | 1..N    | X     |
| Ey, H̃x   | sin     | cos     | 1..M    | 0..N    | Y     |
| Ez        | sin     | sin     | 1..M    | 1..N    | Z     |
| H̃z       | cos     | cos     | 0..M    | 0..N    | W     |

Mode count per lead: N_TE + N_TM = [(M+1)(N+1) − 1] + MN = dim(X) + dim(Y). Exactly
the dimension of the transverse field space — the modal basis is complete at every truncation.

## Fourier factorization (Li's rules, adapted to the sin/cos basis)

- ε·Ex (Dx continuous across x-normal edges): **inverse rule in x, direct rule in y**
  (Li 1997 crossed-grating rule ⌈⌊1/ε⌋_x⁻¹⌉_y), built strip-wise for piecewise-constant ε.
- ε·Ey: inverse rule in y, direct rule in x.
- Ez elimination, ε⁻¹(∂xH̃y − ∂yH̃x): Ez is continuous (tangential to all lateral edges),
  so use the **inverse of the direct-rule matrix**: (⟦ε⟧_ZZ)⁻¹.

## Units

SI throughout the public API: meters, Hz (frequency f; ω = 2πf), dimensionless ε.

## ASR (adaptive spatial resolution)

With Solver(..., asr=AsrConfig(eta)) the problem is solved in mapped coordinates
u, v with x = X(u), y = Y(v) compressing resolution near dielectric edges
(X' = eta there).  Implementation is the transformation-optics equivalent: the
undeformed box filled with eps~ = eps*diag(Y'/X', X'/Y', X'Y') and
mu~ = diag(Y'/X', X'/Y', X'Y'), acting on the transformed fields
E~ = (X' Ex, Y' Ey, Ez)  (Ward-Pendry, J = diag(1/X', 1/Y', 1)).  Because
dx dy = X'Y' du dv absorbs exactly the two metric factors of the transverse
components, the plain coefficient pseudo-flux e^T v EQUALS the physical
pseudo-flux: the ASR S-matrix is the physical S-matrix with no normalization
gauge.  Port modes are numerical eigenvectors of the transformed uniform lead,
matched to analytic TE/TM labels by field-pattern overlap (eigenvalue proximity
is NOT reliable in the deep-evanescent tail), aligned inside degenerate TE/TM
clusters, and flux-orthonormalized -- S = S^T and lossless unitarity hold to
machine precision.
