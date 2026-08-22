# Comparing against other solvers and measurements

Every mismatch we ever chased against an external reference came down to
conventions or comparison protocol, never the physics. In order of
likelihood:

1. **Time convention.** SCEPTRE is e^{−iωt}; COMSOL, VNAs, and most
   engineering tools are e^{+jωt}. Conjugate their S-data for phase-resolved
   comparison (this also swaps handedness labels RCP ↔ LCP). Magnitudes
   compare directly.
2. **Reference planes.** SCEPTRE references S at the obstacle faces with
   zero lead length. De-embed external data through its leads (÷ e^{2iβL})
   before comparing phases. Magnitudes need none of this.
3. **Per-mode sign gauge.** A relative sign on a mode (e.g. TE01) between
   two codes is normalization, not error. It flips cross-terms and swaps
   enantiomer labels (T_RR ↔ T_LL). Determine it once per dataset by trying
   both and keeping the match; diagonal elements and |S| are
   gauge-invariant.
4. **Never compare single frequencies near a resonance.** A converged-but-
   shifted line makes fixed-frequency S-elements disagree wildly while the
   physics agrees. Real case: reference |S31|/|S41| = 0.68/0.64 vs SCEPTRE
   0.89/0.39 at the same frequency looked broken — evaluated 18 MHz up the
   flank (the actual line offset), SCEPTRE read 0.68/0.63. **Score
   spectrally, at each code's own resonance line**; treat the common line
   shift as one number that converges away with N (or vanishes under NVF).
   The same rule applies to derived quantities: an absorption-vs-tanδ slope
   evaluated at a fixed frequency on a truncation-shifted flank was 3×
   wrong; evaluated at the same *relative* spectral position it matched the
   reference to ~10%.
5. **Track lines by identity, not by argmax.** Real spectra are denser than
   any single reference window suggests (we found three real disk modes
   below the reference's own frequency window). Identify a line as the
   nearest strong peak to a reference position with an amplitude floor;
   global argmax jumps between modes as truncation or parameters shift.

## Resonances: complex-frequency poles, not peak fitting

Q factors and linewidths from `Solver.det_port_s` + `poles.find_zeros_poles`
beat FWHM fits of |S(f)|:

- no frequency grid fine enough to resolve a narrow line is ever needed
  (a Q ≈ 50 000 lossless line is invisible on any practical real-axis grid
  — its pole is found in a handful of evaluations);
- at realistic loss the transmission line can top out at |T| ~ 10⁻³ where
  peak fitting is hopeless, while the pole is unambiguous;
- when resonances overlap, peak fitting reports mode *spacing* as mode
  *splitting* — a published-retraction-grade mistake the pole route cannot
  make.

Practicalities: seed contours from a cheap real-axis scan; keep contours
away from lead-mode cutoffs (branch points of S(ω)); for narrow poles use a
winding-number box around the seed rather than Newton/secant from far away
(a Γ ≲ 1 MHz pole's convergence basin is smaller than typical seed error);
pass a fixed port-index set to `det_port_s` for analyticity across the
region (the default set is chosen at Re f — see its docstring). Rational
(AAA) continuation of S data from a real-axis window agrees with direct
complex-frequency solves to machine precision on validated cases and makes
excellent pole seeds.
