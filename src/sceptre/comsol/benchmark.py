"""The COMSOL cross-verification benchmark: single source of truth.

Geometry: WR-90 rectangular waveguide (PEC walls), partial-height full-width
dielectric block (the convergence-study obstacle of tests/test_convergence.py),
TE10 rectangular ports across the single-mode band.

Both the SCEPTRE reference solution and every COMSOL route (MPh-driven or the
generated Java model) import these numbers -- change them here only.
"""

from __future__ import annotations

import numpy as np

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import C0, Solver

# --- geometry [m] ---
A = 0.02286  # WR-90 width
B = 0.01016  # WR-90 height
BLOCK_LEN = 0.008  # obstacle length along z
BLOCK_HEIGHT = 0.45 * B  # partial height -> genuinely hybrid (LSE/LSM) fields
EPS_BLOCK = 9.0  # lossless dielectric
LEAD_LEN = 0.015  # vacuum leads in the COMSOL model (evanescent decay before ports)

# --- sweep ---
F_MIN = 8.5e9
F_MAX = 12.0e9
N_FREQ = 15

# --- comparison thresholds (VALIDATION.md) ---
MAX_DS = 0.01  # |Delta S| < 1% away from resonances
MAX_DF_RES = 1e-3  # resonance-frequency agreement < 0.1%


def frequencies() -> np.ndarray:
    return np.linspace(F_MIN, F_MAX, N_FREQ)


def sceptre_solver(n_order: int = 24) -> Solver:
    block = Box(0.0, A, 0.0, BLOCK_HEIGHT, 0.0, BLOCK_LEN, EPS_BLOCK)
    return Solver(
        Structure(Waveguide(A, B), [block]), M=1, N=n_order, factorization="li"
    )


def sceptre_s11_s21(freqs=None, n_order: int = 24):
    """SCEPTRE TE10 S-parameters, phase-referenced at the obstacle faces."""
    freqs = frequencies() if freqs is None else np.asarray(freqs, dtype=float)
    solver = sceptre_solver(n_order)
    s11 = np.empty(len(freqs), dtype=complex)
    s21 = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        res = solver.smatrix(f)
        s11[i] = res.coeff(1, ("TE", 1, 0), 1, ("TE", 1, 0))
        s21[i] = res.coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0))
    return freqs, s11, s21


def te10_beta(freqs) -> np.ndarray:
    """TE10 propagation constant in the empty leads (for de-embedding)."""
    k0 = 2 * np.pi * np.asarray(freqs, dtype=float) / C0
    return np.sqrt((k0**2 - (np.pi / A) ** 2).astype(complex))


def deembed_comsol(freqs, s11_port, s21_port):
    """Adapt raw COMSOL S-parameters to SCEPTRE's convention and reference planes.

    Three steps:
    1. COMSOL RF uses the engineering time convention e^{+j omega t}; SCEPTRE uses
       the physics convention e^{-i omega t} (refs/CONVENTIONS.md), so conjugate.
    2. COMSOL references S at the ports, LEAD_LEN away from each obstacle face:
       S11_port = e^{2 i beta L} S11_face and S21_port = e^{2 i beta L} S21_face
       (L = LEAD_LEN on both sides), so divide the lead phases out.
    3. Port-mode polarity gauge: COMSOL's two rectangular-port TE10 modes and
       SCEPTRE's two lead modes differ by a global sign at port 2 (measured
       constant ratio S21_comsol/S21_sceptre = -1.000 +- 0.005 across the band,
       i.e. S -> D S D with D = diag(1, -1); no physical content).  Flip it.
    """
    s11 = np.conj(np.asarray(s11_port))
    s21 = -np.conj(np.asarray(s21_port))
    phase = np.exp(1j * te10_beta(freqs) * LEAD_LEN)
    return s11 / phase**2, s21 / phase**2
