"""The COMSOL cross-verification benchmarks: single source of truth.

Two cases share the WR-90 guide, PEC walls and TE10 rectangular ports:

* STANDARD -- the partial-height eps = 9 block of tests/test_convergence.py,
  solved with plain Li factorization.
* CERAMIC  -- a short partial-height eps = 80 block (microwave-ceramic
  contrast), solved WITH ASR; this is the FEM verification of the ASR mode.

Both the SCEPTRE reference solution and every COMSOL route (MPh-driven or the
generated Java model) read these numbers -- change them here only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sceptre.asr import AsrConfig
from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import C0, Solver

# --- shared geometry [m] ---
A = 0.02286  # WR-90 width
B = 0.01016  # WR-90 height
LEAD_LEN = 0.015  # vacuum leads in the COMSOL model (evanescent decay before ports)

# --- comparison thresholds (VALIDATION.md) ---
MAX_DS = 0.01  # |Delta S| < 1% away from resonances
MAX_DF_RES = 1e-3  # resonance-frequency agreement < 0.1%


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    eps_block: complex
    block_len: float
    block_height: float
    f_min: float
    f_max: float
    n_freq: int
    n_order: int  # SCEPTRE truncation N (M = 1: the obstacle is full-width)
    use_asr: bool
    mesh_air: float  # COMSOL hmax outside the block
    mesh_diel: float  # COMSOL hmax inside the block


STANDARD = BenchmarkCase(
    name="standard",
    eps_block=9.0,
    block_len=0.008,
    block_height=0.45 * B,
    f_min=8.5e9,
    f_max=12.0e9,
    n_freq=15,
    n_order=24,
    use_asr=False,
    mesh_air=2.4e-3,
    mesh_diel=1.0e-3,
)

CERAMIC = BenchmarkCase(
    name="ceramic",
    eps_block=80.0,
    block_len=0.003,
    block_height=0.45 * B,
    f_min=8.5e9,
    f_max=12.0e9,
    n_freq=29,  # denser grid: sharp high-Q resonances at this contrast
    n_order=40,
    use_asr=True,
    mesh_air=2.2e-3,
    mesh_diel=0.35e-3,  # ~ lambda_diel/9 at 12 GHz in eps = 80
)

# Legacy aliases (STANDARD values); prefer passing a BenchmarkCase explicitly.
BLOCK_LEN = STANDARD.block_len
BLOCK_HEIGHT = STANDARD.block_height
EPS_BLOCK = STANDARD.eps_block
F_MIN = STANDARD.f_min
F_MAX = STANDARD.f_max
N_FREQ = STANDARD.n_freq


def frequencies(case: BenchmarkCase = STANDARD) -> np.ndarray:
    return np.linspace(case.f_min, case.f_max, case.n_freq)


def sceptre_solver(
    n_order: int | None = None, case: BenchmarkCase = STANDARD
) -> Solver:
    block = Box(0.0, A, 0.0, case.block_height, 0.0, case.block_len, case.eps_block)
    return Solver(
        Structure(Waveguide(A, B), [block]),
        M=1,
        N=n_order if n_order is not None else case.n_order,
        factorization="li",
        asr=AsrConfig() if case.use_asr else None,
    )


def sceptre_s11_s21(
    freqs=None, n_order: int | None = None, case: BenchmarkCase = STANDARD
):
    """SCEPTRE TE10 S-parameters, phase-referenced at the obstacle faces."""
    freqs = frequencies(case) if freqs is None else np.asarray(freqs, dtype=float)
    solver = sceptre_solver(n_order, case)
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
