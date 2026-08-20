"""Example 1: empty WR-90 guide -- modal dispersion and pure-phase S-matrix.

Run:  uv run python examples/01_empty_guide.py
Output: examples/output/empty_guide_dispersion.png + console summary.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sceptre import Box, Structure, Solver, Waveguide
from sceptre.solver import C0

A, B = 0.02286, 0.01016  # WR-90
L = 0.020
OUT = Path(__file__).parent / "output"


def main() -> None:
    struct = Structure(Waveguide(A, B), [Box(0, A, 0, B, 0, L, 1.0)])
    solver = Solver(struct, M=3, N=3)

    freqs = np.linspace(5e9, 20e9, 121)
    betas = []
    for f in freqs:
        res = solver.smatrix(f)
        betas.append(res.lead.beta)
    betas = np.array(betas)
    labels = solver.smatrix(freqs[0]).lead.labels

    fig, ax = plt.subplots(figsize=(8, 5))
    shown = 0
    for i, lab in enumerate(labels):
        if shown >= 6:
            break
        kc = np.sqrt((lab[1] * np.pi / A) ** 2 + (lab[2] * np.pi / B) ** 2)
        fc = kc * C0 / (2 * np.pi)
        if fc < freqs[-1]:
            ax.plot(freqs / 1e9, betas[:, i].real, label=f"{lab[0]}{lab[1]}{lab[2]}")
            shown += 1
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"Re $\beta$ [rad/m]")
    ax.set_title("Empty WR-90: modal dispersion (analytic lead modes)")
    ax.grid(alpha=0.3)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "empty_guide_dispersion.png", dpi=150)

    # S-matrix of an empty section is a pure diagonal phase:
    res = solver.smatrix(12e9)
    s21 = res.smatrix.s21
    err_offdiag = np.max(np.abs(s21 - np.diag(np.diag(s21))))
    err_phase = np.max(np.abs(np.diag(s21) - np.exp(1j * res.lead.beta * L)))
    print(f"empty guide at 12 GHz, section length {L * 1e3:.0f} mm")
    print(f"  max |S21 off-diagonal|      = {err_offdiag:.3e}")
    print(f"  max |S21 - exp(i beta L)|   = {err_phase:.3e}")
    print(f"  max |S11|                   = {np.max(np.abs(res.smatrix.s11)):.3e}")
    print(f"plot: {OUT / 'empty_guide_dispersion.png'}")


if __name__ == "__main__":
    main()
