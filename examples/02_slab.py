"""Example 2: full-cross-section dielectric slab -- SCEPTRE vs the analytic solution.

Run:  uv run python examples/02_slab.py
Output: examples/output/slab_sparams.png + console error summary.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sceptre import Box, Structure, Solver, Waveguide
from sceptre.solver import C0

A, B = 0.02286, 0.01016  # WR-90
L = 0.0061
EPS = 4.0
OUT = Path(__file__).parent / "output"


def analytic_te10(f_hz: float):
    k0 = 2 * np.pi * f_hz / C0
    kc2 = (np.pi / A) ** 2
    b1 = np.sqrt(k0**2 - kc2 + 0j)
    b2 = np.sqrt(EPS * k0**2 - kc2 + 0j)
    r = (b1 - b2) / (b1 + b2)  # zeta_TE = beta / k0
    X = np.exp(1j * b2 * L)
    s11 = r * (1 - X**2) / (1 - r**2 * X**2)
    s21 = (1 - r**2) * X / (1 - r**2 * X**2)
    return s11, s21


def main() -> None:
    struct = Structure(Waveguide(A, B), [Box(0, A, 0, B, 0, L, EPS)])
    solver = Solver(struct, M=2, N=2)

    freqs = np.linspace(7e9, 12.4e9, 109)
    s11_f, s21_f, s11_a, s21_a = [], [], [], []
    for f in freqs:
        res = solver.smatrix(f)
        s11_f.append(res.coeff(1, ("TE", 1, 0), 1, ("TE", 1, 0)))
        s21_f.append(res.coeff(2, ("TE", 1, 0), 1, ("TE", 1, 0)))
        a11, a21 = analytic_te10(f)
        s11_a.append(a11)
        s21_a.append(a21)
    s11_f, s21_f = np.array(s11_f), np.array(s21_f)
    s11_a, s21_a = np.array(s11_a), np.array(s21_a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(freqs / 1e9, np.abs(s11_f), "-", label="SCEPTRE |S11|")
    ax1.plot(freqs / 1e9, np.abs(s21_f), "-", label="SCEPTRE |S21|")
    ax1.plot(
        freqs[::6] / 1e9, np.abs(s11_a)[::6], "o", mfc="none", label="analytic |S11|"
    )
    ax1.plot(
        freqs[::6] / 1e9, np.abs(s21_a)[::6], "s", mfc="none", label="analytic |S21|"
    )
    ax1.set_ylabel("|S|")
    ax1.set_title(f"TE10 through an eps={EPS:g} slab, L={L * 1e3:.1f} mm (WR-90)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.semilogy(freqs / 1e9, np.abs(s11_f - s11_a), label="|dS11|")
    ax2.semilogy(freqs / 1e9, np.abs(s21_f - s21_a), label="|dS21|")
    ax2.set_xlabel("frequency [GHz]")
    ax2.set_ylabel("deviation from analytic")
    ax2.grid(alpha=0.3)
    ax2.legend()
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "slab_sparams.png", dpi=150)

    print(f"max |dS11| = {np.max(np.abs(s11_f - s11_a)):.3e}")
    print(f"max |dS21| = {np.max(np.abs(s21_f - s21_a)):.3e}")
    print(
        f"energy balance max ||S11|^2+|S21|^2-1| = "
        f"{np.max(np.abs(np.abs(s11_f) ** 2 + np.abs(s21_f) ** 2 - 1)):.3e}"
    )
    print(f"plot: {OUT / 'slab_sparams.png'}")


if __name__ == "__main__":
    main()
