"""Compare COMSOL S-parameters against SCEPTRE on the shared benchmark.

Acceptance criteria (VALIDATION.md):
* |Delta S| < 1% for S11 and S21 away from resonances,
* resonance-frequency agreement < 0.1% (when a resonance dip lies in the band).

"Away from resonances" = grid points more than RES_EXCLUSION_STEPS grid steps from
any located |S21| dip: near a sharp dip a sub-permille frequency shift of the FEM
model produces a large pointwise |Delta S| that says nothing about model agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import benchmark as bm

RES_EXCLUSION_STEPS = 2
DIP_THRESHOLD = 0.7  # |S21| below this counts as a resonance dip


@dataclass
class ComparisonReport:
    freqs: np.ndarray
    s11_sceptre: np.ndarray
    s21_sceptre: np.ndarray
    s11_comsol: np.ndarray  # de-embedded to the obstacle faces
    s21_comsol: np.ndarray
    max_ds11: float  # away from resonances
    max_ds21: float
    resonances_sceptre: list[float] = field(default_factory=list)
    resonances_comsol: list[float] = field(default_factory=list)
    max_res_shift: float | None = None  # relative frequency shift
    excluded: np.ndarray | None = None  # mask of near-resonance points
    plots: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        ok = self.max_ds11 < bm.MAX_DS and self.max_ds21 < bm.MAX_DS
        if self.max_res_shift is not None:
            ok = ok and self.max_res_shift < bm.MAX_DF_RES
        return ok

    def summary(self) -> str:
        lines = [
            f"max |dS11| (off-resonance): {self.max_ds11:.4%}  (limit {bm.MAX_DS:.0%})",
            f"max |dS21| (off-resonance): {self.max_ds21:.4%}  (limit {bm.MAX_DS:.0%})",
        ]
        if self.max_res_shift is not None:
            lines.append(
                f"resonance shift: {self.max_res_shift:.5%}  (limit {bm.MAX_DF_RES:.1%})"
            )
        else:
            lines.append("no |S21| dip below threshold in band: resonance check n/a")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _dips(freqs: np.ndarray, mag: np.ndarray) -> list[float]:
    """Interior local minima of |S21| below DIP_THRESHOLD, parabolic refined."""
    out = []
    for i in range(1, len(mag) - 1):
        if mag[i] < DIP_THRESHOLD and mag[i] <= mag[i - 1] and mag[i] <= mag[i + 1]:
            y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
            denom = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-15 else 0.0
            df = freqs[1] - freqs[0]
            out.append(float(freqs[i] + np.clip(shift, -1, 1) * df))
    return out


def compare_sparams(
    freqs,
    s11_port,
    s21_port,
    *,
    deembed: bool = True,
    n_order: int = 24,
    out_dir: str | Path | None = None,
) -> ComparisonReport:
    """Full comparison; pass COMSOL S-parameters referenced at the ports."""
    freqs = np.asarray(freqs, dtype=float)
    if deembed:
        s11_c, s21_c = bm.deembed_comsol(
            freqs, np.asarray(s11_port), np.asarray(s21_port)
        )
    else:
        s11_c, s21_c = np.asarray(s11_port), np.asarray(s21_port)
    _, s11_s, s21_s = bm.sceptre_s11_s21(freqs, n_order=n_order)

    res_s = _dips(freqs, np.abs(s21_s))
    res_c = _dips(freqs, np.abs(s21_c))
    max_shift = None
    if res_s and res_c:
        max_shift = max(min(abs(fc - fs) / fs for fc in res_c) for fs in res_s)

    excl = np.zeros(len(freqs), dtype=bool)
    df = freqs[1] - freqs[0]
    for fr in res_s + res_c:
        excl |= np.abs(freqs - fr) <= RES_EXCLUSION_STEPS * df
    keep = ~excl
    if not np.any(keep):
        raise RuntimeError("resonance exclusion swallowed every grid point")

    report = ComparisonReport(
        freqs=freqs,
        s11_sceptre=s11_s,
        s21_sceptre=s21_s,
        s11_comsol=s11_c,
        s21_comsol=s21_c,
        max_ds11=float(np.max(np.abs(s11_c - s11_s)[keep])),
        max_ds21=float(np.max(np.abs(s21_c - s21_s)[keep])),
        resonances_sceptre=res_s,
        resonances_comsol=res_c,
        max_res_shift=max_shift,
        excluded=excl,
    )
    if out_dir is not None:
        report.plots = _plots(report, Path(out_dir))
    return report


def _plots(rep: ComparisonReport, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ghz = rep.freqs / 1e9
    paths = []

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    for ax, name, ss, sc in (
        (axes[0], "S11", rep.s11_sceptre, rep.s11_comsol),
        (axes[1], "S21", rep.s21_sceptre, rep.s21_comsol),
    ):
        ax.plot(ghz, np.abs(ss), "-", label=f"SCEPTRE |{name}|")
        ax.plot(ghz, np.abs(sc), "o", mfc="none", label=f"COMSOL |{name}|")
        ax.set_ylabel(f"|{name}|")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_xlabel("frequency [GHz]")
    fig.suptitle("SCEPTRE vs COMSOL: partial-height dielectric block in WR-90")
    p = out_dir / "comsol_overlay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogy(ghz, np.abs(rep.s11_comsol - rep.s11_sceptre), "s-", label="|dS11|")
    ax.semilogy(ghz, np.abs(rep.s21_comsol - rep.s21_sceptre), "o-", label="|dS21|")
    if rep.excluded is not None and np.any(rep.excluded):
        ax.semilogy(
            ghz[rep.excluded],
            np.abs(rep.s21_comsol - rep.s21_sceptre)[rep.excluded],
            "x",
            color="gray",
            label="excluded (near resonance)",
        )
    ax.axhline(bm.MAX_DS, color="r", ls="--", label="1% limit")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|Delta S|")
    ax.grid(alpha=0.3)
    ax.legend()
    p = out_dir / "comsol_deviation.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))
    return paths
