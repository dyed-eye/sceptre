"""Compare a COMSOL-exported S-parameter CSV against SCEPTRE on the shared benchmark.

Usage:
    uv run python scripts/compare_comsol.py sceptre_comsol_sparams.csv [out_dir]

CSV columns (COMSOL table export, '%' comment lines ignored):
    freq [Hz], Re S11, Im S11, Re S21, Im S21

Asserts |Delta S| < 1% away from resonances and resonance-frequency agreement
< 0.1%; writes overlay/deviation plots to out_dir (default: comsol/output).
"""

import sys
from pathlib import Path

import numpy as np

from sceptre.comsol.compare import compare_sparams


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    csv_path = Path(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else Path("comsol/output")
    if not csv_path.is_file():
        print(f"no such file: {csv_path}")
        return 2

    try:
        # Accept both COMSOL table exports ('%' comments) and our own CSVs ('#').
        rows = [
            ln
            for ln in csv_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith(("%", "#"))
        ]
        data = np.genfromtxt(rows, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 5:
            data = np.genfromtxt(rows)  # whitespace-separated fallback
    except (ValueError, OSError) as exc:
        print(f"could not parse {csv_path}: {exc}")
        return 2
    if data.ndim != 2 or data.shape[1] < 5 or not np.all(np.isfinite(data[:, :5])):
        print(f"could not parse 5 finite numeric columns from {csv_path}")
        return 2

    freqs = data[:, 0]
    s11 = data[:, 1] + 1j * data[:, 2]
    s21 = data[:, 3] + 1j * data[:, 4]

    report = compare_sparams(freqs, s11, s21, out_dir=out_dir)
    print(report.summary())
    for p in report.plots:
        print(f"plot: {p}")

    assert report.max_ds11 < 0.01, f"|dS11| = {report.max_ds11:.4%} exceeds 1%"
    assert report.max_ds21 < 0.01, f"|dS21| = {report.max_ds21:.4%} exceeds 1%"
    if report.max_res_shift is not None:
        assert report.max_res_shift < 1e-3, (
            f"resonance shift {report.max_res_shift:.5%} exceeds 0.1%"
        )
    print("COMSOL cross-verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
