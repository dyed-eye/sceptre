"""COMSOL cross-verification driver.

1. Detect local COMSOL installations.
2. Always emit the ready-to-run Java model (manual route) into comsol/.
3. If COMSOL is present, attempt the MPh route: build + solve the benchmark,
   de-embed, compare against SCEPTRE, save overlay/deviation plots and a CSV.

Exit codes: 0 = MPh comparison ran and PASSED, 1 = ran and FAILED,
3 = COMSOL missing or MPh route failed (Java fallback emitted).
"""

import sys
import traceback
from pathlib import Path

import numpy as np

from sceptre.comsol import detect_comsol
from sceptre.comsol.model_gen import write_java

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "comsol"


def main() -> int:
    installs = detect_comsol()
    if installs:
        for ins in installs:
            print(f"found COMSOL: {ins.version} at {ins.root}")
            print(f"  mphserver: {ins.mphserver_exe}")
    else:
        print("no COMSOL installation found on this machine")

    comsol_bin = (
        str(installs[0].comsol_exe.parent)
        if installs
        else r"C:\Program Files\COMSOL\COMSOL61\Multiphysics\bin\win64"
    )
    java_path = write_java(OUT / "SceptreBenchmark.java", comsol_bin=comsol_bin)
    print(f"emitted Java fallback model: {java_path}")

    if not installs:
        print("SKIPPING live comparison: no COMSOL. Run the Java model manually,")
        print(
            "then: uv run python scripts/compare_comsol.py sceptre_comsol_sparams.csv"
        )
        return 3

    try:
        from sceptre.comsol.mph_driver import run_benchmark

        print("starting COMSOL via MPh (this builds, meshes and sweeps the model)...")
        sweep = run_benchmark(save_model=OUT / "sceptre_benchmark.mph")
        print(f"sweep done: {len(sweep.freqs)} frequencies")
        csv_path = OUT / "sceptre_comsol_sparams.csv"
        np.savetxt(
            csv_path,
            np.column_stack(
                [
                    sweep.freqs,
                    sweep.s11_port.real,
                    sweep.s11_port.imag,
                    sweep.s21_port.real,
                    sweep.s21_port.imag,
                ]
            ),
            delimiter=",",
            header="freq_Hz, Re S11(port), Im S11(port), Re S21(port), Im S21(port)",
        )
        print(f"saved COMSOL S-parameters: {csv_path}")
    except Exception:
        print("MPh route FAILED:")
        traceback.print_exc()
        print()
        print("Fallback: compile and run the emitted Java model, e.g.")
        print(f'  "{comsol_bin}\\comsolcompile.exe" {java_path}')
        print(
            f'  "{comsol_bin}\\comsolbatch.exe" -inputfile {java_path.with_suffix(".class")}'
        )
        print(
            "then: uv run python scripts/compare_comsol.py sceptre_comsol_sparams.csv"
        )
        return 3

    from sceptre.comsol.compare import compare_sparams

    report = compare_sparams(
        sweep.freqs, sweep.s11_port, sweep.s21_port, out_dir=OUT / "output"
    )
    print(report.summary())
    for p in report.plots:
        print(f"plot: {p}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
