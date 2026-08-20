"""Example 3: exceptional-point hunt in a lossy two-resonator obstacle.

Two coupled dielectric-slab resonators in WR-90; slab B has tunable thickness
(geometry) and Im eps (loss).  The script:
  1. locates the coupled pole pair of det S_TE10(f) at zero loss,
  2. tracks the pair while ramping the loss (pole trajectories),
  3. Newton-solves the double-root system for the EP (sceptre.ep.find_ep),
  4. confirms the Puiseux sqrt-splitting and plots it on log-log axes.

Run:  uv run python examples/03_ep_hunt.py     (takes ~1 min)
Output: examples/output/ep_trajectories.png, ep_puiseux.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sceptre import Box, Structure, Solver, Waveguide
from sceptre.ep import find_ep, pole_pair, puiseux_fit
from sceptre.poles import find_zeros_poles, refine_pole

A, B = 0.02286, 0.01016  # WR-90
EPS_A = 9.0
T_A = 0.006
GAP = 0.012
Z_B = T_A + GAP
OUT = Path(__file__).parent / "output"


def det_te10(t_b: float, eps_b_im: float):
    boxes = [
        Box(0, A, 0, B, 0.0, T_A, EPS_A),
        Box(0, A, 0, B, Z_B, Z_B + t_b, EPS_A + 1j * eps_b_im),
    ]
    solver = Solver(Structure(Waveguide(A, B), boxes), M=1, N=1)
    return lambda f: solver.det_port_s(f, np.array([0]))  # TE10 block


def main() -> None:
    OUT.mkdir(exist_ok=True)

    print("1. surveying poles of det S_TE10 (lossless symmetric dimer)...")
    survey = find_zeros_poles(
        det_te10(T_A, 0.0), 10.0e9 - 0.7e9j, 2.4e9, 1.6e9, rel_tol=1e-11
    )
    poles = sorted(survey.poles, key=lambda p: p.real)
    print(
        f"   found {len(poles)} poles: "
        + ", ".join(f"{p / 1e9:.3f} GHz" for p in poles)
    )
    pairs = [(abs(p - q), p, q) for i, p in enumerate(poles) for q in poles[i + 1 :]]
    _, p1, p2 = min(pairs, key=lambda t: t[0])

    print("2. ramping loss, tracking the pair...")
    loss_grid = np.linspace(0.0, 3.4, 18)
    traj1, traj2 = [p1], [p2]
    best = (abs(p1 - p2), 0.0)
    for eps_im in loss_grid[1:]:
        det = det_te10(T_A, eps_im)
        traj1.append(refine_pole(det, traj1[-1], tol=1e-2))
        traj2.append(refine_pole(det, traj2[-1], tol=1e-2))
        if abs(traj1[-1] - traj2[-1]) < best[0]:
            best = (abs(traj1[-1] - traj2[-1]), eps_im)
    print(f"   closest approach at Im eps_B = {best[1]:.2f}")

    print("3. Newton on the double-root system (t_B, Im eps_B, f)...")
    i_best = int(np.argmin(np.abs(loss_grid - best[1])))
    result = find_ep(
        lambda tb, ei, f: 1.0 / det_te10(tb, ei)(f),
        p0=(T_A, best[1]),
        omega0=0.5 * (traj1[i_best] + traj2[i_best]),
        scales=(T_A, 1.0, 1e10),
    )
    t_ep, ei_ep = result.p
    print(
        f"   EP: t_B = {t_ep * 1e3:.5f} mm, Im eps_B = {ei_ep:.5f}, "
        f"f_EP = {result.omega / 1e9:.5f} GHz (converged={result.converged})"
    )

    print("4. Puiseux confirmation...")
    direction = (0.4e-3, 0.25)
    ts = np.geomspace(3e-3, 8e-2, 8)
    splits = []
    for t in ts:
        det = det_te10(t_ep + t * direction[0], ei_ep + t * direction[1])
        box = 0.45e9 * np.sqrt(t / ts[-1]) + 0.02e9
        w1, w2 = pole_pair(det, result.omega, box, min_cell=0.05 * box)
        splits.append(w1 - w2)
    fit = puiseux_fit(ts, splits)
    print(
        f"   fitted exponent {fit.exponent:.4f} (EP requires 1/2), "
        f"|c| = {abs(fit.coefficient) / 1e9:.3f} GHz"
    )

    # --- plots ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tr1, tr2 = np.array(traj1), np.array(traj2)
    ax.plot(tr1.real / 1e9, tr1.imag / 1e9, "o-", ms=4, label="pole 1")
    ax.plot(tr2.real / 1e9, tr2.imag / 1e9, "s-", ms=4, label="pole 2")
    ax.plot(
        result.omega.real / 1e9,
        result.omega.imag / 1e9,
        "k*",
        ms=14,
        label=f"EP (Im eps_B = {ei_ep:.3f})",
    )
    ax.set_xlabel("Re f [GHz]")
    ax.set_ylabel("Im f [GHz]")
    ax.set_title("Pole trajectories of the coupled dimer under loss ramp")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ep_trajectories.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ts, np.abs(splits) / 1e9, "o", label="pole splitting")
    ax.loglog(
        ts, 2 * abs(fit.coefficient) * np.sqrt(ts) / 1e9, "--", label=r"$2|c|\sqrt{t}$"
    )
    ax.set_xlabel("parameter perturbation t")
    ax.set_ylabel(r"$|\omega_+-\omega_-|$ [GHz]")
    ax.set_title(f"Puiseux fit: exponent {fit.exponent:.3f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ep_puiseux.png", dpi=150)
    print(f"plots: {OUT / 'ep_trajectories.png'}, {OUT / 'ep_puiseux.png'}")


if __name__ == "__main__":
    main()
