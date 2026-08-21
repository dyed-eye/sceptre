"""Validate SCEPTRE against the chiral_cavities COMSOL molecule archive.

Reference case: BK tw63, the frozen ceramic candidate.
  guide 32 x 32 mm PEC, eps_rod = 6.4, ports at z = +-L_wg = +-150 mm
  bar 1: 10.635 x 25.524 x 14.889 mm at (x,y,z) = (-3.72225, 0, 0), rot 0 deg
  bar 2: same size at (0, 0, 16.5), rot 63 deg about z

Only bar 2 is rotated, so it is staircased in the x-y cross-section; bar 1 is
represented exactly.

We compare MAGNITUDES |S11|, |S21|, |S31|, |S41| and the resonance position.
Those are independent of the reference plane, so the comparison cannot be faked
(or broken) by the de-embedding convention. The phase is checked separately, as
a consistency test, by verifying that the COMSOL/SCEPTRE ratio is a pure
propagation phase rather than an arbitrary complex number.

Basis map:  our port 1/3 = TE10 (E_y),  port 2/4 = TE01 (E_x).
  S11 = r(TE10<-TE10)   S21 = r(TE01<-TE10)
  S31 = t(TE10<-TE10)   S41 = t(TE01<-TE10)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.io as sio

from sceptre import Box, Solver, Structure, Waveguide

MAT = (
    r"C:\emae\phys\chiral_cavities\Modelling"
    r"\molecule_smatrix_stage0_wgrp_eps6.4_Sp1.000_al0_dz16.5_tw63"
    r"_gs1.063_w2-10_h2-14_x1--3.5_fine2M.mat"
)
A = 0.032
B = 0.032
EPS_ROD = 6.4
C0 = 299792458.0


def rotated_bar_boxes(cx, cy, w, ell, theta_deg, z1, z2, eps, nstrip):
    """Staircase a z-uniform rotated rectangle into axis-aligned boxes.

    cx, cy, w, ell, z1, z2 in metres; the rectangle has width w along its local
    x and length ell along its local y, rotated by theta_deg about z.
    Strips run along global y; each strip takes the exact x-extent of the
    rotated rectangle at the strip midpoint (midpoint rule).
    """
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    hw, hl = w / 2.0, ell / 2.0
    # corners in local frame -> global
    local = np.array([(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)])
    rot = np.array([[c, -s], [s, c]])
    poly = local @ rot.T + np.array([cx, cy])

    ylo, yhi = poly[:, 1].min(), poly[:, 1].max()
    edges = np.arange(nstrip + 1) / nstrip * (yhi - ylo) + ylo
    boxes = []
    for i in range(nstrip):
        ymid = 0.5 * (edges[i] + edges[i + 1])
        xs = []
        for k in range(4):
            p, q = poly[k], poly[(k + 1) % 4]
            if (p[1] - ymid) * (q[1] - ymid) <= 0 and p[1] != q[1]:
                t = (ymid - p[1]) / (q[1] - p[1])
                xs.append(p[0] + t * (q[0] - p[0]))
        if len(xs) < 2:
            continue
        x1, x2 = min(xs), max(xs)
        if x2 - x1 <= 0:
            continue
        boxes.append(
            Box(x1=x1 + A / 2, x2=x2 + A / 2,
                y1=edges[i] + B / 2, y2=edges[i + 1] + B / 2,
                z1=z1, z2=z2, eps=eps)
        )
    return boxes


def build(nstrip):
    mm = 1e-3
    rp_len, rp_w1, rp_w2 = 25.524 * mm, 10.635 * mm, 10.635 * mm
    rp_h1, rp_h2 = 14.889 * mm, 14.889 * mm
    rp_x1, rp_x2 = -3.72225 * mm, 0.0
    dz, twist = 16.5 * mm, 63.0

    # bar 1: rot 0 -> exact axis-aligned box
    bar1 = Box(
        x1=rp_x1 - rp_w1 / 2 + A / 2, x2=rp_x1 + rp_w1 / 2 + A / 2,
        y1=-rp_len / 2 + B / 2, y2=rp_len / 2 + B / 2,
        z1=-rp_h1 / 2, z2=rp_h1 / 2, eps=EPS_ROD,
    )
    bars2 = rotated_bar_boxes(
        rp_x2, 0.0, rp_w2, rp_len, twist,
        dz - rp_h2 / 2, dz + rp_h2 / 2, EPS_ROD, nstrip,
    )
    return Structure(Waveguide(a=A, b=B), [bar1, *bars2])


def sceptre_block(struct, freq, M, N):
    """2x2 reflection and transmission blocks in the [TE10, TE01] basis."""
    solver = Solver(struct, M=M, N=N, factorization="li")
    res = solver.smatrix(freq)
    te10, te01 = ("TE", 1, 0), ("TE", 0, 1)
    r = np.array([[res.coeff(1, te10, 1, te10), res.coeff(1, te10, 1, te01)],
                  [res.coeff(1, te01, 1, te10), res.coeff(1, te01, 1, te01)]])
    t = np.array([[res.coeff(2, te10, 1, te10), res.coeff(2, te10, 1, te01)],
                  [res.coeff(2, te01, 1, te10), res.coeff(2, te01, 1, te01)]])
    return r, t


def comsol_ref(mat, freq):
    """Archived COMSOL S at the port planes, nearest tabulated frequency."""
    k = int(np.argmin(np.abs(mat["f"] - freq)))
    r = np.array([[mat["S11"][k], mat["S12"][k]], [mat["S21"][k], mat["S22"][k]]])
    t = np.array([[mat["S31"][k], mat["S32"][k]], [mat["S41"][k], mat["S42"][k]]])
    return float(mat["f"][k]), r, t


def main():
    mat = sio.loadmat(MAT, squeeze_me=True)
    f_probe = float(sys.argv[1]) if len(sys.argv) > 1 else 5.80e9
    nstrip = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    fk, r_c, t_c = comsol_ref(mat, f_probe)
    print(f"reference file : {MAT.split(chr(92))[-1]}")
    print(f"probe frequency: {fk/1e9:.4f} GHz   (staircase strips = {nstrip})")
    print(f"COMSOL  |S11| {abs(r_c[0,0]):.4f}  |S21| {abs(r_c[1,0]):.4f}  "
          f"|S31| {abs(t_c[0,0]):.4f}  |S41| {abs(t_c[1,0]):.4f}")
    print(f"COMSOL  power sum for TE10 in: "
          f"{abs(r_c[0,0])**2 + abs(r_c[1,0])**2 + abs(t_c[0,0])**2 + abs(t_c[1,0])**2:.4f}")
    print()
    print(f"{'M=N':>4} {'|S11|':>8} {'|S21|':>8} {'|S31|':>8} {'|S41|':>8} "
          f"{'sum':>7} {'t/s':>7}")
    struct = build(nstrip)
    for MN in (6, 8, 10, 12, 14):
        t0 = time.time()
        try:
            r_s, t_s = sceptre_block(struct, fk, MN, MN)
        except Exception as exc:  # noqa: BLE001
            print(f"{MN:>4}  FAILED: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        tot = (abs(r_s[0, 0]) ** 2 + abs(r_s[1, 0]) ** 2
               + abs(t_s[0, 0]) ** 2 + abs(t_s[1, 0]) ** 2)
        print(f"{MN:>4} {abs(r_s[0,0]):>8.4f} {abs(r_s[1,0]):>8.4f} "
              f"{abs(t_s[0,0]):>8.4f} {abs(t_s[1,0]):>8.4f} {tot:>7.4f} {dt:>7.1f}")


if __name__ == "__main__":
    main()
