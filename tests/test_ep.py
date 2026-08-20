"""Spec test 6: exceptional point of a lossy two-resonator obstacle.

Two dielectric slabs (coupled Fabry-Perot resonators) in WR-90; slab B carries
tunable loss.  Search space: (t_B, Im eps_B) = (geometry, loss).  The EP is found by
Newton on the double-root system h = h' = 0 with h = 1/det S(omega) (ep.find_ep),
then confirmed by the Puiseux square-root splitting of the pole pair under a generic
parameter perturbation: |omega_+ - omega_-| ~ 2|c| sqrt(t).
"""

import numpy as np
import pytest

from sceptre.ep import find_ep, pole_pair, puiseux_fit
from sceptre.geometry import Box, Structure, Waveguide
from sceptre.poles import find_zeros_poles
from sceptre.solver import Solver

A, B = 0.02286, 0.01016  # WR-90
EPS_A = 9.0
T_A = 0.006
GAP = 0.012
Z_B = T_A + GAP  # slab B start (fixed); its thickness t_B is a search parameter


def _det_te10(t_b: float, eps_b_im: float):
    """det of the TE10 port S-matrix as a function of complex frequency [Hz]."""
    boxes = [
        Box(0, A, 0, B, 0.0, T_A, EPS_A),
        Box(0, A, 0, B, Z_B, Z_B + t_b, EPS_A + 1j * eps_b_im),
    ]
    solver = Solver(Structure(Waveguide(A, B), boxes), M=1, N=1)
    idx = np.array([0])  # TE10 is always the lowest-cutoff lead mode (a > b)
    return lambda f: solver.det_port_s(f, idx)


@pytest.fixture(scope="module")
def ep():
    # 1+2. Loss scan with boxed pole surveys: the coupled pair approaches with
    # increasing Im eps_B (0.97 GHz split lossless -> merge near Im eps_B ~ 2.3);
    # start Newton from the closest CLEANLY RESOLVED approach.
    best = None
    for eps_im in (0.0, 0.75, 1.5, 2.0, 2.25):
        survey = find_zeros_poles(
            _det_te10(T_A, eps_im), 8.7e9 - 1.2e9j, 1.6e9, 1.15e9, rel_tol=1e-10
        )
        if len(survey.poles) == 2:
            split = abs(survey.poles[0] - survey.poles[1])
            if best is None or split < best[0]:
                best = (split, eps_im, survey.poles[0], survey.poles[1])
    assert best is not None, "loss scan never resolved the coupled pole pair"
    _, eps_im0, p1, p2 = best

    # 3. Nested Newton on the double-root system in (t_B, Im eps_B, complex f).
    def h(t_b, eps_im, f):
        return 1.0 / _det_te10(t_b, eps_im)(f)

    result = find_ep(
        h,
        p0=(T_A, eps_im0),
        omega0=0.5 * (p1 + p2),
        scales=(T_A, 1.0, 1e10),
        bounds=((0.004, 0.008), (0.5, 4.0)),  # keep geometry sane and loss positive
    )
    assert result.converged, (
        f"EP Newton did not converge: p={result.p}, omega={result.omega}, "
        f"splitting={result.splitting:.3e}, iterations={result.iterations}"
    )
    assert result.splitting < 1e-7 * 1e10  # coalesced below tol_split_rel * scale
    return result


def test_ep_is_physical(ep):
    t_b, eps_im = ep.p
    assert 0.5 * T_A < t_b < 2.0 * T_A  # geometry stayed in a sane range
    assert eps_im > 0.0  # the EP requires loss
    assert ep.omega.imag < 0.0  # decaying resonance (e^{-i omega t} convention)


def test_pole_pair_coalesces_at_ep(ep):
    """Approaching the EP along the loss axis, the splitting collapses."""
    t_b, eps_im = ep.p
    far = pole_pair(
        _det_te10(t_b, eps_im * 0.7), ep.omega, 0.6e9, min_cell=0.05 * 0.6e9
    )
    near = pole_pair(
        _det_te10(t_b, eps_im * 0.985), ep.omega, 0.3e9, min_cell=0.05 * 0.3e9
    )
    split_far = abs(far[0] - far[1])
    split_near = abs(near[0] - near[1])
    assert split_near < 0.35 * split_far


def test_puiseux_square_root_splitting(ep):
    """|omega_+ - omega_-| ~ t^(1/2) along a generic parameter-space direction."""
    t_b, eps_im = ep.p
    direction = (0.4e-3, 0.25)  # (d t_B [m], d Im eps_B) -- generic, not tangent
    ts = np.geomspace(3e-3, 8e-2, 6)
    splittings = []
    for t in ts:
        det = _det_te10(t_b + t * direction[0], eps_im + t * direction[1])
        box = 0.45e9 * np.sqrt(t / ts[-1]) + 0.02e9
        w1, w2 = pole_pair(det, ep.omega, box, min_cell=0.05 * box)
        splittings.append(w1 - w2)
    fit = puiseux_fit(ts, splittings)
    assert abs(fit.exponent - 0.5) < 0.06, f"Puiseux exponent {fit.exponent:.3f} != 0.5"
    # consistency of the sqrt law across the scan range:
    ratio = abs(splittings[0]) / abs(splittings[-1])
    expected = np.sqrt(ts[0] / ts[-1])
    assert 0.5 * expected < ratio < 2.0 * expected
