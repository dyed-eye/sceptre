"""API-robustness regression tests (from code review findings)."""

import numpy as np
import pytest

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import Solver

A, B = 0.02286, 0.01016  # WR-90


def _dimer_solver():
    boxes = [
        Box(0, A, 0, B, 0.0, 0.006, 9.0),
        Box(0, A, 0, B, 0.018, 0.024, 9.0),
    ]
    return Solver(Structure(Waveguide(A, B), boxes), M=1, N=1)


def test_det_port_s_default_indices_work_at_complex_freq():
    """Without explicit indices, the port set must come from Re(freq) -- never from
    the complex frequency itself, where the propagating test is empty and det of a
    0 x 0 block would silently be 1.0 everywhere."""
    solver = _dimer_solver()
    f = 9.5e9 - 0.4e9j
    d_default = solver.det_port_s(f)
    d_explicit = solver.det_port_s(f, np.array([0]))  # TE10 only in band
    assert d_default == pytest.approx(d_explicit, rel=1e-12)
    assert abs(d_default - 1.0) > 1e-3  # genuinely frequency-dependent


def test_det_port_s_raises_below_cutoff():
    solver = _dimer_solver()
    with pytest.raises(ValueError, match="no propagating"):
        solver.det_port_s(3.0e9)  # below the TE10 cutoff (6.56 GHz)


@pytest.mark.unit
def test_solver_rejects_unknown_factorization():
    struct = Structure(Waveguide(A, B), [Box(0, A, 0, B, 0, 0.005, 4.0)])
    with pytest.raises(ValueError, match="factorization"):
        Solver(struct, M=2, N=2, factorization="Li")  # typo'd case


@pytest.mark.unit
def test_zero_permittivity_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        Box(0, A, 0, B, 0, 0.005, 0.0)
    with pytest.raises(ValueError, match="nonzero"):
        Structure(Waveguide(A, B), [], background=0.0)
