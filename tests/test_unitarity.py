"""Spec test 3: lossless obstacle -- unitarity, reciprocity, energy conservation.

A lossless partial-height / partial-width block exercises the full vectorial
machinery (TE/TM coupling, Li factorization).  Structural identities that must hold
to ~1e-10 at ANY truncation:

* S restricted to propagating ports is unitary: S^dagger S = I,
* reciprocity in the pseudo-flux normalization: S = S^T (all modes, both ports),
* energy conservation: sum_i |S_ij|^2 = 1 for every propagating input j.
"""

import numpy as np
import pytest

from sceptre.geometry import Box, Structure, Waveguide
from sceptre.solver import Solver

A, B = 0.02286, 0.01016  # WR-90


@pytest.fixture(scope="module")
def result():
    block = Box(0.3 * A, 0.85 * A, 0.0, 0.55 * B, 0.0, 0.006, 6.0)
    struct = Structure(Waveguide(A, B), [block])
    solver = Solver(struct, M=6, N=6, factorization="li")
    return solver.smatrix(18.0e9)  # several propagating modes


def test_propagating_block_is_unitary(result):
    sp = result.port_smatrix()  # propagating modes only
    gram = sp.conj().T @ sp
    assert np.max(np.abs(gram - np.eye(sp.shape[0]))) < 1e-10


def test_energy_conservation_row_sums(result):
    sp = result.port_smatrix()
    sums = np.sum(np.abs(sp) ** 2, axis=0)
    assert np.max(np.abs(sums - 1.0)) < 1e-10


def test_reciprocity_full_smatrix(result):
    s = result.smatrix.full()
    denom = np.max(np.abs(s))
    assert np.max(np.abs(s - s.T)) < 1e-10 * denom
