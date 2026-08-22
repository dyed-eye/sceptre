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


# ---- tensor factorization plumbing (nvf / kfj) -------------------------------

_AWG = 0.032


def _cyl():
    from sceptre.shapes import Cylinder

    return Cylinder(_AWG / 2, _AWG / 2 + 1e-3, 15e-3, 0.0, 5e-3, 80.0 + 0j, k=32)


@pytest.mark.unit
@pytest.mark.parametrize("fac", ["nvf", "kfj"])
def test_tensor_factorizations_require_shapes(fac):
    struct = Structure(
        Waveguide(_AWG, _AWG), [Box(0.01, 0.02, 0.01, 0.02, 0, 5e-3, 4.0)]
    )
    with pytest.raises(ValueError, match="[Ss]hape"):
        Solver(struct, 4, 4, factorization=fac)


@pytest.mark.unit
@pytest.mark.parametrize("fac", ["nvf", "kfj"])
def test_tensor_factorizations_reject_mixed_boxes_and_shapes(fac):
    struct = Structure(
        Waveguide(_AWG, _AWG),
        [Box(0.001, 0.004, 0.001, 0.004, 0, 5e-3, 4.0)],
        shapes=[_cyl()],
    )
    with pytest.raises(ValueError, match="box"):
        Solver(struct, 4, 4, factorization=fac)


@pytest.mark.unit
def test_tensor_factorizations_exclusive_with_asr():
    from sceptre.asr import AsrConfig

    struct = Structure(Waveguide(_AWG, _AWG), shapes=[_cyl()])
    for fac in ("nvf", "kfj"):
        with pytest.raises(ValueError, match="ASR"):
            Solver(struct, 4, 4, factorization=fac, asr=AsrConfig())


@pytest.mark.unit
def test_tensor_config_requires_matching_factorization():
    from sceptre.kfj import KfjConfig
    from sceptre.nvf import NvfConfig

    struct = Structure(Waveguide(_AWG, _AWG), shapes=[_cyl()])
    with pytest.raises(ValueError, match="nvf"):
        Solver(struct, 4, 4, nvf=NvfConfig())
    with pytest.raises(ValueError, match="kfj"):
        Solver(struct, 4, 4, kfj=KfjConfig())


@pytest.mark.unit
def test_high_contrast_curved_li_recommends_nvf():
    struct = Structure(Waveguide(_AWG, _AWG), shapes=[_cyl()])
    with pytest.warns(UserWarning, match="nvf"):
        Solver(struct, 4, 4)


@pytest.mark.integration
def test_nvf_and_kfj_solve_through_solver_with_symmetry():
    """End-to-end wiring: sectored solve equals the full solve per tensor
    factorization (the kfj exy sector path gets its own equality check)."""
    import warnings

    struct = Structure(Waveguide(_AWG, _AWG), shapes=[_cyl()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for fac in ("nvf", "kfj"):
            s_full = Solver(struct, 12, 12, factorization=fac).smatrix(5.44e9)
            s_sect = Solver(struct, 12, 12, factorization=fac, symmetry="x").smatrix(
                5.44e9
            )
            a = s_full.smatrix.full()
            b = s_sect.smatrix.full()
            assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(a))
