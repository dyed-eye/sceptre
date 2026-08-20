"""COMSOL-free unit tests for the COMSOL bridge: detection, Java model generation,
de-embedding, and the comparison logic (fed with synthetic 'COMSOL' data).

The live MPh route (mph_driver.py) requires a licensed COMSOL installation and is
exercised by scripts/run_comsol_validation.py, not by the unit suite.
"""

import numpy as np
import pytest

from sceptre.comsol import benchmark as bm
from sceptre.comsol import detect_comsol
from sceptre.comsol.compare import compare_sparams
from sceptre.comsol.model_gen import write_java

N_ORDER = 12  # keep the SCEPTRE reference cheap in unit tests


@pytest.mark.unit
def test_detect_comsol_returns_well_formed_entries():
    installs = detect_comsol()  # may be empty on machines without COMSOL
    for ins in installs:
        assert ins.comsol_exe.is_file()
        assert ins.version


@pytest.mark.unit
def test_model_gen_writes_compilable_looking_java(tmp_path):
    path = write_java(tmp_path / "SceptreBenchmark.java")
    text = path.read_text(encoding="utf-8")
    assert "public class SceptreBenchmark" in text
    assert f'"{bm.EPS_BLOCK}"' in text  # dielectric constant reaches the material
    assert "PortType" in text and "Rectangular" in text
    assert "sceptre_comsol_sparams.csv" in text
    assert text.count("{") == text.count("}")  # no format-template leftovers


def _embed_to_ports(freqs, s11_face, s21_face):
    """Exact inverse of bm.deembed_comsol: SCEPTRE-convention face S-parameters ->
    raw COMSOL port values (conjugate for e^{+j omega t}, lead phases, port-2 sign)."""
    phase = np.exp(1j * bm.te10_beta(freqs) * bm.LEAD_LEN)
    return np.conj(s11_face * phase**2), -np.conj(s21_face * phase**2)


@pytest.mark.unit
def test_deembed_inverts_embedding():
    freqs = bm.frequencies()
    s11_face = 0.3 * np.exp(0.7j) * np.ones(len(freqs))
    s21_face = 0.9 * np.exp(-0.2j) * np.ones(len(freqs))
    s11_back, s21_back = bm.deembed_comsol(
        freqs, *_embed_to_ports(freqs, s11_face, s21_face)
    )
    assert np.allclose(s11_back, s11_face, atol=1e-14)
    assert np.allclose(s21_back, s21_face, atol=1e-14)


@pytest.fixture(scope="module")
def synthetic_comsol():
    """SCEPTRE's own solution re-embedded to the ports + small 'FEM noise'."""
    freqs, s11, s21 = bm.sceptre_s11_s21(n_order=N_ORDER)
    rng = np.random.default_rng(7)
    noise = 2e-4 * np.exp(2j * np.pi * rng.random(len(freqs)))
    s11_port, s21_port = _embed_to_ports(freqs, s11 + noise, s21 + noise)
    return freqs, s11_port, s21_port


def test_compare_passes_on_selfconsistent_data(synthetic_comsol, tmp_path):
    freqs, s11_port, s21_port = synthetic_comsol
    report = compare_sparams(
        freqs, s11_port, s21_port, n_order=N_ORDER, out_dir=tmp_path
    )
    assert report.max_ds11 < 1e-3
    assert report.max_ds21 < 1e-3
    assert report.passed
    assert len(report.plots) == 2  # overlay + deviation figures written


def test_compare_fails_on_wrong_data(synthetic_comsol):
    freqs, s11_port, s21_port = synthetic_comsol
    report = compare_sparams(freqs, s11_port, 1.05 * s21_port, n_order=N_ORDER)
    assert not report.passed
