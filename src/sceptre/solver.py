"""Top-level SCEPTRE solver: geometry + frequency -> modal scattering matrix.

Pipeline per frequency (real or complex):
  1. analytic lead modes at both ports (uniform background permittivity),
  2. per z-uniform segment: modal eigenproblem (numerical, or analytic if uniform),
  3. interface S-matrices from modal continuity + intra-slice propagation,
  4. stable Redheffer/Li cascading into the total two-port S-matrix.

The permittivity operators of each distinct cross-section layout are frequency
independent and cached, so frequency sweeps and complex-plane pole hunts only pay
for the eigensolves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .basis import ModeBasis
from .fourier import EpsOperators, build_eps_operators
from .geometry import CrossSection, Structure
from .modes import LeadModes, lead_modes
from .slicesolver import solve_slice
from .smatrix import SMatrix, cascade, interface_smatrix, propagation_smatrix

C0 = 299792458.0  # vacuum speed of light, m/s


@dataclass(frozen=True)
class SResult:
    """Scattering result at one frequency: full modal S plus the port-mode table."""

    freq: complex
    smatrix: SMatrix  # blocks are T x T over the full lead-mode set
    lead: LeadModes

    def propagating_indices(self) -> np.ndarray:
        return np.flatnonzero(self.lead.propagating())

    def port_smatrix(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Assembled 2p x 2p S over the selected lead modes (default: propagating)."""
        if indices is None:
            indices = self.propagating_indices()
        ix = np.ix_(indices, indices)
        s = self.smatrix
        return np.block([[s.s11[ix], s.s12[ix]], [s.s21[ix], s.s22[ix]]])

    def coeff(
        self,
        out_port: int,
        out_mode: tuple[str, int, int],
        in_port: int,
        in_mode: tuple[str, int, int],
    ) -> complex:
        """Single S-matrix element, e.g. coeff(2, ("TE",1,0), 1, ("TE",1,0)) = S21 of TE10."""
        i = self.lead.mode_index(*out_mode)
        j = self.lead.mode_index(*in_mode)
        block = {
            (1, 1): self.smatrix.s11,
            (1, 2): self.smatrix.s12,
            (2, 1): self.smatrix.s21,
            (2, 2): self.smatrix.s22,
        }[(out_port, in_port)]
        return complex(block[i, j])


class Solver:
    """Fourier-modal-method solver for one Structure at fixed truncation (M, N)."""

    def __init__(
        self,
        structure: Structure,
        M: int,
        N: int,
        factorization: str = "li",
        lead_eps: complex | None = None,
    ):
        if factorization not in ("li", "direct"):
            raise ValueError(f"unknown factorization {factorization!r}")
        self.structure = structure
        self.basis = ModeBasis(structure.waveguide.a, structure.waveguide.b, M, N)
        self.factorization = factorization
        self.lead_eps = complex(
            lead_eps if lead_eps is not None else structure.background
        )
        self.segments = structure.segments()
        self._ops_cache: dict[object, EpsOperators] = {}

    def _segment_ops(self, layout: CrossSection) -> EpsOperators | None:
        if layout.is_uniform:
            return None  # solve_slice takes the analytic path
        key = (layout.key(), self.factorization)
        if key not in self._ops_cache:
            self._ops_cache[key] = build_eps_operators(
                layout, self.basis, self.factorization
            )
        return self._ops_cache[key]

    def smatrix(self, freq: complex) -> SResult:
        """Total S-matrix at frequency freq [Hz] (complex allowed for continuation)."""
        k0 = 2.0 * np.pi * freq / C0
        lead = lead_modes(self.basis, k0, self.lead_eps)

        parts: list[SMatrix] = []
        prev_W, prev_V = lead.W, lead.V
        for seg in self.segments:
            modes = solve_slice(
                seg.cross_section,
                self.basis,
                k0,
                self.factorization,
                ops=self._segment_ops(seg.cross_section),
            )
            parts.append(interface_smatrix(prev_W, prev_V, modes.W, modes.V))
            parts.append(propagation_smatrix(modes.beta, seg.length))
            prev_W, prev_V = modes.W, modes.V
        parts.append(interface_smatrix(prev_W, prev_V, lead.W, lead.V))
        return SResult(freq, cascade(parts), lead)

    def sweep(self, freqs: Sequence[complex] | np.ndarray) -> list[SResult]:
        return [self.smatrix(f) for f in np.asarray(freqs).ravel()]

    def det_port_s(self, freq: complex, indices: np.ndarray | None = None) -> complex:
        """det of the port S-matrix -- the analytic function fed to the pole finder.

        indices: fixed lead-mode index set defining the ports.  For sweeps over a
        region, pass one index set determined at a real reference frequency so the
        function stays analytic across the region.  When omitted, the propagating
        set at Re(freq) is used -- NOT at the complex freq itself, where the
        near-real-beta test would come back empty and det of a 0 x 0 block would
        silently evaluate to 1.0 everywhere.
        """
        if indices is None:
            k0_ref = 2.0 * np.pi * float(np.real(freq)) / C0
            lead_ref = lead_modes(self.basis, k0_ref, self.lead_eps)
            indices = np.flatnonzero(lead_ref.propagating())
        if len(indices) == 0:
            raise ValueError(
                "no propagating lead modes at this frequency: the port S-matrix "
                "is empty (pass an explicit mode index set)"
            )
        res = self.smatrix(freq)
        return complex(np.linalg.det(res.port_smatrix(indices)))
