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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .asr import AsrConfig, build_asr_operators, build_maps
from .basis import ModeBasis
from .fourier import EpsOperators, build_eps_operators
from .geometry import CrossSection, Segment, Structure, Waveguide
from .kfj import KfjConfig, build_kfj_operators
from .nvf import NvfConfig, build_nvf_operators
from .modes import LeadModes, lead_modes
from .slicesolver import solve_slice
from .smatrix import SMatrix, cascade, interface_smatrix, propagation_smatrix
from .symmetry import Sector, lead_columns, require_x_symmetric, x_sectors
from .threads import blas_thread_limit

C0 = 299792458.0  # vacuum speed of light, m/s

# Above this permittivity contrast the plain Li factorization needs a large
# truncation order for percent-level accuracy (measured: eps = 80 still ~1% at
# N = 40); recommend ASR instead of failing silently slowly.
ASR_RECOMMEND_CONTRAST = 25.0

# Relative; cross-section edges must reach both PEC walls. Deliberately 1000x
# looser than geometry._GEOM_TOL: that one merges internally-computed
# breakpoints, this one validates caller-supplied grids, which accumulate more
# round-off. Still orders of magnitude below any physical feature size.
_SPAN_TOL = 1e-9


def _require_spanning_sections(
    segments: Sequence[Segment], waveguide: Waveguide
) -> None:
    """Every cross-section must tile the whole guide cross-section.

    Structure builds layouts seeded with [0, a] x [0, b], so this only bites on
    explicitly supplied grids (docs/inverse-design.md), where a map covering
    part of the guide otherwise leaves the remainder undefined and solves
    silently against the wrong geometry."""
    a, b = waveguide.a, waveguide.b
    for seg in segments:
        cs = seg.cross_section
        if (
            abs(cs.x_edges[0]) > _SPAN_TOL * a
            or abs(cs.x_edges[-1] - a) > _SPAN_TOL * a
            or abs(cs.y_edges[0]) > _SPAN_TOL * b
            or abs(cs.y_edges[-1] - b) > _SPAN_TOL * b
        ):
            raise ValueError(
                f"cross-section of the segment at z = [{seg.z1:g}, {seg.z2:g}] "
                f"must span the full guide: x_edges span "
                f"[{cs.x_edges[0]:g}, {cs.x_edges[-1]:g}] and y_edges span "
                f"[{cs.y_edges[0]:g}, {cs.y_edges[-1]:g}], expected "
                f"[0, {a:g}] x [0, {b:g}]"
            )


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
        asr: AsrConfig | None = None,
        symmetry: str | None = None,
        blas_threads: int | None = None,
        nvf: NvfConfig | None = None,
        kfj: KfjConfig | None = None,
    ):
        if factorization not in ("li", "direct", "nvf", "kfj"):
            raise ValueError(f"unknown factorization {factorization!r}")
        if blas_threads is not None and blas_threads < 1:
            raise ValueError("blas_threads must be >= 1 (or None to leave BLAS alone)")
        self.blas_threads = blas_threads
        if nvf is not None and factorization != "nvf":
            raise ValueError("an NvfConfig requires factorization='nvf'")
        if kfj is not None and factorization != "kfj":
            raise ValueError("a KfjConfig requires factorization='kfj'")
        if factorization in ("nvf", "kfj"):
            if asr is not None:
                raise ValueError(
                    f"factorization='{factorization}' is not supported together "
                    "with ASR (exclusive treatments of the same boundaries)"
                )
            if not structure.shapes:
                raise ValueError(
                    f"factorization='{factorization}' needs Shape geometry "
                    "(e.g. Cylinder) — a box staircase carries no normal field"
                )
            if structure.boxes:
                raise ValueError(
                    "mixed boxes + shapes are not supported with tensor "
                    "factorizations: outside the shape windows plain box edges "
                    "would lose Li's inverse rule (a Rectangle shape is the "
                    "planned path for box-like geometry)"
                )
        self.nvf_config = nvf or (NvfConfig() if factorization == "nvf" else None)
        self.kfj_config = kfj or (KfjConfig() if factorization == "kfj" else None)
        if asr is not None and factorization != "li":
            raise ValueError("ASR requires factorization='li'")
        if symmetry not in (None, "x"):
            raise ValueError(f"unknown symmetry {symmetry!r} (supported: 'x')")
        if symmetry is not None and asr is not None:
            raise ValueError("symmetry='x' is not supported together with ASR")
        self.structure = structure
        self.basis = ModeBasis(structure.waveguide.a, structure.waveguide.b, M, N)
        self.factorization = factorization
        self.lead_eps = complex(
            lead_eps if lead_eps is not None else structure.background
        )
        self.segments = structure.segments()
        _require_spanning_sections(self.segments, structure.waveguide)
        # Fixed for the object's lifetime: _lead/_segment_ops and the ops cache
        # assume it never changes after construction.
        self.asr = asr
        self.symmetry = symmetry
        self._sectors: tuple[Sector, Sector] | None = None
        if symmetry == "x":
            a = structure.waveguide.a
            for shape in structure.shapes:
                bx1, bx2, _y1, _y2 = shape.bbox
                if abs(0.5 * (bx1 + bx2) - 0.5 * a) > 1e-9 * a:
                    raise ValueError(
                        f"symmetry='x' needs x-centred shapes; {shape} is not "
                        "mirror-symmetric about a/2"
                    )
            for seg in self.segments:
                require_x_symmetric(seg.cross_section, structure.waveguide.a)
            self._sectors = x_sectors(self.basis)
        self._ops_cache: dict[object, EpsOperators] = {}
        if asr is not None:
            # Compression intervals shorter than the basis half-period 2a/M
            # (2b/N) put metric content beyond the representable bandwidth --
            # aliased operators and an unstable numerical lead stage. The Li
            # inverse-rule PRODUCTS double the bandwidth need, hence the extra
            # factor 2 (measured on a staircased eps=80 disk: max column-energy
            # error 1.1e-3 at 2a/M vs 1e-6 at 4a/M).
            self._xmap, self._ymap = build_maps(
                structure,
                asr.eta,
                min_x=4.0 * structure.waveguide.a / M,
                min_y=4.0 * structure.waveguide.b / N,
            )
            self._patterns = None  # built lazily (import kept local to _lead)
        self._maybe_recommend_high_contrast_cure()

    def _maybe_recommend_high_contrast_cure(self) -> None:
        if self.factorization in ("nvf", "kfj"):
            return
        mags = [abs(self.lead_eps)] + [
            abs(v) for seg in self.segments for v in seg.cross_section.eps_cells.ravel()
        ]
        contrast = max(mags) / min(mags)
        if self.asr is None and contrast >= ASR_RECOMMEND_CONTRAST:
            if self.structure.shapes:
                hint = (
                    "this structure has curved Shape geometry: "
                    "factorization='nvf' reaches the converged line position "
                    "at N ~ 16-20 in one solve where plain Li needs an "
                    "N-ladder plus Richardson extrapolation"
                )
            else:
                hint = (
                    "consider Solver(..., asr=AsrConfig()) or verify "
                    "convergence by increasing N"
                )
            warnings.warn(
                f"permittivity contrast {contrast:.0f} is high: the plain Li "
                f"factorization converges slowly here — {hint}.",
                UserWarning,
                stacklevel=3,
            )

    def _lead(self, k0: complex) -> LeadModes:
        if self.asr is None:
            return lead_modes(self.basis, k0, self.lead_eps)
        from .asr_modes import asr_lead_modes, lead_patterns

        if self._patterns is None:
            self._patterns = lead_patterns(self.basis, self._xmap, self._ymap)
        layout = CrossSection(
            np.array([0.0, self.basis.a]),
            np.array([0.0, self.basis.b]),
            np.array([[self.lead_eps]], dtype=complex),
        )
        ops = self._segment_ops(layout)
        assert ops is not None  # asr is set, so _segment_ops always builds
        return asr_lead_modes(self.basis, k0, self.lead_eps, ops, self._patterns)

    def _segment_ops(self, layout: CrossSection) -> EpsOperators | None:
        if self.asr is None and layout.is_uniform:
            return None  # solve_slice takes the analytic path
        key = (layout.key(), self.factorization, self.asr)
        if key not in self._ops_cache:
            if self.asr is not None:
                self._ops_cache[key] = build_asr_operators(
                    layout, self.basis, self._xmap, self._ymap
                )
            else:
                self._ops_cache[key] = build_eps_operators(
                    layout, self.basis, self.factorization
                )
        return self._ops_cache[key]

    def _ops_for(self, seg: Segment) -> EpsOperators | None:
        """Per-segment operator dispatch.  Tensor factorizations route shape
        segments to their builders; shape-free segments in a shapes-only
        structure are uniform background (analytic path, ops None)."""
        if self.factorization not in ("nvf", "kfj"):
            return self._segment_ops(seg.cross_section)
        if not seg.shapes:
            return None
        config = self.nvf_config if self.factorization == "nvf" else self.kfj_config
        key = (
            seg.cross_section.key(),
            self.factorization,
            config,
            tuple(id(s) for s in seg.shapes),  # Structure is fixed for the
            # Solver's lifetime, so id() is a safe per-instance identity and
            # imposes no hashability requirement on user Shape subclasses
        )
        if key not in self._ops_cache:
            if self.factorization == "nvf":
                ops = build_nvf_operators(
                    seg.shapes,
                    seg.cross_section,
                    self.structure.waveguide,
                    self.basis,
                    self.nvf_config,
                )
            else:
                ops = build_kfj_operators(
                    seg.shapes,
                    seg.cross_section,
                    self.structure.waveguide,
                    self.basis,
                    self.kfj_config,
                    background=self.structure.background,
                )
            self._ops_cache[key] = ops
        return self._ops_cache[key]

    def smatrix(self, freq: complex) -> SResult:
        """Total S-matrix at frequency freq [Hz] (complex allowed for continuation)."""
        with blas_thread_limit(self.blas_threads):
            return self._smatrix_impl(freq)

    def _smatrix_impl(self, freq: complex) -> SResult:
        k0 = 2.0 * np.pi * freq / C0
        lead = self._lead(k0)
        if self._sectors is None:
            return SResult(freq, self._cascade(lead, k0, None, lead.W, lead.V), lead)

        # x-mirror sectorization: solve the two half-size parity classes
        # independently and scatter them back into the full lead-mode ordering
        # (cross-sector scattering is exactly zero for a symmetric structure).
        T = self.basis.size_t
        merged = [np.zeros((T, T), dtype=complex) for _ in range(4)]
        for sec in self._sectors:
            cols = lead_columns(lead.labels, sec)
            rc = np.ix_(sec.t, cols)
            s = self._cascade(lead, k0, sec, lead.W[rc], lead.V[rc])
            ix = np.ix_(cols, cols)
            for dst, block in zip(merged, (s.s11, s.s12, s.s21, s.s22)):
                dst[ix] = block
        return SResult(freq, SMatrix(*merged), lead)

    def _cascade(
        self,
        lead: LeadModes,
        k0: complex,
        sector: Sector | None,
        lead_W: np.ndarray,
        lead_V: np.ndarray,
    ) -> SMatrix:
        parts: list[SMatrix] = []
        prev_W, prev_V = lead_W, lead_V
        for seg in self.segments:
            modes = solve_slice(
                seg.cross_section,
                self.basis,
                k0,
                self.factorization,
                ops=self._ops_for(seg),
                sector=sector,
            )
            parts.append(interface_smatrix(prev_W, prev_V, modes.W, modes.V))
            parts.append(propagation_smatrix(modes.beta, seg.length))
            prev_W, prev_V = modes.W, modes.V
        parts.append(interface_smatrix(prev_W, prev_V, lead_W, lead_V))
        return cascade(parts)

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
            lead_ref = self._lead(k0_ref)
            indices = np.flatnonzero(lead_ref.propagating())
        if len(indices) == 0:
            raise ValueError(
                "no propagating lead modes at this frequency: the port S-matrix "
                "is empty (pass an explicit mode index set)"
            )
        res = self.smatrix(freq)
        return complex(np.linalg.det(res.port_smatrix(indices)))
