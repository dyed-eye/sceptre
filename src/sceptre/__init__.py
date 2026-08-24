"""SCEPTRE -- S-matrix Cascading Eigenmode Propagation Through Rectangular Enclosures.

Fourier modal method (FMM / mode matching) for closed rectangular PEC waveguides
containing piecewise-constant dielectric obstacles, with numerically stable S-matrix
cascading, complex-frequency pole/zero location, and exceptional-point search.

Conventions: refs/CONVENTIONS.md (e^{-i omega t}; Im eps > 0 lossy; forward e^{+i beta z}).
"""

from .asr import AsrConfig
from .basis import ModeBasis
from .geometry import Box, CrossSection, Segment, Structure, Waveguide
from .kfj import KfjConfig
from .modes import LeadModes, lead_modes
from .nvf import NvfConfig
from .shapes import Cylinder, Shape
from .smatrix import SMatrix, cascade, interface_smatrix, propagation_smatrix, redheffer
from .solver import C0, SResult, Solver
from .threads import blas_thread_limit, recommended_blas_threads, set_blas_threads

__version__ = "0.2.0"

__all__ = [
    "AsrConfig",
    "Box",
    "C0",
    "CrossSection",
    "Cylinder",
    "KfjConfig",
    "LeadModes",
    "ModeBasis",
    "NvfConfig",
    "SMatrix",
    "Shape",
    "SResult",
    "Segment",
    "Solver",
    "Structure",
    "Waveguide",
    "blas_thread_limit",
    "cascade",
    "interface_smatrix",
    "lead_modes",
    "propagation_smatrix",
    "recommended_blas_threads",
    "redheffer",
    "set_blas_threads",
]
