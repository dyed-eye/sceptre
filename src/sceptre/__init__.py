"""SCEPTRE -- S-matrix Cascading Eigenmode Propagation Through Rectangular Enclosures.

Fourier modal method (FMM / mode matching) for closed rectangular PEC waveguides
containing piecewise-constant dielectric obstacles, with numerically stable S-matrix
cascading, complex-frequency pole/zero location, and exceptional-point search.

Conventions: refs/CONVENTIONS.md (e^{-i omega t}; Im eps > 0 lossy; forward e^{+i beta z}).
"""

from .asr import AsrConfig
from .basis import ModeBasis
from .geometry import Box, CrossSection, Segment, Structure, Waveguide
from .modes import LeadModes, lead_modes
from .smatrix import SMatrix, cascade, interface_smatrix, propagation_smatrix, redheffer
from .solver import C0, SResult, Solver

__version__ = "0.1.0"

__all__ = [
    "AsrConfig",
    "Box",
    "C0",
    "CrossSection",
    "LeadModes",
    "ModeBasis",
    "SMatrix",
    "SResult",
    "Segment",
    "Solver",
    "Structure",
    "Waveguide",
    "cascade",
    "interface_smatrix",
    "lead_modes",
    "propagation_smatrix",
    "redheffer",
]
