"""COMSOL cross-verification bridge.

detect_comsol()  -- find local COMSOL installations (standard paths + PATH).
benchmark        -- the shared benchmark geometry (single source of truth).
mph_driver       -- drive COMSOL through the MPh package (preferred route).
model_gen        -- emit a ready-to-run COMSOL Java model file (manual route).
"""

from .detect import ComsolInstall, detect_comsol

__all__ = ["ComsolInstall", "detect_comsol"]
