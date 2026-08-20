"""Locate local COMSOL Multiphysics installations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

_STANDARD_ROOTS = [
    Path(r"C:\Program Files\COMSOL"),
    Path(r"C:\Program Files (x86)\COMSOL"),
    Path("/usr/local/comsol"),
    Path("/opt/comsol"),
]


@dataclass(frozen=True)
class ComsolInstall:
    version: str  # e.g. "COMSOL61"
    root: Path  # .../COMSOL61/Multiphysics
    comsol_exe: Path
    mphserver_exe: Path | None
    batch_exe: Path | None


def detect_comsol() -> list[ComsolInstall]:
    """All COMSOL installations found, newest version first."""
    installs: list[ComsolInstall] = []
    for root in _STANDARD_ROOTS:
        if not root.is_dir():
            continue
        for verdir in sorted(root.iterdir(), reverse=True):
            mp = verdir / "Multiphysics"
            exe = _first_existing(
                mp / "bin" / "win64" / "comsol.exe", mp / "bin" / "comsol"
            )
            if exe is None:
                continue
            installs.append(
                ComsolInstall(
                    version=verdir.name,
                    root=mp,
                    comsol_exe=exe,
                    mphserver_exe=_first_existing(
                        mp / "bin" / "win64" / "comsolmphserver.exe",
                        mp / "bin" / "comsolmphserver",
                    ),
                    batch_exe=_first_existing(
                        mp / "bin" / "win64" / "comsolbatch.exe",
                        mp / "bin" / "comsolbatch",
                    ),
                )
            )
    on_path = shutil.which("comsol")
    if on_path and not any(i.comsol_exe == Path(on_path) for i in installs):
        p = Path(on_path)
        installs.append(
            ComsolInstall(
                version="(PATH)",
                root=p.parent,
                comsol_exe=p,
                mphserver_exe=(
                    Path(w) if (w := shutil.which("comsolmphserver")) else None
                ),
                batch_exe=Path(w) if (w := shutil.which("comsolbatch")) else None,
            )
        )
    return installs


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.is_file():
            return c
    return None
