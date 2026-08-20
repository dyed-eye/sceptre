"""Drive COMSOL through MPh (https://mph.readthedocs.io): build the benchmark model,
run the frequency sweep, return port-referenced S11/S21.

The model: WR-90 guide with LEAD_LEN vacuum sections on both sides of the
partial-height dielectric block, PEC walls (default exterior boundary condition of
the emw interface), two Rectangular (analytic TE10) ports, direct frequency sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import benchmark as bm


def _eps_str(eps: complex) -> str:
    """COMSOL expression for a (possibly complex) relative permittivity."""
    eps = complex(eps)
    if eps.imag == 0:
        return f"{eps.real:.12g}"
    return f"{eps.real:.12g}+{eps.imag:.12g}*i"


@dataclass(frozen=True)
class ComsolSweep:
    freqs: np.ndarray
    s11_port: np.ndarray  # phase-referenced at the ports (lead ends)
    s21_port: np.ndarray
    model_path: str


def run_benchmark(
    save_model: str | Path | None = None,
    cores: int | None = None,
    case: bm.BenchmarkCase = bm.STANDARD,
) -> ComsolSweep:
    """Build + solve the benchmark in COMSOL via MPh.  Raises on any failure."""
    import mph  # deferred: optional dependency

    # Stand-alone mode loads COMSOL into this process: no comsolmphserver child
    # that would outlive a crash and hold license seats hostage.
    mph.option("session", "stand-alone")  # pyright: ignore[reportPrivateImportUsage]
    client = (
        mph.start(cores=cores) if cores else mph.start()  # pyright: ignore[reportPrivateImportUsage]
    )
    try:
        model = client.create(f"sceptre_benchmark_{case.name}")
        _build(model.java, case)
        # Run via the Java API tag: mph's Model.solve() resolves studies by their
        # LABEL ("Study 1"), not the tag we created ("std1").
        try:
            model.java.study("std1").run()
        except Exception:
            # The physics-controlled iterative (multigrid/SOR) solver can fail on
            # this high-contrast resonant model ("NaN or Inf ... using SOR").
            # Point the fully-coupled step at the default direct solver instead.
            model.java.sol("sol1").feature("s1").feature("fc1").set("linsolver", "dDef")
            model.java.sol("sol1").runAll()
        # The sweep grid is what we set in plist; re-deriving it through
        # model.evaluate("freq") only adds a unit-conversion failure mode.
        freqs = bm.frequencies(case)
        s11 = _complex_eval(model, "emw.S11")
        s21 = _complex_eval(model, "emw.S21")
        if len(s11) != len(freqs) or len(s21) != len(freqs):
            raise RuntimeError(
                f"expected {len(freqs)} sweep points, got {len(s11)}/{len(s21)}"
            )
        path = str(save_model) if save_model else ""
        if save_model:
            model.save(str(save_model))
        return ComsolSweep(freqs, s11, s21, path)
    finally:
        try:  # best-effort cleanup; API name differs across mph versions
            client.clear()
        except Exception:  # noqa: BLE001 -- never mask the primary error
            pass


def _complex_eval(model, expr: str) -> np.ndarray:
    re = np.atleast_1d(model.evaluate(f"real({expr})"))
    im = np.atleast_1d(model.evaluate(f"imag({expr})"))
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


def _build(java, case: bm.BenchmarkCase = bm.STANDARD) -> None:
    a, b = bm.A, bm.B
    lead, blen, bh = bm.LEAD_LEN, case.block_len, case.block_height
    tol = 1e-6

    java.component().create("comp1", True)
    comp = java.component("comp1")
    comp.geom().create("geom1", 3)
    geom = comp.geom("geom1")
    geom.lengthUnit("m")
    g1 = geom.create("blk_guide", "Block")
    g1.set("size", [a, b, blen + 2 * lead])
    g1.set("pos", [0.0, 0.0, -lead])
    g2 = geom.create("blk_diel", "Block")
    g2.set("size", [a, bh, blen])
    g2.set("pos", [0.0, 0.0, 0.0])
    geom.run()

    # JPype cannot disambiguate set(String, int) from set(String, boolean) for a
    # plain Python int -- wrap integer property values explicitly.
    from jpype import JInt

    def box_selection(tag, dim, zmin, zmax, ymax=None):
        sel = comp.selection().create(tag, "Box")
        sel.set("entitydim", JInt(dim))
        sel.set("xmin", -tol)
        sel.set("xmax", a + tol)
        sel.set("ymin", -tol)
        sel.set("ymax", (b if ymax is None else ymax) + tol)
        sel.set("zmin", zmin)
        sel.set("zmax", zmax)
        sel.set("condition", "inside")
        return sel

    box_selection("sel_port1", 2, -lead - tol, -lead + tol)
    box_selection("sel_port2", 2, blen + lead - tol, blen + lead + tol)
    box_selection("sel_diel", 3, -tol, blen + tol, ymax=bh)

    mat_air = comp.material().create("mat_air", "Common")
    # A material created through the Java API covers NOTHING by default; without
    # this the air domain is undefined and the solver dies with "NaN or Inf".
    mat_air.selection().all()
    mat_air.propertyGroup("def").set("relpermittivity", ["1"])
    mat_air.propertyGroup("def").set("relpermeability", ["1"])
    mat_air.propertyGroup("def").set("electricconductivity", ["0"])
    mat_d = comp.material().create("mat_diel", "Common")
    mat_d.selection().named("sel_diel")
    mat_d.propertyGroup("def").set("relpermittivity", [_eps_str(case.eps_block)])
    mat_d.propertyGroup("def").set("relpermeability", ["1"])
    mat_d.propertyGroup("def").set("electricconductivity", ["0"])

    emw = comp.physics().create("emw", "ElectromagneticWaves", "geom1")
    p1 = emw.create("port1", "Port", 2)
    p1.selection().named("sel_port1")
    p1.set("PortType", "Rectangular")
    p1.set("PortExcitation", "on")
    p2 = emw.create("port2", "Port", 2)
    p2.selection().named("sel_port2")
    p2.set("PortType", "Rectangular")
    # API-created ports default to PortExcitation = on (unlike the GUI, which only
    # excites the first); an excited output port silently ruins the S-parameters.
    p2.set("PortExcitation", "off")

    mesh = comp.mesh().create("mesh1")
    size = mesh.feature("size")
    size.set("custom", "on")
    size.set("hmax", case.mesh_air)
    fdiel = mesh.create("siz_diel", "Size")
    fdiel.selection().geom("geom1", 3)
    fdiel.selection().named("sel_diel")
    fdiel.set("custom", "on")
    fdiel.set("hmaxactive", True)
    fdiel.set("hmax", case.mesh_diel)
    mesh.create("ftet", "FreeTet")
    mesh.run()

    std = java.study().create("std1")
    frq = std.create("freq", "Frequency")
    # EXPLICIT frequency unit: the Frequency study step interprets bare numbers
    # in its own default unit (GHz in COMSOL 6.x), which silently shifts the
    # sweep by 9 orders of magnitude and yields a zero field.
    plist = " ".join(f"{f:.10g}[Hz]" for f in bm.frequencies(case))
    frq.set("plist", plist)

    # Bind the ports to the study step, matching what the GUI does when a study
    # exists (GUI-built reference models carry StudyStep = std1/freq).
    emw.feature("port1").set("StudyStep", "std1/freq")
    emw.feature("port2").set("StudyStep", "std1/freq")
