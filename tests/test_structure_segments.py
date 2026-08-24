"""Structure.from_segments: explicit z-uniform slices instead of boxes.

Beyond ergonomics it closes three silent-wrong-answer classes: Solver._cascade
walks the segment list in order and only ever uses seg.length, so a gap between
consecutive segments is dropped, an overlap is double-counted, and a
mis-ordered list cascades in the wrong order -- none of which raised.
"""

import numpy as np
import pytest

from sceptre import Box, CrossSection, Segment, Solver, Structure, Waveguide

A, B = 0.032, 0.032
F = 5.85e9


def _uniform(eps, n=4):
    xe = np.linspace(0.0, A, n + 1)
    ye = np.linspace(0.0, B, n + 1)
    return CrossSection(xe, ye, np.full((n, n), complex(eps)))


def test_accepts_and_solves_a_single_segment():
    st = Structure.from_segments(Waveguide(A, B), [Segment(0.0, 0.01, _uniform(4.0))])
    res = Solver(st, M=4, N=4).smatrix(F)
    assert np.isfinite(res.smatrix.s11).all()


def test_exposes_the_structure_contract_the_solver_reads():
    wg = Waveguide(A, B)
    st = Structure.from_segments(wg, [Segment(0.0, 0.01, _uniform(4.0))])
    assert st.waveguide is wg
    assert st.boxes == ()
    assert st.shapes == ()
    assert st.background == 1.0 + 0.0j
    assert st.z_span == (0.0, 0.01)
    assert len(st.segments()) == 1


def test_segments_are_not_aliased_to_the_caller_list():
    segs = [Segment(0.0, 0.01, _uniform(4.0))]
    st = Structure.from_segments(Waveguide(A, B), segs)
    segs.append(Segment(0.05, 0.06, _uniform(9.0)))
    assert len(st.segments()) == 1


def test_matches_an_equivalent_box_structure():
    """The whole point: same physics, different way of describing it."""
    boxes = [
        Box(0, A, 0, B, 0.0, 0.005, 4.0),
        Box(0, A, 0, B, 0.005, 0.012, 9.0),
    ]
    ref = Solver(Structure(Waveguide(A, B), boxes), M=4, N=4).smatrix(F)
    seg = Structure.from_segments(
        Waveguide(A, B),
        [
            Segment(0.0, 0.005, _uniform(4.0)),
            Segment(0.005, 0.012, _uniform(9.0)),
        ],
    )
    got = Solver(seg, M=4, N=4).smatrix(F)
    assert got.smatrix.s11 == pytest.approx(ref.smatrix.s11, rel=1e-12, abs=1e-14)
    assert got.smatrix.s21 == pytest.approx(ref.smatrix.s21, rel=1e-12, abs=1e-14)


def test_rejects_empty_segment_list():
    with pytest.raises(ValueError, match="at least one segment"):
        Structure.from_segments(Waveguide(A, B), [])


def test_rejects_non_positive_segment_length():
    with pytest.raises(ValueError, match="positive z-extent"):
        Structure.from_segments(Waveguide(A, B), [Segment(0.01, 0.01, _uniform(4.0))])


def test_rejects_a_gap_between_segments():
    """Solver._cascade would silently omit the gap entirely."""
    with pytest.raises(ValueError, match="contiguous"):
        Structure.from_segments(
            Waveguide(A, B),
            [
                Segment(0.0, 0.005, _uniform(4.0)),
                Segment(0.008, 0.012, _uniform(9.0)),  # 3 mm of nothing
            ],
        )


def test_rejects_overlapping_segments():
    with pytest.raises(ValueError, match="contiguous"):
        Structure.from_segments(
            Waveguide(A, B),
            [
                Segment(0.0, 0.006, _uniform(4.0)),
                Segment(0.005, 0.012, _uniform(9.0)),
            ],
        )


def test_rejects_out_of_order_segments():
    with pytest.raises(ValueError, match="contiguous"):
        Structure.from_segments(
            Waveguide(A, B),
            [
                Segment(0.005, 0.012, _uniform(9.0)),
                Segment(0.0, 0.005, _uniform(4.0)),
            ],
        )


def test_rejects_a_gap_on_a_structure_far_from_the_origin():
    """The join tolerance must scale with the segments, not with the distance
    from the origin: a 0.1 mm gap in a 10 mm structure is a real gap whether it
    sits at z = 0 or z = 1000 km."""
    z0 = 1.0e6
    with pytest.raises(ValueError, match="contiguous"):
        Structure.from_segments(
            Waveguide(A, B),
            [
                Segment(z0, z0 + 0.005, _uniform(4.0)),
                Segment(z0 + 0.005 + 1e-4, z0 + 0.010 + 1e-4, _uniform(9.0)),
            ],
        )


def test_accepts_exact_joins_far_from_the_origin():
    """...but last-bit representation slack at large z must still be tolerated."""
    z0 = 1.0e6
    mid = z0 + 0.005
    st = Structure.from_segments(
        Waveguide(A, B),
        [
            Segment(z0, mid, _uniform(4.0)),
            Segment(np.nextafter(mid, np.inf), z0 + 0.010, _uniform(9.0)),
        ],
    )
    assert len(st.segments()) == 2


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_rejects_non_finite_segment_bounds(bad):
    """inf passes `z2 > z1`, then propagation_smatrix's overflow guard computes
    -Im(beta)*d = 0*inf = NaN and never fires -- S came back all NaN, no error."""
    with pytest.raises(ValueError, match="finite|positive z-extent"):
        Structure.from_segments(Waveguide(A, B), [Segment(0.0, bad, _uniform(4.0))])


def test_rejects_zero_background():
    with pytest.raises(ValueError, match="nonzero"):
        Structure.from_segments(
            Waveguide(A, B), [Segment(0.0, 0.01, _uniform(4.0))], background=0.0
        )


def test_tolerates_floating_point_joins():
    """Segment boundaries built by accumulation must not be rejected."""
    z = [i * 0.001 for i in range(4)]
    segs = [Segment(z[i], z[i + 1] + 1e-18, _uniform(4.0 + i)) for i in range(3)]
    st = Structure.from_segments(Waveguide(A, B), segs)
    assert len(st.segments()) == 3


def test_works_with_x_symmetry_and_still_validates_the_map():
    sym = Structure.from_segments(Waveguide(A, B), [Segment(0.0, 0.01, _uniform(4.0))])
    assert Solver(sym, M=4, N=4, symmetry="x") is not None

    xe = np.linspace(0.0, A, 5)
    ye = np.linspace(0.0, B, 5)
    eps = np.full((4, 4), 4.0 + 0j)
    eps[0, :] = 9.0  # break mirror symmetry
    asym = Structure.from_segments(
        Waveguide(A, B), [Segment(0.0, 0.01, CrossSection(xe, ye, eps))]
    )
    with pytest.raises(ValueError, match="symmetric"):
        Solver(asym, M=4, N=4, symmetry="x")


def test_works_with_asr():
    """build_maps reads only .waveguide and .segments(), so ASR composes -- but
    nothing proved it until now."""
    from sceptre import AsrConfig

    st = Structure.from_segments(Waveguide(A, B), [Segment(0.0, 0.01, _uniform(9.0))])
    res = Solver(st, M=4, N=4, asr=AsrConfig(eta=0.3)).smatrix(F)
    assert np.isfinite(res.smatrix.s11).all()


def test_tensor_factorizations_are_rejected_with_the_shape_message():
    st = Structure.from_segments(Waveguide(A, B), [Segment(0.0, 0.01, _uniform(4.0))])
    with pytest.raises(ValueError, match="Shape geometry"):
        Solver(st, M=4, N=4, factorization="nvf")
