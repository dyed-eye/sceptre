"""Spec test 5: contour-integration pole/zero finder on manufactured rationals.

A rational function with known poles and zeros (times an entire, nonvanishing
background factor, which must not disturb anything) is fed to the finder; every
location must be recovered to 1e-12.
"""

import numpy as np

from sceptre.poles import find_zeros_poles, refine_pole, refine_zero

ZEROS = [0.62 + 0.31j, -0.24 + 0.71j, 0.48 - 0.46j]
POLES = [0.33 - 0.21j, -0.47 - 0.36j, -0.72 + 0.44j]


def _rational(z):
    num = np.prod([z - w for w in ZEROS])
    den = np.prod([z - w for w in POLES])
    return np.exp(0.2 * z) * num / den  # entire background: no extra zeros/poles


def _match(found, expected, tol):
    assert len(found) == len(expected), (found, expected)
    for w in expected:
        assert min(abs(w - v) for v in found) < tol, (w, found)


def test_manufactured_rational_recovered_to_1e12():
    res = find_zeros_poles(_rational, 0.0 + 0.0j, 1.0, 1.0, rel_tol=1e-13)
    _match(res.zeros, ZEROS, 1e-12)
    _match(res.poles, POLES, 1e-12)


def test_close_pole_pair_resolved():
    """Two poles separated by 1e-4 (the EP-tracking workload)."""
    p1, p2 = 0.4 + 0.1j, 0.4 + 0.1j + 1.2e-4 * np.exp(0.7j)

    def f(z):
        return 1.0 / ((z - p1) * (z - p2))

    res = find_zeros_poles(f, 0.35 + 0.05j, 0.4, rel_tol=1e-13)
    _match(res.poles, [p1, p2], 1e-12)
    assert res.zeros == []


def test_hidden_zero_pole_pair_not_lost():
    """A region whose total winding is zero (one zero + one pole) must not come back
    empty -- the first-moment test forces subdivision until they separate."""
    z0, p0 = 0.21 + 0.17j, -0.19 - 0.12j

    def f(z):
        return (z - z0) / (z - p0)

    res = find_zeros_poles(f, 0.0 + 0.0j, 0.6, rel_tol=1e-13)
    _match(res.zeros, [z0], 1e-12)
    _match(res.poles, [p0], 1e-12)


def test_newton_refiners():
    z = refine_zero(_rational, ZEROS[0] + 0.01 - 0.02j, tol=1e-14)
    assert abs(z - ZEROS[0]) < 1e-13
    p = refine_pole(_rational, POLES[2] + 0.015j, tol=1e-14)
    assert abs(p - POLES[2]) < 1e-13


def test_scaled_domain_like_frequencies():
    """Same recovery on a physically-scaled domain (poles ~ 1e10, like f in Hz)."""
    scale = 1e10
    zeros = [complex(1.02, -0.004) * scale, complex(0.97, -0.011) * scale]
    poles = [complex(0.99, -0.007) * scale, complex(1.045, -0.002) * scale]

    def f(z):
        num = np.prod([z - w for w in zeros])
        den = np.prod([z - w for w in poles])
        return num / den

    res = find_zeros_poles(f, 1.0 * scale + 0j, 0.1 * scale, rel_tol=1e-13)
    _match(res.zeros, zeros, 1e-12 * scale)
    _match(res.poles, poles, 1e-12 * scale)
