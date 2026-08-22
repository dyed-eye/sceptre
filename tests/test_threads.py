"""Runtime BLAS thread management (threadpoolctl — works AFTER numpy import)."""

import numpy as np
import pytest

from sceptre.threads import (
    blas_thread_limit,
    recommended_blas_threads,
    set_blas_threads,
)


def _current_blas_limits() -> list[int]:
    from threadpoolctl import threadpool_info

    return [
        info["num_threads"]
        for info in threadpool_info()
        if info.get("user_api") == "blas"
    ]


@pytest.mark.unit
def test_recommended_blas_threads_caps_at_four(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert recommended_blas_threads() == 4
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert recommended_blas_threads() == 2
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert recommended_blas_threads() == 1


@pytest.mark.unit
def test_set_blas_threads_applies_globally():
    before = _current_blas_limits()
    if not before:
        pytest.skip("no BLAS detected by threadpoolctl")
    try:
        set_blas_threads(2)
        assert all(n <= 2 for n in _current_blas_limits())
    finally:
        set_blas_threads(max(before))  # restore


@pytest.mark.unit
def test_blas_thread_limit_scope_restores():
    before = _current_blas_limits()
    if not before:
        pytest.skip("no BLAS detected by threadpoolctl")
    with blas_thread_limit(1):
        inside = _current_blas_limits()
        assert all(n == 1 for n in inside)
        np.linalg.inv(np.eye(8))  # a solve under the limit works
    assert _current_blas_limits() == before


@pytest.mark.unit
def test_blas_thread_limit_none_is_noop():
    before = _current_blas_limits()
    with blas_thread_limit(None):
        assert _current_blas_limits() == before


@pytest.mark.unit
def test_set_blas_threads_rejects_nonpositive():
    with pytest.raises(ValueError):
        set_blas_threads(0)


@pytest.mark.unit
def test_solver_blas_threads_wiring():
    from sceptre import Box, Solver, Structure, Waveguide

    struct = Structure(
        Waveguide(0.02286, 0.01016), [Box(0.005, 0.01, 0.0, 0.005, 0.0, 0.003, 4.0)]
    )
    before = _current_blas_limits()
    solver = Solver(struct, 3, 3, blas_threads=1)
    observed: list[list[int]] = []

    import sceptre.solver as solver_mod

    original = solver_mod.blas_thread_limit

    def recording(n):
        class _Ctx:
            def __enter__(self):
                self._inner = original(n)
                self._inner.__enter__()
                observed.append(_current_blas_limits())

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

        return _Ctx()

    solver_mod.blas_thread_limit = recording
    try:
        solver.smatrix(15e9)
    finally:
        solver_mod.blas_thread_limit = original
    # the limit was ACTIVE inside smatrix (n=1), and restored afterwards
    assert observed and all(n == 1 for n in observed[0])
    assert _current_blas_limits() == before
    with pytest.raises(ValueError):
        Solver(struct, 3, 3, blas_threads=0)
