"""Runtime BLAS thread control via threadpoolctl.

Why this exists: OpenBLAS at its default thread count THRASHES on the
matrix sizes sceptre produces (measured on a 16-core laptop: inv at n=312
is 217 ms at 16 threads vs 10 ms at 4).  Environment variables like
``OPENBLAS_NUM_THREADS`` only work when set BEFORE numpy first loads — a
silent no-op afterwards.  threadpoolctl talks to the loaded BLAS directly,
so these helpers work at any point, including inside notebooks.

NOT THREAD-SAFE: threadpoolctl mutates process-global BLAS state.  Python
threads running solves concurrently with different limits race (last exit
wins on restore).  Process-based parallelism — the library's recommended
sweep pattern, where each worker process sets its own limit (or the env
vars, set before numpy imports in the worker) — is unaffected.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from threadpoolctl import threadpool_limits

# Measured knee on 16-core OpenBLAS at sceptre's LAPACK sizes (n ~ 300-2000):
# above ~4 threads the LU/eig calls slow down 10-20x.  Other BLAS builds
# (MKL, Accelerate) or very different matrix sizes may knee elsewhere —
# treat this as a measured default, not a law; sweep once on your machine.
_MEASURED_KNEE = 4


def recommended_blas_threads() -> int:
    """min(4, cpu_count): the measured OpenBLAS sweet spot for this workload."""
    return min(_MEASURED_KNEE, os.cpu_count() or 1)


def set_blas_threads(n: int) -> None:
    """Set the BLAS thread count process-wide, effective immediately."""
    if n < 1:
        raise ValueError("BLAS thread count must be >= 1")
    threadpool_limits(limits=n, user_api="blas")


@contextmanager
def blas_thread_limit(n: int | None) -> Iterator[None]:
    """Scoped BLAS thread limit; ``None`` is a no-op (current setting kept)."""
    if n is None:
        yield
        return
    if n < 1:
        raise ValueError("BLAS thread count must be >= 1")
    with threadpool_limits(limits=n, user_api="blas"):
        yield
