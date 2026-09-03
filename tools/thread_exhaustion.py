"""Detection for OS thread/resource exhaustion at spawn boundaries."""

from __future__ import annotations

import errno


def is_thread_start_exhaustion(exc: BaseException) -> bool:
    """Return True only for failures indicating no OS thread is available."""
    message = str(exc).lower()
    if isinstance(exc, RuntimeError):
        return "can't start new thread" in message
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
    }:
        return (
            "can't start new thread" in message
            or "resource temporarily unavailable" in message
        )
    return False
