"""Shared spurious stdin-EOF recovery for the TUI gateway entry point and slash worker.

When a child inherits fd 0 and sets ``O_NONBLOCK``, the flag lands on the SHARED open
file description, not just the child's descriptor. The next ``read()`` returns
``EAGAIN``, which CPython's buffered ``TextIOWrapper`` converts to ``b''`` (apparent
EOF), killing the gateway. Recovery is POSIX-only (``fcntl``); on Windows the guard
just reports a genuine EOF and lets the caller exit.
"""

from __future__ import annotations

import os
import socket
import struct
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

# Recoveries per 60s window. A child aggressively flipping the flag would otherwise
# create a tight busy-loop; exceeding the cap exits so the parent respawns us fresh.
MAX_RECOVERIES_PER_MINUTE = 10


def _stdin_nonblock() -> bool:
    try:
        return bool(fcntl.fcntl(0, fcntl.F_GETFL) & os.O_NONBLOCK)  # type: ignore[union-attr]
    except Exception:
        return False


def _stdin_sockopt(getter):
    """Run ``getter(sock)`` against a dup of fd 0 as a socket; None on failure.

    ``fromfd`` dups the fd, so ``close`` releases the dup without touching fd 0.
    """
    try:
        s = socket.fromfd(0, socket.AF_UNIX, socket.SOCK_STREAM)
    except Exception:
        return None
    try:
        return getter(s)
    except Exception:
        return None
    finally:
        s.close()


def diagnose_stdin_state() -> str:
    """Diagnostic string (``O_NONBLOCK`` / ``SO_RCVTIMEO``) for crash-log forensics.

    ``SO_RCVTIMEO`` is equally shared on the open file description; a child's
    ``setsockopt`` launders into the same spurious-EOF path with ``O_NONBLOCK`` clear.
    """
    parts: list[str] = []
    if fcntl is None:
        parts.append("O_NONBLOCK=n/a (no fcntl)")
    else:
        try:
            flags = fcntl.fcntl(0, fcntl.F_GETFL)
            parts.append(f"O_NONBLOCK={'1' if flags & os.O_NONBLOCK else '0'}")
        except Exception as e:
            parts.append(f"F_GETFL error: {e}")
    tv = _stdin_sockopt(lambda s: s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO))
    if tv is not None:
        parts.append(f"SO_RCVTIMEO={tv!r}")
    return ", ".join(parts) if parts else "unknown"


def handle_spurious_eof(recovery_times: list[float], log_fn: object) -> bool:
    """Check whether an empty ``readline()`` is spurious; recover if so.

    Returns True if the caller should ``continue`` the read loop (recovered), False if it
    should ``break`` (genuine peer-close or rate limit exceeded). ``log_fn`` receives a
    diagnostic string.
    """
    # Without fcntl (Windows) we can't check the flag and the issue is POSIX-specific
    # anyway; a clear flag means a genuine peer-close.
    if fcntl is None or not _stdin_nonblock():
        log_fn("stdin EOF (peer closed)")  # type: ignore[operator]
        return False
    now = time.time()
    recovery_times.append(now)
    recovery_times[:] = [t for t in recovery_times if t > now - 60]
    if len(recovery_times) > MAX_RECOVERIES_PER_MINUTE:
        log_fn(  # type: ignore[operator]
            f"stdin spurious-EOF recovery rate exceeded "
            f"({len(recovery_times)}/min, cap {MAX_RECOVERIES_PER_MINUTE})")
        return False
    log_fn(f"stdin spurious EOF (subprocess O_NONBLOCK flip), recovering: {diagnose_stdin_state()}")  # type: ignore[operator]
    # Restore blocking mode on the shared description, and clear SO_RCVTIMEO too: a
    # non-zero timeout would make the next readline() return '' again until the limiter fires.
    os.set_blocking(0, True)
    # "ll" = struct timeval {tv_sec, tv_usec}; zero timeval disables the timeout.
    _stdin_sockopt(lambda s: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", 0, 0)))
    # TextIOWrapper.readline returns '' on EAGAIN but does NOT stick EOF; the next call
    # blocks until data arrives or the peer truly closes.
    return True
