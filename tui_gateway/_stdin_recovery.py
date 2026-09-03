"""Shared spurious stdin-EOF recovery for the TUI gateway entry point and slash worker.

When a child process inherits fd 0 (stdin) and sets ``O_NONBLOCK``, the flag
lands on the **shared open file description** — not just the child's descriptor.
The gateway's next ``read()`` returns ``EAGAIN``, which CPython's buffered
``TextIOWrapper`` converts to ``b''`` (apparent EOF), killing the gateway.

This module provides:
- :func:`diagnose_stdin_state` — forensic diagnostic (``O_NONBLOCK`` / ``SO_RCVTIMEO``)
- :func:`handle_spurious_eof` — check whether an empty ``readline()`` is a genuine
  peer-close or a spurious EOF, and recover if spurious.

The recovery is **POSIX-only** (``fcntl``).  On Windows, ``O_NONBLOCK`` on a
shared file description is not a concern, so the guard simply reports a
genuine EOF and lets the caller exit.
"""

from __future__ import annotations

import os
import time

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

try:
    import socket as _socket
    _HAS_SOCKET = True
except ImportError:
    _socket = None  # type: ignore[assignment]
    _HAS_SOCKET = False

import struct


# Rate-limit: at most this many spurious-EOF recoveries per 60-second window.
# A child aggressively flipping ``O_NONBLOCK`` on the shared fd would otherwise
# create a tight busy-loop burning CPU.  Exceeding the cap exits the process —
# the parent (TUI / gateway) respawns it with fresh state, which is safer than
# fighting forever.
MAX_RECOVERIES_PER_MINUTE = 10


def diagnose_stdin_state() -> str:
    """Return a diagnostic string about stdin's current state.

    Used for crash-log forensics when stdin iteration falls through.
    Distinguishes genuine peer-close (flag clear) from spurious EOF
    caused by a child setting ``O_NONBLOCK`` on the shared file description.
    """
    parts: list[str] = []
    if _HAS_FCNTL and _fcntl is not None:
        try:
            flags = _fcntl.fcntl(0, _fcntl.F_GETFL)
            parts.append(f"O_NONBLOCK={'1' if flags & os.O_NONBLOCK else '0'}")
        except Exception as e:
            parts.append(f"F_GETFL error: {e}")
    else:
        parts.append("O_NONBLOCK=n/a (no fcntl)")
    # ``SO_RCVTIMEO`` is a socket option (not a file-status flag), equally
    # shared on the open file description.  A child setting it via
    # ``setsockopt`` launders into the same spurious-EOF path with
    # ``O_NONBLOCK`` clear, so we report it alongside the flag.
    if _HAS_SOCKET and _socket is not None:
        try:
            s = _socket.fromfd(0, _socket.AF_UNIX, _socket.SOCK_STREAM)
            try:
                tv = s.getsockopt(_socket.SOL_SOCKET, _socket.SO_RCVTIMEO)
                parts.append(f"SO_RCVTIMEO={tv!r}")
            finally:
                # ``fromfd`` duped the fd; ``close`` releases the dup without
                # touching the original fd 0.
                s.close()
        except Exception:
            pass
    return ", ".join(parts) if parts else "unknown"


def handle_spurious_eof(
    recovery_times: list[float],
    log_fn: object,
) -> bool:
    """Check whether an empty ``readline()`` is spurious; recover if so.

    Returns ``True`` if the caller should ``continue`` the read loop
    (spurious EOF was recovered), ``False`` if it should ``break`` (genuine
    peer-close or rate limit exceeded).

    ``log_fn`` is called with a diagnostic string — ``_log_exit`` in
    ``entry.py``, ``print(file=sys.stderr)`` in ``slash_worker.py``.
    """
    # Without ``fcntl`` (Windows) we can't check the flag, and the
    # ``O_NONBLOCK`` shared-description issue is POSIX-specific anyway —
    # treat it as a genuine EOF.
    if not (_HAS_FCNTL and _fcntl is not None):
        log_fn("stdin EOF (peer closed)")  # type: ignore[operator]
        return False

    try:
        flags = _fcntl.fcntl(0, _fcntl.F_GETFL)
        is_nonblock = bool(flags & os.O_NONBLOCK)
    except Exception:
        is_nonblock = False

    if not is_nonblock:
        # Genuine peer-close — no subprocess flag tampering detected.
        log_fn("stdin EOF (peer closed)")  # type: ignore[operator]
        return False

    # Spurious EOF: a child set ``O_NONBLOCK`` (and/or ``SO_RCVTIMEO``) on
    # the shared file description, laundered into ``b''`` / ``EAGAIN`` by
    # CPython's buffered layer.  Restore blocking mode and resume.
    now = time.time()
    recovery_times.append(now)
    recovery_times[:] = [t for t in recovery_times if t > now - 60]
    if len(recovery_times) > MAX_RECOVERIES_PER_MINUTE:
        log_fn(  # type: ignore[operator]
            f"stdin spurious-EOF recovery rate exceeded "
            f"({len(recovery_times)}/min, cap {MAX_RECOVERIES_PER_MINUTE})"
        )
        return False

    diag = diagnose_stdin_state()
    log_fn(f"stdin spurious EOF (subprocess O_NONBLOCK flip), recovering: {diag}")  # type: ignore[operator]

    # Clear ``O_NONBLOCK`` on the shared file description.
    os.set_blocking(0, True)

    # Also clear ``SO_RCVTIMEO`` if it was set by a child on the shared
    # description.  A non-zero timeout would cause the next ``readline()``
    # to time out and return ``''`` again, looping until the rate limiter
    # kicks in.  Clearing it restores fully blocking semantics.
    if _HAS_SOCKET and _socket is not None:
        try:
            s = _socket.fromfd(0, _socket.AF_UNIX, _socket.SOCK_STREAM)
            try:
                # Zero timeval: tv_sec=0, tv_usec=0 (struct timeval on most platforms)
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVTIMEO, struct.pack("ll", 0, 0))
            finally:
                s.close()
        except Exception:
            pass

    # ``_io.TextIOWrapper.readline`` returns an empty string on ``EAGAIN``
    # but does NOT stick EOF; after restoring blocking, the next call will
    # block until data arrives or the peer truly closes.
    return True
