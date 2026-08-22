"""Shared spurious stdin-EOF recovery for the TUI gateway entry point and slash worker.

When a child process inherits fd 0 (stdin) and sets ``O_NONBLOCK``, the flag
lands on the **shared open file description** — not just the child's descriptor.
The gateway's next ``read()`` returns ``EAGAIN``, which CPython's buffered
``TextIOWrapper`` converts to ``b''`` (apparent EOF), killing the gateway.

``SO_RCVTIMEO`` is the second route to the same symptom.  It is a socket option
rather than a file-status flag, but it lives on the same shared description, and
when it expires the read returns ``''`` with ``O_NONBLOCK`` **clear** — so the
flag alone is not enough to tell tampering from a real peer close.

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


# ``struct timeval`` — ``tv_sec``, ``tv_usec`` — the payload of ``SO_RCVTIMEO``
# on the platforms this POSIX-only module runs on.  Named once so the probe and
# the recovery below cannot drift apart.
_TIMEVAL_FORMAT = "ll"
_TIMEVAL_SIZE = struct.calcsize(_TIMEVAL_FORMAT)


# Rate-limit: at most this many spurious-EOF recoveries per 60-second window.
# A child aggressively flipping ``O_NONBLOCK`` on the shared fd would otherwise
# create a tight busy-loop burning CPU.  Exceeding the cap exits the process —
# the parent (TUI / gateway) respawns it with fresh state, which is safer than
# fighting forever.
MAX_RECOVERIES_PER_MINUTE = 10


def _read_stdin_rcvtimeo() -> bytes | None:
    """Return stdin's raw ``SO_RCVTIMEO`` timeval, or ``None`` if unreadable.

    ``SO_RCVTIMEO`` is a socket option (not a file-status flag), equally shared
    on the open file description.  A child setting it via ``setsockopt``
    launders into the same spurious-EOF path as ``O_NONBLOCK`` — but with
    ``O_NONBLOCK`` clear.

    ``None`` means **"cannot tell"**, not "no timeout": either the ``socket``
    module is unavailable, or fd 0 is not a socket at all (a tty, or an
    anonymous pipe), in which case ``fromfd``/``getsockopt`` raises
    ``ENOTSOCK``.  Callers must treat ``None`` as "nothing detected" so that
    non-socket stdin keeps its existing behaviour.
    """
    if not (_HAS_SOCKET and _socket is not None):
        return None
    try:
        s = _socket.fromfd(0, _socket.AF_UNIX, _socket.SOCK_STREAM)
    except Exception:
        # ``ENOTSOCK`` surfaces here on platforms whose socket constructor
        # validates the descriptor (macOS).
        return None
    try:
        # ``getsockopt`` without an explicit buffer length assumes an *int*
        # option and reads only ``sizeof(int)`` bytes, truncating
        # ``struct timeval`` to the low half of ``tv_sec`` — a 500 ms timeout
        # would then be reported as ``0``.  Read the whole struct.
        return s.getsockopt(_socket.SOL_SOCKET, _socket.SO_RCVTIMEO, _TIMEVAL_SIZE)
    except Exception:
        # ...and here on platforms that defer validation to the option call.
        return None
    finally:
        # ``fromfd`` duped the fd; ``close`` releases the dup without touching
        # the original fd 0.
        s.close()


def _stdin_rcvtimeo_is_set() -> bool:
    """Return ``True`` only when stdin carries a **non-zero** receive timeout.

    A zeroed timeval means "block forever", which is the default and is not
    tampering.  An unreadable probe (``None``) is deliberately reported as
    ``False``: on a tty or an anonymous pipe there is no timeout to detect, and
    guessing ``True`` there would make every genuine peer-close look spurious.

    The byte-wise test avoids unpacking, so it holds regardless of the
    platform's timeval width or endianness.
    """
    tv = _read_stdin_rcvtimeo()
    return tv is not None and any(tv)


def diagnose_stdin_state() -> str:
    """Return a diagnostic string about stdin's current state.

    Used for crash-log forensics when stdin iteration falls through.
    Distinguishes a genuine peer-close from a spurious EOF caused by a child
    mutating the shared file description — either ``O_NONBLOCK`` or a non-zero
    ``SO_RCVTIMEO``.
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
    # Report the shared-description socket timeout alongside the flag.
    tv = _read_stdin_rcvtimeo()
    if tv is not None:
        parts.append(f"SO_RCVTIMEO={tv!r}")
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

    # ``SO_RCVTIMEO`` reaches the same symptom by a different route: the read
    # expires and returns ``''`` with ``O_NONBLOCK`` **clear**, so the flag
    # alone cannot distinguish it from a peer close.  Probe it too.
    if not is_nonblock and not _stdin_rcvtimeo_is_set():
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
    log_fn(f"stdin spurious EOF (subprocess O_NONBLOCK / SO_RCVTIMEO), recovering: {diag}")  # type: ignore[operator]

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
                s.setsockopt(
                    _socket.SOL_SOCKET,
                    _socket.SO_RCVTIMEO,
                    struct.pack(_TIMEVAL_FORMAT, 0, 0),
                )
            finally:
                s.close()
        except Exception:
            pass

    # ``_io.TextIOWrapper.readline`` returns an empty string on ``EAGAIN``
    # but does NOT stick EOF; after restoring blocking, the next call will
    # block until data arrives or the peer truly closes.
    return True
