"""Regression tests for spurious stdin-EOF detection in the TUI gateway.

``tui_gateway/_stdin_recovery.py`` guards the gateway against a child process
that mutates the **shared open file description** behind fd 0.  Two distinct
mutations produce the same symptom — an empty ``readline()`` that looks like a
peer close:

- ``O_NONBLOCK`` set via ``fcntl`` (``EAGAIN`` laundered into ``b''``), and
- a non-zero ``SO_RCVTIMEO`` set via ``setsockopt``, which expires and returns
  ``''`` with ``O_NONBLOCK`` **clear**.

The module's docstring and its recovery tail both name the second case, but the
decision gate only ever inspected ``O_NONBLOCK``, so an ``SO_RCVTIMEO``-only
mutation was reported as ``stdin EOF (peer closed)`` and the gateway exited
mid-session.

These tests pin the decision function itself.  ``handle_spurious_eof`` is a pure
predicate over fd-0 state, so the collaborators (``fcntl``, ``socket``,
``os.set_blocking``) are stubbed at the module boundary rather than reproducing
a real kernel ``EAGAIN``.

The fail-safe is the load-bearing case: when fd 0 is **not** a socket the probe
must report "cannot tell" and the caller must still see a genuine EOF.  Getting
that wrong would make every real peer close look spurious, so it is covered from
both directions (``fromfd`` raising ``ENOTSOCK``, as on macOS, and ``getsockopt``
raising it after a successful ``fromfd``, as on Linux).
"""

from __future__ import annotations

import errno
import os
import struct

from tui_gateway import _stdin_recovery

# Same layout the production recovery tail packs with.
_TIMEVAL = "ll"

# ``os.O_NONBLOCK`` is POSIX-only — the repo's platform-safe idiom is
# ``getattr(os, "O_NONBLOCK", ...)`` (see ``cron/lifecycle_guard.py:430``).
# Every OS collaborator here is stubbed, so these tests are pure logic and can
# run anywhere; they only need the stub and the module under test to agree on
# which bit means "non-blocking", which ``_install`` arranges below.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0o4000)


class _FakeSocket:
    """Stand-in for the ``socket.fromfd`` dup of fd 0."""

    def __init__(self, rcvtimeo: bytes, getsockopt_error: OSError | None = None) -> None:
        self.rcvtimeo = rcvtimeo
        self._getsockopt_error = getsockopt_error
        self.setsockopt_calls: list[tuple[int, int, bytes]] = []
        self.close_count = 0

    def getsockopt(self, level: int, optname: int, buflen: int | None = None):
        if self._getsockopt_error is not None:
            raise self._getsockopt_error
        if buflen is None:
            # Mirrors CPython: without an explicit length an *int* option is
            # assumed, so only the first ``sizeof(int)`` bytes are read.
            return struct.unpack("i", self.rcvtimeo[: struct.calcsize("i")])[0]
        return self.rcvtimeo[:buflen]

    def setsockopt(self, level: int, optname: int, value: bytes) -> None:
        self.setsockopt_calls.append((level, optname, value))
        if optname == _FakeSocketModule.SO_RCVTIMEO:
            # Real ``setsockopt`` mutates the description; later probes must see it.
            self.rcvtimeo = value

    def close(self) -> None:
        self.close_count += 1


class _FakeSocketModule:
    """Stand-in for the ``socket`` module as bound in ``_stdin_recovery``."""

    AF_UNIX = 1
    SOCK_STREAM = 1
    SOL_SOCKET = 0xFFFF
    SO_RCVTIMEO = 0x1006

    def __init__(
        self,
        sock: _FakeSocket | None = None,
        fromfd_error: OSError | None = None,
    ) -> None:
        self.sock = sock
        self._fromfd_error = fromfd_error
        self.fromfd_calls = 0

    def fromfd(self, fd: int, family: int, type_: int) -> _FakeSocket:
        self.fromfd_calls += 1
        if self._fromfd_error is not None:
            raise self._fromfd_error
        assert self.sock is not None
        return self.sock


class _FakeFcntl:
    """Stand-in for the ``fcntl`` module: ``F_GETFL`` returns fixed flags."""

    F_GETFL = 3

    def __init__(self, flags: int) -> None:
        self.flags = flags

    def fcntl(self, fd: int, op: int) -> int:
        return self.flags


def _install(monkeypatch, *, nonblock: bool, socket_mod: _FakeSocketModule) -> list:
    """Bind stub collaborators onto the module; return the set_blocking log."""
    # The module reads ``os.O_NONBLOCK`` to mask the flags; bind the same value
    # it is being handed so the pair agrees even where the attribute is absent.
    monkeypatch.setattr(_stdin_recovery.os, "O_NONBLOCK", _O_NONBLOCK, raising=False)
    flags = os.O_RDWR | (_O_NONBLOCK if nonblock else 0)
    monkeypatch.setattr(_stdin_recovery, "_HAS_FCNTL", True)
    monkeypatch.setattr(_stdin_recovery, "_fcntl", _FakeFcntl(flags))
    monkeypatch.setattr(_stdin_recovery, "_HAS_SOCKET", True)
    monkeypatch.setattr(_stdin_recovery, "_socket", socket_mod)
    set_blocking_calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        _stdin_recovery.os,
        "set_blocking",
        lambda fd, blocking: set_blocking_calls.append((fd, blocking)),
    )
    return set_blocking_calls


def _enotsock() -> OSError:
    return OSError(errno.ENOTSOCK, "Socket operation on non-socket")


def test_rcvtimeo_only_is_spurious(monkeypatch):
    """A child-set receive timeout with ``O_NONBLOCK`` clear is not a peer close."""
    sock = _FakeSocket(struct.pack(_TIMEVAL, 0, 500_000))
    mod = _FakeSocketModule(sock=sock)
    set_blocking_calls = _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []

    result = _stdin_recovery.handle_spurious_eof([], logs.append)

    assert result is True, f"expected spurious-EOF recovery; log_fn saw {logs!r}"
    assert set_blocking_calls == [(0, True)]
    cleared = [v for _, opt, v in sock.setsockopt_calls if opt == mod.SO_RCVTIMEO]
    assert cleared, "recovery must clear SO_RCVTIMEO"
    assert struct.unpack(_TIMEVAL, cleared[-1]) == (0, 0)
    assert sock.close_count == mod.fromfd_calls, "every fromfd dup must be closed"


def test_no_flags_is_genuine_eof(monkeypatch):
    """Negative control: both mutations absent still exits the read loop."""
    sock = _FakeSocket(struct.pack(_TIMEVAL, 0, 0))
    mod = _FakeSocketModule(sock=sock)
    set_blocking_calls = _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []

    result = _stdin_recovery.handle_spurious_eof([], logs.append)

    assert result is False
    assert logs == ["stdin EOF (peer closed)"]
    assert set_blocking_calls == []
    assert sock.setsockopt_calls == []


def test_non_socket_stdin_is_genuine_eof(monkeypatch):
    """Fail-safe: ``fromfd`` raising ``ENOTSOCK`` must not fake a recovery."""
    mod = _FakeSocketModule(fromfd_error=_enotsock())
    set_blocking_calls = _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []

    result = _stdin_recovery.handle_spurious_eof([], logs.append)

    assert result is False
    assert logs == ["stdin EOF (peer closed)"]
    assert set_blocking_calls == []


def test_getsockopt_enotsock_is_genuine_eof(monkeypatch):
    """Fail-safe, other shape: ``fromfd`` succeeds and ``getsockopt`` raises."""
    sock = _FakeSocket(b"", getsockopt_error=_enotsock())
    mod = _FakeSocketModule(sock=sock)
    set_blocking_calls = _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []

    result = _stdin_recovery.handle_spurious_eof([], logs.append)

    assert result is False
    assert logs == ["stdin EOF (peer closed)"]
    assert set_blocking_calls == []


def test_nonblock_on_non_socket_still_recovers(monkeypatch):
    """The pre-existing ``O_NONBLOCK`` path is unaffected by the socket probe."""
    mod = _FakeSocketModule(fromfd_error=_enotsock())
    set_blocking_calls = _install(monkeypatch, nonblock=True, socket_mod=mod)
    logs: list[str] = []

    result = _stdin_recovery.handle_spurious_eof([], logs.append)

    assert result is True
    assert set_blocking_calls == [(0, True)]


def test_genuine_close_under_preset_rcvtimeo_costs_one_iteration(monkeypatch):
    """Bounded worst case: a pre-set timeout delays a real EOF by one pass.

    If the peer really did close while stdin carried a non-zero
    ``SO_RCVTIMEO``, the first pass recovers and zeroes the timeout; the very
    next empty ``readline()`` then finds both mutations clear and exits.
    """
    sock = _FakeSocket(struct.pack(_TIMEVAL, 5, 0))
    mod = _FakeSocketModule(sock=sock)
    _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []
    times: list[float] = []

    assert _stdin_recovery.handle_spurious_eof(times, logs.append) is True
    assert _stdin_recovery.handle_spurious_eof(times, logs.append) is False
    assert logs[-1] == "stdin EOF (peer closed)"


def test_rcvtimeo_recovery_respects_the_rate_limit(monkeypatch):
    """A child re-arming the timeout cannot spin the read loop forever."""
    class _RearmingSocket(_FakeSocket):
        def setsockopt(self, level, optname, value):
            super().setsockopt(level, optname, value)
            self.rcvtimeo = struct.pack(_TIMEVAL, 5, 0)

    sock = _RearmingSocket(struct.pack(_TIMEVAL, 5, 0))
    mod = _FakeSocketModule(sock=sock)
    _install(monkeypatch, nonblock=False, socket_mod=mod)
    logs: list[str] = []
    times: list[float] = []

    results = [_stdin_recovery.handle_spurious_eof(times, logs.append) for _ in range(12)]

    assert results[: _stdin_recovery.MAX_RECOVERIES_PER_MINUTE] == [
        True
    ] * _stdin_recovery.MAX_RECOVERIES_PER_MINUTE
    assert results[_stdin_recovery.MAX_RECOVERIES_PER_MINUTE] is False
    assert "recovery rate exceeded" in logs[-1]


def test_diagnostic_reports_the_full_receive_timeout(monkeypatch):
    """Forensics must show sub-second timeouts, not a truncated ``tv_sec``."""
    packed = struct.pack(_TIMEVAL, 0, 500_000)
    sock = _FakeSocket(packed)
    mod = _FakeSocketModule(sock=sock)
    _install(monkeypatch, nonblock=False, socket_mod=mod)

    diag = _stdin_recovery.diagnose_stdin_state()

    assert "O_NONBLOCK=0" in diag
    assert f"SO_RCVTIMEO={packed!r}" in diag
