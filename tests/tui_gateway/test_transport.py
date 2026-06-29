"""Tests for tui_gateway.transport — WebSocket transport error handling."""

from __future__ import annotations

import errno

from tui_gateway.transport import _PEER_GONE_ERRNOS


def test_einval_in_peer_gone_errnos():
    """EINVAL (errno 22) must be treated as a clean disconnect on Windows.

    On Windows, detached stdout raises OSError(22, 'Invalid argument')
    instead of POSIX EPIPE. Without EINVAL in the set, this was re-raised
    as a fatal error causing WebSocket 1011 crash loops.

    Regression test for #55119.
    """
    assert errno.EINVAL in _PEER_GONE_ERRNOS


def test_epipe_in_peer_gone_errnos():
    """EPIPE must always be treated as a clean disconnect."""
    assert errno.EPIPE in _PEER_GONE_ERRNOS


def test_peer_gone_errnos_is_frozen():
    """The set must be immutable to prevent accidental mutation."""
    assert isinstance(_PEER_GONE_ERRNOS, frozenset)
