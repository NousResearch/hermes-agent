"""Regression test: TUI gateway shutdown calls interrupt_all().

Before the fix, `_shutdown_sessions` only closed open WS sessions but never
signalled running async delegations to stop.  CLI and gateway both call
`tools.async_delegation.interrupt_all()` on shutdown; the TUI path was the
only entry point that didn't, leaving background delegations running as
orphaned daemon threads until the process exited.
"""

from unittest.mock import MagicMock, patch

from tui_gateway import server


def test_shutdown_sessions_calls_interrupt_all():
    """_shutdown_sessions must call interrupt_all before closing sessions."""
    interrupt_mock = MagicMock(return_value=0)

    with patch("tui_gateway.server._sessions_lock"), \
         patch.dict("tui_gateway.server._sessions", {}), \
         patch("tools.async_delegation.interrupt_all", interrupt_mock):
        server._shutdown_sessions()

    interrupt_mock.assert_called_once_with(reason="tui_shutdown")


def test_shutdown_sessions_interrupt_error_does_not_prevent_close():
    """If interrupt_all raises, session close still runs."""
    closed = []

    def _fake_close(sid, *, end_reason):
        closed.append(sid)

    with patch("tui_gateway.server._sessions_lock"), \
         patch.dict("tui_gateway.server._sessions", {"sid1": {}, "sid2": {}}), \
         patch("tui_gateway.server._close_session_by_id", side_effect=_fake_close), \
         patch(
             "tools.async_delegation.interrupt_all",
             side_effect=RuntimeError("boom"),
         ):
        server._shutdown_sessions()

    assert set(closed) == {"sid1", "sid2"}
