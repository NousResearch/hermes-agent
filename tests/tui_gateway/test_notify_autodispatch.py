"""Tests for the notification-poller auto-dispatch gate.

The TUI notification poller always EMITS a completed background process to the
client, but only auto-STARTS an agent turn (rid ``__notif__``) to react to it
when ``display.notify_autodispatch`` allows. Without the gate, a plain idle chat
"responds by itself" whenever a backgrounded job (subagent, terminal background
task, watch match) completes. See ``_notify_should_autodispatch``.
"""

from unittest.mock import patch

from tui_gateway.server import (
    _load_notify_autodispatch,
    _notify_should_autodispatch,
)


class _Agent:
    def __init__(self, autopilot_mode: bool) -> None:
        self.autopilot_mode = autopilot_mode


def _cfg(mode: str | None) -> dict:
    if mode is None:
        return {"display": {}}
    return {"display": {"notify_autodispatch": mode}}


def test_default_mode_is_autopilot():
    with patch("tui_gateway.server._load_cfg", return_value={"display": {}}):
        assert _load_notify_autodispatch() == "autopilot"


def test_unknown_value_falls_back_to_autopilot():
    with patch("tui_gateway.server._load_cfg", return_value=_cfg("bogus")):
        assert _load_notify_autodispatch() == "autopilot"


def test_explicit_values_pass_through():
    for mode in ("always", "autopilot", "never"):
        with patch("tui_gateway.server._load_cfg", return_value=_cfg(mode)):
            assert _load_notify_autodispatch() == mode


def test_default_plain_idle_chat_does_not_autodispatch():
    # The reported bug: a non-autopilot chat must NOT auto-react.
    with patch("tui_gateway.server._load_cfg", return_value={"display": {}}):
        assert _notify_should_autodispatch({}) is False
        assert _notify_should_autodispatch({"agent": _Agent(False)}) is False


def test_default_autopilot_session_autodispatches():
    with patch("tui_gateway.server._load_cfg", return_value={"display": {}}):
        # autopilot flagged on the session dict
        assert _notify_should_autodispatch({"autopilot": True}) is True
        # or on the live agent
        assert _notify_should_autodispatch({"agent": _Agent(True)}) is True


def test_always_autodispatches_even_without_autopilot():
    with patch("tui_gateway.server._load_cfg", return_value=_cfg("always")):
        assert _notify_should_autodispatch({}) is True
        assert _notify_should_autodispatch({"agent": _Agent(False)}) is True


def test_never_blocks_even_with_autopilot():
    with patch("tui_gateway.server._load_cfg", return_value=_cfg("never")):
        assert _notify_should_autodispatch({"autopilot": True}) is False
        assert _notify_should_autodispatch({"agent": _Agent(True)}) is False
