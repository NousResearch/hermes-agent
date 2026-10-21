"""Regression tests for the Windows integrity-level input gate.

Covers NousResearch/hermes-agent#142661: on Windows, when the sticky target's
process sits at a higher integrity level (or is protected), synthetic input is
silently discarded by UIPI while the driver still reports an unverifiable
success — so the verify → escalate ladder climbs forever. The fix refuses
determinately, on the first input action, before anything is sent.

The gate is wiring-tested here (probe stubbed per test); the probe's own
win32 ctypes path only runs on Windows and is fail-open by construction.
Stdlib + pytest + unittest.mock only. No live cua-driver, no network.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import patch

import pytest


class _FakeSession:
    """Minimal cua-driver session stub recording every tool call."""

    def __init__(self, out: Optional[dict[str, Any]] = None):
        self._out = out or {
            "isError": False,
            "data": {"message": "ok"},
            "structuredContent": {},
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, args: dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return self._out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return True

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        return True

    def _has_tool(self, name: str) -> bool:
        return name == "bring_to_front"


def _make_backend(session: _FakeSession, pid: int = 4242):
    from tools.computer_use.cua_backend import CuaDriverBackend

    be = CuaDriverBackend.__new__(CuaDriverBackend)
    be._session = session  # type: ignore[attr-defined]
    be._session_id = "test-run"  # type: ignore[attr-defined]
    be._snapshot_tokens = {}  # type: ignore[attr-defined]
    be._active_pid = pid  # type: ignore[attr-defined]
    be._active_window_id = 7  # type: ignore[attr-defined]
    return be


# ---------------------------------------------------------------------------
# _token_denial_is_determinate: the pure decision (probe-independent)
# ---------------------------------------------------------------------------


def test_decision_refuses_only_when_target_denied_and_own_readable():
    from tools.computer_use.cua_backend_input import _token_denial_is_determinate

    assert (
        _token_denial_is_determinate(lambda: True, lambda: False) is True
    )  # blocked: target above us
    assert (
        _token_denial_is_determinate(lambda: True, lambda: True) is False
    )  # target at/below us: inject
    assert (
        _token_denial_is_determinate(lambda: False, lambda: False) is False
    )  # permission-blind: degrade


def test_probe_returns_false_off_windows():
    """Off Windows (and under any ctypes failure) the probe must never claim a block."""
    from tools.computer_use import cua_backend_input

    assert cua_backend_input._win_input_blocked_by_integrity(1234) is False


# ---------------------------------------------------------------------------
# Wiring: the gate refuses before the driver is touched
# ---------------------------------------------------------------------------


def _windows_gate(probe):
    """Patch the module into "Windows mode" with *probe* as the integrity probe."""
    from tools.computer_use import cua_backend_input

    return (
        patch.object(cua_backend_input, "_WINDOWS", True),
        patch.object(cua_backend_input, "_win_input_blocked_by_integrity", probe),
    )


def test_blocked_target_refuses_first_input_without_calling_driver():
    session = _FakeSession()
    be = _make_backend(session)
    gate1, gate2 = _windows_gate(lambda pid: True)
    with gate1, gate2:
        res = be.click(x=10, y=20)

    assert res.ok is False
    assert res.code == "input_blocked_il"
    assert res.effect == "suspected_noop"
    assert res.verified is False
    # Acceptance: the refusal carries no escalation, so the ladder terminates.
    assert res.escalation is None
    assert "No input was sent" in res.message
    assert session.calls == []  # nothing was dispatched: not even one wasted SendInput


def test_blocked_target_refuses_keyboard_paths_too():
    be = _make_backend(_FakeSession())
    gate1, gate2 = _windows_gate(lambda pid: True)
    with gate1, gate2:
        typed = be.type_text("z")
        pressed = be.key("ctrl+v")

    assert typed.ok is False and typed.code == "input_blocked_il"
    assert pressed.ok is False and pressed.code == "input_blocked_il"


def test_unblocked_target_still_sends_to_driver():
    session = _FakeSession()
    be = _make_backend(session)
    gate1, gate2 = _windows_gate(lambda pid: False)
    with gate1, gate2:
        res = be.click(x=10, y=20)

    assert res.ok is True
    assert session.calls and session.calls[0][0] == "click"


def test_probe_result_is_cached_per_pid():
    session = _FakeSession()
    be = _make_backend(session, pid=4242)
    calls = []

    def probe(pid):
        calls.append(pid)
        return False

    gate1, gate2 = _windows_gate(probe)
    with gate1, gate2:
        be.click(x=1, y=2)
        be.type_text("a")
        be.scroll(direction="up")

    assert calls == [4242]  # one probe for the sticky target, not one per action


def test_new_pid_is_reprobed_and_can_pass():
    """A blocked verdict must not stick to the next, lower-integrity target."""
    session = _FakeSession()
    be = _make_backend(session, pid=111)
    gate1, gate2 = _windows_gate(lambda pid: pid == 111)
    with gate1, gate2:
        blocked = be.click(x=1, y=2)
        be._active_pid = 222
        allowed = be.click(x=1, y=2)

    assert blocked.ok is False and blocked.code == "input_blocked_il"
    assert allowed.ok is True


def test_off_windows_the_gate_is_inert():
    session = _FakeSession()
    be = _make_backend(session)
    # _WINDOWS stays False on this host: the probe must never be consulted.
    from tools.computer_use import cua_backend_input

    with patch.object(
        cua_backend_input,
        "_win_input_blocked_by_integrity",
        side_effect=AssertionError("probe consulted off Windows"),
    ):
        res = be.click(x=10, y=20)
    assert res.ok is True


def test_no_active_target_skips_gate():
    be = _make_backend(_FakeSession())
    be._active_pid = None
    gate1, gate2 = _windows_gate(lambda pid: True)
    with gate1, gate2:
        res = be.click(x=10, y=20)
    # The pre-existing no-target refusal wins; the probe is never reached.
    assert res.ok is False
    assert res.code is None and "capture()" in res.message
