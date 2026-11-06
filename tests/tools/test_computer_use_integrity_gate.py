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

import os
import sys
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
# _il_decision: the pure decision (probe-independent)
# ---------------------------------------------------------------------------


def test_decision_compares_rids_when_both_readable():
    """SeDebugPrivilege case: token readability is not the verdict — an elevated caller can open
    even a System token, so the RID comparison must be the thing that refuses."""
    from tools.computer_use.cua_backend_input import _il_decision

    reason = _il_decision(0x3000, 0x4000)  # System above High: refuse
    assert reason is not None
    assert "0x4000" in reason and "0x3000" in reason  # the refusal message carries both levels
    assert _il_decision(0x3000, 0x3000) is None  # same level: inject
    assert _il_decision(0x3000, 0x2000) is None  # below us: inject


def test_decision_blocks_unreadable_target_but_not_a_blind_probe():
    from tools.computer_use.cua_backend_input import _il_decision

    assert _il_decision(0x3000, None) is not None  # target token denied, ours readable: block
    assert _il_decision(None, 0x4000) is None  # our own token unreadable: probe is blind, degrade
    assert _il_decision(None, None) is None  # permission-blind probe: degrade


def test_probe_returns_none_off_windows(monkeypatch):
    """Off Windows (and under any ctypes failure) the probe must never claim a block. ``_WINDOWS``
    is forced False so the test also holds on a Windows host instead of calling the real probe."""
    from tools.computer_use import cua_backend_input

    monkeypatch.setattr(cua_backend_input, "_WINDOWS", False)
    assert cua_backend_input._win_input_blocked_by_integrity(1234) is None


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


_BLOCKED_REASON = ("the target process runs at integrity 0x4000 (System), above this process at "
                   "0x3000 (High)")


def test_blocked_target_refuses_first_input_without_calling_driver():
    session = _FakeSession()
    be = _make_backend(session)
    gate1, gate2 = _windows_gate(lambda pid: _BLOCKED_REASON)
    with gate1, gate2:
        res = be.click(x=10, y=20)

    assert res.ok is False
    assert res.code == "input_blocked_il"
    assert res.effect == "suspected_noop"
    assert res.verified is False
    # Acceptance: the refusal carries no escalation, so the ladder terminates.
    assert res.escalation is None
    assert "No input was sent" in res.message
    assert "0x4000" in res.message  # the refusal names the measured levels, not a guess
    assert session.calls == []  # nothing was dispatched: not even one wasted SendInput


def test_blocked_target_refuses_keyboard_paths_too():
    be = _make_backend(_FakeSession())
    gate1, gate2 = _windows_gate(lambda pid: _BLOCKED_REASON)
    with gate1, gate2:
        typed = be.type_text("z")
        pressed = be.key("ctrl+v")

    assert typed.ok is False and typed.code == "input_blocked_il"
    assert pressed.ok is False and pressed.code == "input_blocked_il"


def test_unblocked_target_still_sends_to_driver():
    session = _FakeSession()
    be = _make_backend(session)
    gate1, gate2 = _windows_gate(lambda pid: None)
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
    gate1, gate2 = _windows_gate(lambda pid: _BLOCKED_REASON if pid == 111 else None)
    with gate1, gate2:
        blocked = be.click(x=1, y=2)
        be._active_pid = 222
        allowed = be.click(x=1, y=2)

    assert blocked.ok is False and blocked.code == "input_blocked_il"
    assert allowed.ok is True


def test_off_windows_the_gate_is_inert(monkeypatch):
    session = _FakeSession()
    be = _make_backend(session)
    from tools.computer_use import cua_backend_input

    # Force "off Windows" regardless of host so the test is green on a Windows host too.
    monkeypatch.setattr(cua_backend_input, "_WINDOWS", False)
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
    gate1, gate2 = _windows_gate(lambda pid: _BLOCKED_REASON)
    with gate1, gate2:
        res = be.click(x=10, y=20)
    # The pre-existing no-target refusal wins; the probe is never reached.
    assert res.ok is False
    assert res.code is None and "capture()" in res.message


# ---------------------------------------------------------------------------
# The probe has to actually parse a RID (regression: a dead probe is invisible
# to the wiring tests above, because they stub this level)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="live integrity probe is Windows-only")
def test_live_probe_parses_a_rid_for_its_own_process():
    """``GetTokenInformation``'s sizing call always returns 0 (ERROR_INSUFFICIENT_BUFFER) while writing
    the required length into ``needed``. A probe that returns early on that return value answers
    ``None`` for every process, and the gate silently never fires — which no stubbed test can see.
    Our own pid must therefore come back with two readable, equal RIDs."""
    from tools.computer_use.cua_backend_input import _win_integrity_rids, _win_input_blocked_by_integrity

    rids = _win_integrity_rids(os.getpid())
    assert rids.probed is True
    assert rids.own is not None, "our own token must be readable"
    assert rids.target is not None, "our own process must be probeable"
    assert rids.own == rids.target
    assert rids.own in {0x1000, 0x2000, 0x3000, 0x4000, 0x5000}
    # Same level: the gate must not fire for ourselves.
    assert _win_input_blocked_by_integrity(os.getpid()) is None


def test_rids_tuple_is_fail_open_off_windows():
    """Force the flag rather than relying on the host, so this holds on Windows CI too."""
    from tools.computer_use import cua_backend_input

    original = cua_backend_input._WINDOWS
    try:
        cua_backend_input._WINDOWS = False
        rids = cua_backend_input._win_integrity_rids(1234)
    finally:
        cua_backend_input._WINDOWS = original
    assert rids.probed is False
    assert rids.own is None and rids.target is None
    assert cua_backend_input._win_input_blocked_by_integrity(1234) is None
