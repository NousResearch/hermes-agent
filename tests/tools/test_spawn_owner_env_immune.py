"""Regression: env-immune spawn-owner capture for the WebUI Option 3
cross-session wakeup misroute safety net.

Background (RCA: rca-xsession-wakeup-misroute.md): a
``terminal(background=true, notify_on_complete=true)`` spawn historically
captured its owner session from the process-global
``os.environ['HERMES_SESSION_KEY']`` via ``get_current_session_key``. Two
concurrent WebUI turns race on that single global slot, so session A's
watcher could wake session B.

This module proves the core-side primitive that makes WebUI Option 3
(``api/background_process._resolve_wakeup_target``) actually able to
detect that misroute:

* ``tools.approval.get_env_immune_session_key()`` reads context-local
  state ONLY — never ``os.environ`` — so a concurrent turn overwriting
  the global slot cannot corrupt it.
* ``ProcessSession.spawn_session_id`` is stamped from that env-immune
  value at spawn and is one of the duck-typed attribute names Option 3
  probes.
* When no per-turn identity contextvar is bound (CLI / cron / plain
  tests / pre-Option-1 WebUI), the value is ``""`` so Option 3 stays a
  pure pass-through and never suppresses a valid wakeup.

These are behavioral invariants exercised against the REAL modules, not
source-text change-detectors.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.approval import (
    _approval_session_key,
    get_current_session_key,
    get_env_immune_session_key,
)
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


# ---------------------------------------------------------------------------
# Invariant 1: env-immune helper ignores the process-global env slot.
# ---------------------------------------------------------------------------

def test_env_immune_key_ignores_process_global_env(monkeypatch):
    """``get_env_immune_session_key`` must NOT read os.environ.

    ``get_current_session_key`` (the historically racy path) DOES fall
    back to it — assert that contrast so the test is not a tautology.
    """
    monkeypatch.setenv("HERMES_SESSION_KEY", "polluted-by-other-turn")
    # No contextvar bound (conftest resets _approval_session_key to "").

    # Racy path: falls through to the polluted global slot.
    assert get_current_session_key(default="") == "polluted-by-other-turn"

    # Env-immune path: ignores the global slot entirely -> "".
    assert get_env_immune_session_key() == ""


def test_env_immune_key_prefers_bound_contextvar(monkeypatch):
    """When the per-turn contextvar IS bound, the env-immune helper
    returns it and still ignores a conflicting global env slot."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "polluted-by-other-turn")
    token = _approval_session_key.set("real-owner-sid")
    try:
        assert get_env_immune_session_key() == "real-owner-sid"
    finally:
        _approval_session_key.reset(token)


def test_env_immune_key_reads_session_context_contextvar(monkeypatch):
    """Second source: an explicitly-bound gateway.session_context
    ``_SESSION_KEY`` contextvar is honored, but its ``_UNSET`` sentinel
    (the env-fallback trigger) is NOT."""
    from gateway import session_context as sc

    monkeypatch.setenv("HERMES_SESSION_KEY", "polluted-by-other-turn")

    # _UNSET (never bound) -> env-immune helper must NOT fall back to env.
    assert sc._SESSION_KEY.get() is sc._UNSET
    assert get_env_immune_session_key() == ""

    # Explicitly bound -> honored.
    tok = sc._SESSION_KEY.set("ctx-owner-sid")
    try:
        assert get_env_immune_session_key() == "ctx-owner-sid"
    finally:
        sc._SESSION_KEY.reset(tok)


# ---------------------------------------------------------------------------
# Invariant 2: spawn stamps the env-immune owner and it is NOT corrupted
# by a later env mutation (exercises the real race window, not a literal).
# ---------------------------------------------------------------------------

def _patched_spawn(registry, command="echo hi", **kw):
    """Run spawn_local with Popen/threads patched so no real process is
    created; returns the ProcessSession."""
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("subprocess.Popen", return_value=fake_proc), \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
        return registry.spawn_local(command, cwd="/tmp", **kw)


def test_spawn_session_id_field_default_empty(registry):
    """Compat: a spawn with no env-immune owner leaves spawn_session_id
    empty so Option 3 stays pass-through (never suppresses a wakeup)."""
    s = _patched_spawn(registry)
    assert isinstance(s, ProcessSession)
    assert s.spawn_session_id == ""


def test_spawn_session_id_stamped_and_immune_to_later_env_race(registry, monkeypatch):
    """The value stamped at spawn must reflect the spawning turn's
    env-immune identity and must NOT change when a concurrent turn later
    overwrites the process-global env slot.

    This mirrors the RCA timeline: lock released, agent runs, OTHER
    session stamps os.environ — but the already-captured owner stays put.
    """
    token = _approval_session_key.set("turn-A-sid")
    try:
        # terminal_tool.py computes this exact value at spawn time.
        captured = get_env_immune_session_key()
        s = _patched_spawn(registry, spawn_session_id=captured)
        assert s.spawn_session_id == "turn-A-sid"

        # Concurrent turn B overwrites the racy global slot mid-flight.
        monkeypatch.setenv("HERMES_SESSION_KEY", "turn-B-sid")
        # The already-spawned session's env-immune owner is unchanged.
        assert s.spawn_session_id == "turn-A-sid"
    finally:
        _approval_session_key.reset(token)


def test_concurrent_turns_capture_their_own_spawn_owner_under_env_race():
    """Deterministic two-turn race: each turn binds its own
    ``_approval_session_key`` and, even when the other turn stamps the
    process-global env slot in the documented "lock released, agent still
    running" window, each turn's env-immune capture is ITS OWN sid.

    Pre-fix (capture via get_current_session_key -> env fallback) this
    cross-captured the other turn's sid (see repro_xsession_wakeup.py in
    t_f62ff1e8). Post-fix it must not.
    """
    captured: dict = {}
    start = threading.Barrier(2)
    a_bound = threading.Event()

    def turn(my_sid: str, label: str, other_started: threading.Event):
        # Each turn runs in its own thread with its own contextvar copy,
        # exactly as agent/tool_executor.py copy_context() worker does.
        tok = _approval_session_key.set(my_sid)
        try:
            # Mimic streaming.py turn-start writing the racy global slot.
            os.environ["HERMES_SESSION_KEY"] = my_sid
            start.wait()
            if label == "A":
                a_bound.set()
                # Let B run its turn-start (stamping the global slot to B)
                # while A is still "mid-turn" about to spawn its watcher.
                other_started.wait(timeout=2)
            else:
                a_bound.wait(timeout=2)
                os.environ["HERMES_SESSION_KEY"] = my_sid
                other_started.set()
            # The exact call terminal_tool makes at spawn:
            captured[label] = get_env_immune_session_key()
        finally:
            _approval_session_key.reset(tok)

    b_started = threading.Event()
    prev_env = os.environ.get("HERMES_SESSION_KEY")
    ta = threading.Thread(target=turn, args=("sid-A", "A", b_started))
    tb = threading.Thread(target=turn, args=("sid-B", "B", b_started))
    try:
        ta.start(); tb.start(); ta.join(timeout=5); tb.join(timeout=5)
    finally:
        if prev_env is None:
            os.environ.pop("HERMES_SESSION_KEY", None)
        else:
            os.environ["HERMES_SESSION_KEY"] = prev_env

    assert captured.get("A") == "sid-A", (
        f"Turn A captured {captured.get('A')!r} (expected 'sid-A'). "
        "Env-immune spawn-owner capture is leaking the process-global "
        "env race — the xsession misroute is NOT prevented."
    )
    assert captured.get("B") == "sid-B", (
        f"Turn B captured {captured.get('B')!r} (expected 'sid-B')."
    )


# ---------------------------------------------------------------------------
# Invariant 3: Option 3 duck-typing contract — spawn_session_id is one of
# the names the WebUI resolver probes, and the pass-through/re-route
# semantics hold when it is read off a real ProcessSession.
# ---------------------------------------------------------------------------

def test_spawn_session_id_satisfies_option3_ducktype_contract(registry):
    """Replicates api/background_process._env_immune_spawn_owner /
    _resolve_wakeup_target against a REAL core ProcessSession to prove
    the cross-repo contract end to end (core side).
    """
    _ENV_IMMUNE_OWNER_ATTRS = (
        "spawn_session_id",
        "owner_session_id",
        "turn_session_id",
    )

    def _env_immune_spawn_owner(proc_session) -> str:
        if proc_session is None:
            return ""
        for attr in _ENV_IMMUNE_OWNER_ATTRS:
            val = getattr(proc_session, attr, "")
            if val:
                return str(val)
        return ""

    def _resolve_wakeup_target(*, session_key_resolved_sid, proc_session):
        resolved = str(session_key_resolved_sid or "")
        owner = _env_immune_spawn_owner(proc_session)
        if not owner or owner == resolved:
            return resolved
        return owner

    # (a) No env-immune owner -> pure pass-through (today's CLI/cron, and
    #     keeps a valid Option Z wakeup working).
    s_empty = _patched_spawn(registry)
    assert _resolve_wakeup_target(
        session_key_resolved_sid="whatever-resolved",
        proc_session=s_empty,
    ) == "whatever-resolved"

    # (b) Owner == resolved -> pass-through (the post-Option-1 norm).
    token = _approval_session_key.set("owner-X")
    try:
        s_match = _patched_spawn(
            registry, spawn_session_id=get_env_immune_session_key()
        )
    finally:
        _approval_session_key.reset(token)
    assert _resolve_wakeup_target(
        session_key_resolved_sid="owner-X", proc_session=s_match
    ) == "owner-X"

    # (c) Owner != resolved -> POSITIVE mismatch -> re-route to the true
    #     env-immune owner (the exact misroute Option 3 must block).
    assert _resolve_wakeup_target(
        session_key_resolved_sid="wrong-session-from-env-race",
        proc_session=s_match,
    ) == "owner-X"
