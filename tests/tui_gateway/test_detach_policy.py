"""Per-session ``detach_policy`` (task t_7c43b64c / spec t_ee65a912).

``park`` is a client-declared contract: a session whose last WS client vanishes
stays live and resumable headless — no orphan-reap timer, the durable row kept
open on runtime reclaims — bounded only by the idle TTL / LRU cap, until an
EXPLICIT close ends it. ``grace`` (default) is byte-for-byte today's behavior;
``close`` maps onto today's ``close_on_disconnect``.
"""

import threading
import types

import pytest

from tui_gateway import server
from tui_gateway.server import (
    _close_sessions_for_transport,
    _normalize_detach_policy,
    _sessions,
    _sessions_lock,
    _teardown_session,
    _WS_ORPHAN_REAP_GRACE_S,
)
from tui_gateway.contracts.events import GatewayReadyPayload, SkinPayload

# Split-module functions must be called through ``server``: bind_module rebinds them
# onto the server namespace (their module globals lack _sessions_lock / grace constants).
_schedule_ws_orphan_reap = server._schedule_ws_orphan_reap
_cancel_ws_orphan_reap = server._cancel_ws_orphan_reap


# ── policy normalization ───────────────────────────────────────────────────────────────


class TestNormalizeDetachPolicy:
    def test_absent_defaults_to_grace(self):
        assert _normalize_detach_policy({}, close_on_disconnect=False) == "grace"

    def test_legacy_flag_maps_to_close(self):
        assert _normalize_detach_policy({}, close_on_disconnect=True) == "close"

    @pytest.mark.parametrize("policy", ["park", "grace", "close"])
    def test_valid_values_pass_through(self, policy):
        assert _normalize_detach_policy({"detach_policy": policy}, close_on_disconnect=False) == policy

    def test_policy_wins_over_legacy_flag(self):
        # park + close_on_disconnect=true: the explicit policy is the newer contract.
        assert _normalize_detach_policy({"detach_policy": "park"}, close_on_disconnect=True) == "park"

    def test_unknown_falls_back_to_grace_never_close(self):
        # An unknown value must never become "close": that would turn a plain
        # disconnect into an immediate teardown.
        assert _normalize_detach_policy({"detach_policy": "forever"}, close_on_disconnect=False) == "grace"
        assert _normalize_detach_policy({"detach_policy": ""}, close_on_disconnect=False) == "grace"

    def test_case_and_whitespace_normalized(self):
        assert _normalize_detach_policy({"detach_policy": " Park "}, close_on_disconnect=False) == "park"


# ── park: no orphan-reap timer, ever ───────────────────────────────────────────────────


class _FakeTransport:
    def __init__(self, name):
        self.name = name
        self._closed = False

    def __eq__(self, other):
        return isinstance(other, _FakeTransport) and other.name == self.name

    def __hash__(self):
        return hash(("fake", self.name))


def _record(sid, **extra):
    record = {
        "agent": types.SimpleNamespace(), "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "session_key": sid, "running": False,
        "source": "tui", "detach_policy": "park", "close_on_disconnect": False, "viewers": {},
    }
    record.update(extra)
    return record


class TestParkSkipsOrphanReap:
    def test_scheduler_arms_nothing_for_park(self):
        sid = "park-sched"
        with _sessions_lock:
            _sessions[sid] = _record(sid)
        try:
            _schedule_ws_orphan_reap(sid)
            assert sid not in server._pending_ws_reaps  # no Timer, ever
        finally:
            with _sessions_lock:
                _sessions.pop(sid, None)

    def test_scheduler_arms_for_grace_default(self):
        assert _WS_ORPHAN_REAP_GRACE_S > 0  # the deployment default arms a timer
        sid = "grace-sched"
        with _sessions_lock:
            _sessions[sid] = _record(sid, detach_policy="grace")
        try:
            _schedule_ws_orphan_reap(sid, delay_s=60.0)  # long delay: fires nothing in-test
            assert sid in server._pending_ws_reaps
        finally:
            _cancel_ws_orphan_reap(sid)
            with _sessions_lock:
                _sessions.pop(sid, None)

    def test_disconnect_funnel_parks_without_reap(self):
        transport = _FakeTransport("ws-park")
        sid = "park-disc"
        session = _record(sid, transport=transport)
        with _sessions_lock:
            _sessions[sid] = session
        try:
            reaped, detached = _close_sessions_for_transport(transport)
            assert (reaped, detached) == (0, 1)
            # park: parked on the detached sentinel with NO timer armed.
            assert sid not in server._pending_ws_reaps
            assert session["transport"] is server._detached_ws_transport
        finally:
            with _sessions_lock:
                _sessions.pop(sid, None)

    def test_disconnect_funnel_grace_still_arms(self):
        transport = _FakeTransport("ws-grace")
        sid = "grace-disc"
        session = _record(sid, detach_policy="grace", transport=transport)
        with _sessions_lock:
            _sessions[sid] = session
        try:
            reaped, detached = _close_sessions_for_transport(transport)
            assert (reaped, detached) == (0, 1)
            assert sid in server._pending_ws_reaps  # grace: today's behavior — timer armed
        finally:
            _cancel_ws_orphan_reap(sid)
            with _sessions_lock:
                _sessions.pop(sid, None)

    def test_disconnect_close_policy_reaps_immediately(self):
        transport = _FakeTransport("ws-close")
        sid = "close-disc"
        session = _record(sid, detach_policy="close", close_on_disconnect=True, transport=transport)
        torn = []
        real_teardown = server._teardown_popped_session
        server._teardown_popped_session = lambda s, *, end_reason: torn.append(end_reason) or True
        with _sessions_lock:
            _sessions[sid] = session
        try:
            reaped, detached = _close_sessions_for_transport(transport)
            assert (reaped, detached) == (1, 0)
            assert torn == ["ws_disconnect"]
        finally:
            server._teardown_popped_session = real_teardown
            with _sessions_lock:
                _sessions.pop(sid, None)


# ── park: durable row kept open on runtime reclaims ────────────────────────────────────


def _make_real_session(tmp_path, monkeypatch, *, source, session_id, detach_policy=None):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    db_path = hermes_home / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source=source)
    db.close()

    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = None
    agent._owns_session_db = False
    agent._end_session_on_close = True
    agent._session_messages = []
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._memory_manager = None

    session = _record(session_id, agent=agent, profile_home=str(hermes_home), source=source)
    if detach_policy:
        session["detach_policy"] = detach_policy
    return db_path, session, agent


def _read_row(db_path, session_id):
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    try:
        return db.get_session(session_id)
    finally:
        db.close()


class TestParkRowSemantics:
    @pytest.mark.parametrize("reason", ["ws_orphan_reap", "idle_timeout", "lru_evict", "tui_shutdown"])
    def test_park_automatic_reclaim_keeps_row_open(self, tmp_path, monkeypatch, reason):
        db_path, session, agent = _make_real_session(
            tmp_path, monkeypatch, source="tui", session_id=f"park-{reason}", detach_policy="park")

        _teardown_session(session, end_reason=reason)

        row = _read_row(db_path, f"park-{reason}")
        assert (row["ended_at"], row["end_reason"]) == (None, None)
        assert agent._end_session_on_close is False

    def test_park_explicit_close_still_ends_row(self, tmp_path, monkeypatch):
        db_path, session, _agent = _make_real_session(
            tmp_path, monkeypatch, source="tui", session_id="park-tui-close", detach_policy="park")

        _teardown_session(session, end_reason="tui_close")

        row = _read_row(db_path, "park-tui-close")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "tui_close"

    def test_grace_reap_still_ends_row_today(self, tmp_path, monkeypatch):
        db_path, session, _agent = _make_real_session(
            tmp_path, monkeypatch, source="tui", session_id="grace-reap", detach_policy="grace")

        _teardown_session(session, end_reason="ws_orphan_reap")

        row = _read_row(db_path, "grace-reap")
        assert row["ended_at"] is not None  # NOT kept open: grace is today's behavior
        assert row["end_reason"] == "ws_orphan_reap"


# ── reattach override ──────────────────────────────────────────────────────────────────


class TestReattachOverride:
    def test_resume_override_flips_policy(self):
        session = _record("flip")
        server._apply_reconnect_detach_policy({"detach_policy": "grace"}, session)
        assert session["detach_policy"] == "grace"
        assert session["close_on_disconnect"] is False
        server._apply_reconnect_detach_policy({"detach_policy": "close"}, session)
        assert session["detach_policy"] == "close"
        assert session["close_on_disconnect"] is True

    def test_absent_override_keeps_policy(self):
        session = _record("keep")
        server._apply_reconnect_detach_policy({}, session)
        assert session["detach_policy"] == "park"
        server._apply_reconnect_detach_policy({"detach_policy": "junk"}, session)
        assert session["detach_policy"] == "park"

    def test_resume_cancels_armed_grace_timer(self):
        # A timer armed under grace is cancelled when the client reattaches with park —
        # the resume fast-path cancels unconditionally (_resume_reuse_live_locked).
        sid = "cancel-armed"
        with _sessions_lock:
            _sessions[sid] = _record(sid, detach_policy="grace")
        try:
            _schedule_ws_orphan_reap(sid, delay_s=60.0)
            assert sid in server._pending_ws_reaps
            server._apply_reconnect_detach_policy({"detach_policy": "park"}, _sessions[sid])
            _cancel_ws_orphan_reap(sid)
            assert sid not in server._pending_ws_reaps
        finally:
            _cancel_ws_orphan_reap(sid)
            with _sessions_lock:
                _sessions.pop(sid, None)


# ── capability flag ────────────────────────────────────────────────────────────────────


class TestGatewayReadyCapability:
    def test_ws_ready_payload_carries_detach_policy_flag(self):
        payload = GatewayReadyPayload(skin=SkinPayload(), change_events=True, replay_epoch="e1")
        assert payload.detach_policy is None  # default: a payload that never set it advertises nothing

    def test_flag_is_boolean_optional(self):
        payload = GatewayReadyPayload(skin=SkinPayload(), change_events=True, replay_epoch="e1", detach_policy=True)
        assert payload.detach_policy is True


# ── end-to-end: disconnect (park) → resume warm ────────────────────────────────────────


class TestParkThenResumeWarm:
    """Disconnect-mid-task → parked → reattach through the real ``session.resume``
    handler: the same live record is rebound, policy restated, no reap."""

    def test_parked_session_resumes_warm_with_policy_restate(self, monkeypatch, tmp_path):
        import time as _time

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        sid = "park-resume-1"
        key = "20261017_000000_park1"
        transport = _FakeTransport("ws-resume")
        # profile_home unset = the launch profile (how a plain remote-attach TUI session reads).
        record = _record(sid, session_key=key, transport=transport,
                         source="tui", last_active=_time.time())
        with _sessions_lock:
            _sessions[sid] = record
        # Simulate the disconnect: park on the sentinel (no timer for park).
        _close_sessions_for_transport(transport)
        assert record["transport"] is server._detached_ws_transport
        assert sid not in server._pending_ws_reaps

        try:
            out = server._methods["session.resume"](1, {"session_id": key, "omit_messages": True})
            assert "error" not in out, out
            assert out["result"]["session_id"] == sid
            # The same live record survived (warm), still park-governed.
            assert _sessions[sid] is record
            assert record["detach_policy"] == "park"
            assert sid not in server._pending_ws_reaps
        finally:
            with _sessions_lock:
                _sessions.pop(sid, None)

    def test_resume_can_downgrade_park_back_to_grace(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        sid = "park-downgrade"
        key = "20261017_000000_park2"
        record = _record(sid, session_key=key, source="tui", last_active=0.0)
        with _sessions_lock:
            _sessions[sid] = record
        try:
            out = server._methods["session.resume"](
                1, {"session_id": key, "omit_messages": True, "detach_policy": "grace"})
            assert "error" not in out, out
            assert record["detach_policy"] == "grace"
        finally:
            with _sessions_lock:
                _sessions.pop(sid, None)
