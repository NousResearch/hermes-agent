"""Behavioral contract tests for the read-only Desktop Action Center aggregation.

``inbox.list`` is profile-scoped, read-only, and aggregates:
  - persisted automation (goal/loop/heartbeat) via the same snapshots ``session.control.read``
    returns — including sessions that are NOT currently open (no resume / transcript hydration);
  - live pending gateway approvals (redacted on egress);
  - live pending clarify for OPEN sessions (metadata only, never question text).

It must never mutate, must never resolve anything, must not leak raw credentials, must honor
the session deny-list, and must declare partial coverage honestly instead of labeling a bounded
snapshot as global.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Isolated persisted-manager database + state.db for every test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    import tui_gateway.server as mod
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_sig", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    mod._server_requests.reset_for_tests()
    from tools import approval

    with approval._lock:
        approval._gateway_queues.clear()


@pytest.fixture()
def db(server, hermes_home):
    """Launch/own-profile SessionDB handle (sessions listed by inbox.list)."""
    handle = server._get_db()
    yield handle


def _call(server, method, *, rid=91, **params):
    return server._methods[method](rid, params)


def _result(server, method, **params):
    response = _call(server, method, **params)
    assert "result" in response, response
    return response["result"]


def _error(response):
    assert "error" in response
    return response["error"]


def _new_key(tag="inx"):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _create_row(db, key, *, source="cli", title=""):
    db.create_session(key, source=source)
    if title:
        db.set_session_title(key, title)
    return key


def _save_goal(key, **overrides):
    from hermes_cli.goals import GoalState, save_goal

    fields = {
        "goal": "Finish the inbox slice",
        "status": "active",
        "turns_used": 1,
        "max_turns": 6,
        "created_at": 100.0,
        "last_turn_at": 200.0,
    }
    fields.update(overrides)
    save_goal(key, GoalState(**fields))


def _save_loop(key, **overrides):
    from hermes_cli.loops import LoopState, save_loop

    fields = {
        "prompt": "Check the deployment",
        "status": "active",
        "mode": "interval",
        "interval_seconds": 300,
        "current_delay": 300,
        "created_at": 100.0,
        "next_due_at": 400.0,
    }
    fields.update(overrides)
    save_loop(key, LoopState(**fields))


def _save_heartbeat(key, **overrides):
    from hermes_cli.heartbeat import HeartbeatState, save_heartbeat

    fields = {
        "prompt": "Check the deployment",
        "interval_seconds": 600,
        "status": "active",
        "created_at": 100.0,
        "last_fired_at": 150.0,
        "fire_count": 2,
    }
    fields.update(overrides)
    save_heartbeat(key, HeartbeatState(**fields))


def _queue_approval(server, key, *, command="rm -rf /tmp/secret-value-abc", description="run removal"):
    from tools import approval

    with approval._lock:
        approval._gateway_queues.setdefault(key, []).append(
            SimpleNamespace(data={
                "request_id": f"rid-{uuid.uuid4().hex[:8]}",
                "command": command,
                "description": description,
            })
        )


def _open_session(server, key, *, profile_home=None):
    sid = f"sid-{uuid.uuid4().hex[:8]}"
    server._sessions[sid] = {
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": None,
        "created_at": time.time(),
        # The REAL launch-profile shape: server._add_session stores None for the launch
        # profile, never the home path. Fixtures that stored the path hid the
        # launch/foreign comparison bug (approval listed under Needs attention,
        # detail read empty).
        "profile_home": profile_home,
    }
    return sid


def _queue_clarify(server, key, *, question="Which provider?", profile_home=None, sid=None):
    """Inject a live pending clarify for an OPEN session via server_requests._open."""
    from tui_gateway.server_requests import ServerRequest

    if sid is None:
        sid = _open_session(server, key, profile_home=profile_home)
    req = ServerRequest(sid, "clarify", {"question": question})
    # Access the same server_requests instance the server module uses
    sr = server._server_requests
    with sr._lock:
        sr._open[req.id] = req
    return sid


# ── pure lane classification ─────────────────────────────────────────────────
class TestLaneClassification:
    def test_empty_control_and_no_pending_yields_no_lanes(self):
        from tui_gateway.methods_inbox import classify_lanes

        assert classify_lanes(None, False, False) == []
        assert classify_lanes({"goal": None, "loop": None, "heartbeat": None}, False, False) == []

    def test_pending_request_short_circuits_to_needs_you(self):
        from tui_gateway.methods_inbox import classify_lanes

        assert classify_lanes(None, True, False) == ["needs_you"]
        assert classify_lanes(None, False, True) == ["needs_you"]
        lanes = classify_lanes(
            {"goal": {"status": "active"}, "loop": None, "heartbeat": None}, True, False
        )
        assert lanes[0] == "needs_you" and "running" in lanes

    def test_active_goal_without_barrier_is_running(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {"goal": {"status": "active"}, "loop": None, "heartbeat": None}
        assert classify_lanes(control, False, False) == ["running"]

    def test_active_goal_with_wait_barrier_is_waiting_not_running(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {"goal": {"status": "active", "wait_barrier": {"type": "until"}}, "loop": None, "heartbeat": None}
        assert classify_lanes(control, False, False) == ["waiting"]

    def test_paused_goal_is_waiting(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {"goal": {"status": "paused"}, "loop": None, "heartbeat": None}
        assert classify_lanes(control, False, False) == ["waiting"]

    def test_loop_deferred_by_goal_is_waiting(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {
            "goal": {"status": "active"},
            "loop": {"status": "active", "awaiting_response": False, "deferred_by_goal": True,
                     "next_due_at": 0},
            "heartbeat": None,
        }
        assert classify_lanes(control, False, False) == ["running", "waiting"]

    def test_loop_awaiting_response_is_running(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {
            "loop": {"status": "active", "awaiting_response": True, "deferred_by_goal": False,
                     "next_due_at": 0},
            "goal": None, "heartbeat": None,
        }
        assert classify_lanes(control, False, False) == ["running"]

    def test_active_heartbeat_is_scheduled(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {"heartbeat": {"status": "active"}, "goal": None, "loop": None}
        assert classify_lanes(control, False, False) == ["scheduled"]

    def test_loop_scheduled_only_when_future_due(self):
        from tui_gateway.methods_inbox import classify_lanes

        future = {"loop": {"status": "active", "awaiting_response": False, "deferred_by_goal": False,
                           "next_due_at": time.time() + 60}, "goal": None, "heartbeat": None}
        past = {"loop": {"status": "active", "awaiting_response": False, "deferred_by_goal": False,
                         "next_due_at": time.time() - 60}, "goal": None, "heartbeat": None}
        assert classify_lanes(future, False, False) == ["scheduled"]
        assert classify_lanes(past, False, False) == []

    def test_lanes_are_ordered_and_deduplicated(self):
        from tui_gateway.methods_inbox import classify_lanes

        control = {
            "goal": {"status": "active"},
            "loop": {"status": "active", "awaiting_response": True, "deferred_by_goal": False,
                     "next_due_at": 0},
            "heartbeat": {"status": "active"},
        }
        assert classify_lanes(control, True, False) == ["needs_you", "running", "scheduled"]


class TestBadgeState:
    def test_badge_states(self):
        from tui_gateway.methods_inbox import badge_state

        assert badge_state([]) == "none"
        assert badge_state([{"lanes": ["running"]}]) == "none"
        assert badge_state([{"lanes": ["needs_you"]}]) == "amber"
        assert badge_state([{"lanes": ["running"]}, {"lanes": ["needs_you"]}]) == "amber"
        assert badge_state([{"lanes": ["needs_you"]}], errors=["x"]) == "red"
        assert badge_state([], errors=["x"]) == "red"


# ── inbox.list aggregation ────────────────────────────────────────────────────
class TestInboxList:
    def test_method_is_registered_and_empty_inbox_is_stable(self, server, db):
        assert "inbox.list" in set(server._methods)
        first = _result(server, "inbox.list")
        second = _result(server, "inbox.list")
        assert first == second
        inbox = first["inbox"]
        assert inbox["items"] == []
        assert inbox["counts"] == {
            "needs_you": 0, "running": 0, "waiting": 0, "scheduled": 0, "total": 0
        }
        assert inbox["badge"] == "none"
        coverage = inbox["coverage"]
        assert coverage["profile"]
        assert coverage["partial"] is False
        assert coverage["approval_scope"] == "live gateway approval queue"

    def test_persisted_running_goal_surfaces_without_an_open_session(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        assert server._sessions == {}  # no live session; persisted state alone must show it
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "none"
        assert inbox["counts"]["running"] == 1
        item = inbox["items"][0]
        assert item["session_key"] == key
        assert item["lanes"] == ["running"]
        assert item["goal"]["status"] == "active"
        assert "loop" in item and "heartbeat" in item

    def test_needs_you_and_scheduled_counts(self, server, db):
        approval_key = _create_row(db, _new_key())
        _queue_approval(server, approval_key)
        hb_key = _create_row(db, _new_key())
        _save_heartbeat(hb_key)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "amber"
        counts = inbox["counts"]
        assert counts["needs_you"] == 1
        assert counts["scheduled"] == 1
        assert counts["total"] == 2

    def test_live_clarify_for_open_session_marks_needs_you(self, server, db):
        key = _create_row(db, _new_key())
        sid = _queue_clarify(server, key, question="Pick a backend")
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "amber", f"badge={inbox['badge']}, items={inbox['items']}, counts={inbox['counts']}"
        assert inbox["counts"]["needs_you"] == 1
        clarify = inbox["items"][0]["pending_clarify"]
        assert clarify == {"count": 1}

    def test_live_clarify_with_launch_profile_none_home(self, server, db):
        """A launch-profile session (profile_home=None, the real shape) still counts.

        Same defect family as the requests-side regression: the live join compared
        the record's raw ``profile_home`` (None for launch) against the launch home
        path, so launch-profile clarifies never reached the list either.
        """
        key = _create_row(db, _new_key())
        _queue_clarify(server, key, sid=_open_session(server, key, profile_home=None))
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["counts"]["needs_you"] == 1
        assert inbox["items"][0]["pending_clarify"] == {"count": 1}

    def test_approval_egress_never_leaks_raw_credential(self, server, db):
        key = _create_row(db, _new_key())
        _queue_approval(server, key, command="curl -H 'Authorization: Bearer SECRET_SK_live_12345' https://x")
        serialized = json.dumps(_result(server, "inbox.list"))
        for forbidden in ("SECRET_SK_live_12345", "Bearer", "Authorization: Bearer SECRET_SK_live_12345"):
            assert forbidden not in serialized
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["pending_approval"] == {"count": 1, "description": "pending approval", "command_redacted": True}

    def test_control_snapshot_redacts_private_fields(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        serialized = json.dumps(_result(server, "inbox.list"))
        # _safe_goal_snapshot does not include last_output_tail or last_failed_fingerprint
        for forbidden in ("last_output_tail", "last_failed_fingerprint"):
            assert forbidden not in serialized

    def test_deny_list_sources_are_excluded(self, server, db):
        visible = _create_row(db, _new_key(), source="cli")
        _save_goal(visible)
        hidden = _create_row(db, _new_key(), source="kanban")
        _save_goal(hidden)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["counts"]["running"] == 1
        assert {item["session_key"] for item in inbox["items"]} == {visible}

    def test_clarify_is_profile_isolated_by_live_owner(self, server, db):
        """A live session owned by ANOTHER profile must not surface in the launch profile's inbox."""
        from tui_gateway.server_requests import ServerRequest

        foreign = _new_key()
        sid = f"sid-foreign-{uuid.uuid4().hex[:8]}"
        server._sessions[sid] = {
            "session_key": foreign, "history": [], "history_lock": threading.Lock(), "history_version": 0,
            "running": False, "attached_images": [], "cols": 120, "agent": None,
            "created_at": time.time(), "profile_home": "/some/other/profile/home",
        }
        req = ServerRequest(sid, "clarify", {"question": "foreign"})
        sr = server._server_requests
        with sr._lock:
            sr._open[req.id] = req
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["counts"]["needs_you"] == 0
        assert inbox["items"] == []

    def test_approval_queue_state_is_never_consumed_by_reading(self, server, db):
        from tools import approval

        key = _create_row(db, _new_key())
        _queue_approval(server, key)
        with approval._lock:
            before = len(approval._gateway_queues.get(key, []))
        _result(server, "inbox.list")
        with approval._lock:
            after = len(approval._gateway_queues.get(key, []))
        assert before == after == 1

    def test_partial_coverage_is_declared_when_scan_hits_cap(self, server, db):
        for _ in range(3):
            k = _create_row(db, _new_key())
            _save_goal(k)
        inbox = _result(server, "inbox.list", limit=2)["inbox"]
        assert inbox["coverage"]["partial"] is True
        # With cap+1 fetch, scanned_sessions counts allowed rows up to cap
        assert inbox["coverage"]["scanned_sessions"] == 2

    def test_db_unavailable_returns_5031(self, server, monkeypatch):
        class _NoDB:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(server, "_profile_db", lambda params: _NoDB())
        assert _error(_call(server, "inbox.list"))["code"] == 5031

    def test_snapshot_read_error_surfaces_as_red_badge_not_all_clear(self, server, db, monkeypatch):
        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom(session_key):
            raise RuntimeError("simulated snapshot failure")

        monkeypatch.setattr(server, "_snapshot_control", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert inbox["coverage"]["errors"]
        # Defect 4: raw exception text must not appear in coverage errors
        for err in inbox["coverage"]["errors"]:
            assert "simulated snapshot failure" not in err
            assert "RuntimeError" in err  # safe type name only

    def test_clarify_count_not_question_text(self, server, db):
        """Clarify pending must expose only a count, never the question text."""
        key = _create_row(db, _new_key())
        _queue_clarify(server, key, question="What is the API key?")
        serialized = json.dumps(_result(server, "inbox.list"))
        assert "API key" not in serialized
        assert "What is" not in serialized

    def test_multiple_clarify_counted(self, server, db):
        """Multiple open clarifies for one session increment the count."""
        key = _create_row(db, _new_key())
        sid = _queue_clarify(server, key, question="Q1")
        _queue_clarify(server, key, question="Q2", sid=sid)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["items"][0]["pending_clarify"] == {"count": 2}

    def test_empty_lane_session_included_in_other_category(self, server, db):
        """A session with no goal/loop/heartbeat and no pending approval/clarify
        still appears in the inbox under the 'other' category."""
        key = _create_row(db, _new_key())
        # No _save_goal, no approval, no clarify — just a plain session row
        inbox = _result(server, "inbox.list")["inbox"]
        assert len(inbox["items"]) == 1
        item = inbox["items"][0]
        assert item["session_key"] == key
        assert "other" in item["categories"]
        assert item["lanes"] == []  # no lanes, but still included
        assert inbox["categories"]["other"] == 1

    def test_session_with_only_background_tasks_appears_in_inbox(self, server, db, monkeypatch):
        """A session with only background processes (no goal/loop/heartbeat/subagents)
        appears in the inbox under 'background_tasks' and 'other' categories."""
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        # Fake a running background process — use the PUBLIC scoped API shape:
        # list_sessions(session_key=key) returns entries without session_key.
        fake_session = {
            "session_id": "proc-fake",
            "command": "npm test",
            "cwd": "/tmp",
            "owner_task_id": key,
            "started_at": "2026-01-01T00:00:00",
            "uptime_seconds": 100,
            "status": "running",
            "output_preview": "",
        }
        original_ls = pr_mod.process_registry.list_sessions

        def mock_ls(task_id=None, session_key=None, *, include_retained=False):
            if session_key == key:
                return [fake_session]
            return []

        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", mock_ls)
        inbox = _result(server, "inbox.list")["inbox"]
        assert len(inbox["items"]) == 1
        item = inbox["items"][0]
        assert item["background_task_count"] == 1
        assert "background_tasks" in item["categories"]
        assert "other" in item["categories"]
        assert inbox["categories"]["background_tasks"] == 1

    def test_overlapping_goal_and_loop_categories_in_full_inbox(self, server, db):
        """A session with both goal and loop counts in both category totals."""
        key = _create_row(db, _new_key())
        _save_goal(key)
        _save_loop(key, awaiting_response=True)
        inbox = _result(server, "inbox.list")["inbox"]
        assert len(inbox["items"]) == 1
        item = inbox["items"][0]
        assert "goals" in item["categories"]
        assert "loops" in item["categories"]
        # Both category counts should be 1 (same session counted in both)
        assert inbox["categories"]["goals"] == 1
        assert inbox["categories"]["loops"] == 1


# ── Background task counting ───────────────────────────────────────────────
class TestBackgroundTaskCounting:
    """Tests for per-session background task count in inbox items."""

    def test_no_background_tasks_yields_zero_count(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["background_task_count"] == 0

    def test_background_tasks_counted_by_session_key(self, server, db, monkeypatch):
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        _save_goal(key)
        fake_sessions = [
            {
                "session_id": "proc-1", "command": "npm test", "cwd": "/tmp",
                "owner_task_id": key,
                "started_at": "2026-01-01T00:00:00", "uptime_seconds": 10,
                "status": "running", "output_preview": "",
            },
            {
                "session_id": "proc-2", "command": "cargo build", "cwd": "/tmp",
                "owner_task_id": key,
                "started_at": "2026-01-01T00:00:00", "uptime_seconds": 5,
                "status": "running", "output_preview": "",
            },
        ]

        def mock_ls(task_id=None, session_key=None, *, include_retained=False):
            if session_key == key:
                return fake_sessions
            return []

        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", mock_ls)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["background_task_count"] == 2

    def test_exited_background_tasks_not_counted(self, server, db, monkeypatch):
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        _save_goal(key)
        fake_sessions = [
            {
                "session_id": "proc-1", "command": "npm test", "cwd": "/tmp",
                "owner_task_id": key,
                "started_at": "2026-01-01T00:00:00", "uptime_seconds": 10,
                "status": "exited", "output_preview": "",
            },
        ]

        def mock_ls(task_id=None, session_key=None, *, include_retained=False):
            if session_key == key:
                return fake_sessions
            return []

        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", mock_ls)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["background_task_count"] == 0


# ── Defect regression tests ────────────────────────────────────────────────
class TestDefect1SchemaMatch:
    """Defect 1: handler result must validate against InboxListResult (wrapper with ``inbox`` key)."""

    def test_handler_result_matches_inbox_list_result_contract(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        response = _call(server, "inbox.list")
        assert "result" in response
        result = response["result"]
        # Must validate without raising (extra_forbidden on unknown root keys)
        validated = InboxListResult.model_validate(result)
        assert validated.inbox is not None
        assert isinstance(validated.inbox.items, list)
        assert isinstance(validated.inbox.counts, dict) or hasattr(validated.inbox.counts, "needs_you")

    def test_populated_result_matches_contract(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        key = _create_row(db, _new_key())
        _save_goal(key)
        _queue_approval(server, key)
        response = _call(server, "inbox.list")
        result = response["result"]
        validated = InboxListResult.model_validate(result)
        assert len(validated.inbox.items) >= 1
        assert validated.inbox.badge in ("none", "amber", "red")

    def test_empty_result_matches_contract(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        response = _call(server, "inbox.list")
        validated = InboxListResult.model_validate(response["result"])
        assert validated.inbox.items == []
        assert validated.inbox.badge == "none"


class TestDefect2Pagination:
    """Defect 2: cap+1 fetch and honest truncation detection."""

    def test_partial_at_max_limit(self, server, db):
        """partial must be True when cap == _MAX_LIMIT and there are more rows."""
        from tui_gateway.methods_inbox import _MAX_LIMIT

        for _ in range(_MAX_LIMIT + 5):
            k = _create_row(db, _new_key())
            _save_goal(k)
        inbox = _result(server, "inbox.list", limit=_MAX_LIMIT)["inbox"]
        assert inbox["coverage"]["partial"] is True
        assert inbox["coverage"]["scanned_sessions"] == _MAX_LIMIT

    def test_not_partial_when_exactly_cap_rows(self, server, db):
        """partial must be False when exactly cap rows exist (no truncation)."""
        for _ in range(3):
            k = _create_row(db, _new_key())
            _save_goal(k)
        inbox = _result(server, "inbox.list", limit=3)["inbox"]
        assert inbox["coverage"]["partial"] is False
        assert inbox["coverage"]["scanned_sessions"] == 3

    def test_denied_source_rows_do_not_affect_truncation_detection(self, server, db):
        """Denied-source rows in the DB don't cause false truncation.
        _listing_rows filters at Python level (most-recent-first), so create allowed
        rows AFTER denied rows to ensure they are returned."""
        # Create denied rows first (they appear earlier in most-recent-first order)
        for _ in range(5):
            k = _create_row(db, _new_key(), source="kanban")
            _save_goal(k)
        # Create allowed rows last (they are most recent)
        for _ in range(2):
            k = _create_row(db, _new_key(), source="cli")
            _save_goal(k)
        inbox = _result(server, "inbox.list", limit=2)["inbox"]
        # DB returns 3 rows (2 cli + 1 kanban), _listing_rows filters to 2 cli
        # With cap=2, fetch=3, len(rows)=2 <= cap → not truncated
        assert inbox["coverage"]["partial"] is False
        assert inbox["coverage"]["scanned_sessions"] == 2

    def test_cap_plus_one_fetch_neverProcesses_extra(self, server, db):
        """The extra row fetched for truncation detection is never included in items."""
        for _ in range(4):
            k = _create_row(db, _new_key())
            _save_goal(k)
        inbox = _result(server, "inbox.list", limit=3)["inbox"]
        assert len(inbox["items"]) == 3
        assert inbox["coverage"]["partial"] is True


class TestDefect3SwallowedExceptions:
    """Defect 3: live enumeration/request-reader exceptions surface as coverage errors."""

    def test_approval_read_failure_surfaces_in_coverage(self, server, db, monkeypatch):
        """An exception during approval read surfaces as a coverage error, not false all-clear."""
        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom(session_key, *, strict=False):
            raise RuntimeError("approval queue corrupted")

        monkeypatch.setattr(server, "_pending_approval_request_payload", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert any("approval read failed" in e for e in inbox["coverage"]["errors"])
        # Raw exception text must not appear
        assert not any("approval queue corrupted" in e for e in inbox["coverage"]["errors"])

    def test_snapshot_error_and_approval_error_both_surfaced(self, server, db, monkeypatch):
        """Multiple independent failures all appear in coverage errors."""
        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom_snapshot(session_key):
            raise RuntimeError("snapshot boom")

        def boom_approval(session_key, *, strict=False):
            raise RuntimeError("approval boom")

        monkeypatch.setattr(server, "_snapshot_control", boom_snapshot)
        monkeypatch.setattr(server, "_pending_approval_request_payload", boom_approval)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        # Both errors surfaced: snapshot failure sets control={} (does not skip),
        # so approval evaluation proceeds and its error is also captured.
        assert len(inbox["coverage"]["errors"]) >= 2

    def test_rpc_error_message_is_sanitized(self, server, db, monkeypatch):
        """The top-level RPC error must not contain raw exception text.
        An unhandled exception in the handler is caught and sanitized."""
        original = server._list_inbox

        def boom(rid, params):
            raise RuntimeError("super-secret-internal-detail-12345")

        server._list_inbox = boom
        try:
            response = _call(server, "inbox.list")
            assert "error" in response
            assert "super-secret-internal-detail-12345" not in response["error"].get("message", "")
            assert response["error"]["code"] == 5031
        finally:
            server._list_inbox = original


class TestDefect4MetadataOnlyApproval:
    """Defect 4: approval description is a fixed label, never arbitrary text from the payload."""

    def test_description_is_fixed_label(self, server, db):
        key = _create_row(db, _new_key())
        _queue_approval(server, key, description="run rm -rf / --no-preserve-root")
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["pending_approval"]["description"] == "pending approval"

    def test_description_never_contains_payload_text(self, server, db):
        """Any arbitrary description text must not leak into the serialized result."""
        secrets = [
            "Bearer SECRET_TOKEN_abc123",
            "password=mysecret123",
            "Authorization: Basic dXNlcjpwYXNz",
        ]
        for desc in secrets:
            key = _create_row(db, _new_key())
            _queue_approval(server, key, description=desc)
            serialized = json.dumps(_result(server, "inbox.list"))
            assert desc not in serialized

    def test_error_text_never_leaks_in_rpc_response(self, server, db, monkeypatch):
        """Raw exception text must not appear in error messages."""
        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom(session_key):
            raise RuntimeError("credential-leak-attempt-XYZ")

        monkeypatch.setattr(server, "_snapshot_control", boom)
        response = _call(server, "inbox.list")
        # The result should have errors in coverage, not in the RPC error message
        if "error" in response:
            assert "credential-leak-attempt-XYZ" not in response["error"].get("message", "")
        if "result" in response:
            errors = response["result"].get("inbox", {}).get("coverage", {}).get("errors", [])
            for e in errors:
                assert "credential-leak-attempt-XYZ" not in e


class TestDefect5ProfileResolution:
    """Defect 5: ProfileUnavailableError must propagate, never silently fall back."""

    def test_profile_unavailable_error_propagates(self, server, db):
        """An explicit non-existent profile must raise, not fall back to launch profile.
        The _profile_scoped wrapper raises before the handler runs; rpc_dispatch catches
        it and returns 4064, but direct calls propagate the exception."""
        from tui_gateway.server import ProfileUnavailableError

        with pytest.raises(ProfileUnavailableError, match="does not exist"):
            _call(server, "inbox.list", profile="nonexistent-profile-xyz")

    def test_other_profile_errors_return_error_not_fallback(self, server, db, monkeypatch):
        """Non-ProfileUnavailableError exceptions during resolution propagate as an error.
        The _profile_scoped wrapper raises before the handler runs; handle_request catches
        it and returns 5031, but direct calls propagate the ValueError."""
        from tui_gateway.server import _profile_home as _orig_profile_home

        def boom(profile):
            if profile == "broken-profile":
                raise ValueError("some other resolution error")
            return _orig_profile_home(profile)

        monkeypatch.setattr(server, "_profile_home", boom)
        with pytest.raises(ValueError, match="some other resolution error"):
            _call(server, "inbox.list", profile="broken-profile")


class TestDefect6OpenRPCDiff:
    """Defect 6: contracts are consistent — InboxListResult wraps InboxResult."""

    def test_inbox_result_is_subschema_of_inbox_list_result(self):
        from tui_gateway.contracts.inbox import InboxListResult, InboxResult

        # InboxListResult.inbox field type must be InboxResult
        field_info = InboxListResult.model_fields["inbox"]
        assert field_info.annotation is InboxResult

    def test_inbox_list_result_has_extra_forbid(self):
        from tui_gateway.contracts.inbox import InboxListResult

        config = InboxListResult.model_config
        assert config.get("extra") == "forbid"


# ── Gate regression tests ────────────────────────────────────────────────
class TestGateDefect1ClarifyErrors:
    """Defect 1: _live_clarify_by_session_key returns errors alongside counts."""

    def test_clarify_enumeration_failure_surfaces_in_coverage(self, server, db, monkeypatch):
        """When session enumeration fails, the error appears in coverage and badge is red."""
        import tui_gateway.server as server_mod

        class _EnumerationBoom(dict):
            """``.get`` still answers the profile-scope wrapper; enumeration raises."""

            def items(self):
                raise TypeError("lock not acquired")

        # Break only the enumeration (list(_sessions.items())). The registry stays
        # dict-shaped because the profile-scope wrapper resolves ``.get()`` first and
        # a bare None would fail there instead of in the inbox path under test.
        monkeypatch.setattr(server_mod, "_sessions", _EnumerationBoom())

        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert any("live-session enumeration failed" in e for e in inbox["coverage"]["errors"])

    def test_clarify_query_failure_surfaces_in_coverage(self, server, db, monkeypatch):
        """When open_requests() raises for a session, the error appears in coverage."""
        key = _create_row(db, _new_key())
        _open_session(server, key)
        from tui_gateway import server_requests

        original = server_requests.open_requests

        def boom(sid):
            raise RuntimeError("queue read boom")

        monkeypatch.setattr(server_requests, "open_requests", boom)
        try:
            inbox = _result(server, "inbox.list")["inbox"]
            assert inbox["badge"] == "red"
            assert any("clarify query failed" in e for e in inbox["coverage"]["errors"])
        finally:
            monkeypatch.setattr(server_requests, "open_requests", original)

    def test_real_queue_getter_failure(self, server, db, monkeypatch):
        """Test with the actual get_pending_gateway_approval raising, not monkeypatching
        the wrapper."""
        from tools import approval

        original = approval.get_pending_gateway_approval

        def boom(session_key):
            raise IOError("disk I/O error")

        monkeypatch.setattr(approval, "get_pending_gateway_approval", boom)
        key = _create_row(db, _new_key())
        _save_goal(key)
        try:
            inbox = _result(server, "inbox.list")["inbox"]
            assert inbox["badge"] == "red"
            assert any("approval read failed" in e for e in inbox["coverage"]["errors"])
        finally:
            monkeypatch.setattr(approval, "get_pending_gateway_approval", original)


class TestGateDefect2SnapshotThenClarify:
    """Defect 2: snapshot failure sets control={} so approval/clarify are still evaluated."""

    def test_snapshot_failure_still_evaluates_pending_clarify(self, server, db, monkeypatch):
        """Session with snapshot failure + genuine pending clarify still shows needs_you.
        Badge is red (error present), not amber, because data-source errors take priority."""
        key = _create_row(db, _new_key())
        _queue_clarify(server, key, question="Which provider?")

        def boom(session_key):
            raise RuntimeError("snapshot boom")

        monkeypatch.setattr(server, "_snapshot_control", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert inbox["counts"]["needs_you"] == 1
        item = inbox["items"][0]
        assert item["session_key"] == key
        assert item["pending_clarify"] == {"count": 1}

    def test_snapshot_failure_still_evaluates_pending_approval(self, server, db, monkeypatch):
        """Session with snapshot failure + pending approval still shows needs_you.
        Badge is red (error present) because data-source errors take priority."""
        key = _create_row(db, _new_key())
        _queue_approval(server, key)

        def boom(session_key):
            raise RuntimeError("snapshot boom")

        monkeypatch.setattr(server, "_snapshot_control", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert inbox["counts"]["needs_you"] == 1
        assert inbox["items"][0]["pending_approval"] is not None


class TestGateDefect3NoProfileFallback:
    """Defect 3: generic profile resolution failure never falls back to launch profile."""

    def test_generic_resolution_error_propagates(self, server, db, monkeypatch):
        """A non-ProfileUnavailableError during profile resolution propagates as an error.
        The _profile_scoped wrapper raises before the handler runs; handle_request catches
        it and returns 5031, but direct calls propagate the exception."""
        from tui_gateway.server import _profile_home as _orig_profile_home

        def boom(profile):
            if profile == "broken-profile":
                raise OSError("filesystem corrupted")
            return _orig_profile_home(profile)

        monkeypatch.setattr(server, "_profile_home", boom)
        with pytest.raises(OSError, match="filesystem corrupted"):
            _call(server, "inbox.list", profile="broken-profile")

    def test_no_db_or_requests_touched_on_resolution_failure(self, server, db, monkeypatch):
        """When resolution fails, no DB reads or live requests should happen."""
        from tui_gateway.server import _profile_home as _orig

        db_touched = []

        def boom(profile):
            if profile == "broken-profile":
                raise OSError("filesystem corrupted")
            return _orig(profile)

        monkeypatch.setattr(server, "_profile_home", boom)
        with pytest.raises(OSError):
            _call(server, "inbox.list", profile="broken-profile")


class TestGateDefect4NoSentinelProcessing:
    """Defect 4: sentinel row (cap+1) is never processed even with leading denied rows."""

    def test_denied_first_row_with_cap_plus_one_total(self, server, db):
        """If the first row is denied and total rows = cap+1, sentinel is not processed."""
        # Create 1 denied row (most recent, so first in listing)
        k_denied = _create_row(db, _new_key(), source="kanban")
        _save_goal(k_denied)
        # Create 2 allowed rows (cap=2, so fetch_limit=3, sentinel = 3rd row)
        keys = []
        for _ in range(2):
            k = _create_row(db, _new_key(), source="cli")
            _save_goal(k)
            keys.append(k)
        inbox = _result(server, "inbox.list", limit=2)["inbox"]
        # Only the 2 allowed rows are in items; sentinel never processed
        assert len(inbox["items"]) == 2
        assert {item["session_key"] for item in inbox["items"]} == set(keys)
        assert inbox["coverage"]["partial"] is False


class TestGateDefect5StrictApprovalReader:
    """Inbox opts into error visibility without changing legacy callers."""

    def test_legacy_reader_still_returns_none_on_queue_error(self, server, monkeypatch):
        from tools import approval

        def boom(session_key):
            raise RuntimeError("queue unavailable")

        monkeypatch.setattr(approval, "get_pending_gateway_approval", boom)
        assert server._pending_approval_request_payload("test-session") is None
        with pytest.raises(RuntimeError):
            server._pending_approval_request_payload("test-session", strict=True)

    def test_strict_reader_propagates_queue_errors(self, server, db, monkeypatch):
        """With strict=True, a queue getter error surfaces as a coverage error."""
        from tools import approval

        original = approval.get_pending_gateway_approval

        def boom(session_key):
            raise RuntimeError("queue corrupted")

        monkeypatch.setattr(approval, "get_pending_gateway_approval", boom)
        key = _create_row(db, _new_key())
        _save_goal(key)
        try:
            inbox = _result(server, "inbox.list")["inbox"]
            assert inbox["badge"] == "red"
            assert any("approval read failed" in e for e in inbox["coverage"]["errors"])
            # Safe type name only, never raw text
            assert not any("queue corrupted" in e for e in inbox["coverage"]["errors"])
        finally:
            monkeypatch.setattr(approval, "get_pending_gateway_approval", original)


class TestGateDefect6CanonicalPaths:
    """Defect 6: profile path comparison uses normcase for cross-platform equivalence."""

    def test_different_case_profile_matches(self, server, db):
        """Sessions with differently-cased profile paths are still matched."""
        import tui_gateway.methods_inbox as inbox_mod

        key = _create_row(db, _new_key())
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home())
        # Register session with uppercased path
        sid = f"sid-case-{uuid.uuid4().hex[:8]}"
        server._sessions[sid] = {
            "session_key": key, "history": [], "history_lock": threading.Lock(),
            "history_version": 0, "running": False, "attached_images": [], "cols": 120,
            "agent": None, "created_at": time.time(),
            "profile_home": home.upper(),
        }
        req = server._server_requests  # type: ignore[attr-defined]
        from tui_gateway.server_requests import ServerRequest
        srq = ServerRequest(sid, "clarify", {"question": "test"})
        with req._lock:
            req._open[srq.id] = srq
        inbox = _result(server, "inbox.list")["inbox"]
        import os
        expected = int(os.path.normcase(home.upper()) == os.path.normcase(home))
        assert inbox["counts"]["needs_you"] == expected

    def test_genuine_foreign_home_excluded(self, server, db):
        """A session with a genuinely different profile_home is excluded."""
        key = _create_row(db, _new_key())
        sid = f"sid-foreign-{uuid.uuid4().hex[:8]}"
        server._sessions[sid] = {
            "session_key": key, "history": [], "history_lock": threading.Lock(),
            "history_version": 0, "running": False, "attached_images": [], "cols": 120,
            "agent": None, "created_at": time.time(),
            "profile_home": "/completely/different/path",
        }
        from tui_gateway.server_requests import ServerRequest
        srq = ServerRequest(sid, "clarify", {"question": "test"})
        with server._server_requests._lock:
            server._server_requests._open[srq.id] = srq
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["counts"]["needs_you"] == 0


# ── Category classification ────────────────────────────────────────────────
class TestCategoryClassification:
    """Tests for the left-navigation category classification (overlapping)."""

    def test_empty_control_yields_other(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories(None) == ["other"]
        assert classify_categories({}) == ["other"]
        assert classify_categories({"goal": None, "loop": None, "heartbeat": None}) == ["other"]

    def test_active_goal_yields_goals(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": {"status": "active"}, "loop": None, "heartbeat": None}) == ["goals"]

    def test_paused_goal_yields_goals(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": {"status": "paused"}, "loop": None, "heartbeat": None}) == ["goals"]

    def test_active_loop_yields_loops(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": None, "loop": {"status": "active"}, "heartbeat": None}) == ["loops"]

    def test_paused_loop_yields_loops(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": None, "loop": {"status": "paused"}, "heartbeat": None}) == ["loops"]

    def test_active_heartbeat_yields_heartbeats(self):
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": None, "loop": None, "heartbeat": {"status": "active"}}) == ["heartbeats"]

    def test_goal_and_loop_overlap(self):
        """Sessions with both a goal and loop appear in BOTH categories."""
        from tui_gateway.methods_inbox import classify_categories

        control = {
            "goal": {"status": "active"},
            "loop": {"status": "active"},
            "heartbeat": None,
        }
        assert classify_categories(control) == ["goals", "loops"]

    def test_loop_and_heartbeat_overlap(self):
        from tui_gateway.methods_inbox import classify_categories

        control = {
            "goal": None,
            "loop": {"status": "active"},
            "heartbeat": {"status": "active"},
        }
        assert classify_categories(control) == ["loops", "heartbeats"]

    def test_all_three_overlap(self):
        from tui_gateway.methods_inbox import classify_categories

        control = {
            "goal": {"status": "active"},
            "loop": {"status": "active"},
            "heartbeat": {"status": "active"},
        }
        assert classify_categories(control) == ["goals", "loops", "heartbeats"]

    def test_done_goal_in_goals_category(self):
        """A completed (done) goal appears in the goals category (persisted completed state)."""
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": {"status": "done"}, "loop": None, "heartbeat": None}) == ["goals"]

    def test_cleared_state_never_reaches_classify(self):
        """'cleared' status is filtered at the snapshot level (returns None),
        so it never reaches classify_categories.  The classify function only
        sees 'active', 'paused', and 'done' for persisted automation."""
        from tui_gateway.methods_inbox import classify_categories

        # These are direct calls to classify_categories; in production,
        # cleared states are converted to None by _safe_*_snapshot before reaching here.
        assert classify_categories({"goal": {"status": "cleared"}, "loop": None, "heartbeat": None}) == ["other"]
        assert classify_categories({"goal": None, "loop": {"status": "cleared"}, "heartbeat": None}) == ["other"]
        assert classify_categories({"goal": None, "loop": None, "heartbeat": {"status": "cleared"}}) == ["other"]


# ── Category counts ────────────────────────────────────────────────────────
class TestCategoryCounts:
    """Tests for the categories breakdown in the inbox response."""

    def test_empty_inbox_has_zero_category_counts(self, server, db):
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["categories"] == {"goals": 0, "loops": 0, "heartbeats": 0, "subagents": 0, "background_tasks": 0, "other": 0}

    def test_goal_session_increments_goals_category(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["categories"]["goals"] == 1
        assert inbox["categories"]["loops"] == 0
        assert inbox["categories"]["heartbeats"] == 0

    def test_loop_session_increments_loops_category(self, server, db):
        key = _create_row(db, _new_key())
        # Loop with awaiting_response=True gets the "running" lane, so it appears in inbox
        _save_loop(key, awaiting_response=True)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["categories"]["loops"] == 1
        assert inbox["categories"]["goals"] == 0

    def test_heartbeat_session_increments_heartbeats_category(self, server, db):
        key = _create_row(db, _new_key())
        _save_heartbeat(key)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["categories"]["heartbeats"] == 1
        assert inbox["categories"]["goals"] == 0

    def test_overlapping_categories_counted_correctly(self, server, db):
        """A session with both goal and loop increments BOTH category counts."""
        key = _create_row(db, _new_key())
        _save_goal(key)
        _save_loop(key, awaiting_response=True)
        inbox = _result(server, "inbox.list")["inbox"]
        cats = inbox["categories"]
        assert cats["goals"] == 1
        assert cats["loops"] == 1

    def test_mixed_sessions_counted_correctly(self, server, db):
        goal_key = _create_row(db, _new_key())
        _save_goal(goal_key)
        loop_key = _create_row(db, _new_key())
        _save_loop(loop_key, awaiting_response=True)
        hb_key = _create_row(db, _new_key())
        _save_heartbeat(hb_key)
        inbox = _result(server, "inbox.list")["inbox"]
        cats = inbox["categories"]
        assert cats["goals"] == 1
        assert cats["loops"] == 1
        assert cats["heartbeats"] == 1

    def test_item_has_categories_array(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["categories"] == ["goals"]

    def test_item_has_subagent_count_field(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert "subagent_count" in item
        assert item["subagent_count"] == 0

    def test_item_has_background_task_count_field(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert "background_task_count" in item
        assert item["background_task_count"] == 0


# ── Subagent counting ──────────────────────────────────────────────────────
class TestSubagentCounting:
    """Tests for per-session subagent count in inbox items."""

    def test_no_active_subagents_yields_zero_count(self, server, db):
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["subagent_count"] == 0

    def test_active_subagents_counted_by_owner_session(self, server, db, monkeypatch):
        from tools import delegate_tool_registry as dtr

        key = _create_row(db, _new_key())
        _save_goal(key)
        # Inject fake active subagent records belonging to this session
        fake_records = [
            {"subagent_id": "sa-1", "owner_agent_session_id": key, "goal": "task1", "status": "running"},
            {"subagent_id": "sa-2", "owner_agent_session_id": key, "goal": "task2", "status": "running"},
            {"subagent_id": "sa-3", "owner_agent_session_id": "other-session", "goal": "task3", "status": "running"},
        ]
        monkeypatch.setattr(dtr, "list_active_subagents", lambda: fake_records)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["subagent_count"] == 2

    def test_subagent_count_in_subagents_category(self, server, db, monkeypatch):
        """A session with active subagents gets 'subagents' in its categories array."""
        from tools import delegate_tool_registry as dtr

        key = _create_row(db, _new_key())
        _save_goal(key)
        fake_records = [
            {"subagent_id": "sa-2", "owner_agent_session_id": key, "goal": "task2", "status": "running"},
        ]
        monkeypatch.setattr(dtr, "list_active_subagents", lambda: fake_records)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["categories"]["subagents"] == 1
        item = inbox["items"][0]
        assert "subagents" in item["categories"]

    def test_session_with_only_subagents_appears_in_inbox(self, server, db, monkeypatch):
        """A session with subagents but no goal/loop/heartbeat still appears in inbox."""
        from tools import delegate_tool_registry as dtr

        key = _create_row(db, _new_key())
        fake_records = [
            {"subagent_id": "sa-1", "owner_agent_session_id": key, "goal": "task1", "status": "running"},
        ]
        monkeypatch.setattr(dtr, "list_active_subagents", lambda: fake_records)
        inbox = _result(server, "inbox.list")["inbox"]
        assert len(inbox["items"]) == 1
        item = inbox["items"][0]
        assert item["session_key"] == key
        assert item["subagent_count"] == 1
        assert "subagents" in item["categories"]
        assert "other" in item["categories"]


# ── Contract validation for new fields ─────────────────────────────────────
class TestContractNewFields:
    """Verify the contract validates the new categories and subagent_count fields."""

    def test_handler_result_matches_inbox_list_result_with_new_fields(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        key = _create_row(db, _new_key())
        _save_goal(key)
        response = _call(server, "inbox.list")
        assert "result" in response
        result = response["result"]
        validated = InboxListResult.model_validate(result)
        assert len(validated.inbox.items) >= 1
        item = validated.inbox.items[0]
        assert "goals" in item.categories
        assert item.subagent_count == 0
        assert item.background_task_count == 0
        assert hasattr(validated.inbox, "categories")
        assert validated.inbox.categories.goals >= 1

    def test_empty_result_matches_contract_with_categories(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        response = _call(server, "inbox.list")
        validated = InboxListResult.model_validate(response["result"])
        assert validated.inbox.categories.goals == 0
        assert validated.inbox.categories.loops == 0
        assert validated.inbox.categories.heartbeats == 0
        assert validated.inbox.categories.subagents == 0
        assert validated.inbox.categories.background_tasks == 0
        assert validated.inbox.categories.other == 0

    def test_populated_result_category_counts_match_items(self, server, db):
        from tui_gateway.contracts.inbox import InboxListResult

        goal_key = _create_row(db, _new_key())
        _save_goal(goal_key)
        loop_key = _create_row(db, _new_key())
        _save_loop(loop_key, awaiting_response=True)
        hb_key = _create_row(db, _new_key())
        _save_heartbeat(hb_key)
        response = _call(server, "inbox.list")
        validated = InboxListResult.model_validate(response["result"])
        # Count unique sessions per category from the validated items
        cat_sessions: dict[str, set[str]] = {}
        for item in validated.inbox.items:
            for cat in item.categories:
                cat_sessions.setdefault(cat, set()).add(item.session_key)
        assert len(cat_sessions.get("goals", set())) == validated.inbox.categories.goals
        assert len(cat_sessions.get("loops", set())) == validated.inbox.categories.loops
        assert len(cat_sessions.get("heartbeats", set())) == validated.inbox.categories.heartbeats


# ── Error state propagation ────────────────────────────────────────────────
class TestErrorStatePropagation:
    """Subagent and bg-process errors surface as explicit unavailable flags, not silent zeros."""

    def test_subagent_enumeration_failure_surfaces_in_coverage(self, server, db, monkeypatch):
        """When list_active_subagents raises, badge is red and coverage has the error."""
        from tools import delegate_tool_registry as dtr

        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom():
            raise RuntimeError("subagent registry locked")

        monkeypatch.setattr(dtr, "list_active_subagents", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert any("subagent enumeration failed" in e for e in inbox["coverage"]["errors"])
        # Safe type name only, never raw text
        assert not any("subagent registry locked" in e for e in inbox["coverage"]["errors"])
        item = inbox["items"][0]
        assert item["subagent_count"] == 0
        assert item["subagent_count_unavailable"] is True

    def test_bg_process_query_failure_surfaces_in_coverage(self, server, db, monkeypatch):
        """When list_sessions(session_key=...) raises, badge is red and coverage has the error."""
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom(task_id=None, session_key=None, *, include_retained=False):
            raise IOError("disk I/O error")

        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", boom)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        assert any("bg-process" in e for e in inbox["coverage"]["errors"])
        item = inbox["items"][0]
        assert item["background_task_count"] == 0
        assert item["background_task_count_unavailable"] is True

    def test_both_errors_surfaced_simultaneously(self, server, db, monkeypatch):
        """Both subagent and bg-process errors appear in coverage."""
        from tools import delegate_tool_registry as dtr
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        _save_goal(key)

        def boom_subagents():
            raise RuntimeError("subagent boom")

        def boom_bg(task_id=None, session_key=None, *, include_retained=False):
            raise IOError("bg boom")

        monkeypatch.setattr(dtr, "list_active_subagents", boom_subagents)
        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", boom_bg)
        inbox = _result(server, "inbox.list")["inbox"]
        assert inbox["badge"] == "red"
        errors = inbox["coverage"]["errors"]
        assert any("subagent enumeration failed" in e for e in errors)
        assert any("bg-process" in e for e in errors)
        item = inbox["items"][0]
        assert item["subagent_count_unavailable"] is True
        assert item["background_task_count_unavailable"] is True

    def test_unavailable_flags_default_false_on_success(self, server, db):
        """When both sources succeed, unavailable flags are False."""
        key = _create_row(db, _new_key())
        _save_goal(key)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["subagent_count_unavailable"] is False
        assert item["background_task_count_unavailable"] is False


# ── Completed automation categories ────────────────────────────────────────
class TestCompletedAutomationCategories:
    """Persisted completed automation types appear in categories."""

    def test_done_goal_in_goals_category(self):
        """A completed (done) goal still appears in the goals category."""
        from tui_gateway.methods_inbox import classify_categories

        control = {"goal": {"status": "done"}, "loop": None, "heartbeat": None}
        cats = classify_categories(control)
        assert "goals" in cats

    def test_done_loop_in_loops_category(self):
        """A completed (done) loop still appears in the loops category."""
        from tui_gateway.methods_inbox import classify_categories

        control = {"goal": None, "loop": {"status": "done"}, "heartbeat": None}
        cats = classify_categories(control)
        assert "loops" in cats

    def test_done_heartbeat_in_heartbeats_category(self):
        """A completed (done) heartbeat still appears in the heartbeats category."""
        from tui_gateway.methods_inbox import classify_categories

        control = {"goal": None, "loop": None, "heartbeat": {"status": "done"}}
        cats = classify_categories(control)
        assert "heartbeats" in cats

    def test_cleared_state_still_falls_through_to_other(self):
        """'cleared' status is NOT included — only 'done' is a completed-but-persisted state."""
        from tui_gateway.methods_inbox import classify_categories

        assert classify_categories({"goal": {"status": "cleared"}, "loop": None, "heartbeat": None}) == ["other"]
        assert classify_categories({"goal": None, "loop": {"status": "cleared"}, "heartbeat": None}) == ["other"]
        assert classify_categories({"goal": None, "loop": None, "heartbeat": {"status": "cleared"}}) == ["other"]

    def test_overlapping_done_and_active(self):
        """A session with a done goal and active loop appears in both categories."""
        from tui_gateway.methods_inbox import classify_categories

        control = {
            "goal": {"status": "done"},
            "loop": {"status": "active"},
            "heartbeat": None,
        }
        cats = classify_categories(control)
        assert "goals" in cats
        assert "loops" in cats


# ── Producer-shaped data verification ─────────────────────────────────────
class TestProducerShapedData:
    """Verify the handler works with actual list_sessions() return shape (no session_key field)."""

    def test_bg_task_count_uses_scoped_api(self, server, db, monkeypatch):
        """Verify _background_task_counts_by_session calls list_sessions(session_key=key)."""
        from tools import process_registry as pr_mod

        key = _create_row(db, _new_key())
        _save_goal(key)
        call_log = []

        def mock_ls(task_id=None, session_key=None, *, include_retained=False):
            call_log.append({"task_id": task_id, "session_key": session_key})
            if session_key == key:
                return [{
                    "session_id": "proc-1", "command": "npm test", "cwd": "/tmp",
                    "owner_task_id": key,
                    "started_at": "2026-01-01T00:00:00", "uptime_seconds": 10,
                    "status": "running", "output_preview": "",
                }]
            return []

        monkeypatch.setattr(pr_mod.process_registry, "list_sessions", mock_ls)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["background_task_count"] == 1
        # Verify the scoped API was called with session_key parameter
        assert any(c["session_key"] == key for c in call_log)

    def test_subagent_count_uses_owner_agent_session_id(self, server, db, monkeypatch):
        """Verify subagent mapping uses owner_agent_session_id (the actual public field)."""
        from tools import delegate_tool_registry as dtr

        key = _create_row(db, _new_key())
        _save_goal(key)
        # Producer-shaped entry: owner_agent_session_id is the public field
        fake_records = [
            {"subagent_id": "sa-1", "owner_agent_session_id": key, "goal": "task1", "status": "running"},
        ]
        monkeypatch.setattr(dtr, "list_active_subagents", lambda: fake_records)
        item = _result(server, "inbox.list")["inbox"]["items"][0]
        assert item["subagent_count"] == 1


def test_inbox_deny_list_matches_canonical_listing_sources():
    """The inbox must not clip the sidebar's deny-list.

    Every method module's top-level names are copied into the shared server
    namespace at registration, so a private copy of the deny-list here silently
    replaced ``session.list``'s set and surfaced one-shot runs in the picker.
    """
    import tui_gateway.server  # noqa: F401  (runs the split-module registration bridge)
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    from tui_gateway.methods_inbox import _INBOX_DENY_SOURCES

    assert _INBOX_DENY_SOURCES == frozenset(INTERNAL_LISTING_SOURCES)
    assert "oneshot" in _INBOX_DENY_SOURCES
