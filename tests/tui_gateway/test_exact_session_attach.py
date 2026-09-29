"""Exact attachment is observation, never authorization to execute old work."""

import concurrent.futures
import json
import time
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from tui_gateway import server, server_requests
from tui_gateway.turn_marker import _marker_path, record_turn_start


@pytest.fixture
def store(tmp_path, monkeypatch):
    from hermes_cli import profiles

    home = tmp_path / "default"
    home.mkdir()
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    db = SessionDB(db_path=home / "state.db")
    other = SessionDB(db_path=secondary / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    for database, text in ((db, "default transcript"), (other, "secondary transcript")):
        database.create_session(session_id="exact-parent", source="tui", model="test")
        database.append_message("exact-parent", "user", text)
        database.append_message("exact-parent", "assistant", "recorded answer")
        database.set_session_title("exact-parent", "human alias")
    yield db, other, home, secondary
    db.close()
    other.close()


def rpc(method, **params):
    return server.handle_request({"id": "test", "method": method, "params": params})


def attach(profile="default", target="exact-parent"):
    return rpc("session.attach", session_id=target, profile=profile)


def inert_spies(monkeypatch, db):
    names = (
        "_schedule_agent_build",
        "_start_agent_build",
        "_schedule_resume_hydration",
        "_maybe_schedule_auto_continue",
        "_enable_gateway_prompts",
        "_start_session_services",
        "_run_prompt_submit",
        "_drain_queued_prompt",
        "_make_agent",
        "_schedule_session_cap_enforcement",
    )
    spies = []
    for obj, name in [(server, n) for n in names] + [
        (db, "reopen_session"),
        (db, "get_resume_conversations"),
        (db, "get_messages_as_conversation"),
    ]:
        spy = Mock(side_effect=AssertionError(f"attachment invoked {name}"))
        monkeypatch.setattr(obj, name, spy)
        spies.append(spy)
    return spies


@pytest.mark.parametrize("marker", [True, False])
def test_cold_attach_is_inert_exact_and_repeatable(store, monkeypatch, marker):
    db, _, home, _ = store
    db.end_session("exact-parent", "compression")
    db.create_session(
        session_id="descendant",
        source="tui",
        model="test",
        parent_session_id="exact-parent",
    )
    before = db.get_session("exact-parent")
    if marker:
        record_turn_start(home, "exact-parent", "old private work")
    path = _marker_path(home)
    marker_before = path.read_bytes() if path.exists() else None
    spies = inert_spies(monkeypatch, db)
    first = attach()["result"]
    second = attach()["result"]
    assert first["stored_session_id"] == first["requested_session_id"] == "exact-parent"
    assert first["profile"] == "default"
    # The writer of a just-recorded marker is still alive. Its presence does not prove a crash.
    assert first["disposition"] == "unknown"
    assert first["fence_reason"] == (
        "marker_writer_alive" if marker else "no_terminal_receipt"
    )
    assert first["execution_fenced"] and not first["can_submit_prompt"]
    assert not first["recovery"]["complete"]
    assert first["recovery"]["request_lifecycle"] == "unknown"
    assert first["recovery"]["replay_high_water"] is None
    assert second == {**first, "reused_runtime": True}
    assert len(server._sessions) == 1
    assert db.get_session("exact-parent") == before
    assert (path.read_bytes() if path.exists() else None) == marker_before
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize(
    "target", ["human alias", "exact", " exact-parent", "exact-parent ", "missing"]
)
def test_no_alias_prefix_whitespace_or_missing_substitution(store, target):
    assert "error" in attach(target=target)
    assert server._sessions == {}


def test_profile_identity_and_same_id_isolation_a_b_a(store):
    a = attach()["result"]
    b = attach("secondary")["result"]
    again = attach()["result"]
    assert a["session_id"] != b["session_id"]
    assert again["session_id"] == a["session_id"]
    assert b["profile"] == "secondary"
    assert server._sessions[b["session_id"]]["profile_home"] == str(store[3])
    assert "error" in attach("missing")
    assert "error" in attach(".hermes")
    assert "error" in attach("Secondary")
    assert len(server._sessions) == 2


@pytest.mark.parametrize(
    "params",
    [
        {"session_id": "exact-parent"},
        {"session_id": 123, "profile": "default"},
        {"session_id": "exact-parent", "profile": "default", "lazy": True},
        {"session_id": "exact-parent", "profile": "default", "defer_history": True},
    ],
)
def test_wire_rejects_unsafe_modes_and_malformed_identity(store, params):
    assert rpc("session.attach", **params)["error"]["code"] == 4000
    assert not server._sessions


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        '{"exact-parent": null}',
        '{"exact-parent": {}, "exact-parent": {}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":NaN}}',
        '{"exact-parent": {"prompt":"x","attempts":true,"started_at":1}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"notification_category":{}}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":true}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":0}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":"42"}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_start_time":1}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":42,"writer_start_time":null}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":42,"writer_start_time":NaN}}',
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":'
        + "9" * 400
        + "}}",
        '{"exact-parent": {"prompt":"x","attempts":0,"started_at":1,"writer_pid":42,"writer_start_time":'
        + "9" * 400
        + "}}",
    ],
)
def test_malformed_recovery_remains_unknown_and_fenced(store, raw):
    path = _marker_path(store[2])
    path.parent.mkdir(exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    result = attach()["result"]
    assert result["disposition"] == "unknown"
    assert result["fence_reason"] == "invalid_recovery_state"
    assert result["execution_fenced"]
    assert path.read_text(encoding="utf-8-sig") == raw


def test_dead_writer_marker_is_interrupted_but_stays_fenced(store):
    home = store[2]
    record_turn_start(home, "exact-parent", "prior work")
    path = _marker_path(home)
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    # A recycled PID cannot identify its former writer: the start time proves it is gone.
    entries["exact-parent"]["writer_start_time"] = time.time() - 86400
    path.write_text(json.dumps(entries), encoding="utf-8")
    before = path.read_bytes()
    result = attach()["result"]
    assert result["disposition"] == "interrupted"
    assert result["fence_reason"] == "interrupted_turn"
    assert result["execution_fenced"] and not result["can_submit_prompt"]
    assert path.read_bytes() == before


def test_legacy_marker_without_writer_identity_remains_unknown(store):
    home = store[2]
    record_turn_start(home, "exact-parent", "prior work")
    path = _marker_path(home)
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    entries["exact-parent"].pop("writer_pid")
    entries["exact-parent"].pop("writer_start_time", None)
    path.write_text(json.dumps(entries), encoding="utf-8")
    result = attach()["result"]
    assert result["disposition"] == "unknown"
    assert result["fence_reason"] == "writer_liveness_unknown"
    assert result["execution_fenced"]


def test_current_writer_maximum_entry_count_is_valid(store):
    home = store[2]
    record_turn_start(home, "exact-parent", "prior work")
    path = _marker_path(home)
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    marker = entries["exact-parent"]
    for index in range(32):
        entries[f"other-{index}"] = dict(marker)
    path.write_text(json.dumps(entries), encoding="utf-8")
    result = attach()["result"]
    assert result["disposition"] == "unknown"
    assert result["fence_reason"] == "marker_writer_alive"
    assert result["execution_fenced"]


def test_bom_prefixed_current_marker_is_read_strictly(store):
    home = store[2]
    record_turn_start(home, "exact-parent", "prior work")
    path = _marker_path(home)
    raw = path.read_text(encoding="utf-8-sig")
    path.write_text("\ufeff" + raw, encoding="utf-8")
    result = attach()["result"]
    assert result["disposition"] == "unknown"
    assert result["fence_reason"] == "marker_writer_alive"
    assert result["execution_fenced"]


def test_later_prompt_and_every_synthetic_execution_path_are_fenced(store, monkeypatch):
    db, _, home, _ = store
    record_turn_start(home, "exact-parent", "interrupted tool work")
    result = attach()["result"]
    sid = result["session_id"]
    session = server._sessions[sid]
    queue = {"text": "previously queued work", "transport": None}
    pending_switch = {"raw": "anthropic/claude-sonnet-4.6"}
    session.update(
        queued_prompt=queue,
        queued_prompts=[{"text": "another old prompt"}],
        pending_model_switch=pending_switch,
    )
    row_before, marker_before = (
        db.get_session("exact-parent"),
        _marker_path(home).read_bytes(),
    )
    forbidden = Mock(side_effect=AssertionError("execution reached"))
    monkeypatch.setattr(server, "_make_agent", forbidden)
    monkeypatch.setattr(server, "_ensure_active_session_slot", forbidden)
    monkeypatch.setattr(server, "_ensure_session_db_row", forbidden)
    monkeypatch.setattr(server, "_emit", forbidden)
    monkeypatch.setattr(server, "_tts_stream_stop", forbidden)
    monkeypatch.setattr(server, "_resume_wake_after_interrupt", forbidden)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: True)
    monkeypatch.setattr(server, "_interrupt_session_turn", forbidden)
    assert rpc("session.interrupt", session_id=sid)["error"]["code"] == 4091
    refused = rpc("prompt.submit", session_id=sid, text="unrelated new question")
    assert refused["error"]["code"] == 4091
    assert refused["error"]["data"]["reason"] == "attachment_execution_fenced"
    server._start_agent_build(sid, session)
    assert server._sess({"session_id": sid}, "test")[1]["error"]["code"] == 4091
    assert server._maybe_schedule_auto_continue(sid, session, "exact-parent") is None
    assert not server._drain_queued_prompt("old", sid, session)
    assert not server._run_prompt_submit(
        "old", sid, session, "synthesized goal/notification"
    )
    assert (
        server._submit_prompt_to_compute_host("old", sid, session, "isolated old work")[
            "error"
        ]["code"]
        == 4091
    )
    server._notif_submit("old", sid, session, "old notification", "test")
    assert not server._notif_claim_turn(session)
    assert rpc("session.resume", session_id="exact-parent")["error"]["code"] == 4091
    assert rpc("session.activate", session_id=sid)["error"]["code"] == 4091
    with server._session_turn_admission(session) as admitted:
        assert not admitted
    assert session["queued_prompt"] is queue
    assert len(session["queued_prompts"]) == 1
    assert session["pending_model_switch"] is pending_switch
    assert not session["running"]
    assert session["agent"] is None
    assert not session.get("agent_build_started")
    assert db.get_session("exact-parent") == row_before
    assert _marker_path(home).read_bytes() == marker_before
    forbidden.assert_not_called()


def test_close_inert_attachment_does_not_end_repair_or_retire_prior_work(
    store, monkeypatch
):
    db, _, home, _ = store
    record_turn_start(home, "exact-parent", "old work")
    sid = attach()["result"]["session_id"]
    row = db.get_session("exact-parent")
    raw = _marker_path(home).read_bytes()
    hook = Mock()
    monkeypatch.setattr(server, "_notify_session_boundary", hook)
    monkeypatch.setattr("tools.approval.unregister_gateway_notify", hook)
    assert rpc("session.close", session_id=sid)["result"]["closed"]
    assert db.get_session("exact-parent") == row
    assert _marker_path(home).read_bytes() == raw
    hook.assert_not_called()


def live_record(store, **fields):
    record = server._deferred_session_record(
        "exact-parent", cols=80, cwd=str(store[2]), history=[], lease=None
    )
    record.update(fields)
    server._sessions["surviving-runtime"] = record
    return record


@pytest.mark.parametrize(
    "method,params",
    [
        ("clarify", {"questions": [{"qid": "q1", "question": "Choose?"}]}),
        ("approval", {"request_id": "approval-one", "choices": ["once", "deny"]}),
    ],
)
def test_surviving_runtime_queue_and_pending_request_untouched(
    store, monkeypatch, method, params
):
    session = live_record(store, running=True, queued_prompt={"text": "old queue"})
    monkeypatch.setattr(server_requests, "_open", {})
    monkeypatch.setattr(server_requests, "_write", lambda _: None)
    monkeypatch.setattr(server_requests, "_answerable", lambda _: True)
    callback = Mock()
    settle = server_requests.send_async(method, "surviving-runtime", params, callback)
    request = next(iter(server_requests._open.values()))
    request_before = request.snapshot()
    spies = inert_spies(monkeypatch, store[0])
    result = attach()["result"]
    assert result["session_id"] == "surviving-runtime"
    assert result["disposition"] == "live"
    assert not result["execution_fenced"]
    assert session["running"] and session["queued_prompt"]["text"] == "old queue"
    assert request.snapshot() == request_before
    assert not request.answered and not request.event.is_set()
    callback.assert_not_called()
    for spy in spies:
        spy.assert_not_called()
    settle("test cleanup")


@pytest.mark.parametrize(
    "fields",
    [
        {"_closing": True},
        {"_client_gone_interrupt_requested": True},
        {"_lease_taken_over": True},
        {"lazy": True},
        {"agent": type("Rotated", (), {"session_id": "descendant"})()},
    ],
)
def test_unsafe_live_runtime_refuses_without_replacement(store, fields):
    record = live_record(store, **fields)
    assert "error" in attach()
    assert server._sessions == {"surviving-runtime": record}


def test_conflicting_live_lease_is_not_taken_over(store):
    from hermes_cli.active_sessions import (
        active_session_registry_snapshot,
        try_acquire_active_session,
    )

    lease, error = try_acquire_active_session(
        session_id="exact-parent", surface="cli", config={}, registry_home=store[2]
    )
    assert lease is not None and error is None
    try:
        before = active_session_registry_snapshot(store[2], strict=True)
        assert attach()["error"]["code"] == 4090
        assert not server._sessions
        assert active_session_registry_snapshot(store[2], strict=True) == before
    finally:
        lease.release()


def test_ambiguous_runtime_and_unreadable_registry_fail_closed(store, monkeypatch):
    record = live_record(store)
    server._sessions["other-runtime"] = dict(record)
    assert attach()["error"]["code"] == 4090
    server._sessions.clear()

    def unavailable(*args, **kwargs):
        raise OSError("unreadable registry")

    monkeypatch.setattr(
        "hermes_cli.active_sessions.active_session_liveness_guard", unavailable
    )
    assert attach()["error"]["data"]["reason"] == "recovery_unknown"
    assert not server._sessions


def test_concurrent_attach_is_idempotent_and_restart_changes_incarnation(store):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: attach()["result"], range(8)))
    assert len({r["session_id"] for r in results}) == 1
    assert len(server._sessions) == 1
    old = results[0]
    server._sessions.clear()  # owner/runtime loss; durable identity remains
    new = attach()["result"]
    assert new["runtime_incarnation"] != old["runtime_incarnation"]
    assert new["stored_session_id"] == old["stored_session_id"]
    assert new["disposition"] == "unknown" and new["execution_fenced"]


def test_attach_reuses_live_transport_without_displacing_owner_and_activate_still_works(
    store, monkeypatch
):
    from tui_gateway.transport import bind_transport, reset_transport

    class Peer:
        closed = False

        def write(self, frame):
            return True

        def close(self):
            self.closed = True

    original, joining = Peer(), Peer()
    session = live_record(store, transport=original, running=True)
    monkeypatch.setattr(server, "_fallback_session_info", lambda _: {"model": "test"})
    token = bind_transport(joining)
    try:
        attached = attach()["result"]
        assert attached["session_id"] == "surviving-runtime"
        assert server._session_transport_contains(session, original)
        assert server._session_transport_contains(session, joining)
        activated = rpc(
            "session.activate", session_id="surviving-runtime", omit_messages=True
        )
        assert activated["result"]["running"]
        assert session["running"]
        assert not original.closed
    finally:
        reset_transport(token)
        session["transport"].close()


def test_pending_legacy_hydration_is_observed_not_restarted(store, monkeypatch):
    import threading

    ready = threading.Event()
    session = live_record(store, resume_hydrating=True, resume_history_ready=ready)
    spies = inert_spies(monkeypatch, store[0])
    assert attach()["result"]["disposition"] == "live"
    assert session["resume_hydrating"] and not ready.is_set()
    for spy in spies:
        spy.assert_not_called()


def test_store_identity_mismatch_and_missing_store_fail_closed(store, monkeypatch):
    monkeypatch.setattr(store[0], "get_session", lambda _: {"id": "other"})
    assert attach()["error"]["code"] == 4007
    assert not server._sessions
    monkeypatch.setattr(server, "_get_db", lambda: None)
    assert "error" in attach()
    assert not server._sessions


@pytest.mark.parametrize("auto_continue", [True, False])
def test_stale_or_disabled_marker_never_grants_execution(
    store, monkeypatch, auto_continue
):
    monkeypatch.setattr("tui_gateway.turn_marker.time.time", lambda: 1)
    record_turn_start(store[2], "exact-parent", "old work", auto_continue=auto_continue)
    before = _marker_path(store[2]).read_bytes()
    result = attach()["result"]
    assert result["disposition"] == "unknown"  # the marker writer is still alive
    assert result["execution_fenced"]
    assert _marker_path(store[2]).read_bytes() == before
