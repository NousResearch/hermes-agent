"""Durable prompt intent regressions for #62203: retries, route assertions and queue transport."""

import threading
import types

import pytest

from tui_gateway import server
from tui_gateway.prompt_intents import PromptIntentLedger


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, **_thread_options):
        self._target = target

    def start(self):
        self._target()


def _defer_prompt_submit_thread(monkeypatch):
    """Keep prompt.submit deterministic without running an agent turn."""
    threads = []

    class _DeferredThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(server.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args, **_kwargs: None)
    return threads


def test_prompt_submit_expected_stored_session_id_matches_session_key(monkeypatch):
    """Desktop can bind prompt.submit to the durable conversation it intended."""
    session = _session(session_key="stored-a")
    server._sessions["runtime-a"] = session
    threads = _defer_prompt_submit_thread(monkeypatch)
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: pytest.fail("exact durable matches must not open the DB"),
    )

    try:
        response = server.handle_request(
            {
                "id": "submit-a",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-a",
                    "expected_stored_session_id": "stored-a",
                    "text": "hello",
                },
            }
        )

        assert response["result"] == {"status": "streaming"}
        assert session["running"] is True
        assert len(threads) == 1
    finally:
        server._sessions.pop("runtime-a", None)


def test_prompt_submit_rejects_wrong_expected_stored_session_id_without_mutation(monkeypatch):
    """A stale durable destination fails before inflight, persistence, or build."""
    history = [{"role": "user", "content": "before"}]
    session = _session(session_key="stored-a", history=list(history))
    original_transport = object()
    session["transport"] = original_transport
    server._sessions["runtime-a"] = session
    monkeypatch.setattr(
        server,
        "_claim_prompt_submit_intent",
        lambda *_args, **_kwargs: pytest.fail("must not claim client intent"),
    )
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args, **_kwargs: pytest.fail("must not start inflight turn"),
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda *_args, **_kwargs: pytest.fail("must not persist row"),
    )
    monkeypatch.setattr(
        server,
        "_persist_branch_seed",
        lambda *_args, **_kwargs: pytest.fail("must not persist branch"),
    )
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args, **_kwargs: pytest.fail("must not build agent"),
    )

    try:
        response = server.handle_request(
            {
                "id": "submit-wrong",
                "method": "prompt.submit",
                "params": {
                    "client_request_id": "wrong-destination-intent",
                    "session_id": "runtime-a",
                    "expected_stored_session_id": "stored-b",
                    "text": "wrong tab prompt",
                },
            }
        )

        assert response["error"]["code"] == 4019
        assert "stored session mismatch" in response["error"]["message"]
        assert session["history"] == history
        assert session["running"] is False
        assert session["transport"] is original_transport
    finally:
        server._sessions.pop("runtime-a", None)


def test_prompt_submit_omitted_expected_stored_session_id_remains_compatible(monkeypatch):
    """Older clients that only send the runtime session id still work."""
    session = _session(session_key="stored-a")
    server._sessions["runtime-a"] = session
    threads = _defer_prompt_submit_thread(monkeypatch)

    try:
        response = server.handle_request(
            {
                "id": "legacy-submit",
                "method": "prompt.submit",
                "params": {"session_id": "runtime-a", "text": "legacy client"},
            }
        )

        assert response["result"] == {"status": "streaming"}
        assert len(threads) == 1
    finally:
        server._sessions.pop("runtime-a", None)


def test_prompt_submit_expected_stored_session_id_accepts_compression_parent(monkeypatch):
    """A routed parent remains valid when its live session is a continuation tip."""
    session = _session(
        agent=types.SimpleNamespace(session_id="continuation-tip"),
        session_key="continuation-tip",
    )
    server._sessions["runtime-tip"] = session
    threads = _defer_prompt_submit_thread(monkeypatch)

    class _Db:
        def resolve_resume_session_id(self, session_id):
            return "continuation-tip" if session_id == "lineage-root" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _Db())

    try:
        response = server.handle_request(
            {
                "id": "continuation-submit",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-tip",
                    "expected_stored_session_id": "lineage-root",
                    "text": "continue",
                },
            }
        )

        assert response["result"] == {"status": "streaming"}
        assert len(threads) == 1
    finally:
        server._sessions.pop("runtime-tip", None)


def test_prompt_submit_session_lineage_failure_is_retryable_without_mutation(monkeypatch):
    """A failed lineage read is temporary, not proof of a destination mismatch."""
    history = [{"role": "user", "content": "before"}]
    session = _session(session_key="continuation-tip", history=list(history))
    server._sessions["runtime-tip"] = session

    class _BrokenDb:
        def resolve_resume_session_id(self, _session_id):
            raise RuntimeError("lineage database unavailable")

    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDb())
    monkeypatch.setattr(
        server,
        "_claim_prompt_submit_intent",
        lambda *_args, **_kwargs: pytest.fail("must not claim client intent"),
    )
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args, **_kwargs: pytest.fail("must not start inflight turn"),
    )

    try:
        response = server.handle_request(
            {
                "id": "lineage-unavailable",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-tip",
                    "expected_stored_session_id": "lineage-root",
                    "client_request_id": "lineage-intent",
                    "text": "continue",
                },
            }
        )

        assert response["error"] == {
            "code": 4022,
            "message": "stored session lineage is temporarily unavailable; retry later",
        }
        assert session["history"] == history
        assert session["running"] is False
    finally:
        server._sessions.pop("runtime-tip", None)


def test_prompt_submit_unavailable_ledger_returns_temporary_error(monkeypatch):
    """The first idempotent submit opens the ledger without making import fatal."""
    history = [{"role": "user", "content": "before"}]
    session = _session(session_key="stored-a", history=list(history))
    server._sessions["runtime-a"] = session
    monkeypatch.setattr(server, "_prompt_intents", None)

    def _unavailable_ledger(**_kwargs):
        raise PermissionError("state directory is read-only")

    monkeypatch.setattr(server, "PromptIntentLedger", _unavailable_ledger)
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args, **_kwargs: pytest.fail("must not start inflight turn"),
    )

    try:
        response = server.handle_request(
            {
                "id": "ledger-unavailable",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-a",
                    "expected_stored_session_id": "stored-a",
                    "client_request_id": "unavailable-ledger-intent",
                    "text": "hello",
                },
            }
        )

        assert response["error"] == {
            "code": 4022,
            "message": "prompt idempotency is temporarily unavailable; retry later",
        }
        assert session["history"] == history
        assert session["running"] is False
    finally:
        server._sessions.pop("runtime-a", None)


def test_prompt_submit_expected_stored_session_id_uses_profile_database(monkeypatch, tmp_path):
    """Continuation resolution must use the live session's profile state.db."""
    from hermes_state import SessionDB

    profile_home = tmp_path / "profiles" / "coder"
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session("profile-root", source="tui")
    profile_db.end_session("profile-root", "compression")
    profile_db.create_session("profile-tip", source="tui", parent_session_id="profile-root")

    default_db = SessionDB(db_path=tmp_path / "default.db")
    default_db.create_session("profile-root", source="tui")
    default_db.end_session("profile-root", "compression")
    default_db.create_session("default-tip", source="tui", parent_session_id="profile-root")

    session = _session(
        agent=types.SimpleNamespace(session_id="profile-tip"),
        session_key="profile-tip",
        profile_home=str(profile_home),
    )
    server._sessions["runtime-profile"] = session
    threads = _defer_prompt_submit_thread(monkeypatch)
    monkeypatch.setattr(server, "_get_db", lambda: default_db)

    try:
        assert profile_db.resolve_resume_session_id("profile-root") == "profile-tip"
        assert default_db.resolve_resume_session_id("profile-root") == "default-tip"

        response = server.handle_request(
            {
                "id": "profile-submit",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-profile",
                    "expected_stored_session_id": "profile-root",
                    "text": "continue profile",
                },
            }
        )

        assert response["result"] == {"status": "streaming"}
        assert len(threads) == 1
    finally:
        server._sessions.pop("runtime-profile", None)
        profile_db.close()
        default_db.close()


def test_prompt_submit_expected_stored_session_id_lazy_watch_requires_exact_child(monkeypatch):
    """An unupgraded lazy watch binds to its exact child, not its parent lineage."""
    session = _session(agent=None, session_key="child-tip", lazy=True)
    session["agent"] = None
    server._sessions["runtime-child"] = session

    class _Db:
        def resolve_resume_session_id(self, session_id):
            return "child-tip" if session_id == "parent-root" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _Db())
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args, **_kwargs: pytest.fail("must not start inflight turn"),
    )

    try:
        response = server.handle_request(
            {
                "id": "watch-submit",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-child",
                    "expected_stored_session_id": "parent-root",
                    "text": "wrong watch target",
                },
            }
        )

        assert response["error"]["code"] == 4019
        assert session["running"] is False
    finally:
        server._sessions.pop("runtime-child", None)


def test_prompt_submit_deduplicates_lost_ack_retry_within_stored_session(
    monkeypatch, tmp_path
):
    """One intent survives a lost ack, runtime replacement, and continuation."""
    from hermes_state import SessionDB
    from tui_gateway.prompt_intents import PromptIntentLedger

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stored-a", source="tui")
    session = _session(
        agent=types.SimpleNamespace(session_id="stored-a"),
        session_key="stored-a",
    )
    server._sessions["runtime-a"] = session
    ledger = PromptIntentLedger()
    monkeypatch.setattr(server, "_prompt_intents", ledger)
    counts = {"build": 0, "execute": 0, "inflight": 0, "persist": 0, "seed": 0}

    def _run_once(_rid, _sid, live_session, _text, **_kwargs):
        counts["execute"] += 1
        live_session["running"] = False

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_run_prompt_submit", _run_once)
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args, **_kwargs: counts.__setitem__("inflight", counts["inflight"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda *_args, **_kwargs: counts.__setitem__("persist", counts["persist"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_persist_branch_seed",
        lambda *_args, **_kwargs: counts.__setitem__("seed", counts["seed"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args, **_kwargs: counts.__setitem__("build", counts["build"] + 1),
    )
    params = {
        "session_id": "runtime-a",
        "expected_stored_session_id": "stored-a",
        "client_request_id": "intent-lost-ack-1",
        "text": f"only once {'x' * 10_000}",
    }

    try:
        first = server.handle_request({"id": "rpc-1", "method": "prompt.submit", "params": params})
        db.end_session("stored-a", "compression")
        db.create_session("stored-b", source="tui", parent_session_id="stored-a")
        assert db.resolve_resume_session_id("stored-a") == "stored-b"
        server._sessions.pop("runtime-a")
        server._sessions["runtime-a-recovered"] = _session(
            agent=types.SimpleNamespace(session_id="stored-b"),
            session_key="stored-b",
        )
        retry = server.handle_request(
            {
                "id": "rpc-2",
                "method": "prompt.submit",
                "params": {**params, "session_id": "runtime-a-recovered"},
            }
        )
        conflicting_reuse = server.handle_request(
            {
                "id": "rpc-3",
                "method": "prompt.submit",
                "params": {
                    **params,
                    "session_id": "runtime-a-recovered",
                    "text": "different prompt",
                },
            }
        )
        changed_route_reuse = server.handle_request(
            {
                "id": "rpc-route-conflict",
                "method": "prompt.submit",
                "params": {
                    **params,
                    "session_id": "runtime-a-recovered",
                    "expected_stored_session_id": "stored-b",
                },
            }
        )
        wrong_destination = server.handle_request(
            {
                "id": "rpc-4",
                "method": "prompt.submit",
                "params": {
                    **params,
                    "session_id": "runtime-a-recovered",
                    "expected_stored_session_id": "stored-c",
                },
            }
        )
        oversized_id = server.handle_request(
            {
                "id": "rpc-5",
                "method": "prompt.submit",
                "params": {
                    **params,
                    "session_id": "runtime-a-recovered",
                    "client_request_id": "i" * 257,
                },
            }
        )
        unrelated_intent = server.handle_request(
            {
                "id": "rpc-6",
                "method": "prompt.submit",
                "params": {
                    **params,
                    "session_id": "runtime-a-recovered",
                    "client_request_id": "another-live-intent",
                },
            }
        )

        assert first["result"] == {"status": "streaming"}
        assert retry["result"] == {
            "duplicate": True,
            "messages": [],
            "status": "complete",
        }
        assert conflicting_reuse["error"]["code"] == 4020
        assert changed_route_reuse["error"]["code"] == 4020
        assert wrong_destination["error"]["code"] == 4019
        assert oversized_id["error"]["code"] == 4021
        assert unrelated_intent["result"] == {"status": "streaming"}
        assert counts == {"build": 2, "execute": 2, "inflight": 2, "persist": 2, "seed": 2}
        assert len(ledger) == 2
    finally:
        server._sessions.pop("runtime-a", None)
        server._sessions.pop("runtime-a-recovered", None)
        db.close()


def test_prompt_submit_releases_intent_when_agent_setup_fails(monkeypatch):
    """A reserved id can retry when no agent execution ever began."""
    from tui_gateway.prompt_intents import PromptIntentLedger

    ledger = PromptIntentLedger()
    monkeypatch.setattr(server, "_prompt_intents", ledger)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args, **_kwargs: None)
    # The deferred prompt path waits via _wait_agent_for_prompt, not the flat
    # _wait_agent ceiling it replaced (#63078).
    monkeypatch.setattr(
        server,
        "_wait_agent_for_prompt",
        lambda _session, rid, _sid: server._err(rid, 5000, "build failed"),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: pytest.fail("agent execution must not begin"),
    )
    session = _session(session_key="stored-a")
    server._sessions["runtime-a"] = session
    inflight = []
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_args: inflight.append(True))
    params = {
        "session_id": "runtime-a",
        "expected_stored_session_id": "stored-a",
        "client_request_id": "retry-after-build-failure",
        "text": "try after setup recovers",
    }

    try:
        first = server.handle_request({"id": "setup-1", "method": "prompt.submit", "params": params})
        retry = server.handle_request({"id": "setup-2", "method": "prompt.submit", "params": params})

        assert first["result"] == {"status": "streaming"}
        assert retry["result"] == {"status": "streaming"}
        assert inflight == [True, True]
        assert len(ledger) == 0
        assert session["running"] is False
    finally:
        server._sessions.pop("runtime-a", None)


def test_duplicate_retry_rebinds_matching_queued_prompt_transport(monkeypatch):
    monkeypatch.setattr(server, "_prompt_intents", PromptIntentLedger())
    session = _session(running=True, transport="ws-old")
    params = {
        "session_id": "runtime-a",
        "expected_stored_session_id": "session-key",
        "client_request_id": "intent-queued-1",
        "text": "run this next",
    }

    assert server._claim_prompt_submit_intent(params, "rpc-1", session) is None
    server._enqueue_prompt(
        session,
        params["text"],
        session["transport"],
        client_request_id=params["client_request_id"],
    )

    session["transport"] = "ws-reconnected"
    session["running"] = False
    retry = server._claim_prompt_submit_intent(params, "rpc-2", session)

    assert retry["result"] == {"duplicate": True, "status": "queued"}
    assert session["queued_prompt"]["transport"] == "ws-reconnected"


def test_busy_rewind_is_retried_instead_of_queued_without_truncation(monkeypatch):
    ledger = PromptIntentLedger()
    monkeypatch.setattr(server, "_prompt_intents", ledger)
    session = _session(
        running=True,
        history=[{"role": "user", "content": "original"}],
    )
    server._sessions["runtime-rewind"] = session

    try:
        response = server.handle_request(
            {
                "id": "rewind-busy",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-rewind",
                    "expected_stored_session_id": "session-key",
                    "client_request_id": "intent-rewind-1",
                    "text": "edited",
                    "truncate_before_user_ordinal": 0,
                },
            }
        )

        assert response["error"]["code"] == 4009
        assert session.get("queued_prompt") is None
        assert len(ledger) == 0
    finally:
        server._sessions.pop("runtime-rewind", None)


