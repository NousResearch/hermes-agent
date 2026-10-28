"""Session ownership must survive cold discovery and refusals must reach the client."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB

KEY = "20260101_000000_deadbe"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    homes = [launch, launch / "profiles" / "bot", launch / "profiles" / "other"]
    for home in homes:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    from tui_gateway import server
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    dbs = [SessionDB(db_path=home / "state.db") for home in homes]
    monkeypatch.setattr(server, "_get_db", lambda: dbs[0])
    dbs[1].create_session(KEY, source="desktop")
    yield server, homes, dbs
    for db in reversed(dbs):
        db.close()
    server._sessions.clear()


@pytest.mark.parametrize("cached,key,local_exists,unavailable,explicit,allowed", [
    pytest.param((), KEY, False, False, False, False, id="cold"),
    pytest.param((2,), KEY, False, False, False, False, id="partial"),
    pytest.param((1,), KEY, False, False, False, False, id="discovered"),
    pytest.param((), KEY, True, False, False, False, id="ambiguous"),
    pytest.param((), "unverified-local", False, True, False, False, id="unavailable"),
    pytest.param((), "local-session", False, False, False, True, id="new-local"),
    pytest.param((), "local-session", True, False, False, True, id="existing-local"),
    pytest.param((), KEY, False, False, True, True, id="explicit"),
])
def test_durable_ownership_is_independent_of_process_discovery(
        stores, cached, key, local_exists, unavailable, explicit, allowed):
    server, homes, dbs = stores
    server._served_profile_homes.update(homes[index] for index in cached)
    if local_exists:
        dbs[0].create_session(key, source="desktop")
    if unavailable:
        dbs[2].close()
        (homes[2] / "state.db").write_bytes(b"not a sqlite database")
    session = {"session_key": key, "source": "desktop"}
    if explicit:
        session["profile_home"] = str(homes[1])
    if allowed:
        assert server._ensure_session_db_row(session) is True
        frame = server._compute_host_turn_frame("request", "ui", {**session, "history_lock": threading.RLock()}, "hello")
        assert Path(frame["profile_home"]) == homes[1 if explicit else 0]
    else:
        with pytest.raises(server.SessionProfileOwnershipError):
            server._ensure_session_db_row(session)
    assert (dbs[0].get_session(key) is not None) == (local_exists or (allowed and not explicit))
    assert dbs[1].get_session(KEY) is not None


@pytest.mark.parametrize("entry", ["submit", "busy", "synthetic", "queue", "compute"])
def test_profile_refusal_has_an_observable_disposition(stores, monkeypatch, entry):
    server, homes, dbs = stores
    server._served_profile_homes.add(homes[1])
    events, dispatched = [], []
    monkeypatch.setattr(server, "_emit", lambda name, sid, payload=None: events.append((name, payload)))
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *a, **kw: dispatched.append("agent"))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session, cfg=None: False)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda cfg: SimpleNamespace(
        submit_turn=lambda frame, **kw: dispatched.append("compute")))
    session = {"session_key": KEY, "source": "desktop", "running": True, "agent": None,
               "history_lock": threading.RLock(), "attached_images": [], "history": [],
               "queued_prompts": [], "_queued_prompt_generation": 0,
               "inflight_turn": {"user": "hello", "assistant": "", "streaming": True}}
    server._sessions["ui"] = session
    calls = {
        "submit": lambda: server.handle_request({"id": "request", "method": "prompt.submit",
                    "params": {"session_id": "ui", "text": "hello"}}),
        "busy": lambda: server._handle_busy_submit("request", "ui", session, "hello", transport=None),
        "synthetic": lambda: server._run_prompt_submit("request", "ui", session, "hello"),
        "queue": lambda: server._drain_queued_prompt("request", "ui", session),
        "compute": lambda: server._submit_prompt_to_compute_host("request", "ui", session, "hello"),
    }
    if entry in {"queue", "submit"}:
        session["running"] = False
    if entry == "queue":
        session["queued_prompt"] = {"text": "hello"}
    response = calls[entry]()
    if isinstance(response, dict):
        assert response["error"]["code"] == 4095
    else:
        assert response is (entry == "queue")
        assert not session["running"]
        assert any(name in {"message.complete", "error"} for name, payload in events)
    if entry == "synthetic":
        assert server._inflight_snapshot(session)["status"] == "error"
    if entry == "queue":
        assert session["queued_prompt"]["text"] == "hello"
    if entry == "busy":
        assert not session.get("queued_prompt")
    assert dispatched == []
    assert dbs[0].get_session(KEY) is None
