"""Regression coverage for persistent Pi delegate_session semantics."""

from __future__ import annotations

import json
import threading
import time

import pytest

import tools.delegate_session_tool as ds


class Parent:
    def __init__(self, session_id: str = "parent-session") -> None:
        self.session_id = session_id


class FakePiClient:
    instances = []

    def __init__(self, *, persistent_session=False, session_id=None, session_name=None, acp_cwd=None, **_kwargs):
        self.persistent_session = persistent_session
        self.session_id = session_id
        self.session_name = session_name
        self.cwd = acp_cwd
        self.is_closed = False
        self.messages = []
        self.steers = []
        self.started_turn = threading.Event()
        self.release_turn = threading.Event()
        self.block_turns = False
        self._proc = None
        self.__class__.instances.append(self)

    def start(self, *, timeout=30.0):
        return {
            "sessionId": self.session_id,
            "sessionFile": f"/tmp/{self.session_id}.jsonl",
            "messageCount": len(self.messages),
            "isStreaming": False,
        }

    def run_session_prompt(self, message, *, timeout_seconds=900.0):
        self.messages.append(message)
        self.started_turn.set()
        if self.block_turns:
            assert self.release_turn.wait(timeout=5)
        return {
            "success": True,
            "text": f"done:{message}",
            "reasoning": "",
            "duration_s": 0.01,
            "state": {
                "sessionId": self.session_id,
                "messageCount": len(self.messages),
                "isStreaming": False,
            },
        }

    def get_messages(self, *, timeout=30.0):
        return [{"role": "user", "content": m} for m in self.messages]

    def steer(self, message, *, timeout=30.0):
        self.steers.append(message)
        return {"success": True, "command": "steer"}

    def abort(self, *, timeout=30.0):
        self.release_turn.set()
        return {"success": True, "command": "abort"}

    def close(self):
        self.is_closed = True
        self.release_turn.set()


@pytest.fixture(autouse=True)
def clean_sessions(monkeypatch, tmp_path):
    with ds._SESSION_LOCK:
        for record in ds._SESSIONS.values():
            try:
                record["client"].close()
            except Exception:
                pass
        ds._SESSIONS.clear()
    FakePiClient.instances.clear()
    monkeypatch.setattr(ds, "PiRPCClient", FakePiClient)
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: tmp_path)
    monkeypatch.setattr(ds, "pending_question_for_owner", lambda _client: None)
    yield
    with ds._SESSION_LOCK:
        for record in ds._SESSIONS.values():
            try:
                record["client"].close()
            except Exception:
                pass
        ds._SESSIONS.clear()


def payload(raw: str) -> dict:
    return json.loads(raw)


def wait_for_status(parent: Parent, sid: str, wanted: str, timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    latest = {}
    while time.time() < deadline:
        latest = payload(ds.delegate_session(action="status", session_id=sid, parent_agent=parent))
        if latest.get("status") == wanted:
            return latest
        time.sleep(0.01)
    raise AssertionError(f"session {sid} never reached {wanted}: {latest}")


def test_start_creates_native_persistent_pi_session():
    parent = Parent()
    result = payload(ds.delegate_session(action="start", parent_agent=parent))

    assert result["success"] is True
    assert result["created"] is True
    assert result["status"] == "idle"
    assert result["session_id"] == result["pi_session_id"]
    client = FakePiClient.instances[-1]
    assert client.persistent_session is True
    assert client.session_id == result["session_id"]
    assert client.cwd == result["cwd"]


def test_send_reuses_same_client_and_preserves_followup_history():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]

    first = payload(ds.delegate_session(action="send", session_id=sid, message="first", parent_agent=parent))
    assert first["accepted"] is True
    wait_for_status(parent, sid, "idle")

    second = payload(ds.delegate_session(action="send", session_id=sid, message="second", parent_agent=parent))
    assert second["accepted"] is True
    final = wait_for_status(parent, sid, "idle")

    assert FakePiClient.instances == [client]
    assert client.messages == ["first", "second"]
    assert final["last_result"]["text"] == "done:second"


def test_steer_targets_live_session_instead_of_spawning_child_agent():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.block_turns = True

    payload(ds.delegate_session(action="send", session_id=sid, message="long task", parent_agent=parent))
    assert client.started_turn.wait(timeout=1)

    steered = payload(ds.delegate_session(action="steer", session_id=sid, message="focus on tests", parent_agent=parent))
    assert steered["success"] is True
    assert client.steers == ["focus on tests"]

    client.release_turn.set()
    wait_for_status(parent, sid, "idle")


def test_sessions_are_scoped_to_owning_hermes_conversation():
    owner = Parent("owner")
    stranger = Parent("stranger")
    started = payload(ds.delegate_session(action="start", parent_agent=owner))
    sid = started["session_id"]

    denied = ds.delegate_session(action="status", session_id=sid, parent_agent=stranger)
    assert "not found for this conversation" in denied.lower()

    owner_list = payload(ds.delegate_session(action="list", parent_agent=owner))
    stranger_list = payload(ds.delegate_session(action="list", parent_agent=stranger))
    assert [row["session_id"] for row in owner_list["sessions"]] == [sid]
    assert stranger_list["sessions"] == []


def test_stop_closes_client_but_native_id_can_be_resumed():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    first_client = FakePiClient.instances[-1]

    stopped = payload(ds.delegate_session(action="stop", session_id=sid, parent_agent=parent))
    assert stopped["closed"] is True
    assert first_client.is_closed is True

    # Simulate a gateway restart/process-local registry loss while Pi's native
    # session remains durable on disk, then reopen the same native session id.
    with ds._SESSION_LOCK:
        ds._SESSIONS.pop(sid, None)
    resumed = payload(ds.delegate_session(action="resume", session_id=sid, parent_agent=parent))
    assert resumed["created"] is True
    assert resumed["session_id"] == sid
    assert resumed["pi_session_id"] == sid
    assert FakePiClient.instances[-1] is not first_client
    assert FakePiClient.instances[-1].session_id == sid
