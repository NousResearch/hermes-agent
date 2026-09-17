"""Regression coverage for persistent Pi delegate_session semantics."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

import tools.delegate_session_tool as ds


class Parent:
    def __init__(self, session_id: str = "parent-session") -> None:
        self.session_id = session_id


class FakePiClient:
    instances = []

    def __init__(
        self,
        *,
        persistent_session=False,
        session_id=None,
        session_name=None,
        acp_cwd=None,
        question_answerer=None,
        **_kwargs,
    ):
        self.persistent_session = persistent_session
        self.session_id = session_id
        self.session_name = session_name
        self.cwd = acp_cwd
        self.question_answerer = question_answerer
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
    monkeypatch.setattr(
        ds, "_session_store_root", lambda: tmp_path / "delegate-session-store"
    )
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


def wait_for_status(
    parent: Parent, sid: str, wanted: str, timeout: float = 2.0
) -> dict:
    deadline = time.time() + timeout
    latest = {}
    while time.time() < deadline:
        latest = payload(
            ds.delegate_session(action="status", session_id=sid, parent_agent=parent)
        )
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


def test_start_installs_hermes_auto_question_answerer(monkeypatch):
    parent = Parent()
    seen = []

    def fake_answerer(parent_arg, method, title, options):
        seen.append((parent_arg, method, title, options))
        return "Hermes chose this"

    monkeypatch.setattr(ds, "_auto_answer_pi_question", fake_answerer)
    result = payload(ds.delegate_session(action="start", parent_agent=parent))
    client = FakePiClient.instances[-1]

    assert result["success"] is True
    assert callable(client.question_answerer)
    assert (
        client.question_answerer("input", "Which path?", ["A", "B"])
        == "Hermes chose this"
    )
    assert seen == [(parent, "input", "Which path?", ["A", "B"])]


def test_auto_answer_uses_supervising_context_and_main_runtime(monkeypatch):
    parent = Parent()
    parent._session_messages = [
        {"role": "user", "content": "Use PostgreSQL for this project."},
        {"role": "assistant", "content": "I will keep the existing database choice."},
        {"role": "tool", "content": "Detected database port 5432."},
    ]
    parent._current_main_runtime = lambda: {
        "provider": "test-provider",
        "model": "test-model",
        "base_url": "https://example.invalid/v1",
        "api_key": "secret-not-logged",
    }
    captured = {}

    def fake_oneshot(**kwargs):
        captured.update(kwargs)
        return "5432"

    monkeypatch.setattr("agent.oneshot.run_oneshot", fake_oneshot)
    answer = ds._auto_answer_pi_question(parent, "input", "Which DB port?", [])

    assert answer == "5432"
    assert "Use PostgreSQL for this project" in captured["user_input"]
    assert "Detected database port 5432" in captured["user_input"]
    assert "Never ask the user" in captured["instructions"]
    assert captured["task"] == "delegate_session_question"
    assert captured["main_runtime"]["model"] == "test-model"


def test_auto_answer_normalizes_confirm_and_select(monkeypatch):
    parent = Parent()
    answers = iter([
        "Proceed, yes.",
        "use grpc",
        "2",
        "not one of the options",
        "maybe",
    ])
    monkeypatch.setattr("agent.oneshot.run_oneshot", lambda **_kwargs: next(answers))

    assert ds._auto_answer_pi_question(parent, "confirm", "Continue?", []) == "yes"
    assert (
        ds._auto_answer_pi_question(
            parent, "select", "Transport?", ["Use REST", "Use gRPC"]
        )
        == "Use gRPC"
    )
    assert (
        ds._auto_answer_pi_question(
            parent, "select", "Transport?", ["Use REST", "Use gRPC"]
        )
        == "Use gRPC"
    )
    assert (
        ds._auto_answer_pi_question(
            parent, "select", "Transport?", ["Use REST", "Use gRPC"]
        )
        is None
    )
    assert ds._auto_answer_pi_question(parent, "confirm", "Continue?", []) is None


def test_send_reuses_same_client_and_preserves_followup_history():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]

    first = payload(
        ds.delegate_session(
            action="send", session_id=sid, message="first", parent_agent=parent
        )
    )
    assert first["accepted"] is True
    wait_for_status(parent, sid, "idle")

    second = payload(
        ds.delegate_session(
            action="send", session_id=sid, message="second", parent_agent=parent
        )
    )
    assert second["accepted"] is True
    final = wait_for_status(parent, sid, "idle")

    assert FakePiClient.instances == [client]
    assert client.messages == ["first", "second"]
    assert final["last_result"]["text"] == "done:second"


def test_start_on_live_session_with_goal_dispatches_followup_turn():
    """Re-start on a live session must run the goal, not drop it (conductor v5/v6)."""
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    wait_for_status(parent, sid, "idle")

    reused = payload(
        ds.delegate_session(
            action="start",
            session_id=sid,
            goal="phase two goal",
            parent_agent=parent,
        )
    )

    assert reused["reused"] is True
    assert reused.get("turn_dispatched") is True
    wait_for_status(parent, sid, "idle")
    assert any("phase two goal" in m for m in client.messages), client.messages


def test_start_with_goal_on_dead_client_reopens_and_dispatches():
    """R69 (2026-09-17): the spool server drives follow-up phase turns with
    action=start + session_id + goal. When the pi client behind that handle
    has died (is_closed — e.g. the RPC process crashed mid-run), the start
    must REOPEN the session on a fresh client and dispatch the goal there —
    never dispatch into the closed client, which raises "pi rpc client is
    closed" on every retry and strands the phase permanently."""
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    dead_client = FakePiClient.instances[-1]
    wait_for_status(parent, sid, "idle")

    # Simulate the RPC process dying: client closed but the session record
    # still live in the registry (exactly the spool server's view after a
    # crashed turn).
    dead_client.close()

    result = payload(
        ds.delegate_session(
            action="start",
            session_id=sid,
            goal="reland goal",
            parent_agent=parent,
        )
    )

    assert result["success"] is True
    fresh_client = FakePiClient.instances[-1]
    assert fresh_client is not dead_client
    assert fresh_client.is_closed is False
    # The goal is dispatched on the REOPENED client, and the dead client
    # never saw it.
    wait_for_status(parent, sid, "idle")
    assert any("reland goal" in m for m in fresh_client.messages), fresh_client.messages
    assert not any("reland goal" in m for m in dead_client.messages)


def test_steer_on_idle_session_degrades_to_send_instead_of_erroring():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]

    payload(
        ds.delegate_session(
            action="send", session_id=sid, message="first", parent_agent=parent
        )
    )
    wait_for_status(parent, sid, "idle")

    # Race window: turn already ended, but the caller tries to steer.
    steered = payload(
        ds.delegate_session(
            action="steer",
            session_id=sid,
            message="focus on tests",
            parent_agent=parent,
        )
    )
    assert steered["success"] is True
    assert steered["degraded_to_send"] is True
    assert "follow-up" in steered["note"]
    # No steer was attempted (session was idle); message became a new turn instead.
    assert client.steers == []
    final = wait_for_status(parent, sid, "idle")
    assert client.messages == ["first", "focus on tests"]
    assert final["last_result"]["text"] == "done:focus on tests"


def test_steer_on_closed_session_still_errors_with_resume_hint():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]

    payload(ds.delegate_session(action="stop", session_id=sid, parent_agent=parent))

    result = payload(
        ds.delegate_session(
            action="steer", session_id=sid, message="hello", parent_agent=parent
        )
    )
    assert result.get("error") or result.get("success") is not True


def test_steer_targets_live_session_instead_of_spawning_child_agent():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.block_turns = True

    payload(
        ds.delegate_session(
            action="send", session_id=sid, message="long task", parent_agent=parent
        )
    )
    assert client.started_turn.wait(timeout=1)

    steered = payload(
        ds.delegate_session(
            action="steer",
            session_id=sid,
            message="focus on tests",
            parent_agent=parent,
        )
    )
    assert steered["success"] is True
    assert client.steers == ["focus on tests"]

    client.release_turn.set()
    wait_for_status(parent, sid, "idle")


def test_sessions_are_recoverable_by_new_supervisor_in_same_workspace():
    owner = Parent("owner")
    replacement = Parent("replacement")
    started = payload(ds.delegate_session(action="start", parent_agent=owner))
    sid = started["session_id"]

    status = payload(
        ds.delegate_session(action="status", session_id=sid, parent_agent=replacement)
    )
    assert status["success"] is True
    assert status["session_id"] == sid

    replacement_list = payload(
        ds.delegate_session(action="list", parent_agent=replacement)
    )
    assert [row["session_id"] for row in replacement_list["sessions"]] == [sid]


def test_stop_closes_client_but_native_id_can_be_resumed():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    first_client = FakePiClient.instances[-1]

    stopped = payload(
        ds.delegate_session(action="stop", session_id=sid, parent_agent=parent)
    )
    assert stopped["closed"] is True
    assert first_client.is_closed is True

    # Simulate a gateway restart/process-local registry loss while Pi's native
    # session remains durable on disk, then reopen the same native session id.
    with ds._SESSION_LOCK:
        ds._SESSIONS.pop(sid, None)
    resumed = payload(
        ds.delegate_session(action="resume", session_id=sid, parent_agent=parent)
    )
    assert resumed["created"] is True
    assert resumed["session_id"] == sid
    assert resumed["pi_session_id"] == sid
    assert FakePiClient.instances[-1] is not first_client
    assert FakePiClient.instances[-1].session_id == sid


def test_stop_then_resume_reopens_in_same_gateway_process():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    first_client = FakePiClient.instances[-1]

    payload(ds.delegate_session(action="stop", session_id=sid, parent_agent=parent))
    resumed = payload(
        ds.delegate_session(action="resume", session_id=sid, parent_agent=parent)
    )

    assert resumed["success"] is True
    assert resumed["created"] is True
    assert resumed["status"] == "idle"
    assert FakePiClient.instances[-1] is not first_client
    assert FakePiClient.instances[-1].is_closed is False
    assert callable(FakePiClient.instances[-1].question_answerer)


def test_resume_after_registry_loss_requires_same_workspace(monkeypatch, tmp_path):
    parent = Parent()
    original = tmp_path / "original"
    elsewhere = tmp_path / "elsewhere"
    original.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: original)

    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    assert FakePiClient.instances[-1].cwd == str(original.resolve())

    with ds._SESSION_LOCK:
        stale = ds._SESSIONS.pop(sid)
    stale["client"].close()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: elsewhere)

    denied = ds.delegate_session(
        action="resume", session_id=sid, parent_agent=Parent("replacement")
    )
    assert "another profile or workspace" in denied.lower()

    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: original)
    resumed = payload(
        ds.delegate_session(
            action="resume", session_id=sid, parent_agent=Parent("replacement")
        )
    )
    assert resumed["cwd"] == str(original.resolve())
    assert FakePiClient.instances[-1].cwd == str(original.resolve())


def test_list_includes_offline_durable_sessions_after_registry_loss():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    with ds._SESSION_LOCK:
        stale = ds._SESSIONS.pop(sid)
    stale["client"].close()

    listed = payload(ds.delegate_session(action="list", parent_agent=parent))
    row = next(row for row in listed["sessions"] if row["session_id"] == sid)
    assert row["status"] == "offline"
    assert row["cwd"] == started["cwd"]


def test_durable_metadata_cache_prunes_oldest_files(monkeypatch, tmp_path):
    root = ds._session_store_root()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ds, "_MAX_DURABLE_SESSIONS", 3)
    created = []
    for index in range(5):
        path = root / f"session-{index}.json"
        path.write_text(json.dumps({"session_id": str(index)}))
        os.utime(path, (100 + index, 100 + index))
        created.append(path)

    ds._prune_durable_metadata(root)

    assert [path.name for path in ds._metadata_files_newest(root)] == [
        "session-4.json",
        "session-3.json",
        "session-2.json",
    ]
    assert not created[0].exists()
    assert not created[1].exists()


def test_durable_resume_metadata_allows_replacement_supervisor_in_same_workspace():
    owner = Parent("owner")
    replacement = Parent("replacement")
    started = payload(ds.delegate_session(action="start", parent_agent=owner))
    sid = started["session_id"]
    with ds._SESSION_LOCK:
        stale = ds._SESSIONS.pop(sid)
    stale["client"].close()

    resumed = payload(
        ds.delegate_session(action="resume", session_id=sid, parent_agent=replacement)
    )
    assert resumed["success"] is True
    assert resumed["session_id"] == sid


def _write_metadata(sid: str, **fields) -> None:
    """Craft durable metadata directly (as written by an older Hermes)."""
    data = {
        "version": 2,
        "backend": "pi",
        "session_id": sid,
        "native_session_id": sid,
        "pi_session_id": sid,
        "owner": "legacy-supervisor",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    data.update(fields)
    path = ds._metadata_path(sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_legacy_v2_metadata_migrates_on_same_workspace_resume(tmp_path, monkeypatch):
    workspace = tmp_path / "legacy"
    workspace.mkdir()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: workspace)
    sid = "legacy-v2-session"
    _write_metadata(sid, cwd=str(workspace.resolve()))

    resumed = payload(
        ds.delegate_session(
            action="resume", session_id=sid, parent_agent=Parent("new-supervisor")
        )
    )

    assert resumed["success"] is True
    assert resumed["session_id"] == sid
    assert FakePiClient.instances[-1].session_id == sid
    upgraded = json.loads(ds._metadata_path(sid).read_text(encoding="utf-8"))
    assert upgraded["version"] == 3
    assert upgraded["owner_scope"] == ds._scope_for_workspace(workspace)


def test_legacy_v2_metadata_denied_across_workspaces(tmp_path, monkeypatch):
    here = tmp_path / "here"
    elsewhere = tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: here)
    sid = "legacy-v2-foreign"
    _write_metadata(sid, cwd=str(elsewhere.resolve()))

    denied = ds.delegate_session(
        action="resume", session_id=sid, parent_agent=Parent("new-supervisor")
    )

    assert "another profile or workspace" in denied.lower()
    assert not FakePiClient.instances


def test_offline_control_actions_fail_closed_until_resumed():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    with ds._SESSION_LOCK:
        stale = ds._SESSIONS.pop(sid)
    stale["client"].close()

    for action, extra in (
        ("send", {"message": "hello"}),
        ("steer", {"message": "hello"}),
        ("stop", {}),
    ):
        result = payload(
            ds.delegate_session(
                action=action, session_id=sid, parent_agent=parent, **extra
            )
        )
        assert result.get("error"), action
        assert "offline" in result["error"].lower()
        assert "resume" in result["error"].lower()

    status = payload(
        ds.delegate_session(action="status", session_id=sid, parent_agent=parent)
    )
    assert status["success"] is True
    assert status["status"] == "offline"
    assert "resume" in (status.get("note") or "").lower()

    messages = payload(
        ds.delegate_session(action="messages", session_id=sid, parent_agent=parent)
    )
    assert messages["success"] is True
    assert messages["messages_json"] == "[]"
    assert "resume" in (messages.get("note") or "").lower()


def test_summary_preserves_structured_trailer_from_long_last_result():
    trailer = '{"phase_result":{"run_id":"r","phase_id":"p","action_id":"a","attempt_id":"x","status":"succeeded"}}'
    record = {
        "session_id": "s",
        "backend": "pi",
        "status": "idle",
        "client": FakePiClient(session_id="s"),
        "last_result": {"text": ("prefix-" * 2200) + trailer, "duration_s": 1.0},
    }

    summary = ds._summary(record)

    assert len(summary["last_result"]["text"]) <= ds._MAX_TEXT
    assert trailer in summary["last_result"]["text"]


def test_messages_remain_valid_json_when_history_exceeds_bound():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.messages = [(f"message-{i}-" + ("x" * 2500)) for i in range(40)]

    result = payload(
        ds.delegate_session(action="messages", session_id=sid, parent_agent=parent)
    )
    parsed = json.loads(result["messages_json"])

    assert len(result["messages_json"]) <= 40_000
    assert parsed
    assert parsed[-1]["content"].startswith("message-39-")


def test_owner_grant_survives_workspace_move(tmp_path, monkeypatch):
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    original.mkdir()
    moved.mkdir()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: original)
    owner = Parent("owner")
    started = payload(ds.delegate_session(action="start", parent_agent=owner))
    sid = started["session_id"]

    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: moved)

    # The conversation that owns the session keeps control after its cwd moves.
    sent = payload(
        ds.delegate_session(
            action="send", session_id=sid, message="still mine", parent_agent=owner
        )
    )
    assert sent["accepted"] is True
    wait_for_status(owner, sid, "idle")

    # The scope grant does not extend to a replacement supervisor in the moved-to workspace.
    denied = payload(
        ds.delegate_session(
            action="send",
            session_id=sid,
            message="mine now",
            parent_agent=Parent("replacement"),
        )
    )
    assert denied.get("error")
    assert not denied.get("success")


def test_process_local_owner_never_authorizes_durable_metadata(tmp_path, monkeypatch):
    here = tmp_path / "here"
    elsewhere = tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(ds, "resolve_agent_cwd", lambda: here)
    parent = Parent("")  # no conversation id -> process-local owner key
    owner_key = ds._owner_key(parent)
    assert owner_key.startswith("agent:")
    sid = "proc-local-owner"
    # Foreign workspace, but the owner string matches the caller exactly.
    _write_metadata(sid, owner=owner_key, cwd=str(elsewhere.resolve()))

    denied = ds.delegate_session(action="resume", session_id=sid, parent_agent=parent)

    assert "another profile or workspace" in denied.lower()
    assert not FakePiClient.instances


def test_runtime_dispatch_passes_parent_agent(monkeypatch):
    """invoke_tool must pass parent_agent to delegate_session (regression).

    The generic dispatch path never forwards parent_agent, so delegate_session
    used to fail with "requires a parent agent context" on every live call.
    """
    from agent.agent_runtime_helpers import invoke_tool

    calls = {}

    def fake_delegate_session(**kwargs):
        calls.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(
        "tools.delegate_session_tool.delegate_session", fake_delegate_session
    )

    agent = Parent("dispatch-agent")
    # Attributes the dispatch chain touches before reaching the tool.
    setattr(agent, "_memory_manager", None)
    setattr(agent, "_subagent_lifecycle", None)

    raw = invoke_tool(
        agent,
        "delegate_session",
        {"action": "list"},
        effective_task_id="task-1",
    )

    assert calls.get("parent_agent") is agent
    assert calls.get("action") == "list"
    assert raw == '{"ok": true}'


def test_registry_dispatch_resolves_bound_parent(monkeypatch):
    """Registry dispatch (no explicit parent_agent) must fall back to the
    turn-bound active parent instead of failing with 'requires a parent
    agent context'."""
    from agent.subagent_lifecycle import bind_subagent_parent
    from tools import registry as reg

    entry = reg.registry.get_entry("delegate_session")
    owner = Parent("registry-owner")
    with bind_subagent_parent(owner):
        result = entry.handler({"action": "list"})
    payload_result = json.loads(result)
    assert payload_result.get("success") is True, payload_result

    # Without a bound parent the explicit error is preserved.
    err = entry.handler({"action": "list"})
    assert "requires a parent agent context" in err.lower()


# ---------------------------------------------------------------------------
# Backend-configurable sessions (Pi default + OpenCode)
# ---------------------------------------------------------------------------


class FakeOpenCodeClient(FakePiClient):
    """Fake OpenCode backend client mirroring agent.opencode_client surface."""

    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.native_session_id = None

    def pending_question_payload(self):
        return None

    def is_dead(self):
        return False


def test_default_backend_is_pi():
    parent = Parent()
    result = payload(ds.delegate_session(action="start", parent_agent=parent))
    assert result["backend"] == "pi"
    assert (
        result["pi_session_id"] == result["native_session_id"] == result["session_id"]
    )
    assert isinstance(FakePiClient.instances[-1], FakePiClient)


def test_backend_env_var_selects_opencode(monkeypatch):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    monkeypatch.setenv("HERMES_DELEGATE_SESSION_BACKEND", "opencode")
    parent = Parent()
    result = payload(ds.delegate_session(action="start", parent_agent=parent))
    assert result["backend"] == "opencode"
    assert result["native_session_id"]
    assert result["pi_session_id"] == result["native_session_id"]  # compatibility alias


def test_backend_arg_routes_to_opencode_client(monkeypatch):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    parent = Parent()
    result = payload(
        ds.delegate_session(action="start", backend="opencode", parent_agent=parent)
    )
    assert result["backend"] == "opencode"
    client = FakeOpenCodeClient.instances[-1]
    assert client.cwd  # opened in resolved cwd
    # follow-up send reaches the same client
    sid = result["session_id"]
    accepted = payload(
        ds.delegate_session(
            action="send", session_id=sid, message="go", parent_agent=parent
        )
    )
    assert accepted["accepted"] is True
    done = wait_for_status(parent, sid, "idle")
    assert done["last_result"]["text"].startswith("done:")


def test_unknown_backend_is_bounded_error():
    parent = Parent()
    err = ds.delegate_session(action="start", backend="claude", parent_agent=parent)
    assert "unknown backend" in err.lower()


def test_control_action_with_mismatched_backend_errors(monkeypatch):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    parent = Parent()
    started = payload(
        ds.delegate_session(action="start", backend="opencode", parent_agent=parent)
    )
    sid = started["session_id"]
    err = ds.delegate_session(
        action="send", session_id=sid, message="x", backend="pi", parent_agent=parent
    )
    assert "is a opencode delegate session" in err.lower()


def test_resume_of_live_session_with_other_backend_errors(monkeypatch):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))  # pi
    sid = started["session_id"]
    err = ds.delegate_session(
        action="resume", session_id=sid, backend="opencode", parent_agent=parent
    )
    assert "cannot be reopened as a opencode session" in err.lower()


def test_metadata_v2_roundtrip_reopens_correct_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    parent = Parent()
    started = payload(
        ds.delegate_session(action="start", backend="opencode", parent_agent=parent)
    )
    sid = started["session_id"]
    native = started["native_session_id"]
    with ds._SESSION_LOCK:
        stale = ds._SESSIONS.pop(sid)
    stale["client"].close()

    meta = ds._load_metadata(sid)
    assert meta["version"] == 3
    assert meta["backend"] == "opencode"
    assert meta["native_session_id"] == native

    # metadata snapshot keeps the pi_session_id alias for one release
    assert meta["pi_session_id"] == native

    # resume without an explicit backend reopens the stored backend
    resumed = payload(
        ds.delegate_session(action="resume", session_id=sid, parent_agent=parent)
    )
    assert resumed["backend"] == "opencode"
    client = FakeOpenCodeClient.instances[-1]
    assert client.native_session_id == native


def test_v1_metadata_loads_as_pi(monkeypatch, tmp_path):
    parent = Parent()
    sid = "legacy-v1-session"
    root = ds._session_store_root()
    root.mkdir(parents=True, exist_ok=True)
    legacy = {
        "version": 1,
        "session_id": sid,
        "pi_session_id": "pi_native_123",
        "owner": "parent-session",
        "cwd": str(tmp_path),
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    ds._metadata_path(sid).write_text(json.dumps(legacy))

    resumed = payload(
        ds.delegate_session(action="resume", session_id=sid, parent_agent=parent)
    )
    assert resumed["backend"] == "pi"
    # pi session id is reused as the native session on resume
    assert resumed["native_session_id"] == "pi_native_123"
    # re-persisted using the current metadata schema
    assert ds._load_metadata(sid)["version"] == 3


def test_list_includes_backend_field(monkeypatch):
    monkeypatch.setattr(ds, "OpenCodeClient", FakeOpenCodeClient)
    parent = Parent()
    pi_row = payload(ds.delegate_session(action="start", parent_agent=parent))
    oc_row = payload(
        ds.delegate_session(action="start", backend="opencode", parent_agent=parent)
    )

    listed = payload(ds.delegate_session(action="list", parent_agent=parent))
    by_id = {row["session_id"]: row for row in listed["sessions"]}
    assert by_id[pi_row["session_id"]]["backend"] == "pi"
    assert by_id[oc_row["session_id"]]["backend"] == "opencode"
    assert "native_session_id" in by_id[pi_row["session_id"]]
    assert "pi_session_id" in by_id[oc_row["session_id"]]

    # offline durable rows also carry the backend
    for sid in (pi_row["session_id"], oc_row["session_id"]):
        with ds._SESSION_LOCK:
            stale = ds._SESSIONS.pop(sid)
        stale["client"].close()
    listed2 = payload(ds.delegate_session(action="list", parent_agent=parent))
    by_id2 = {row["session_id"]: row for row in listed2["sessions"]}
    assert by_id2[pi_row["session_id"]]["backend"] == "pi"
    assert by_id2[oc_row["session_id"]]["backend"] == "opencode"


def test_registry_handler_forwards_backend(monkeypatch):
    from tools import registry as reg

    entry = reg.registry.get_entry("delegate_session")
    from agent.subagent_lifecycle import bind_subagent_parent

    owner = Parent("backend-forward")
    with bind_subagent_parent(owner):
        err = entry.handler({"action": "start", "backend": "bogus"})
    assert "unknown backend" in err.lower()


def test_wait_returns_immediately_when_session_is_idle():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))

    before = time.monotonic()
    waited = payload(
        ds.delegate_session(
            action="wait",
            session_id=started["session_id"],
            wait_seconds=1,
            parent_agent=parent,
        )
    )

    assert waited["success"] is True
    assert waited["status"] == "idle"
    assert waited["timed_out"] is False
    assert time.monotonic() - before < 0.25


def test_wait_wakes_on_running_to_idle_transition():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.block_turns = True
    payload(ds.delegate_session(action="send", session_id=sid, message="long task", parent_agent=parent))
    assert client.started_turn.wait(timeout=1)

    def release():
        time.sleep(0.05)
        client.release_turn.set()

    thread = threading.Thread(target=release)
    thread.start()
    waited = payload(ds.delegate_session(action="wait", session_id=sid, wait_seconds=1, parent_agent=parent))
    thread.join(timeout=1)

    assert waited["success"] is True
    assert waited["status"] == "idle"
    assert waited["timed_out"] is False
    assert waited["state_changed"] is True


def test_wait_timeout_is_nonfatal_and_does_not_stop_worker():
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.block_turns = True
    payload(ds.delegate_session(action="send", session_id=sid, message="long task", parent_agent=parent))
    assert client.started_turn.wait(timeout=1)

    waited = payload(ds.delegate_session(action="wait", session_id=sid, wait_seconds=0.05, parent_agent=parent))

    assert waited["success"] is True
    assert waited["status"] == "running"
    assert waited["timed_out"] is True
    assert client.is_closed is False
    assert client.release_turn.is_set() is False

    client.release_turn.set()
    assert payload(ds.delegate_session(action="wait", session_id=sid, wait_seconds=1, parent_agent=parent))["status"] == "idle"


def test_wait_seconds_is_separate_from_turn_timeout(monkeypatch):
    parent = Parent()
    started = payload(ds.delegate_session(action="start", parent_agent=parent))
    sid = started["session_id"]
    client = FakePiClient.instances[-1]
    client.block_turns = True
    observed = {}
    original = client.run_session_prompt

    def capture(message, *, timeout_seconds=900.0):
        observed["timeout_seconds"] = timeout_seconds
        return original(message, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(client, "run_session_prompt", capture)
    payload(ds.delegate_session(action="send", session_id=sid, message="task", timeout=777, parent_agent=parent))
    assert client.started_turn.wait(timeout=1)

    waited = payload(ds.delegate_session(action="wait", session_id=sid, wait_seconds=0.01, timeout=11, parent_agent=parent))
    assert waited["timed_out"] is True
    assert observed["timeout_seconds"] == 777

    client.release_turn.set()
    wait_for_status(parent, sid, "idle")


def test_check_requirements_accepts_opencode_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ds.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ds.Path, "home", lambda: tmp_path)  # no ~/.local/bin binaries
    assert ds.check_delegate_session_requirements() is False
    monkeypatch.setenv("HERMES_OPENCODE_SERVER_URL", "http://127.0.0.1:1")
    assert ds.check_delegate_session_requirements() is True


def test_dispatch_turn_uses_non_daemon_thread(monkeypatch):
    captured = {}

    class FakeThread:
        def __init__(self, *, target, args, name, daemon):
            captured.update(target=target, args=args, name=name, daemon=daemon)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(ds.threading, "Thread", FakeThread)
    record = {"session_id": "session-1234", "status": "idle"}

    ds._dispatch_turn(record, "work", 30.0)

    assert captured["started"] is True
    assert captured["daemon"] is False
    assert record["status"] == "running"
