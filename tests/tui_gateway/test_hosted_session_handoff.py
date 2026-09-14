"""Two runtimes share durable history, never a cached writer or an active lease."""

from __future__ import annotations

import multiprocessing
import os
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest


def _runtime(pipe, home):
    """Real RPC/turn/SQLite/registry lifecycle; only model and UI services are fake."""
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HOME"] = str(home.parent)
    from hermes_state import SessionDB
    from hermes_cli.active_sessions import active_session_registry_snapshot
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
    from tools import approval_context
    from tools.approval_gateway_wait import _await_gateway_decision
    from tui_gateway import server
    from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC, HostedRoomSessionError

    patch = pytest.MonkeyPatch()
    patch.setattr(Path, "home", lambda: home.parent)
    dbs = {}
    records = {}
    receipts = []
    observed = []
    emitted = []
    entered = threading.Event()
    proceed = threading.Event()
    terminal = threading.Event()
    finish = threading.Event()
    builds = []
    outcome = "complete"

    def db_for(profile):
        path = home / "profiles" / profile
        path.mkdir(parents=True, exist_ok=True)
        if profile not in dbs:
            dbs[profile] = SessionDB(db_path=path / "state.db")
        return dbs[profile]

    patch.setattr(server, "_profile_home", lambda profile: home / "profiles" / profile)
    patch.setattr(server, "_get_db", lambda: db_for("launch"))
    patch.setattr(server, "_load_cfg", lambda: {})
    patch.setattr(server, "_profile_configured_cwd", lambda _: str(home))
    patch.setattr(server, "_default_session_cwd", lambda: str(home))
    patch.setattr(server, "_completion_cwd", lambda _: str(home))
    for name in (
        "_enable_gateway_prompts", "_wire_callbacks", "_sync_agent_model_with_config",
        "_sync_agent_compression_with_config", "_sync_bot_capabilities",
        "_schedule_session_cap_enforcement", "_maybe_schedule_auto_continue",
        "_emit_settled_session_info",
    ):
        patch.setattr(server, name, lambda *a, **kw: None)
    patch.setattr(server, "_session_info", lambda *a, **kw: {})
    patch.setattr(server, "_get_usage", lambda _: {})
    patch.setattr(server, "_start_turn_voice", lambda: (None, False))
    patch.setattr(server, "_voice_tts_enabled", lambda: False)
    patch.setattr(server, "make_stream_renderer", lambda _: None)
    patch.setattr(server, "render_message", lambda *a: None)
    patch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    patch.setattr(approval_context, "_get_approval_timeout", lambda: 0)

    class Agent:
        model = "test-model"
        provider = "test-provider"
        platform = "bot_room"
        interim_assistant_callback = None

        def __init__(self, record):
            self.record = record
            self.session_id = record["session_key"]
            self._session_db = db_for(Path(record["profile_home"]).name)
            self._session_messages = []
            self._db_flush_scan_prefix = []
            self._owns_session_db = False
            self.interrupted = False
            builds.append(self)

        def clear_interrupt(self):
            self.interrupted = False

        def interrupt(self, *a, **kw):
            self.interrupted = True
            proceed.set()

        def release_clients(self):
            observed.append("soft-release")

        def close(self):
            raise AssertionError("handoff must not hard-close task resources")

        def _persist_session(self, messages):
            if self.record.get("_test_retirement_fault"):
                raise OSError("test final persistence failure")
            # Real SQLite with per-message dedup, including repeated retirement flushes.
            for message in messages:
                if not message.get("_test_durable"):
                    self._session_db.append_message(
                        self.session_id, role=message["role"], content=message["content"])
                    message["_test_durable"] = True
            self._db_flush_scan_prefix = list(messages)
            self._last_flushed_db_idx = len(messages)

        def run_conversation(self, prompt, conversation_history, **kwargs):
            observed.append([m["content"] for m in conversation_history])
            entered.set()
            assert proceed.wait(15), "test did not release model"
            if outcome == "timeout":
                decision = _await_gateway_decision(
                    self.session_id, lambda _: None,
                    {"command": "test-only dangerous operation", "pattern_key": "test"})
                assert decision["resolved"] is False and decision["choice"] is None
                observed.append("approval-timed-out")
            answer = "denied" if outcome == "timeout" else "stopped" if self.interrupted else str(prompt) + " reply"
            history = [dict(m, _test_durable=True) for m in conversation_history]
            history += [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": answer}]
            self._session_messages = history
            self._persist_session(history)
            scope = RoomArtifactScope.from_mapping({
                key: value for key, value in self.record["_hosted_room_task"].items()
                if key not in {"thread_id", "turn_id"}
            })
            RoomArtifactOutbox(Path(self.record["profile_home"]) / "state.db").put_bytes(
                scope=scope, source_name="receipt.txt", data=answer.encode())
            return {"messages": history, "final_response": answer, "interrupted": self.interrupted}

        def _flush_messages_to_session_db(self, messages):
            self._persist_session(messages)
            return True

    def build(sid, record):
        if record.get("agent") is None:
            record["agent"] = Agent(record)
            record["agent_ready"].set()

    patch.setattr(server, "_start_agent_build", build)
    patch.setattr(server, "_schedule_agent_build", lambda sid: build(sid, server._sessions[sid]))
    original_finish = server._finish_turn

    def finish_turn(*args):
        terminal.set()
        assert finish.wait(15), "test did not release finalization"
        original_finish(*args)

    patch.setattr(server, "_finish_turn", finish_turn)
    original_restore = server._Resume.restore

    def restore(ctx):
        # A slow cold read must not look orphaned before its record is registered.
        owners = active_session_registry_snapshot(registry_home=ctx.profile_home)
        local = {entry["lease_id"] for entry in owners if entry["pid"] == os.getpid()}
        assert local <= server._own_live_lease_ids()
        return original_restore(ctx)

    patch.setattr(server._Resume, "restore", restore)
    rpc = HostedRoomServerRPC(server)

    def receive(receipt):
        if outcome == "receipt-error":
            raise OSError("test terminal receipt commit failure")
        receipts.append(dict(receipt))

    pipe.send({"ready": True, "pid": os.getpid()})
    try:
        while True:
            command, data = pipe.recv()
            try:
                profile = data.get("profile", "reviewer")
                if command == "quit":
                    pipe.send({"ok": True})
                    break
                if command == "create":
                    db_for(profile)
                    result = rpc.create(profile=profile, title="Group: room", source="bot_room")
                    records[result["session_id"]] = server._sessions[result["session_id"]]
                    result["key"] = records[result["session_id"]]["session_key"]
                elif command in {"resume", "cache"}:
                    if command == "cache":
                        result = rpc._call("session.resume", {"profile": profile, "session_id": data["key"], "source": "bot_room"})
                    else:
                        result = rpc.resume(profile=profile, session_id=data["key"], source="bot_room")
                    records[result["session_id"]] = server._sessions[result["session_id"]]
                elif command == "submit":
                    outcome = data.get("outcome", "complete")
                    entered.clear()
                    proceed.clear()
                    terminal.clear()
                    finish.clear()
                    task = TaskIdentity("room", data["text"], "thread", "turn")
                    rpc.bind_artifact_scope(task=task, execution_generation=1, member_id="reviewer",
                                            authority_gateway_id="local", authority_epoch=1, profile=profile)
                    result = dict(rpc.submit(profile=profile, session_id=data["sid"], source="bot_room",
                                             prompt=data["text"], task=task, execution_generation=1,
                                             on_terminal=receive))
                    assert entered.wait(15)
                elif command == "terminal":
                    if data.get("stop"):
                        rpc.interrupt(profile=profile, session_id=data["sid"], source="bot_room",
                                      expected_task_id=data["text"])
                    proceed.set()
                    assert terminal.wait(15)
                    result = {"receipts": receipts[:], "running": records[data["sid"]]["running"]}
                elif command == "finish":
                    record = records[data["sid"]]
                    finish.set()
                    record["_run_thread"].join(15)
                    assert not record["_run_thread"].is_alive()
                    result = {"present": data["sid"] in server._sessions, "observed": observed[:],
                              "builds": len(builds), "receipts": receipts[:],
                              "rows": db_for(profile).get_messages_as_conversation(record["session_key"]),
                              "ended": db_for(profile).get_session(record["session_key"])["ended_at"]}
                elif command == "attach":
                    record = records[data["sid"]]
                    record["transport"] = SimpleNamespace(write=lambda _: True, _closed=False)
                    result = {"ok": True}
                elif command == "detach":
                    records[data["sid"]]["transport"] = server._detached_ws_transport
                    server._reap_idle_sessions()
                    result = {"present": data["sid"] in server._sessions}
                elif command == "retire-idle":
                    rpc.retire_idle(session_id=data["sid"])
                    result = {"present": data["sid"] in server._sessions}
                elif command == "building-reservation":
                    record = records[data["sid"]]
                    record["agent_build_started"] = True
                    rpc.retire_idle(session_id=data["sid"])
                    assert data["sid"] in server._sessions
                    assert record["active_session_lease"] is not None
                    record["agent_ready"].set()
                    server._reap_idle_sessions()
                    result = {"present": data["sid"] in server._sessions}
                elif command == "persistence-fault":
                    records[data["sid"]]["_test_retirement_fault"] = True
                    result = {"ok": True}
                elif command == "seed":
                    db = db_for(profile)
                    db.create_session(data["key"], source="bot_room", cwd=str(home), profile_name=profile)
                    db.set_session_title(data["key"], "Group: room")
                    db.set_session_hidden(data["key"], True)
                    result = {"ok": True}
                elif command == "snapshot":
                    result = {"leases": active_session_registry_snapshot(registry_home=home / "profiles" / profile),
                              "builds": len(builds), "observed": observed[:]}
                elif command == "stale-flush":
                    record = records[data["sid"]]
                    record["agent"]._session_messages.append({"role": "user", "content": "stale-write"})
                    result = {"flushed": server._flush_session_messages(record)}
                else:
                    raise AssertionError(command)
                pipe.send(result)
            except HostedRoomSessionError as exc:
                pipe.send({"refused": exc.code, "not_admitted": getattr(exc, "not_admitted", False)})
            except Exception:
                pipe.send({"error": traceback.format_exc()})
    finally:
        proceed.set()
        finish.set()
        for record in records.values():
            if worker := record.get("_run_thread"):
                worker.join(15)
                assert not worker.is_alive()
        patch.undo()
        for db in dbs.values():
            db.close()
        pipe.close()


class Runtime:
    def __init__(self, context, home):
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_runtime, args=(child, home))
        self.process.start()
        child.close()
        assert self.pipe.poll(30)
        assert self.pipe.recv()["ready"]

    def call(self, command, **data):
        self.pipe.send((command, data))
        assert self.pipe.poll(30), command
        result = self.pipe.recv()
        assert "error" not in result, result.get("error")
        return result

    def close(self):
        self.call("quit")
        self.process.join(20)
        assert not self.process.is_alive()
        assert self.process.exitcode == 0
        self.pipe.close()


@pytest.fixture
def runtimes(tmp_path):
    home = tmp_path / "runtime-state"
    home.mkdir()
    context = multiprocessing.get_context("spawn")
    a, b = Runtime(context, home), Runtime(context, home)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("outcome", ["complete", "timeout", "stop"])
def test_completed_hidden_writer_hands_off_only_after_finalization(runtimes, outcome):
    a, b = runtimes
    first = a.call("create")
    sid, key = first["session_id"], first["key"]
    a.call("submit", sid=sid, text="first", outcome=outcome)
    # A second backend may have hydrated the old snapshot through an ordinary viewer RPC.
    cached = b.call("cache", key=key)
    refusal = b.call("submit", sid=cached["session_id"], text="refused")
    assert refusal == {"refused": 4090, "not_admitted": True}
    terminal = a.call("terminal", sid=sid, text="first", stop=outcome == "stop")
    assert terminal["running"] is True
    receipt = terminal["receipts"][0]
    assert receipt["status"] == ("cancelled" if outcome == "stop" else "settled")
    if outcome == "stop":
        assert "artifacts" not in receipt
    else:
        assert receipt["artifacts"]["items"][0]["name"] == "receipt.txt"
    assert len(a.call("snapshot")["leases"]) == 1
    # Receipt publication alone is not permission for another writer.
    refusal = b.call("submit", sid=cached["session_id"], text="still-refused")
    assert refusal["refused"] == 4090
    settled = a.call("finish", sid=sid)
    assert settled["present"] is False, "completed hidden driver retains its exclusive session lease"
    assert settled["ended"] is None, "runtime retirement is not a durable session end"
    assert a.call("snapshot")["leases"] == []
    assert a.call("stale-flush", sid=sid) == {"flushed": False}
    second = b.call("resume", key=key)
    assert second["session_id"] != cached["session_id"], "stale agent must not survive reacquisition"
    owners = b.call("snapshot")["leases"]
    assert len(owners) == 1 and owners[0]["metadata"]["live_session_id"] == second["session_id"]
    if outcome == "complete":
        assert b.call("building-reservation", sid=second["session_id"]) == {"present": False}
        assert b.call("snapshot")["leases"] == []
        second = b.call("resume", key=key)
    b.call("submit", sid=second["session_id"], text="second")
    assert a.call("resume", key=key) == {"refused": 4090, "not_admitted": True}
    b.call("terminal", sid=second["session_id"])
    continued = b.call("finish", sid=second["session_id"])
    expected = "denied" if outcome == "timeout" else "stopped" if outcome == "stop" else "first reply"
    assert ["first", expected] in continued["observed"]
    assert [m["content"] for m in continued["rows"]] == ["first", expected, "second", "second reply"]
    # Return to the original process: it must also cold-resume B's durable continuation.
    third = a.call("resume", key=key)
    a.call("submit", sid=third["session_id"], text="third")
    a.call("terminal", sid=third["session_id"])
    returned = a.call("finish", sid=third["session_id"])
    assert ["first", expected, "second", "second reply"] in returned["observed"]
    assert len(returned["rows"]) == 6


@pytest.mark.parametrize("protection", ["viewer", "receipt", "persistence"])
def test_protected_owner_and_other_profile_stay_independent(runtimes, protection):
    a, b = runtimes
    first = a.call("create")
    sid, key = first["session_id"], first["key"]
    a.call("submit", sid=sid, text="visible", outcome="receipt-error" if protection == "receipt" else "complete")
    if protection == "viewer":
        a.call("attach", sid=sid)
    a.call("terminal", sid=sid)
    if protection == "persistence":
        a.call("persistence-fault", sid=sid)
    settled = a.call("finish", sid=sid)
    assert settled["present"] is True
    assert len(a.call("snapshot")["leases"]) == 1
    # Driver resume must refuse without pre-warming a stale agent in the contender.
    before = b.call("snapshot")["builds"]
    refusal = b.call("resume", key=key)
    assert refusal == {"refused": 4090, "not_admitted": True}
    assert b.call("snapshot")["builds"] == before
    # Same stored id in two profile databases: ownership is profile-local, not global.
    b.call("seed", profile="other", key=key)
    other = b.call("resume", profile="other", key=key)
    b.call("submit", profile="other", sid=other["session_id"], text="independent")
    b.call("terminal", profile="other", sid=other["session_id"])
    result = b.call("finish", profile="other", sid=other["session_id"])
    assert [m["content"] for m in result["rows"]] == ["independent", "independent reply"]
    assert len(a.call("snapshot")["leases"]) == 1
    if protection == "viewer":
        assert a.call("detach", sid=sid) == {"present": False}
        assert a.call("snapshot")["leases"] == []
        reopened = b.call("resume", key=key)
        assert b.call("retire-idle", sid=reopened["session_id"]) == {"present": False}
