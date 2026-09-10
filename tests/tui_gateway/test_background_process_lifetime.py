"""Automatic session cleanup must wait for owned terminal work and its handoff."""

import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tools.process_registry import process_registry


class _Timer:
    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback

    def start(self):
        pass

    def cancel(self):
        pass


def _child_command(directory):
    program = (
        "import pathlib,time; "
        f"p=pathlib.Path({str(directory)!r}); "
        "p.joinpath('started').touch(); "
        "deadline=time.monotonic()+30\n"
        "while not p.joinpath('release').exists() and time.monotonic()<deadline:\n"
        " p.joinpath('progress').write_text(str(time.monotonic()),encoding='utf-8'); time.sleep(.02)\n"
        "print('BACKGROUND_RESULT',flush=True)"
    )
    return shlex.join([sys.executable, "-c", program])


def _wait_for(predicate, label, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError(label() if callable(label) else label)


@pytest.mark.parametrize("reaper", ["orphan", "lru", "ttl", "explicit"])
@pytest.mark.parametrize("ownership", ["owned", "foreign", "transferred"])
@pytest.mark.parametrize("notify", [False, True])
def test_automatic_reap_waits_for_owned_terminal_work(tmp_path, monkeypatch, reaper, ownership, notify):
    from agent.client_lifecycle import ClientLifecycleMixin
    import tools.process_registry as registry_module

    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(sys.modules[__name__], "process_registry", registry)
    kill_receipts = []
    kill = registry.kill_process

    def record_kill(*args, **kwargs):
        result = kill(*args, **kwargs)
        kill_receipts.append(result)
        return result

    monkeypatch.setattr(registry, "kill_process", record_kill)

    sid = "background-owner"
    agent = SimpleNamespace(session_id=sid, _process_owner_task_ids={"owner-turn"})
    agent.close = lambda: ClientLifecycleMixin._close_task_resources(agent, sid)
    session = dict(agent=agent, session_key=sid, history=[], history_lock=threading.Lock(),
                   running=False, transport=server._detached_ws_transport, source="desktop",
                   created_at=0, last_active=0)
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    child = process_registry.spawn_local(
        _child_command(tmp_path), cwd=str(tmp_path), task_id="default", session_key=sid,
        owner_task_id="owner-turn" if ownership == "owned" else "other-turn")
    child.notify_on_complete = notify
    exit_observed, publish_exit = threading.Event(), threading.Event()
    move_to_finished = process_registry._move_to_finished

    def finish(process):
        if process is child and notify and ownership != "foreign" and reaper != "explicit":
            exit_observed.set()
            assert publish_exit.wait(10), "test did not release completion publication"
        move_to_finished(process)

    monkeypatch.setattr(process_registry, "_move_to_finished", finish)
    try:
        _wait_for(lambda: (tmp_path / "started").exists(), "child did not start")
        if ownership == "transferred":
            assert process_registry.transfer_ownership(
                child.id, from_owner="other-turn", to_owner="owner-turn",
                to_task_id="default", to_session_key=sid) is child

        def reap():
            if reaper == "explicit":
                server._close_session_by_id(sid)
            elif reaper == "orphan":
                server._schedule_ws_orphan_reap(sid)
                server._pending_ws_reaps[sid].callback()
            else:
                predicate = (lambda s: server._session_is_lru_evictable(sid, s)) if reaper == "lru" else (
                    lambda s: server._session_is_evictable(sid, s, time.time()))
                server._close_session_by_id(sid, end_reason="lru_evict" if reaper == "lru" else "idle_timeout",
                                            predicate=predicate)

        reap()
        assert (sid in server._sessions) is (ownership != "foreign" and reaper != "explicit")
        if reaper == "explicit" and ownership != "foreign":
            assert child._completion_event.wait(10), kill_receipts
            assert child.termination_source == "agent_close"
            assert child.completion_reason == "killed"
            return
        assert not child.exited, "automatic cleanup killed a live terminal job"
        if ownership == "owned" and reaper == "orphan" and not notify:
            from tools import process_registry_lifecycle

            def unavailable(*args):
                raise RuntimeError("temporary ownership lookup failure")

            with monkeypatch.context() as transient:
                transient.setattr(process_registry_lifecycle, "has_owned_work", unavailable)
                reap()
                assert sid in server._sessions, "indeterminate ownership authorized cleanup"
        (tmp_path / "release").touch()
        if notify and ownership != "foreign":
            assert exit_observed.wait(10), "reader did not observe process exit"
            reap()
            assert sid in server._sessions, "cleanup crossed exit-to-publication boundary"
            publish_exit.set()
        assert child._completion_event.wait(10), "child did not finish"
        assert child.exit_code == 0
        assert "BACKGROUND_RESULT" in child.output_buffer
        if ownership != "foreign":
            if notify:
                from tools.process_registry_notifications import format_process_notification

                # A read-only poll must not suppress or acknowledge the continuation.
                assert process_registry.poll(child.id)["status"] == "exited"
                reap()
                assert sid in server._sessions, "pending notification lost its owner"
                events = process_registry.drain_notifications(owns_event=lambda e: e.get("session_id") == child.id,
                                                              skip_poll_observed=False)
                ready = [event for event, _ in events]
                assert len(ready) == 1
                admitted = []

                def submit(*args, **kwargs):
                    session["running"] = False
                    admitted.append(args[3])
                    return len(admitted) > 1

                monkeypatch.setattr(server, "_run_prompt_submit", submit)
                deferred = []
                server._notif_handle_ready(sid, session, ready, set(), process_registry,
                                          format_process_notification, deferred)
                assert deferred == ready, "refused admission lost the completion"
                reap()
                assert sid in server._sessions
                server._notif_handle_ready(sid, session, deferred, set(), process_registry,
                                          format_process_notification, [])
                assert len(admitted) == 2
            reap()
            assert sid not in server._sessions, "completed work made the session immortal"
    finally:
        publish_exit.set()
        server._cancel_ws_orphan_reap(sid)
        server._close_session_by_id(sid)
        process_registry.kill_process(child.id, source="test", consume_output=True)


def _offline_agent_factory(directory, use_pty):
    from openai.types.chat import ChatCompletion
    from run_agent import AIAgent

    def make(sid, key, **kwargs):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=6,
            enabled_toolsets=["terminal"], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, session_id=key,
            session_db=kwargs.get("session_db") or server._get_db(), platform="desktop")
        agent._disable_streaming = True
        calls = []

        def respond(api_kwargs, **unused):
            calls.append(api_kwargs)
            pending = Path(directory, "provider-calls.tmp")
            pending.write_text(json.dumps(calls, default=str), encoding="utf-8")
            pending.replace(Path(directory, "provider-calls.json"))
            message = {"role": "assistant", "content": "Launched background work"}
            if len(calls) == 1:
                message.update(content=None, tool_calls=[{
                    "id": "launch-job", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({
                        "command": _child_command(directory), "background": True, "notify": True,
                        "pty": use_pty})}}])
            elif len(calls) >= 3:
                message["content"] = "Observed BACKGROUND_RESULT"
            return ChatCompletion.model_validate({
                "id": f"response-{len(calls)}", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if len(calls) == 1 else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

        agent._interruptible_api_call = respond
        agent._interruptible_streaming_api_call = respond
        return agent
    return make


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("use_pty", [False, pytest.param(True, marks=pytest.mark.macos_only, id="macos-pty"),
                                   pytest.param(True, marks=pytest.mark.linux_only, id="linux-pty")])
def test_real_websocket_background_completion_survives_disconnect(tmp_path, monkeypatch, isolated, use_pty):
    import socket
    from fastapi import FastAPI, WebSocket
    import uvicorn
    from websockets.sync.client import connect
    from tui_gateway.host_supervisor import HostSupervisor
    from tui_gateway.ws import handle_ws
    from hermes_state import SessionDB

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("approvals:\n  mode: 'off'\nterminal:\n  env_type: local\n", encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", .2)
    monkeypatch.setattr(server, "_make_agent", _offline_agent_factory(tmp_path, use_pty))
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": isolated})
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    supervisor = None
    host_frames = []
    if isolated:
        # Keep the serving session lazy so this case exercises the child-owned
        # path instead of racing the optional in-process agent pre-warm.
        monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
        supervisor = HostSupervisor(
            argv=[sys.executable, str(Path(__file__).resolve()), "host", str(tmp_path), str(int(use_pty))],
            registry_path=tmp_path / "host.json", env={"HERMES_HOME": os.environ["HERMES_HOME"]},
            expected_hermes_home=os.environ["HERMES_HOME"], rpc_sink=server._relay_compute_host_rpc,
            heartbeat_secs=.1, autostart=False)
        monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)
        monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a: supervisor)
        handle_frame = supervisor._handle_host_frame

        def observe_frame(frame):
            if frame.get("type") in {"turn.end", "turn.error"}:
                host_frames.append(frame)
            return handle_frame(frame)

        monkeypatch.setattr(supervisor, "_handle_host_frame", observe_frame)
    app = FastAPI()

    @app.websocket("/api/ws")
    async def endpoint(ws: WebSocket):
        await handle_ws(ws)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"ws://127.0.0.1:{sock.getsockname()[1]}/api/ws"
    uv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", loop="asyncio"))
    worker = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    sequence = 0

    def rpc(ws, method, params):
        nonlocal sequence
        sequence += 1
        ws.send(json.dumps({"jsonrpc": "2.0", "id": sequence, "method": method, "params": params}))
        while True:
            for line in ws.recv(timeout=15).splitlines():
                reply = json.loads(line)
                if reply.get("id") == sequence:
                    assert "error" not in reply, reply
                    return reply["result"]

    try:
        _wait_for(lambda: uv.started, "WebSocket server did not start")
        with connect(url, max_queue=None) as ws:
            created = rpc(ws, "session.create", {"source": "desktop", "cwd": str(tmp_path)})
            sid = created["session_id"]
            session = server._sessions[sid]
            submitted = rpc(ws, "prompt.submit", {"session_id": sid, "text": "Run the finite background job"})
            assert bool(submitted.get("turn_isolation")) is isolated
            _wait_for(lambda: (tmp_path / "started").exists() and not session["running"],
                      lambda: {"error": "foreground did not finish after launching the real terminal child",
                               "running": session["running"], "history": session["history"],
                               "host_stderr": supervisor._stderr_tail if supervisor else [],
                               "host_frames": host_frames,
                               "calls": (tmp_path / "provider-calls.json").read_text(encoding="utf-8")
                               if (tmp_path / "provider-calls.json").exists() else "none"}, timeout=20)
            stored_id = session["session_key"]
            if isolated:
                assert session["agent"] is None
                assert supervisor.is_running()
        _wait_for(lambda: server._ws_session_is_detached(session), "WebSocket did not detach")
        progress = (tmp_path / "progress").read_text(encoding="utf-8")
        # Long enough to cross ten shortened grace periods, using the actual timers.
        threading.Event().wait(2)
        assert sid in server._sessions, "orphan timer reclaimed the owner of the running job"
        assert (tmp_path / "progress").read_text(encoding="utf-8") != progress, "background child stopped making progress"
        if isolated:
            assert supervisor.has_background_work(sid, session["_compute_host_session_token"])
            assert not supervisor.has_background_work(sid, "foreign-session-generation")
            with monkeypatch.context() as observation:
                def no_startup(*args):
                    raise AssertionError("cleanup attempted host startup/configuration")
                observation.setattr(server, "_get_compute_host_supervisor", no_startup)
                assert server._compute_host_has_background_work(sid, session)
        with connect(url, max_queue=None) as ws:
            resumed = rpc(ws, "session.resume", {"session_id": stored_id})
            assert resumed["session_id"] == sid
            assert server._sessions[sid] is session
        _wait_for(lambda: server._ws_session_is_detached(session), "reconnected client did not detach")
        (tmp_path / "release").touch()
        _wait_for(lambda: len(json.loads((tmp_path / "provider-calls.json").read_text(encoding="utf-8"))) >= 3,
                  "completion did not reach a continuation turn")
        _wait_for(lambda: "Observed BACKGROUND_RESULT" in json.dumps(
            server._get_db().get_messages_as_conversation(stored_id)), "completion was not persisted")
        _wait_for(lambda: sid not in server._sessions, "settled completion retained an idle session forever")
        with connect(url) as ws:
            rpc(ws, "session.resume", {"session_id": stored_id})
            history = server._get_db().get_messages_as_conversation(stored_id)
            assert sum(m.get("content") == "Observed BACKGROUND_RESULT" for m in history) == 1
        calls = json.loads((tmp_path / "provider-calls.json").read_text(encoding="utf-8"))
        assert len(calls) == 3
        assert calls[0]["messages"][0] == calls[2]["messages"][0], "completion rebuilt the cached system prefix"
    finally:
        (tmp_path / "release").touch()
        for sid in list(server._sessions):
            server._cancel_ws_orphan_reap(sid)
            server._close_session_by_id(sid)
        for stop, poller in server._notification_pollers:
            stop.set()
            poller.join(5)
        if supervisor is not None:
            supervisor.shutdown()
        uv.should_exit = True
        worker.join(5)
        sock.close()
        db.close()


if __name__ == "__main__":
    from tui_gateway.compute_host import run_host

    server._make_agent = _offline_agent_factory(Path(sys.argv[2]), bool(int(sys.argv[3])))
    server._sync_agent_model_with_config = lambda *a: None
    # The real serving RPC already holds the lease. Current main tries to
    # acquire it again in the child and refuses the turn before any model call.
    # Model this parent-authorized admission here; this test does NOT certify
    # the separate cross-process lease protocol. All job/notification/teardown
    # behavior and the host pipes below are production code.
    server._ensure_active_session_slot = lambda *a: None
    run_host(stdout=sys.__stdout__)
