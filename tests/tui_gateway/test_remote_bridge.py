from __future__ import annotations

import asyncio
import json
import threading
import types

import pytest

from tui_gateway import server
from tui_gateway import remote_bridge
from tui_gateway import ws as ws_module


class _RecordingTransport:
    def __init__(self, *, ok: bool = True) -> None:
        self.frames: list[dict] = []
        self.ok = ok
        self.closed = False

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return self.ok

    def close(self) -> None:
        self.closed = True


class _FakeWS:
    def __init__(self, *, headers=None, query_params=None) -> None:
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.close_codes: list[int] = []

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)


def _minimal_live_session(transport: _RecordingTransport | None = None, *, session_key: str = "session-key") -> dict:
    return {
        "agent": None,
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.Lock(),
        "last_active": 123.0,
        "running": True,
        "session_key": session_key,
        "transport": transport or _RecordingTransport(),
    }


def test_session_event_mirrors_to_remote_bridge_without_replacing_primary():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    remote = _RecordingTransport()
    try:
        server._sessions.clear()
        server._sessions["sid"] = {"transport": primary}

        assert server.attach_bridge_transport("sid", remote) is True
        assert server.write_json(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": "message.delta", "session_id": "sid"},
            }
        ) is True

        assert len(primary.frames) == 1
        assert len(remote.frames) == 1
        assert server._sessions["sid"]["transport"] is primary
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_attach_bridge_transport_is_copy_on_write():
    """Attaching must rebind a fresh list, never mutate the one in place.

    write_json / _session_event_transports iterate bridge_transports from the
    streaming thread; an in-place append would risk 'list changed size during
    iteration'. A reader holding an earlier snapshot must see it unchanged.
    """
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    first = _RecordingTransport()
    second = _RecordingTransport()
    try:
        server._sessions.clear()
        server._sessions["sid"] = {"transport": primary}

        server.attach_bridge_transport("sid", first)
        snapshot = server._sessions["sid"]["bridge_transports"]  # a reader's view
        assert snapshot == [first]

        server.attach_bridge_transport("sid", second)
        assert snapshot == [first]  # the snapshot the reader holds is untouched
        assert server._sessions["sid"]["bridge_transports"] is not snapshot
        assert server._sessions["sid"]["bridge_transports"] == [first, second]

        server.attach_bridge_transport("sid", second)  # re-attach is a no-op
        assert server._sessions["sid"]["bridge_transports"] == [first, second]
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_failed_remote_bridge_transport_is_pruned_but_primary_still_wins():
    previous = dict(server._sessions)
    primary = _RecordingTransport(ok=True)
    remote = _RecordingTransport(ok=False)
    try:
        server._sessions.clear()
        server._sessions["sid"] = {"transport": primary}
        server.attach_bridge_transport("sid", remote)

        assert server.write_json(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": "message.delta", "session_id": "sid"},
            }
        ) is True

        assert primary.frames
        assert remote.frames
        assert server._sessions["sid"].get("bridge_transports") == []
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_remote_prompt_submitted_event_mirrors_user_turn_to_peers_except_sender():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    remote_sender = _RecordingTransport()
    remote_peer = _RecordingTransport()
    try:
        server._sessions.clear()
        session = {"transport": primary, "bridge_transports": [remote_sender, remote_peer]}
        server._sessions["sid"] = session
        token = server.bind_transport(remote_sender)
        try:
            server._emit_prompt_submitted_to_peers(
                "sid",
                session,
                "hello from phone",
                {"client_id": "mobile-1", "source": "mobile"},
            )
        finally:
            server.reset_transport(token)

        assert remote_sender.frames == []
        assert primary.frames == remote_peer.frames
        frame = primary.frames[0]
        assert frame["method"] == "event"
        assert frame["params"] == {
            "type": "prompt.submitted",
            "session_id": "sid",
            "payload": {"text": "hello from phone", "client_id": "mobile-1", "source": "mobile"},
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_primary_prompt_submitted_event_mirrors_user_turn_to_remote_clients_only():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    remote = _RecordingTransport()
    try:
        server._sessions.clear()
        session = {"transport": primary, "bridge_transports": [remote]}
        server._sessions["sid"] = session
        token = server.bind_transport(primary)
        try:
            server._emit_prompt_submitted_to_peers("sid", session, "local tui prompt", {})
        finally:
            server.reset_transport(token)

        assert primary.frames == []
        frame = remote.frames[0]
        assert frame["method"] == "event"
        assert frame["params"] == {
            "type": "prompt.submitted",
            "session_id": "sid",
            "payload": {"text": "local tui prompt"},
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_session_activate_rehydrates_live_display_journal_as_if_mobile_had_been_connected():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    attached_mobile = _RecordingTransport()
    late_mobile = _RecordingTransport()
    try:
        server._sessions.clear()
        session = _minimal_live_session(primary)
        session["bridge_transports"] = [attached_mobile]
        session["history"] = [
            {"role": "user", "content": "canonical prompt"},
            {"role": "assistant", "content": "canonical answer"},
        ]
        server._sessions["sid"] = session

        server._emit_prompt_submitted_to_peers("sid", session, "local tui prompt", {})
        server._emit(
            "tool.start",
            "sid",
            {"name": "web_search", "context": "web_search(query=remote control)"},
        )
        server._emit(
            "tool.complete",
            "sid",
            {"name": "web_search", "summary": "Did 1 search"},
        )
        server._emit("status.update", "sid", {"kind": "goal", "text": "✓ Goal achieved"})
        server._emit("message.complete", "sid", {"status": "complete", "text": "final answer"})

        resp = server.dispatch(
            {"id": "activate", "method": "session.activate", "params": {"session_id": "sid"}},
            late_mobile,
        )

        assert resp is not None
        assert resp["result"]["messages"] == [
            {"role": "user", "text": "local tui prompt"},
            {"role": "tool", "name": "web_search", "text": "web_search(query=remote control)"},
            {"role": "tool", "name": "web_search", "text": "Did 1 search"},
            {"role": "event", "name": "goal", "text": "✓ Goal achieved"},
            {"role": "assistant", "text": "final answer"},
        ]
        assert resp["result"]["message_count"] == 5
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_mobile_prompt_submit_interrupts_running_turn_and_runs_after_current_turn():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    mobile = _RecordingTransport()

    class _FakeAgent:
        model = "fake/model"
        provider = "fake"
        base_url = ""
        session_id = "session-key"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.interrupts: list[str | None] = []
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self.release_second = threading.Event()

        def interrupt(self, message: str | None = None) -> None:
            self.interrupts.append(message)
            self.release_first.set()

        def run_conversation(self, user_message, conversation_history=None, stream_callback=None):
            self.calls.append(user_message)
            if len(self.calls) == 1:
                self.first_started.set()
                assert self.release_first.wait(1)
                return {
                    "final_response": "interrupted first turn",
                    "interrupted": True,
                    "messages": [{"role": "user", "content": "first prompt"}],
                }
            self.second_started.set()
            assert self.release_second.wait(1)
            return {
                "completed": True,
                "final_response": "second turn complete",
                "messages": [{"role": "user", "content": user_message}],
            }

    agent = _FakeAgent()
    session = {
        "agent": agent,
        "attached_images": [],
        "bridge_transports": [mobile],
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "last_active": 123.0,
        "running": False,
        "session_key": "session-key",
        "transport": primary,
    }

    try:
        server._sessions.clear()
        server._sessions["sid"] = session

        first = server.dispatch(
            {"id": "first", "method": "prompt.submit", "params": {"session_id": "sid", "text": "first prompt"}},
            primary,
        )
        assert first == {"jsonrpc": "2.0", "id": "first", "result": {"status": "streaming"}}
        assert agent.first_started.wait(1)

        second = server.dispatch(
            {
                "id": "second",
                "method": "prompt.submit",
                "params": {
                    "client_id": "mobile-1",
                    "on_busy": "interrupt",
                    "session_id": "sid",
                    "source": "mobile",
                    "text": "mobile followup",
                },
            },
            mobile,
        )

        assert second == {"jsonrpc": "2.0", "id": "second", "result": {"status": "interrupting"}}
        assert session["transport"] is primary
        assert agent.interrupts == ["mobile followup"]
        prompt_frames = [f for f in primary.frames if f.get("params", {}).get("type") == "prompt.submitted"]
        assert prompt_frames == [
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "prompt.submitted",
                    "session_id": "sid",
                    "payload": {"text": "mobile followup", "client_id": "mobile-1", "source": "mobile"},
                },
            }
        ]
        mobile_echoes = [
            f
            for f in mobile.frames
            if f.get("params", {}).get("type") == "prompt.submitted"
            and f.get("params", {}).get("payload", {}).get("text") == "mobile followup"
        ]
        assert mobile_echoes == []
        assert agent.second_started.wait(1)
        assert agent.calls[:2] == ["first prompt", "mobile followup"]
    finally:
        agent.release_first.set()
        agent.release_second.set()
        server._sessions.clear()
        server._sessions.update(previous)


def test_session_activate_attaches_current_transport_to_remote_created_session():
    previous = dict(server._sessions)
    mobile_primary = _RecordingTransport()
    local_tui = _RecordingTransport()
    try:
        server._sessions.clear()
        session = _minimal_live_session(mobile_primary, session_key="persisted-session")
        server._sessions["mobile-live"] = session

        resp = server.dispatch(
            {"id": "activate", "method": "session.activate", "params": {"session_id": "mobile-live"}},
            local_tui,
        )

        assert resp is not None
        assert resp["result"]["session_id"] == "mobile-live"
        assert local_tui in session.get("bridge_transports", [])

        token = server.bind_transport(local_tui)
        try:
            server._emit_prompt_submitted_to_peers("mobile-live", session, "hello from desktop", {})
        finally:
            server.reset_transport(token)

        assert local_tui.frames == []
        frame = mobile_primary.frames[0]
        assert frame["params"] == {
            "type": "prompt.submitted",
            "session_id": "mobile-live",
            "payload": {"text": "hello from desktop"},
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_session_resume_title_reuses_existing_live_session_and_attaches_current_transport(monkeypatch):
    previous = dict(server._sessions)
    mobile_primary = _RecordingTransport()
    local_tui = _RecordingTransport()

    class _FakeDB:
        def get_session(self, target):
            return None

        def get_session_by_title(self, target):
            return {"id": "persisted-session"} if target == "mobile-title" else None

    try:
        server._sessions.clear()
        session = _minimal_live_session(mobile_primary, session_key="persisted-session")
        server._sessions["mobile-live"] = session
        monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())

        token = server.bind_transport(local_tui)
        try:
            resp = server.handle_request(
                {"id": "resume", "method": "session.resume", "params": {"session_id": "mobile-title"}}
            )
        finally:
            server.reset_transport(token)

        assert resp is not None
        assert resp["result"]["session_id"] == "mobile-live"
        assert resp["result"]["resumed"] == "persisted-session"
        assert resp["result"]["live"] is True
        assert list(server._sessions) == ["mobile-live"]
        assert local_tui in session.get("bridge_transports", [])

        token = server.bind_transport(local_tui)
        try:
            server._emit_prompt_submitted_to_peers("mobile-live", session, "hello after resume", {})
        finally:
            server.reset_transport(token)

        assert local_tui.frames == []
        frame = mobile_primary.frames[0]
        assert frame["params"] == {
            "type": "prompt.submitted",
            "session_id": "mobile-live",
            "payload": {"text": "hello after resume"},
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_prompt_resolved_event_mirrors_prompt_answer_to_peers_except_responder():
    previous = dict(server._sessions)
    primary = _RecordingTransport()
    remote_sender = _RecordingTransport()
    remote_peer = _RecordingTransport()
    try:
        server._sessions.clear()
        session = {"transport": primary, "bridge_transports": [remote_sender, remote_peer]}
        server._sessions["sid"] = session
        token = server.bind_transport(remote_sender)
        try:
            server._emit_prompt_resolved_to_peers(
                "sid",
                session,
                "clarify",
                {"answer": "do not mirror this", "client_id": "mobile-1", "source": "mobile"},
                request_id="rid-1",
            )
        finally:
            server.reset_transport(token)

        assert remote_sender.frames == []
        assert primary.frames == remote_peer.frames
        frame = primary.frames[0]
        assert frame["method"] == "event"
        assert frame["params"] == {
            "type": "prompt.resolved",
            "session_id": "sid",
            "payload": {
                "kind": "clarify",
                "resolved": 1,
                "request_id": "rid-1",
                "client_id": "mobile-1",
                "source": "mobile",
            },
        }
        assert "answer" not in frame["params"]["payload"]
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


def test_clarify_respond_mirrors_resolution_to_peer_clients():
    previous_sessions = dict(server._sessions)
    previous_pending = dict(server._pending)
    previous_answers = dict(server._answers)
    primary = _RecordingTransport()
    remote_sender = _RecordingTransport()
    remote_peer = _RecordingTransport()
    ev = threading.Event()
    try:
        server._sessions.clear()
        server._pending.clear()
        server._answers.clear()
        server._sessions["sid"] = {"transport": primary, "bridge_transports": [remote_sender, remote_peer]}
        server._pending["rid-1"] = ("sid", ev)

        resp = server.dispatch(
            {
                "id": "respond",
                "method": "clarify.respond",
                "params": {
                    "answer": "do not mirror this",
                    "client_id": "mobile-1",
                    "request_id": "rid-1",
                    "source": "mobile",
                },
            },
            remote_sender,
        )

        assert resp == {"jsonrpc": "2.0", "id": "respond", "result": {"status": "ok"}}
        assert ev.is_set()
        assert server._answers["rid-1"] == "do not mirror this"
        assert remote_sender.frames == []
        assert primary.frames == remote_peer.frames
        frame = primary.frames[0]
        assert frame["params"]["type"] == "prompt.resolved"
        assert frame["params"]["payload"] == {
            "kind": "clarify",
            "resolved": 1,
            "request_id": "rid-1",
            "client_id": "mobile-1",
            "source": "mobile",
        }
        assert "answer" not in frame["params"]["payload"]
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)
        server._pending.clear()
        server._pending.update(previous_pending)
        server._answers.clear()
        server._answers.update(previous_answers)


def test_approval_respond_mirrors_resolution_to_remote_clients(monkeypatch):
    from tools import approval

    previous = dict(server._sessions)
    primary = _RecordingTransport()
    remote = _RecordingTransport()
    try:
        server._sessions.clear()
        server._sessions["sid"] = {
            "session_key": "sess-key",
            "transport": primary,
            "bridge_transports": [remote],
        }
        monkeypatch.setattr(approval, "resolve_gateway_approval", lambda key, choice, resolve_all=False: 1)
        token = server.bind_transport(primary)
        try:
            resp = server.handle_request(
                {
                    "id": "approve",
                    "method": "approval.respond",
                    "params": {"choice": "once", "session_id": "sid", "source": "tui"},
                }
            )
        finally:
            server.reset_transport(token)

        assert resp == {"jsonrpc": "2.0", "id": "approve", "result": {"resolved": 1}}
        assert primary.frames == []
        frame = remote.frames[0]
        assert frame["params"] == {
            "type": "prompt.resolved",
            "session_id": "sid",
            "payload": {"kind": "approval", "resolved": 1, "choice": "once", "source": "tui"},
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous)


@pytest.mark.parametrize(
    ("event", "payload"),
    [
        ("clarify.request", {"choices": ["yes", "no"], "question": "Proceed?", "request_id": "rid-clarify"}),
        ("sudo.request", {"request_id": "rid-sudo"}),
        ("secret.request", {"env_var": "API_KEY", "prompt": "Enter API key", "request_id": "rid-secret"}),
    ],
)
def test_session_activate_includes_pending_blocking_prompt_for_late_attach(event, payload):
    previous_sessions = dict(server._sessions)
    previous_pending = dict(server._pending)
    previous_prompt_payloads = dict(server._pending_prompt_payloads)
    try:
        server._sessions.clear()
        server._pending.clear()
        server._pending_prompt_payloads.clear()
        server._sessions["sid"] = _minimal_live_session()
        request_id = str(payload["request_id"])
        server._pending[request_id] = ("sid", threading.Event())
        server._pending_prompt_payloads[request_id] = (event, dict(payload))

        resp = server.dispatch(
            {"id": "activate", "method": "session.activate", "params": {"session_id": "sid"}},
            _RecordingTransport(),
        )

        assert resp is not None
        assert resp["result"]["status"] == "waiting"
        assert resp["result"]["pending_prompt"] == {"type": event, "payload": payload}
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)
        server._pending.clear()
        server._pending.update(previous_pending)
        server._pending_prompt_payloads.clear()
        server._pending_prompt_payloads.update(previous_prompt_payloads)


def test_session_activate_includes_pending_approval_for_late_attach():
    from tools import approval

    previous_sessions = dict(server._sessions)
    previous_queues = {key: list(value) for key, value in approval._gateway_queues.items()}
    approval_payload = {
        "command": "rm -rf /tmp/nope",
        "description": "dangerous command",
        "pattern_key": "rm-rf",
        "pattern_keys": ["rm-rf"],
    }
    try:
        server._sessions.clear()
        approval._gateway_queues.clear()
        server._sessions["sid"] = _minimal_live_session(session_key="session-key")
        approval._gateway_queues["session-key"] = [approval._ApprovalEntry(dict(approval_payload))]

        resp = server.dispatch(
            {"id": "activate", "method": "session.activate", "params": {"session_id": "sid"}},
            _RecordingTransport(),
        )

        assert resp is not None
        assert resp["result"]["status"] == "waiting"
        assert resp["result"]["pending_prompt"] == {
            "type": "approval.request",
            "payload": approval_payload,
        }
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)
        approval._gateway_queues.clear()
        approval._gateway_queues.update(previous_queues)


def test_remote_bridge_config_env_enables_and_overrides():
    cfg = remote_bridge.resolve_remote_bridge_config(
        cfg={"tui_remote_bridge": {"enabled": False, "port": 1111}},
        environ={
            "HERMES_TUI_REMOTE_BRIDGE": "1",
            "HERMES_TUI_REMOTE_BRIDGE_HOST": "0.0.0.0",
            "HERMES_TUI_REMOTE_BRIDGE_PORT": "9999",
            "HERMES_TUI_REMOTE_BRIDGE_TOKEN": "secret",
            "HERMES_TUI_REMOTE_BRIDGE_ORIGINS": "https://mobile.example,http://localhost:5174/",
        },
    )

    assert cfg.enabled is True
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9999
    assert cfg.token == "secret"
    assert cfg.trusted_origins == ("https://mobile.example", "http://localhost:5174")
    cfg.validate()


def test_remote_bridge_refuses_non_loopback_without_token():
    cfg = remote_bridge.RemoteBridgeConfig(enabled=True, host="0.0.0.0", token="")

    with pytest.raises(remote_bridge.RemoteBridgeConfigError, match="token is required"):
        cfg.validate()


def test_remote_bridge_host_guard_blocks_rebinding_on_loopback():
    assert remote_bridge.is_accepted_host("localhost:8769", "127.0.0.1")
    assert remote_bridge.is_accepted_host("[::1]:8769", "::1")
    assert not remote_bridge.is_accepted_host("evil.example", "127.0.0.1")
    assert not remote_bridge.is_accepted_host("127.0.0.1.evil.example", "127.0.0.1")


def test_remote_bridge_origin_guard_accepts_native_or_same_origin_only():
    assert remote_bridge.is_accepted_origin(
        "",
        bound_host="127.0.0.1",
        host_header="localhost:8769",
    )
    assert remote_bridge.is_accepted_origin(
        "http://localhost:5174",
        bound_host="127.0.0.1",
        host_header="localhost:8769",
    )
    assert remote_bridge.is_accepted_origin(
        "https://mobile.example",
        bound_host="0.0.0.0",
        host_header="tailnet-host:8769",
        trusted_origins=("https://mobile.example",),
    )
    assert not remote_bridge.is_accepted_origin(
        "http://evil.example",
        bound_host="127.0.0.1",
        host_header="localhost:8769",
    )


def test_remote_bridge_authorize_ws_requires_configured_token():
    cfg = remote_bridge.RemoteBridgeConfig(
        enabled=True,
        host="127.0.0.1",
        token="secret",
    )

    ok = _FakeWS(
        headers={"host": "localhost:8769", "authorization": "Bearer secret"},
    )
    bad = _FakeWS(
        headers={"host": "localhost:8769", "authorization": "Bearer wrong"},
    )

    assert asyncio.run(remote_bridge.authorize_ws(ok, cfg)) is True
    assert ok.close_codes == []
    assert asyncio.run(remote_bridge.authorize_ws(bad, cfg)) is False
    assert bad.close_codes == [4401]


def test_start_remote_bridge_uses_daemon_thread_without_real_uvicorn(monkeypatch):
    class _Config:
        def __init__(self, app, host, port, log_level, lifespan):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level
            self.lifespan = lifespan

    class _Server:
        def __init__(self, config):
            self.config = config
            self.ran = threading.Event()
            self.should_exit = False

        def run(self):
            self.ran.set()

    fake_uvicorn = types.SimpleNamespace(Config=_Config, Server=_Server)
    monkeypatch.setattr(remote_bridge, "_ensure_server_deps", lambda: fake_uvicorn)
    monkeypatch.setattr(remote_bridge, "build_app", lambda config: {"path": config.path})

    handle = remote_bridge.start_remote_bridge(
        remote_bridge.RemoteBridgeConfig(enabled=True, host="127.0.0.1", port=9876)
    )

    assert handle is not None
    assert handle.thread.daemon is True
    assert handle.thread.name == "hermes-tui-remote-bridge"
    assert handle.server.ran.wait(1)
    assert handle.public_info()["url"] == "ws://127.0.0.1:9876/api/tui/ws"
    handle.stop()
    assert handle.server.should_exit is True


class _ScriptedWS:
    """Minimal async WebSocket double: yields scripted requests, then disconnects."""

    def __init__(self, requests: list[dict]) -> None:
        self._queue = [json.dumps(r) for r in requests]
        self.sent: list[dict] = []

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if self._queue:
            return self._queue.pop(0)
        raise ws_module._WebSocketDisconnect()

    async def send_text(self, line: str) -> None:
        self.sent.append(json.loads(line))

    async def close(self, code: int = 1000) -> None:
        pass


def test_dashboard_ws_default_does_not_apply_remote_bridge_allowlist(monkeypatch):
    """Shared dashboard /api/ws keeps the full authenticated gateway surface."""
    dispatched: list[str] = []

    def _fake_dispatch(req, transport=None):
        dispatched.append(req.get("method"))
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"ok": True}}

    monkeypatch.setattr(server, "dispatch", _fake_dispatch)

    dashboard = _ScriptedWS(
        [{"jsonrpc": "2.0", "id": 1, "method": "slash.exec", "params": {}}]
    )
    asyncio.run(ws_module.handle_ws(dashboard))

    assert dispatched == ["slash.exec"]
    assert not [
        m
        for m in dashboard.sent
        if isinstance(m.get("error"), dict) and m["error"].get("code") == 4403
    ]


def test_remote_bridge_ws_passes_bridge_allowlist(monkeypatch):
    seen: dict[str, object] = {}

    async def _allow(_ws, _config):
        return True

    async def _handle_ws(_ws, *, allowed_methods=None):
        seen["allowed_methods"] = allowed_methods

    monkeypatch.setattr(remote_bridge, "authorize_ws", _allow)
    monkeypatch.setattr(ws_module, "handle_ws", _handle_ws)

    cfg = remote_bridge.RemoteBridgeConfig(
        enabled=True,
        host="127.0.0.1",
        port=8769,
    )
    asyncio.run(remote_bridge.handle_remote_ws(object(), cfg))

    assert seen["allowed_methods"] is ws_module.BRIDGE_ALLOWED_METHODS


def test_remote_bridge_allowlist_blocks_dangerous_methods_but_forwards_allowed(monkeypatch):
    """Only allowlisted methods reach dispatch over the remote bridge."""
    assert "prompt.submit" in ws_module.BRIDGE_ALLOWED_METHODS
    assert "image.attach" in ws_module.BRIDGE_ALLOWED_METHODS
    assert "image.detach" in ws_module.BRIDGE_ALLOWED_METHODS
    assert "session.close" in ws_module.BRIDGE_ALLOWED_METHODS
    assert "shell.exec" not in ws_module.BRIDGE_ALLOWED_METHODS
    assert "config.set" not in ws_module.BRIDGE_ALLOWED_METHODS
    assert "slash.exec" not in ws_module.BRIDGE_ALLOWED_METHODS

    dispatched: list[str] = []

    def _fake_dispatch(req, transport=None):
        dispatched.append(req.get("method"))
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"ok": True}}

    monkeypatch.setattr(server, "dispatch", _fake_dispatch)

    blocked = _ScriptedWS(
        [{"jsonrpc": "2.0", "id": 1, "method": "shell.exec", "params": {}}]
    )
    asyncio.run(
        ws_module.handle_ws(blocked, allowed_methods=ws_module.BRIDGE_ALLOWED_METHODS)
    )
    assert "shell.exec" not in dispatched
    rejections = [
        m
        for m in blocked.sent
        if isinstance(m.get("error"), dict) and m["error"].get("code") == 4403
    ]
    assert rejections and rejections[0]["id"] == 1

    allowed = _ScriptedWS(
        [
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "image.attach",
                "params": {"session_id": "x", "path": "/tmp/cat.png"},
            }
        ]
    )
    asyncio.run(
        ws_module.handle_ws(allowed, allowed_methods=ws_module.BRIDGE_ALLOWED_METHODS)
    )
    assert dispatched == ["image.attach"]
