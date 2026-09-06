"""Host identity survives real registry dispatch, MCP loop threading and SDK serialization."""

import asyncio
import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools import mcp_tool as core
from tools.mcp_tool_handlers import _make_tool_handler
from tools.mcp_tool_discovery import _select_new_servers
from tools.registry import ToolRegistry

PREFIX = "com.nousresearch.hermes/"


@pytest.fixture
def wire_dispatch(monkeypatch):
    from mcp import ClientSession, types

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert ready.wait(5)
    captured = []
    session = object.__new__(ClientSession)
    session._call_tool_adapter = None
    session._tool_output_schemas = {"inspect": None}

    async def send_request(request, *_args, **_kwargs):
        assert threading.get_ident() == thread.ident
        wire = request.model_dump(by_alias=True, exclude_none=True)
        captured.append(wire)
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    session.send_request = send_request
    server = core.MCPServerTask("trusted-edge")
    server.session = session
    monkeypatch.setattr(core, "_mcp_loop", loop)
    monkeypatch.setattr(core, "_servers", {"trusted-edge": server})
    monkeypatch.setattr(core, "_session_context_forwarding_servers", set())
    monkeypatch.setattr(core, "_server_error_counts", {})
    registry = ToolRegistry()
    registry.register("inspect", "test", {"name": "inspect", "parameters": {"type": "object"}},
                      _make_tool_handler("trusted-edge", "inspect", 5))
    yield registry, captured
    reset_session_vars()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    assert not thread.is_alive()
    loop.close()


@pytest.mark.parametrize("redact_pii", [False, True])
def test_concurrent_gateway_identity_is_snapshot_outside_arguments(wire_dispatch, monkeypatch, redact_pii):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.run_session_metadata import bind_session_context_for_turn
    from gateway.session import SessionContext, SessionSource, _hash_sender_id
    from hermes_constants import get_hermes_home
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", get_hermes_home())
    (get_hermes_home() / "config.yaml").write_text(
        f"privacy:\n  redact_pii: {str(redact_pii).lower()}\n", encoding="utf-8")
    registry, captured = wire_dispatch
    _select_new_servers({"trusted-edge": {"forward_session_context": True}})
    barrier = threading.Barrier(2)

    def invoke(user):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        source = SessionSource(platform=Platform.TELEGRAM, user_id=user, chat_id="chat-" + user,
                               thread_id="thread-" + user, message_id="message-" + user)
        context = SessionContext(source=source, connected_platforms=[], home_channels={},
                                 session_id="session-" + user, session_key="key-" + user)
        tokens, policy = bind_session_context_for_turn(runner, context)
        assert policy is redact_pii
        barrier.wait(5)
        try:
            result = registry.dispatch("inspect", {"user_id": "model-forgery", "_meta": {PREFIX + "user_id": "forged"}})
            assert json.loads(result)["result"] == "ok"
        finally:
            clear_session_vars(tokens)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(invoke, ["alice", "bob"]))
    assert len(captured) == 2
    for request in captured:
        params = request["params"]
        meta = params["_meta"]
        user = meta[PREFIX + "session_id"].removeprefix("session-")
        assert meta[PREFIX + "user_id"] == (_hash_sender_id(user) if redact_pii else user)
        assert params["arguments"]["user_id"] == "model-forgery"
        assert params["arguments"]["_meta"][PREFIX + "user_id"] == "forged"
        assert set(meta) == {PREFIX + k for k in ("platform", "user_id", "chat_id", "thread_id", "session_id", "session_key", "message_id")}
        if redact_pii:
            assert meta[PREFIX + "session_key"] != "key-" + user
            assert meta[PREFIX + "message_id"] != "message-" + user


@pytest.mark.parametrize("state", ["unbound", "empty", "unknown-policy", "disabled", "name-collision", "valid", "retry",
                                  "incomplete", "invalid-value", "redaction-error", "missing-variable",
                                  "shutdown-empty", "shutdown-scoped"])
def test_only_explicit_bound_policy_can_disclose_metadata(wire_dispatch, monkeypatch, state):
    registry, captured = wire_dispatch
    _select_new_servers({"trusted-edge": {"forward_session_context": state != "disabled"}})
    if state.startswith("shutdown-"):
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        from tools import mcp_tool_loop

        servers = core._servers
        monkeypatch.setattr(core, "_servers", {})
        monkeypatch.setattr(core, "_server_scope_keys", {"trusted-edge": "a", "other": "b"})
        monkeypatch.setattr(mcp_tool_loop, "_stop_mcp_loop", lambda **kwargs: None)
        core._session_context_forwarding_servers.add("other")
        shutdown_mcp_servers(scope="a" if state == "shutdown-scoped" else None)
        assert ("other" in core._session_context_forwarding_servers) is (state == "shutdown-scoped")
        monkeypatch.setattr(core, "_servers", servers)
    if state == "retry":
        from tools import mcp_tool_handlers
        session = core._servers["trusted-edge"].session
        original_send = session.send_request

        async def fail_first(request, *args, **kwargs):
            result = await original_send(request, *args, **kwargs)
            if len(captured) == 1:
                raise RuntimeError("Transport retry required")
            return result

        def recover(_server, _exc, retry_call, _op):
            set_session_vars(platform="telegram", user_id="later-caller", session_id="later", redact_pii=False)
            return retry_call()

        monkeypatch.setattr(session, "send_request", fail_first)
        monkeypatch.setattr(mcp_tool_handlers, "_handle_stdio_child_exited_and_retry", recover)
    for name in ("PLATFORM", "USER_ID", "CHAT_ID", "THREAD_ID", "SESSION_ID", "SESSION_KEY", "MESSAGE_ID"):
        monkeypatch.setenv("HERMES_SESSION_" + name, "stale-ambient")

    def invoke():
        if state != "unbound":
            set_session_vars(platform="telegram", user_id="alice", chat_id="chat", session_id="session",
                             session_key="key", message_id="message", redact_pii=False)
        if state == "empty":
            clear_session_vars([])
        if state == "unknown-policy":
            set_session_vars(platform="telegram", user_id="alice", redact_pii=None)
        if state in {"incomplete", "invalid-value", "missing-variable"}:
            from gateway.session_context import _VAR_MAP
            if state == "missing-variable":
                monkeypatch.delitem(_VAR_MAP, "HERMES_SESSION_MESSAGE_ID")
            else:
                _VAR_MAP["HERMES_SESSION_USER_ID"].set("" if state == "incomplete" else ["alice"])
        if state == "redaction-error":
            import gateway.session
            from gateway.session_context import _SESSION_REDACT_PII
            _SESSION_REDACT_PII.set(True)

            def broken_hash(value):
                raise RuntimeError("Redaction unavailable")

            monkeypatch.setattr(gateway.session, "_hash_chat_id", broken_hash)
        if state == "name-collision":
            # The opt-in is owned by the exact raw server, not its sanitized spelling.
            _select_new_servers({"trusted-edge": {}, "trusted_edge": {"forward_session_context": True}})
        assert json.loads(registry.dispatch("inspect", {}))["result"] == "ok"

    contextvars.Context().run(invoke)
    assert ("_meta" in captured[0]["params"]) is (state in {"valid", "retry"})
    if state == "retry":
        assert len(captured) == 2
        assert captured[0]["params"]["_meta"] == captured[1]["params"]["_meta"]
        assert captured[1]["params"]["_meta"][PREFIX + "user_id"] == "alice"
