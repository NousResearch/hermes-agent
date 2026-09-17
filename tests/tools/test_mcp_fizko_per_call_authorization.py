from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import mcp_tool_config


@pytest.mark.parametrize(
    "config",
    [
        {"url": "http://mcp.fizko.ai/mcp", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://MCP.fizko.ai/mcp", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai:443/mcp", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai/mcp/", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai/mcp?x=1", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai/mcp", "transport": "sse", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai/mcp", "transport": "stdio", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://example.com/mcp", "per_call_authorization": "fizko_person_access_token"},
        {"url": "https://mcp.fizko.ai/mcp", "per_call_authorization": "unknown"},
        {"url": "https://mcp.fizko.ai/mcp", "per_call_authorization": None},
    ],
)
def test_invalid_per_call_authorization_config_is_dropped_fail_closed(config, caplog):
    with patch("hermes_cli.config.load_config", return_value={"mcp_servers": {"fizko": config}}):
        loaded = mcp_tool_config._load_mcp_config()

    assert loaded == {}
    assert "per_call_authorization" in caplog.text


@pytest.mark.parametrize("transport", [None, "streamable_http"])
def test_canonical_streamable_fizko_opt_in_is_retained(transport):
    config = {
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    }
    if transport is not None:
        config["transport"] = transport
    with patch("hermes_cli.config.load_config", return_value={"mcp_servers": {"fizko": config}}):
        loaded = mcp_tool_config._load_mcp_config()

    assert loaded == {"fizko": config}


def test_mcp_200_sdk_dispatcher_applies_local_headers_on_the_real_streamable_http_path():
    assert importlib.metadata.version("mcp") == "2.0.0"

    from mcp import ClientSession
    from mcp.client import streamable_http
    from tools.mcp_per_call_headers import call_tool_with_headers

    seen = []

    async def endpoint(request):
        body = json.loads(request.content)
        seen.append((body["method"], dict(request.headers)))
        result = (
            {"tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]}
            if body["method"] == "tools/list"
            else {"content": [{"type": "text", "text": "ok"}]}
        )
        return streamable_http.httpx2.Response(
            200,
            headers={"content-type": "application/json"},
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
        )

    async def drive():
        transport = streamable_http.httpx2.MockTransport(endpoint)
        async with streamable_http.httpx2.AsyncClient(
            transport=transport,
            headers={"Authorization": "Bearer profile-pat"},
        ) as client:
            async with streamable_http.streamable_http_client(
                "https://mcp.fizko.ai/mcp", http_client=client
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    result = await call_tool_with_headers(
                        session,
                        "whoami",
                        {},
                        headers={"Authorization": "Bearer person-token"},
                    )
                    empty_result = await call_tool_with_headers(
                        session,
                        "whoami",
                        {},
                        headers={"Authorization": ""},
                    )
        return result, empty_result

    result, empty_result = asyncio.run(drive())

    assert result.content[0].text == "ok"
    assert empty_result.content[0].text == "ok"
    call_headers = [headers for method, headers in seen if method == "tools/call"]
    assert call_headers[0]["authorization"] == "Bearer person-token"
    assert call_headers[1]["authorization"] == ""


def _stub_handler_server(config):
    return SimpleNamespace(
        _config=config,
        _rpc_lock=asyncio.Lock(),
        _inflight_tasks=set(),
        _pending_call_context=None,
        session=SimpleNamespace(call_tool=None),
        mark_tool_call=lambda: None,
        _mark_session_proven=lambda: None,
    )


def test_mcp_handler_captures_person_header_before_crossing_to_mcp_loop(monkeypatch):
    from agent.turn_authorization import TurnAuthorization, reset_current_turn_authorization, set_current_turn_authorization
    from tools import mcp_tool_handlers

    recorded = []

    async def fake_call(session, name, arguments, *, headers):
        recorded.append(dict(headers))
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    server = SimpleNamespace(
        _config={
            "url": "https://mcp.fizko.ai/mcp",
            "per_call_authorization": "fizko_person_access_token",
        },
        _rpc_lock=asyncio.Lock(),
        _inflight_tasks=set(),
        _pending_call_context=None,
        session=SimpleNamespace(call_tool=None),
        mark_tool_call=lambda: None,
        _mark_session_proven=lambda: None,
    )
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "call_tool_with_headers", fake_call, raising=False)
    monkeypatch.setattr(
        mcp_tool_handlers._loop,
        "_run_on_mcp_loop",
        lambda call, timeout: asyncio.run(contextvars.Context().run(call)),
    )

    import contextvars

    token = set_current_turn_authorization(TurnAuthorization.from_raw(
        "person-token", expires_at=time.time() + 3600, principal_id="a" * 64
    ))
    try:
        result = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})
    finally:
        reset_current_turn_authorization(token)

    assert json.loads(result) == {"result": "ok"}
    assert recorded == [{"Authorization": "Bearer person-token"}]


def test_opted_in_fizko_without_turn_token_keeps_profile_authorization(monkeypatch):
    from tools import mcp_tool_handlers

    recorded = []

    async def fake_racing(server, server_name, tool_name, args, request_headers=None):
        recorded.append(request_headers)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", fake_racing)
    monkeypatch.setattr(
        mcp_tool_handlers._loop,
        "_run_on_mcp_loop",
        lambda call, timeout: asyncio.run(contextvars.Context().run(call)),
    )

    mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})

    assert recorded == [None]


def test_expired_person_token_blocks_instead_of_falling_back_to_profile_authorization(monkeypatch):
    from agent.turn_authorization import (
        TurnAuthorization,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )
    from tools import mcp_tool_handlers

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(
        mcp_tool_handlers,
        "_call_tool_racing_stdio_death",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("expired personal authority reached the MCP transport")
        ),
    )

    context_token = set_current_turn_authorization(
        TurnAuthorization.from_raw("expired-person", expires_at=1.0, principal_id="a" * 64)
    )
    try:
        result = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})
    finally:
        reset_current_turn_authorization(context_token)

    assert "person authorization expired" in json.loads(result)["error"]


def test_blocked_personal_descendant_does_not_fall_back_to_profile_authorization(monkeypatch):
    from agent.turn_authorization import (
        TurnAuthorization,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )
    from tools import mcp_tool_handlers

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(
        mcp_tool_handlers,
        "_call_tool_racing_stdio_death",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked descendant reached the MCP transport")
        ),
    )

    context_token = set_current_turn_authorization(TurnAuthorization.blocked())
    try:
        result = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})
    finally:
        reset_current_turn_authorization(context_token)

    assert "person authorization unavailable" in json.loads(result)["error"]


def test_person_token_is_revalidated_immediately_before_transport_dispatch(monkeypatch):
    from agent import turn_authorization
    from agent.turn_authorization import (
        TurnAuthorization,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )
    from tools import mcp_tool_handlers

    now = [100.0]
    calls = []

    async def fake_racing(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    def delayed_dispatch(_name, _server, _op, call, _timeout, _handlers, _failure, **_kwargs):
        now[0] = 102.0
        try:
            return asyncio.run(call())
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(turn_authorization.time, "time", lambda: now[0])
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", fake_racing)
    monkeypatch.setattr(mcp_tool_handlers, "_dispatch", delayed_dispatch)
    token = set_current_turn_authorization(
        TurnAuthorization.from_raw("short-lived", expires_at=101.0, principal_id="a" * 64)
    )
    try:
        result = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})
    finally:
        reset_current_turn_authorization(token)

    assert calls == []
    assert "expired" in json.loads(result)["error"]


def test_neighbor_mcp_is_not_given_per_call_headers(monkeypatch):
    from agent.turn_authorization import TurnAuthorization, reset_current_turn_authorization, set_current_turn_authorization
    from tools import mcp_tool_handlers

    recorded = []

    async def fake_racing(server, server_name, tool_name, args, request_headers=None):
        recorded.append(request_headers)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    server = _stub_handler_server({"url": "https://neighbor.example/mcp"})
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", fake_racing)
    monkeypatch.setattr(
        mcp_tool_handlers._loop,
        "_run_on_mcp_loop",
        lambda call, timeout: asyncio.run(contextvars.Context().run(call)),
    )
    token = set_current_turn_authorization(TurnAuthorization.from_raw(
        "person-token", expires_at=time.time() + 3600, principal_id="a" * 64
    ))
    try:
        mcp_tool_handlers._make_tool_handler("neighbor", "ping", 10)({})
    finally:
        reset_current_turn_authorization(token)

    assert recorded == [None]


def test_final_handler_sink_rejects_programmatic_noncanonical_fizko_opt_in(monkeypatch):
    from tools import mcp_tool_handlers

    server = _stub_handler_server({
        "url": "https://attacker.example/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(
        mcp_tool_handlers,
        "_acquire_call_server",
        lambda *args: (server, None),
    )
    monkeypatch.setattr(
        mcp_tool_handlers,
        "_call_tool_racing_stdio_death",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid config reached tools/call transport")
        ),
    )

    result = mcp_tool_handlers._make_tool_handler("portable", "exfiltrate", 10)({})

    assert "invalid per_call_authorization" in json.loads(result)["error"]


def test_concurrent_handlers_keep_person_headers_isolated(monkeypatch):
    from agent.turn_authorization import TurnAuthorization, reset_current_turn_authorization, set_current_turn_authorization
    from tools import mcp_tool_handlers

    recorded = []
    lock = threading.Lock()

    async def fake_racing(server, server_name, tool_name, args, request_headers=None):
        with lock:
            recorded.append(request_headers["Authorization"])
        await asyncio.sleep(0)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", fake_racing)
    monkeypatch.setattr(
        mcp_tool_handlers._loop,
        "_run_on_mcp_loop",
        lambda call, timeout: asyncio.run(contextvars.Context().run(call)),
    )
    handler = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)

    def invoke(person):
        token = set_current_turn_authorization(TurnAuthorization.from_raw(
            person, expires_at=time.time() + 3600, principal_id="a" * 64
        ))
        try:
            handler({})
        finally:
            reset_current_turn_authorization(token)

    threads = [threading.Thread(target=invoke, args=(person,)) for person in ("person-a", "person-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(recorded) == ["Bearer person-a", "Bearer person-b"]


def test_retry_revalidates_the_same_local_person_authorization(monkeypatch):
    from agent import turn_authorization
    from agent.turn_authorization import TurnAuthorization, reset_current_turn_authorization, set_current_turn_authorization
    from tools import mcp_tool_handlers

    recorded = []
    now = [100.0]

    async def fake_racing(server, server_name, tool_name, args, request_headers=None):
        recorded.append(dict(request_headers))
        if len(recorded) == 1:
            now[0] = 102.0
            raise RuntimeError("session expired")
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False, structured_content=None)

    def retry_dispatch(server_name, server, op, call, timeout, handlers, on_failure, **kwargs):
        try:
            asyncio.run(call())
        except RuntimeError:
            ambient = set_current_turn_authorization(
                TurnAuthorization.from_raw(
                    "changed-ambient", expires_at=200.0, principal_id="b" * 64
                )
            )
            try:
                try:
                    return asyncio.run(call())
                except Exception as exc:
                    return json.dumps({"error": str(exc)})
            finally:
                reset_current_turn_authorization(ambient)

    server = _stub_handler_server({
        "url": "https://mcp.fizko.ai/mcp",
        "per_call_authorization": "fizko_person_access_token",
    })
    monkeypatch.setattr(turn_authorization.time, "time", lambda: now[0])
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", fake_racing)
    monkeypatch.setattr(mcp_tool_handlers, "_dispatch", retry_dispatch)
    token = set_current_turn_authorization(
        TurnAuthorization.from_raw(
            "original-person", expires_at=101.0, principal_id="a" * 64
        )
    )
    try:
        result = mcp_tool_handlers._make_tool_handler("fizko", "whoami", 10)({})
    finally:
        reset_current_turn_authorization(token)

    assert recorded == [{"Authorization": "Bearer original-person"}]
    assert "expired" in json.loads(result)["error"]
