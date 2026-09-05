"""Real registry → adapter → mautrix calls on the owning gateway loop."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import secret_scope
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.session_context import (
    _SESSION_CHAT_ID as session_chat_id, _SESSION_PLATFORM as session_platform,
    _SESSION_PROFILE as session_profile,
)
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools import matrix_tool as mt
from tools.registry import registry


@pytest.fixture
def live_path(tmp_path, monkeypatch):
    pytest.importorskip("mautrix.client", reason="Real SDK coverage requires the optional Matrix dependencies")
    from mautrix.client import Client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in tuple(__import__("os").environ):
        if name.startswith("MATRIX_"):
            monkeypatch.delenv(name)
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    loop.call_soon_threadsafe(ready.set)
    assert ready.wait(2)
    calls = []

    async def request(method, path, content=None, **kwargs):
        assert asyncio.get_running_loop() is loop
        calls.append((str(method), str(path), content, kwargs))
        if str(path).endswith("/messages"):
            return {"start": "start", "end": "next", "chunk": [{
                "type": "m.room.message", "event_id": "$history", "sender": "@user:example.org",
                "origin_server_ts": 1, "room_id": "!room:example.org",
                "content": {"msgtype": "m.text", "body": "hello"}}]}
        if str(path).endswith("/createRoom"):
            return {"room_id": "!new:example.org"}
        return {"event_id": "$result"}

    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"homeserver": "https://example.org"}))
    async def make_client():
        return Client(base_url="https://example.org", mxid="@bot:example.org", token="fixture")

    adapter._client = asyncio.run_coroutine_threadsafe(make_client(), loop).result(2)
    adapter._client.api.request = request
    runner = SimpleNamespace(
        _gateway_loop=loop, adapters={Platform.MATRIX: object()},
        _profile_adapters={"secondary": {Platform.MATRIX: adapter}},
        _primary_profile_name="primary",
    )
    # Exercise the production resolver, not a fake function that already promises isolation.
    runner._profile_adapters_map = lambda: runner._profile_adapters
    runner._primary_adapters = lambda: runner.adapters
    runner._authorization_adapter = GatewayAuthorizationMixin._authorization_adapter.__get__(runner)
    tokens = [(v, v.set(value)) for v, value in (
        (session_platform, "matrix"), (session_profile, "secondary"),
        (session_chat_id, "!room:example.org"))]
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            yield SimpleNamespace(loop=loop, adapter=adapter, runner=runner, calls=calls, home=tmp_path)
    finally:
        for v, token in reversed(tokens):
            v.reset(token)
        asyncio.run_coroutine_threadsafe(adapter._client.api.session.close(), loop).result(2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


def invoke(action, **kwargs):
    name = "matrix_admin" if action in mt._ADMIN_ACTIONS else "matrix"
    return json.loads(registry.get_entry(name).handler({"action": action, **kwargs}))


def allow(path, **gates):
    import yaml
    (path / "config.yaml").write_text(yaml.safe_dump({"matrix": {"tools": gates}}))


def test_history_reaches_current_mautrix_api_and_returns_cursor(live_path):
    result = invoke("fetch_history", from_token="previous", limit=200)
    assert result["success"] is True
    assert result["events"][0]["content"]["body"] == "hello"
    assert result["end_token"] == "next"
    params = live_path.calls[-1][3]["query_params"]
    assert params["from"] == "previous"
    assert params["limit"] == "100"
    assert params["dir"] == "b"


@pytest.mark.parametrize("action,args,gate,path_tail", [
    ("send_reaction", {"event_id": "$event", "emoji": "✅"}, None, "/send/m.reaction/"),
    ("set_presence", {"state": "unavailable", "status_msg": "busy"}, None, "/status"),
    ("redact_message", {"event_id": "$event"}, "allow_redaction", "/redact/"),
    ("invite_user", {"user_id": "@user:example.org"}, "allow_invites", "/invite"),
    ("create_room", {"name": "Example"}, "allow_room_create", "/createRoom"),
])
def test_actions_reach_adapter_sdk(live_path, action, args, gate, path_tail):
    if gate:
        assert not invoke(action, **args).get("success")
        assert live_path.calls == []
        allow(live_path.home, **{gate: True})
    assert invoke(action, **args)["success"] is True
    assert path_tail in live_path.calls[-1][1]


def test_public_creation_and_invites_are_independent_gates(live_path, monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOW_PUBLIC_ROOMS", "true")
    args = {"preset": "public_chat", "invite": ["@user:example.org"]}
    allow(live_path.home, allow_room_create=True, allow_public_rooms=False, allow_invites=True)
    assert not invoke("create_room", **args).get("success")
    allow(live_path.home, allow_room_create=True, allow_public_rooms=True, allow_invites=False)
    assert not invoke("create_room", **args).get("success")
    assert live_path.calls == []
    allow(live_path.home, allow_room_create=True, allow_public_rooms=True, allow_invites=True)
    assert invoke("create_room", **args)["success"]
    payload = live_path.calls[-1][2]
    assert payload["preset"] == "public_chat"
    assert payload["invite"] == args["invite"]


def test_room_allowlist_and_destructive_cross_room_gate(live_path):
    args = {"room_id": "!other:example.org", "event_id": "$event"}
    gates = dict(allow_redaction=True, allow_cross_room=True, allowed_rooms=[args["room_id"]])
    allow(live_path.home, **gates)
    assert not invoke("redact_message", **args).get("success")
    gates["allow_cross_room_destructive"] = True
    gates["allowed_rooms"] = ["!different:example.org"]
    allow(live_path.home, **gates)
    assert not invoke("redact_message", **args).get("success")
    assert live_path.calls == []
    gates["allowed_rooms"] = [args["room_id"]]
    allow(live_path.home, **gates)
    assert invoke("redact_message", **args)["success"]


def test_missing_secondary_adapter_does_not_use_primary(live_path):
    live_path.runner._profile_adapters.clear()
    result = invoke("set_presence")
    assert not result.get("success")
    assert "not connected" in result["error"]
    assert live_path.calls == []


def test_secondary_policy_never_borrows_primary_env(live_path, monkeypatch):
    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope({})
    try:
        result = invoke("redact_message", event_id="$event")
        assert not result.get("success")
        assert live_path.calls == []
    finally:
        secret_scope.reset_secret_scope(token)


def test_secondary_yaml_context_survives_loop_handoff(live_path, tmp_path):
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    allow(live_path.home, allow_room_create=False, allow_public_rooms=False)
    allow(secondary, allow_room_create=True, allow_public_rooms=True)
    token = set_hermes_home_override(str(secondary))
    try:
        assert invoke("create_room", preset="public_chat")["success"]
    finally:
        reset_hermes_home_override(token)


def test_stopped_gateway_loop_never_runs_adapter_on_worker():
    loop = asyncio.new_event_loop()
    observed = []

    async def probe():
        observed.append(True)

    coro = probe()
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=SimpleNamespace(_gateway_loop=loop)):
            with pytest.raises(RuntimeError, match="gateway loop"):
                mt._run(coro)
        assert observed == []
        assert coro.cr_frame is None
    finally:
        coro.close()
        loop.close()


def test_unknown_room_stamp_still_requires_destructive_gate(live_path):
    token = session_chat_id.set("")
    try:
        allow(live_path.home, allow_redaction=True, allow_cross_room=True)
        assert not invoke("redact_message", room_id="!room:example.org", event_id="$event").get("success")
        assert live_path.calls == []
    finally:
        session_chat_id.reset(token)

@pytest.mark.parametrize("policy", ["matrix: [", "matrix: {tools: []}", "[]"])
def test_invalid_policy_does_not_enable_legacy_admin_gates(live_path, monkeypatch, policy):
    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
    (live_path.home / "config.yaml").write_text(policy)
    assert not invoke("redact_message", event_id="$event").get("success")
    assert live_path.calls == []


def test_history_failure_is_not_a_successful_empty_page(live_path):
    async def fail(*args, **kwargs):
        raise RuntimeError("transport unavailable")

    live_path.adapter._client.api.request = fail
    result = invoke("fetch_history")
    assert not result.get("success")
    assert "error" in result


def test_sync_tool_on_gateway_loop_fails_without_deadlock(live_path):
    async def probe():
        pytest.fail("must not execute")

    async def call_on_loop():
        coro = probe()
        with pytest.raises(RuntimeError, match="cannot block"):
            mt._run(coro)
        assert coro.cr_frame is None

    asyncio.run_coroutine_threadsafe(call_on_loop(), live_path.loop).result(2)


def test_scheduling_failure_closes_coroutine(live_path):
    async def probe():
        pytest.fail("must not execute")

    coro = probe()
    with patch.object(live_path.loop, "call_soon_threadsafe", side_effect=RuntimeError("closed")):
        with pytest.raises(RuntimeError, match="gateway loop"):
            mt._run(coro)
    assert coro.cr_frame is None


def test_dispatch_timeout_cancels_future(live_path):
    from concurrent.futures import Future

    class TimedOutFuture(Future):
        def result(self, timeout=None):
            assert timeout == 120
            raise TimeoutError()

    future = TimedOutFuture()

    async def probe():
        pass

    def schedule(coro, *args, **kwargs):
        coro.close()
        return future

    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=schedule):
        with pytest.raises(RuntimeError, match="may still be in flight"):
            mt._run(probe())
    assert future.cancelled()
