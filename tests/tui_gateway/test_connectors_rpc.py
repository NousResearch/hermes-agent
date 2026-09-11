"""Connector UI uses the real session/profile and model-tool authority path."""

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, obj: dict) -> bool:
        self.frames.put(obj)
        return True

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from hermes_cli.anon_auth import ANON_AUTH_METHOD

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    launch = tmp_path / "launch"
    profile = tmp_path / "profile"
    for home in (launch, profile):
        home.mkdir()
        (home / "config.yaml").write_text("nous:\n  guest: true\ntools:\n  connectors:\n    enabled: true\n")
        (home / "auth.json").write_text(json.dumps({"providers": {"nous": {
            "auth_method": ANON_AUTH_METHOD, "access_token": f"test-{home.name}-bearer",
            "anon_token": f"test-{home.name}-identity", "expires_at": "2099-01-01T00:00:00Z",
        }}}))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda *a, **kw: pytest.fail("unexpected live HTTP"))
    from tui_gateway import server
    from tools import connections_tool, tool_backend_helpers
    from tools.tool_gateway import client
    import hermes_cli.plugins as plugins

    monkeypatch.setattr(tool_backend_helpers, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(plugins.get_plugin_manager(), "_middleware", {})
    monkeypatch.setattr(connections_tool, "_seen_instructions", set())
    monkeypatch.setattr(connections_tool, "_rendered_links", {})
    peer = Peer()
    agent = SimpleNamespace(enabled_toolsets=["connections"], disabled_toolsets=None,
                            session_id="durable-owner")
    owner = {"session_key": "durable-owner", "agent": agent, "profile_home": str(profile),
             "source": "gui", "transport": peer, "cwd": str(tmp_path), "history": []}
    monkeypatch.setattr(server, "_sessions", {"ui-owner": owner})
    monkeypatch.setattr(server, "_hermes_home", launch)
    wire = []
    rows = [{"connector": "gmail", "enabled": True, "connected": False,
             "connectionStatus": "not_connected", "future_metadata": {"display_name": "Mail"}}]
    response = {"results": [{"connector": "gmail", "status": "initiated",
                              "connect_url": "https://connect.example/start?token=authorization-link",
                              "instruction": "Pick your account."}],
                "summary": {"total": 1, "active": 0, "initiated": 1, "failed": 0}}

    def request(method, url, **kw):
        wire.append((method, url, kw))
        payload = {"items": rows} if method == "GET" else response
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr(client, "_default_transport", lambda: SimpleNamespace(request=request))
    monkeypatch.setenv("CONNECTOR_GATEWAY_URL", "https://connector.example")

    def call(method, *, via=peer, **params):
        result = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": method,
                                  "params": {"session_id": "ui-owner", **params}}, transport=via)
        return result if result is not None else via.frames.get(timeout=10)

    return SimpleNamespace(server=server, peer=peer, owner=owner, agent=agent, call=call,
                           rows=rows, response=response, wire=wire, profile=profile)


def test_connector_rpc_uses_owning_profile_and_real_policy_pipeline(runtime, monkeypatch):
    import hermes_cli.plugins as plugins
    from hermes_constants import get_hermes_home

    r = runtime
    events = []

    def request_policy(**kw):
        assert kw["tool_name"] == "manage_connections"
        assert kw["session_id"] == "durable-owner"
        assert get_hermes_home() == r.profile
        events.append(("request", kw["args"]["action"]))

    def hook(name, args, **kw):
        assert name == "manage_connections"
        events.append(("hook", args["action"]))
        return None, None

    def execution(**kw):
        events.append(("execution", kw["args"]["action"]))
        return kw["next_call"](kw["args"])

    with r.server._session_profile_runtime_scope(r.owner):
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        monkeypatch.setattr(manager, "_middleware",
                            {"tool_request": [request_policy], "tool_execution": [execution]})
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", hook)
    assert r.call("connectors.list") == {"jsonrpc": "2.0", "id": 1,
                                        "result": {"available": True, "connectors": r.rows}}
    result = r.call("connectors.connect", connectors=["gmail"], reconnect=True)["result"]
    assert result["results"][0]["connect_url"] == r.response["results"][0]["connect_url"]
    assert result["results"][0]["instruction"] == "Pick your account."
    assert result["summary"] == r.response["summary"]
    assert events == [(phase, action) for action in ("status", "reconnect")
                      for phase in ("request", "hook", "execution")]
    assert [entry[0] for entry in r.wire] == ["GET", "POST"]
    assert all(entry[2]["headers"]["Authorization"] == "Bearer test-profile-bearer" for entry in r.wire)
    assert r.wire[1][2]["json"] == {"connectors": ["gmail"], "reinitiate": True}
    assert r.owner["history"] == []  # UI consent does not inject a model turn or private-data read.


@pytest.mark.parametrize("gate", ["empty", "restricted", "disabled", "config", "account", "guest_off", "flag_off"])
def test_connector_gates_deny_without_io(runtime, monkeypatch, gate):
    r = runtime
    if gate == "empty":
        r.agent.enabled_toolsets = []
    elif gate == "restricted":
        r.agent.enabled_toolsets = ["web"]
    elif gate == "disabled":
        r.agent.disabled_toolsets = ["connections"]
    elif gate == "config":
        (r.profile / "config.yaml").write_text("tools:\n  connectors: false\n")
    elif gate == "account":
        (r.profile / "auth.json").unlink()
    elif gate == "flag_off":
        monkeypatch.delenv("HERMES_GUEST_ONBOARDING")
    else:
        (r.profile / "config.yaml").write_text("nous:\n  guest: false\n")
    assert r.call("connectors.list")["result"] == {"available": False, "connectors": []}
    assert r.call("connectors.connect", connectors=["gmail"])["error"]["data"]["reason"] == "CONNECTORS_UNAVAILABLE"
    assert not r.wire


@pytest.mark.parametrize("method,params", [
    ("connectors.list", {"session_id": ""}), ("connectors.list", {"profile": "other"}),
    ("connectors.connect", {"connectors": []}), ("connectors.connect", {"connectors": "gmail"}),
    ("connectors.connect", {"connectors": [" Gmail"]}),
    ("connectors.connect", {"connectors": ["gmail"], "reconnect": "false"}),
    ("connectors.connect", {"connectors": ["gmail"], "action": "wait"}),
])
def test_invalid_connector_parameters_do_not_dispatch(runtime, method, params):
    assert runtime.call(method, **params)["error"]["code"] == 4000
    assert not runtime.wire


@pytest.mark.parametrize("method", ["connectors.list", "connectors.connect"])
def test_connector_rpc_requires_bound_live_owner(runtime, method):
    from tui_gateway.transport import bind_transport, reset_transport
    r = runtime
    params = {"connectors": ["gmail"]} if method.endswith("connect") else {}
    assert r.call(method, via=Peer(), **params)["error"]["code"] == 4001
    token = bind_transport(None)
    try:
        assert r.server.handle_request({"id": 1, "method": method,
                                        "params": {"session_id": "ui-owner", **params}})["error"]["code"] == 4001
    finally:
        reset_transport(token)
    assert not r.wire


@pytest.mark.parametrize("phase", ["queued", "inflight"])
def test_retired_connector_request_never_uses_or_returns_replacement(runtime, monkeypatch, phase):
    from threading import Event
    import hermes_cli.plugins as plugins
    r = runtime
    entered, release = Event(), Event()
    pending = []
    if phase == "queued":
        monkeypatch.setattr(r.server, "_pool", SimpleNamespace(submit=pending.append))
    else:
        def execution(**kw):
            entered.set()
            assert release.wait(10)
            return kw["next_call"](kw["args"])
        with r.server._session_profile_runtime_scope(r.owner):
            manager = plugins.get_plugin_manager()
            manager.discover_and_load()
            monkeypatch.setattr(manager, "_middleware", {"tool_execution": [execution]})
    request = {"id": 1, "method": "connectors.list", "params": {"session_id": "ui-owner"}}
    assert r.server.dispatch(request, transport=r.peer) is None
    if phase == "inflight":
        assert entered.wait(10)
    replacement = {**r.owner, "profile_home": str(r.profile.parent / "launch")}
    r.server._sessions["ui-owner"] = replacement
    release.set()
    if phase == "queued":
        pending[0]()
    result = r.peer.frames.get(timeout=10)
    assert result["error"]["code"] == 4001
    if phase == "queued":
        assert not r.wire


@pytest.mark.parametrize("failure", ["http", "policy", "invalid"])
def test_failed_connector_reads_are_typed_errors_not_empty_success(runtime, monkeypatch, failure):
    from tools.tool_gateway import client
    import hermes_cli.plugins as plugins
    r = runtime
    if failure == "policy":
        monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", lambda *a, **kw: ("secret=do-not-echo", None))
    else:
        response = SimpleNamespace(status_code=503 if failure == "http" else 200,
                                   json=lambda: {"error": {"code": "UPSTREAM", "message": "secret=do-not-echo"}})
        monkeypatch.setattr(client, "_default_transport", lambda: SimpleNamespace(request=lambda *a, **kw: response))
    result = r.call("connectors.list")
    assert result["error"]["code"] == 5034
    assert "do-not-echo" not in json.dumps(result)


def test_cold_session_can_list_and_connect_without_building_agent(runtime, monkeypatch):
    r = runtime
    r.owner["agent"] = None
    (r.profile / "config.yaml").write_text(
        "nous:\n  guest: true\nplatform_toolsets:\n  cli: [connections]\ntools:\n  connectors: true\n")
    monkeypatch.setattr(r.server, "_start_agent_build", lambda *a: pytest.fail("RPC built an agent"))
    assert r.call("connectors.list")["result"]["connectors"] == r.rows
    assert r.call("connectors.connect", connectors=["gmail"])["result"]["results"][0]["connect_url"]
    assert r.owner["agent"] is None
    (r.profile / "config.yaml").write_text("platform_toolsets:\n  cli: []\n")
    r.wire.clear()
    assert r.call("connectors.list")["result"] == {"available": False, "connectors": []}
    assert not r.wire


def test_connector_metadata_preserves_additions_but_not_credentials(runtime):
    r = runtime
    r.rows[0]["future_metadata"].update(access_token="private-access", apiKey="private-key", label="Mail")
    result = r.call("connectors.list")["result"]["connectors"][0]
    assert result["future_metadata"]["label"] == "Mail"
    assert "private-access" not in json.dumps(result)
    assert "private-key" not in json.dumps(result)


def test_empty_connect_response_is_not_success(runtime):
    runtime.response.clear()
    assert runtime.call("connectors.connect", connectors=["gmail"])["error"]["data"]["reason"] == "INVALID_CONNECTOR_RESPONSE"


def test_launch_profile_ignores_ambient_sibling_scope(runtime):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    r = runtime
    r.owner["profile_home"] = None
    token = set_hermes_home_override(r.profile)
    try:
        assert r.call("connectors.list")["result"]["available"]
    finally:
        reset_hermes_home_override(token)
    assert r.wire[0][2]["headers"]["Authorization"] == "Bearer test-launch-bearer"


def test_mirrored_owner_can_connect_but_detached_peer_cannot(runtime):
    from tui_gateway.transport import FanoutTransport
    r = runtime
    mirror = Peer()
    fanout = FanoutTransport(r.peer, mirror)
    r.owner["transport"] = fanout
    try:
        assert r.call("connectors.connect", via=mirror, connectors=["gmail"])["result"]["results"]
        fanout.detach(mirror)
        r.wire.clear()
        assert r.call("connectors.list", via=mirror)["error"]["code"] == 4001
        assert not r.wire
        assert r.peer.frames.empty()  # RPC result is never broadcast to another viewer.
    finally:
        fanout.close()


def test_execution_policy_refusal_never_reaches_connector_http(runtime, monkeypatch):
    import hermes_cli.plugins as plugins
    r = runtime
    with r.server._session_profile_runtime_scope(r.owner):
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        monkeypatch.setattr(manager, "_middleware", {
            "tool_execution": [lambda **kw: json.dumps({"error": "policy secret=do-not-echo"})]})
    result = r.call("connectors.connect", connectors=["gmail"])
    assert result["error"]["code"] == 5034
    assert "do-not-echo" not in json.dumps(result)
    assert not r.wire
