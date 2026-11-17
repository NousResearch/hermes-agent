"""The hosted artifacts MCP server attaches only for an entitled Nous profile, only at session start."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli.nous_account import _info_from_account_payload
from tools import hosted_artifacts as ha

BASE = "https://artifacts.example.test"
URL = BASE + ha.MCP_PATH


def _account(**payload):
    body = {"user": {"email": "a@example.test"}, "organisation": {"id": "org_1"},
            "paid_service_access": {"paid_access": True}, "artifacts_access": True}
    body.update(payload)
    return _info_from_account_payload(body, state={"access_token": "t"}, portal_base_url=None)


def _decide(config=None, account=None):
    return ha.decide(config or ha.ArtifactsConfig(base_url=BASE),
                     account_reader=lambda: account if account is not None else _account())


@pytest.fixture(autouse=True)
def _fresh_decisions():
    ha.reset_decisions()
    yield
    ha.reset_decisions()


def test_entitled_account_attaches_at_the_contract_path():
    decision = _decide()
    assert decision.attach and decision.url == URL


@pytest.mark.parametrize("payload, reason", [
    ({"paid_service_access": {"paid_access": False}}, ha.Reason.not_entitled),
    ({"paid_service_access": None}, ha.Reason.not_entitled),
    ({"artifacts_access": False}, ha.Reason.not_enabled),
    ({"artifacts_access": None}, ha.Reason.not_enabled),
    ({"artifacts_access": "true"}, ha.Reason.not_enabled),
])
def test_account_without_both_flags_is_not_attached(payload, reason):
    decision = _decide(account=_account(**payload))
    assert not decision.attach and decision.reason is reason and decision.url is None


def test_signed_out_profile_is_not_attached():
    signed_out = SimpleNamespace(logged_in=False, source="none")
    assert _decide(account=signed_out).reason is ha.Reason.not_logged_in


def test_jwt_snapshot_is_not_an_account_read():
    snapshot = SimpleNamespace(logged_in=True, source="jwt", raw_account=None,
                               paid_service_access_info=SimpleNamespace(paid_access=True))
    assert _decide(account=snapshot).reason is ha.Reason.account_unreadable


def test_account_fetch_failure_fails_closed():
    def boom():
        raise OSError("portal unreachable")
    decision = ha.decide(ha.ArtifactsConfig(base_url=BASE), account_reader=boom)
    assert decision.reason is ha.Reason.account_unreadable and not decision.attach


@pytest.mark.parametrize("raw", [{"enabled": False}, {"enabled": "false"}, {"enabled": "off"}, {"enabled": 0}])
def test_disabled_config_never_reads_the_account(raw):
    def unread():
        raise AssertionError("account read while disabled")
    config = ha.ArtifactsConfig.from_raw({**raw, "base_url": BASE})
    assert ha.decide(config, account_reader=unread).reason is ha.Reason.disabled


def test_enabled_by_default():
    assert ha.ArtifactsConfig.from_raw(None).enabled and ha.ArtifactsConfig.from_raw({}).enabled


@pytest.fixture
def _gateway_env(monkeypatch):
    for name in ("TOOL_GATEWAY_DOMAIN", "TOOL_GATEWAY_SCHEME", "GATEWAY_GATEWAY_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_base_url_is_the_managed_gateway(_gateway_env):
    assert _decide(ha.ArtifactsConfig()).url == "https://gateway-gateway.nousresearch.com" + ha.MCP_PATH


def test_default_base_url_follows_the_tool_gateway_domain_and_scheme(_gateway_env):
    _gateway_env.setenv("TOOL_GATEWAY_DOMAIN", "staging-nousresearch.com")
    assert _decide(ha.ArtifactsConfig()).url == "https://gateway-gateway.staging-nousresearch.com" + ha.MCP_PATH
    _gateway_env.setenv("TOOL_GATEWAY_DOMAIN", "nous.localhost:8080")
    _gateway_env.setenv("TOOL_GATEWAY_SCHEME", "http")
    assert _decide(ha.ArtifactsConfig()).url == "http://gateway-gateway.nous.localhost:8080" + ha.MCP_PATH


def test_configured_base_url_overrides_the_default(_gateway_env):
    _gateway_env.setenv("TOOL_GATEWAY_DOMAIN", "staging-nousresearch.com")
    assert _decide(ha.ArtifactsConfig(base_url=BASE)).url == URL


def test_default_base_url_never_reads_the_relay_connector(_gateway_env):
    from gateway import relay

    def unread(*_a, **_k):
        raise AssertionError("relay config read for the artifacts base URL")
    _gateway_env.setattr(relay, "relay_url", unread)
    assert _decide(ha.ArtifactsConfig()).attach


def test_unusable_tool_gateway_scheme_is_refused(_gateway_env):
    _gateway_env.setenv("TOOL_GATEWAY_SCHEME", "ftp")
    assert _decide(ha.ArtifactsConfig()).reason is ha.Reason.insecure_base_url


@pytest.mark.parametrize("base, url", [
    ("https://a.example.test/some/path/", "https://a.example.test" + ha.MCP_PATH),
    ("http://127.0.0.1:8123", "http://127.0.0.1:8123" + ha.MCP_PATH),
    ("http://gg.localhost:9000", "http://gg.localhost:9000" + ha.MCP_PATH),
    ("http://a.example.test", None),
    ("ftp://a.example.test", None),
    ("not a url", None),
])
def test_bearer_only_goes_over_https_or_to_a_local_host(base, url):
    decision = _decide(ha.ArtifactsConfig(base_url=base))
    assert decision.url == url
    if url is None:
        assert decision.reason is ha.Reason.insecure_base_url


def test_decision_is_cached_per_profile_until_the_ttl(monkeypatch):
    calls = []
    clock = [100.0]

    def reader():
        calls.append(1)
        return _account()
    config = ha.ArtifactsConfig(base_url=BASE)
    first = ha.cached_decision(config, now=lambda: clock[0], account_reader=reader)
    ha.cached_decision(config, now=lambda: clock[0], account_reader=reader)
    assert first.attach and len(calls) == 1
    clock[0] += ha.DECISION_TTL_SECONDS
    ha.cached_decision(config, now=lambda: clock[0], account_reader=reader)
    assert len(calls) == 2


def test_attach_to_adds_and_replaces_only_when_entitled(monkeypatch):
    monkeypatch.setattr(ha, "_read_account", _account)
    servers = {"hermes-artifacts": {"url": "https://evil.example.test/mcp", "auth": "nous"}, "other": {"url": "x"}}
    ha.attach_to(servers, {"artifacts": {"base_url": BASE}})
    assert servers["hermes-artifacts"] == {"url": URL, "auth": "nous"} and servers["other"] == {"url": "x"}

    ha.reset_decisions()
    monkeypatch.setattr(ha, "_read_account", lambda: _account(artifacts_access=False))
    servers = {}
    ha.attach_to(servers, {"artifacts": {"base_url": BASE}})
    assert servers == {}


def _write_config(home, data):
    (home / "config.yaml").write_text(json.dumps(data), encoding="utf-8")  # JSON is YAML


def test_load_mcp_config_carries_the_managed_server_for_an_entitled_profile(tmp_path, monkeypatch):
    from tools import mcp_tool_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, {"artifacts": {"base_url": BASE}, "mcp_servers": {"mine": {"url": "https://m.example.test/mcp"}}})
    monkeypatch.setattr(ha, "_read_account", _account)
    servers = mcp_tool_config._load_mcp_config()
    assert servers[ha.SERVER_NAME] == {"url": URL, "auth": "nous"}
    assert "mine" in servers

    ha.reset_decisions()
    _write_config(tmp_path, {"artifacts": {"enabled": False, "base_url": BASE}})
    assert ha.SERVER_NAME not in mcp_tool_config._load_mcp_config()


def test_nous_auth_is_refused_for_any_other_url(monkeypatch):
    monkeypatch.setattr(ha, "_profile_config", lambda: ha.ArtifactsConfig(base_url=BASE))
    monkeypatch.setattr(ha, "_read_account", _account)
    with pytest.raises(ValueError, match="reserved"):
        ha.nous_bearer_auth("https://elsewhere.example.test" + ha.MCP_PATH)


def test_nous_auth_sends_a_fresh_token_each_request(monkeypatch):
    import httpx

    from hermes_cli import auth as hermes_auth

    monkeypatch.setattr(ha, "_profile_config", lambda: ha.ArtifactsConfig(base_url=BASE))
    monkeypatch.setattr(ha, "_read_account", _account)
    tokens = iter(["tok-1", "tok-2"])
    monkeypatch.setattr(hermes_auth, "resolve_nous_access_token", lambda **_: next(tokens))
    auth = ha.nous_bearer_auth(URL)
    seen = []
    for _ in range(2):
        request = httpx.Request("POST", URL)
        flow = auth.sync_auth_flow(request)
        seen.append(next(flow).headers["Authorization"])
    assert seen == ["Bearer tok-1", "Bearer tok-2"]


def test_nous_auth_is_refused_when_not_entitled(monkeypatch):
    monkeypatch.setattr(ha, "_profile_config", lambda: ha.ArtifactsConfig(base_url=BASE))
    monkeypatch.setattr(ha, "_read_account", lambda: _account(paid_service_access={"paid_access": False}))
    with pytest.raises(ValueError):
        ha.nous_bearer_auth(URL)


@pytest.mark.parametrize("request_url, sends", [
    (URL, True),
    (URL + "?session=1", True),
    ("https://ARTIFACTS.example.test:443" + ha.MCP_PATH, True),
    (BASE + "/unrelated", False),
    (URL + "/extra", False),
    ("http://artifacts.example.test" + ha.MCP_PATH, False),
    ("https://artifacts.example.test:8443" + ha.MCP_PATH, False),
    ("https://other.example.test" + ha.MCP_PATH, False),
])
def test_bearer_goes_only_to_the_exact_endpoint_at_request_time(monkeypatch, request_url, sends):
    import asyncio

    from hermes_cli import auth as hermes_auth
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setattr(ha, "_profile_config", lambda: ha.ArtifactsConfig(base_url=BASE))
    monkeypatch.setattr(ha, "_read_account", _account)
    monkeypatch.setattr(hermes_auth, "resolve_nous_access_token", lambda **_: "TOKEN")
    auth = ha.nous_bearer_auth(URL)
    sync_request = httpx.Request("POST", request_url, headers={"Authorization": "Bearer stale"})
    sync_sent = next(auth.sync_auth_flow(sync_request)).headers.get("Authorization")

    async def first(flow):
        return await flow.__anext__()
    async_request = httpx.Request("POST", request_url)
    async_sent = asyncio.run(first(auth.async_auth_flow(async_request))).headers.get("Authorization")
    expected = "Bearer TOKEN" if sends else None
    assert (sync_sent, async_sent) == (expected, expected)


def test_same_origin_redirect_does_not_receive_the_bearer(monkeypatch):
    """The MCP client follows a same-origin redirect; the redirected request must go without the token."""
    import asyncio

    from mcp.shared._httpx_utils import request_within_origin

    from hermes_cli import auth as hermes_auth
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setattr(ha, "bearer_allowed", lambda _url: True)
    monkeypatch.setattr(hermes_auth, "resolve_nous_access_token", lambda **_: "TOKEN")
    seen = []

    def handle(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(307, headers={"location": "/unrelated"}) if len(seen) == 1 else httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(auth=ha.nous_bearer_auth(URL), transport=httpx.MockTransport(handle)) as client:
            await request_within_origin(client, "POST", URL, json={})
    asyncio.run(run())
    assert seen == [(URL, "Bearer TOKEN"), (BASE + "/unrelated", None)]


@pytest.mark.parametrize("auth, adoptable", [("nous", False), ("NOUS ", False), ("oauth", False), ("", True)])
def test_profile_bound_connections_are_never_adopted_across_profiles(auth, adoptable):
    """The hosted server's connection carries its owner's Nous sign-in: another profile with the same
    route must connect with its own, exactly as for an OAuth server."""
    from tools.mcp_tool_registration import _same_server_route

    config = dict(ha.server_entry(URL), auth=auth)
    server = SimpleNamespace(_config=config, _resolved_identity="digest")
    assert _same_server_route(server, config, cross_profile=False)
    assert _same_server_route(server, config, cross_profile=True, resolved_identity="digest") is adoptable


class _Agent:
    def __init__(self, session_id="s1", *, enabled=None, disabled=None):
        self.session_id = session_id
        self.platform = "cli"
        self.enabled_toolsets = enabled
        self.disabled_toolsets = disabled or []
        self.tools = []
        self.valid_tool_names = set()


SHOW = "mcp__hermes_artifacts__artifact_show"


def _defs(*names):
    return [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in names]


def _names(defs):
    return {tool["function"]["name"] for tool in defs}


def test_session_started_without_the_server_never_gains_it(monkeypatch):
    import toolsets

    agent = _Agent("late")
    assert toolsets.agent_tool_surface(agent, _defs("terminal")) == _defs("terminal")  # first load: no artifacts
    later = _defs("terminal", *ha.contract_tool_names())  # discovery finishes mid-session
    assert _names(toolsets.agent_tool_surface(agent, later)) == {"terminal"}
    rebuilt = _Agent("late")  # same conversation rebuilt in this process
    assert _names(toolsets.agent_tool_surface(rebuilt, later)) == {"terminal"}
    assert _names(toolsets.agent_tool_surface(_Agent("new-session"), later)) == _names(later)


def test_session_started_with_the_server_keeps_it():
    import toolsets

    agent = _Agent("early")
    first = _defs("terminal", *ha.contract_tool_names())
    assert toolsets.agent_tool_surface(agent, first) == first
    assert ha.late_attach_drops(agent) == frozenset()
    assert ha.late_attach_drops(_Agent("early")) == frozenset()  # rebuilt: the session's record holds


def test_unrecorded_session_gets_no_artifact_tools():
    assert ha.contract_tool_names() <= ha.late_attach_drops(_Agent("never-loaded"))


def test_registered_but_disabled_at_start_cannot_be_enabled_mid_session(monkeypatch):
    """Registered in the process, but this session started with the toolset disabled: a refresh that
    enables it (toolset toggle, ``/reload-mcp``) must not add the tools."""
    import model_tools
    import toolsets
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    monkeypatch.setattr(ha, "_registered_tool_names", lambda: frozenset({SHOW}))
    agent = _Agent("disabled-at-start", enabled=[], disabled=[ha.TOOLSET_NAME])
    assert toolsets.agent_tool_surface(agent, []) == []
    assert agent._hosted_artifacts_at_start is False
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_: _defs(SHOW))
    added = refresh_agent_mcp_tools(agent, enabled_override=[ha.TOOLSET_NAME], disabled_override=[])
    assert SHOW not in added and SHOW not in agent.valid_tool_names


def test_deferred_artifact_tools_count_as_the_sessions_surface(monkeypatch):
    """Under Tool Search the first load carries the bridge, not the deferred MCP tools; the session's own
    catalog decides, so such a session keeps artifacts and one without them never reaches them."""
    import model_tools
    import toolsets
    from tools.tool_search_catalog import TOOL_CALL_NAME

    catalog = []
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_: list(catalog))
    catalog[:] = _defs(SHOW)
    with_tools = _Agent("deferred-with")
    toolsets.agent_tool_surface(with_tools, _defs(TOOL_CALL_NAME))
    assert ha.late_attach_drops(with_tools) == frozenset()

    catalog[:] = []
    without = _Agent("deferred-without")
    toolsets.agent_tool_surface(without, _defs(TOOL_CALL_NAME))
    catalog[:] = _defs(SHOW)  # registered mid-session
    assert SHOW in ha.late_attach_drops(without)


def test_tool_search_bridge_refuses_artifacts_to_a_session_started_without_them(monkeypatch):
    import model_tools
    import toolsets
    from agent.tool_executor import _tool_search_scoped_names
    from tools import tool_search

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_: _defs(SHOW, "mcp__other__tool"))
    monkeypatch.setattr(tool_search, "scoped_deferrable_names", lambda defs: frozenset(_names(defs)))
    started_without = _Agent("bridge-without")
    toolsets.agent_tool_surface(started_without, _defs("terminal"))
    assert _tool_search_scoped_names(started_without) == {"mcp__other__tool"}
    started_with = _Agent("bridge-with")
    toolsets.agent_tool_surface(started_with, _defs(SHOW))
    assert SHOW in _tool_search_scoped_names(started_with)


def test_probe_constant_is_stable():
    assert ha.HOSTED_ARTIFACTS_CLIENT_VERSION >= 1
    assert ha.SERVER_NAME == "hermes-artifacts" and ha.MCP_PATH == "/v1/artifacts/mcp"
    assert "mcp__hermes_artifacts__artifact_show" in ha.contract_tool_names()


def test_nous_auth_is_the_mcp_sdks_auth_class(monkeypatch):
    from tools.mcp_tool import sdk_httpx

    monkeypatch.setattr(ha, "_profile_config", lambda: ha.ArtifactsConfig(base_url=BASE))
    monkeypatch.setattr(ha, "_read_account", _account)
    assert isinstance(ha.nous_bearer_auth(URL), sdk_httpx().Auth)


def test_an_attached_server_is_an_enabled_mcp_server_for_sessions():
    from hermes_cli.tools_config import enabled_mcp_server_names

    config = {"artifacts": {"base_url": BASE}}
    artifacts = ha.ArtifactsConfig.from_raw(config["artifacts"])
    assert ha.SERVER_NAME not in enabled_mcp_server_names(config)  # nothing decided yet: no account read here
    ha.cached_decision(artifacts, account_reader=_account)
    assert ha.SERVER_NAME in enabled_mcp_server_names(config)
    ha.reset_decisions()
    ha.cached_decision(artifacts, account_reader=lambda: _account(artifacts_access=False))
    assert ha.SERVER_NAME not in enabled_mcp_server_names(config)
