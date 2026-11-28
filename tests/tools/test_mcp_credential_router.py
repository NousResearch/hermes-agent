"""Which credential an MCP request carries: the service account, or the signed-in user's token.

One connection outlives every chat, so the choice is made per *request* by asking which turn is
running. These tests cover the three things that can go wrong there:

* the wrong credential on a request (a person's call run as the integration account, or the
  reverse for an ``auth_scope: service`` entry);
* the wrong *timing* (a refusal that connects a lazy server, or an approval prompt for a call
  that is then refused);
* a failure mode that tears down the shared connection instead of refusing one call.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent.turn_token import TurnToken, clear_sessions, clear_store, turn_token_scope
from tools import mcp_service_credentials as service
from tools import mcp_credential_router as router_module
from tools.mcp_credential_router import (
    SCOPE_SERVICE,
    SCOPE_UNSET,
    SCOPE_USER,
    CredentialPlan,
    CredentialPolicy,
    credential_router,
    policy,
)

TOKEN_URL = "https://idp.example/realms/r/protocol/openid-connect/token"
CONFIG = {
    "auth_scope": "user",
    "service_credentials": {
        "token_url": TOKEN_URL,
        "client_id": "hermes-mcp-client",
        "client_secret_env": "HERMES_MCP_CLIENT_SECRET",
    },
}


def _user(token: str = "AT-zhangsan", *, expires_in: int = 600) -> TurnToken:
    return TurnToken(access_token=token, issuer="https://sso.example/realms/r",
                     subject="zhangsan", expires_at=int(time.time()) + expires_in)


@pytest.fixture(autouse=True)
def _clean():
    # The binding is a ContextVar that outlives a test in the same thread, so it is cleared on
    # both sides rather than only before.
    handle = turn_token_scope(None)
    handle.__enter__()
    try:
        clear_store()
        clear_sessions()
        yield
    finally:
        clear_store()
        clear_sessions()
        handle.__exit__(None, None, None)


@pytest.fixture
def minted(monkeypatch):
    """Stub the token endpoint; ``calls[0]`` counts how many mints happened."""
    service.clear_minters()
    calls = [0]

    def _post(url, data):
        calls[0] += 1
        return {"access_token": f"SERVICE-AT-{calls[0]}", "expires_in": 600}

    monkeypatch.setattr(service, "_post_form", _post)
    yield calls
    service.clear_minters()


@pytest.fixture
def captured():
    """A real SDK-httpx client whose requests record the Authorization header they carried."""
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    seen: dict[str, str | None] = {}

    async def handler(request):
        seen[request.url.path] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    def _build(config: dict, server: str = "kingdee"):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 auth=credential_router(server, config))

    return _build, seen


def _send(client, path: str, payload: dict):
    async def _drive():
        async with client:
            await client.post(f"https://k3.example{path}", json=payload)

    asyncio.run(_drive())


# --------------------------------------------------------------------------- scope parsing

class TestScope:
    @pytest.mark.parametrize("raw, expected", [
        ("user", SCOPE_USER),
        ("USER", SCOPE_USER),
        (" user ", SCOPE_USER),
        ("service", SCOPE_SERVICE),
        ("", SCOPE_UNSET),
        (None, SCOPE_UNSET),
        ("everyone", SCOPE_UNSET),   # unknown is ignored, never guessed
    ])
    def test_parsing(self, raw, expected):
        assert router_module._auth_scope_of({"auth_scope": raw}) == expected


# --------------------------------------------------------------------------- the plan

class TestPlan:
    def test_user_scope_with_a_service_account_plans_both(self, minted):
        plan = policy.plan_for("kingdee", CONFIG)
        assert plan is not None
        assert plan.user_scope is True
        assert plan.service is not None

    def test_service_scope_plans_only_the_machine_credential(self, minted):
        plan = policy.plan_for("kingdee", {**CONFIG, "auth_scope": "service"})
        assert plan is not None
        assert plan.user_scope is False
        assert plan.service is not None

    def test_an_unconfigured_server_is_not_managed(self):
        assert policy.plan_for("plain", {}) is None

    @pytest.mark.parametrize("config", [
        {"auth_scope": "user", "headers": {"Authorization": "Bearer static"}},
        {"auth_scope": "user", "headers": {"authorization": "Bearer static"}},
        {"auth_scope": "user", "auth": "oauth"},
        # Merged in AFTER this decision, so a router claiming the slot would overwrite it —
        # and "send nothing when the user token is gone" would become "send the static bearer".
        {"auth_scope": "user", "identity_header": {"name": "Authorization",
                                                   "value_from": "static",
                                                   "value": "Bearer static"}},
        {"auth_scope": "user", "identity_header": {"name": "authorization",
                                                   "value_from": "static",
                                                   "value": "Bearer static"}},
    ])
    def test_a_configured_credential_wins(self, config):
        """The operator chose the principal; Hermes must not inject a competing one."""
        assert policy.plan_for("srv", config) is None

    def test_an_identity_header_for_another_name_does_not_block_routing(self, minted):
        plan = policy.plan_for("srv", {**CONFIG, "identity_header": {
            "name": "X-Hermes-User", "value_from": "static", "value": "u1"}})
        assert plan is not None and plan.user_scope is True

    def test_an_incomplete_service_block_is_refused(self):
        """Connecting with no credential reads downstream as an unexplained 401."""
        with pytest.raises(ValueError, match="token_url and client_id"):
            policy.plan_for("srv", {"service_credentials": {"token_url": TOKEN_URL}})

    def test_a_relative_token_url_is_refused(self):
        with pytest.raises(ValueError, match="absolute URL"):
            policy.plan_for("srv", {"service_credentials": {"token_url": "/token",
                                                            "client_id": "c"}})

    def test_describe_never_includes_a_token(self, minted):
        described = policy.plan_for("k", CONFIG).describe()
        assert "service=yes" in described
        assert "AT" not in described


# --------------------------------------------------------------------------- routing

class TestRouting:
    """One connection, many callers: the header must follow the request, not the connection."""

    def test_plumbing_uses_the_service_account(self, minted, captured):
        build, seen = captured
        _send(build(CONFIG), "/mcp", {"method": "initialize"})
        assert seen["/mcp"].startswith("Bearer SERVICE-AT")

    def test_a_user_turn_uses_that_users_token(self, minted, captured):
        build, seen = captured
        with turn_token_scope(_user("AT-zhangsan")):
            _send(build(CONFIG), "/mcp", {"method": "tools/call"})
        assert seen["/mcp"] == "Bearer AT-zhangsan"

    def test_the_service_credential_returns_once_the_turn_ends(self, minted, captured):
        """No user token may leak onto the shared connection after its turn."""
        build, seen = captured

        async def drive():
            async with build(CONFIG) as client:  # one connection, two phases
                with turn_token_scope(_user("AT-zhangsan")):
                    await client.post("https://k3.example/in-turn", json={"method": "tools/call"})
                await client.post("https://k3.example/after", json={"method": "ping"})

        asyncio.run(drive())
        assert seen["/in-turn"] == "Bearer AT-zhangsan"
        assert seen["/after"].startswith("Bearer SERVICE-AT")

    def test_concurrent_users_do_not_share(self, minted, captured):
        """The property that makes this safe for a multi-user gateway."""
        build, seen = captured
        client = build(CONFIG)

        async def one(path: str, token: str):
            with turn_token_scope(_user(token)):
                await client.post(f"https://k3.example{path}", json={"method": "tools/call"})

        async def drive():
            async with client:
                await asyncio.gather(one("/alice", "AT-alice"), one("/bob", "AT-bob"))

        asyncio.run(drive())
        assert seen["/alice"] == "Bearer AT-alice"
        assert seen["/bob"] == "Bearer AT-bob"

    def test_a_service_scoped_server_never_sends_the_user_token(self, minted, captured):
        """``auth_scope: service`` asked to act as the integration account, so a running turn
        must not put its token on that server's requests."""
        build, seen = captured
        with turn_token_scope(_user("AT-zhangsan")):
            _send(build({**CONFIG, "auth_scope": "service"}), "/mcp", {"method": "tools/call"})
        assert seen["/mcp"].startswith("Bearer SERVICE-AT")

    def test_an_expired_turn_token_sends_the_request_bare(self, minted, captured):
        """The token can expire BETWEEN the handler's refusal and this request. The request must
        then go out unauthenticated (the server's own 401) rather than be quietly served by the
        integration account, which would return data the asker is not entitled to."""
        build, seen = captured
        with turn_token_scope(_user(expires_in=-120)):
            _send(build(CONFIG), "/mcp", {"method": "tools/call"})
        assert seen["/mcp"] is None

    def test_a_static_credential_is_left_alone(self):
        assert credential_router("srv", {"auth_scope": "user",
                                         "headers": {"Authorization": "Bearer static"}}) is None


# --------------------------------------------------------------------------- the hook

class TestHook:
    def test_the_hook_never_raises_on_a_user_scoped_server(self, minted, captured):
        """Belt and braces for the teardown hazard: an exception inside the SDK's transport
        TaskGroup tears down the shared connection for every turn on it."""
        build, seen = captured
        with turn_token_scope(_user(expires_in=-120)):
            _send(build(CONFIG), "/mcp", {"method": "tools/call"})
        assert seen["/mcp"] is None   # no exception escaped

    def test_only_the_async_flow_is_overridden(self, minted):
        """Every client mounting this hook is an AsyncClient; the base sync flow is unreachable."""
        hook = credential_router("kingdee", CONFIG)
        own = type(hook).__dict__
        assert "async_auth_flow" in own
        assert "auth_flow" not in own

    def test_the_hook_holds_a_plan_not_a_token(self, minted):
        """It outlives every turn, so capturing a token would pin one user's credential."""
        hook = credential_router("kingdee", CONFIG)
        assert _user("AT-zhangsan").access_token not in repr(hook.__dict__)

    def test_an_oauth_provider_owns_the_slot(self, minted):
        """httpx takes one ``auth=``; two hooks would each write Authorization."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("both")
        server._config = {"auth_scope": "user", "auth": "oauth"}
        sentinel = object()
        assert server._request_auth(sentinel) is sentinel

    def test_no_hook_without_an_opt_in(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("plain")
        server._config = {"url": "https://mcp.example/mcp"}
        assert server._request_auth(None) is None

    def test_a_partially_built_server_gets_no_hook(self):
        """``_request_auth`` is reachable before ``_prepare_run``; it must not raise."""
        from tools.mcp_tool import MCPServerTask

        assert MCPServerTask("bare")._request_auth(None) is None


# --------------------------------------------------------------------------- refusal

class TestPreDispatchRefusal:
    """A user-scoped call with no usable token is refused before any transport work.

    The refusal must NOT live in the HTTP auth hook: that hook runs inside the SDK transport's
    TaskGroup, so raising there tears down the shared connection for every turn on it.
    """

    def test_no_turn_is_not_a_refusal(self, minted):
        # CLI/cron/messaging never run user-scoped calls as a person.
        assert policy.refusal("kingdee", CONFIG) is None

    def test_a_turn_with_a_token_proceeds(self, minted):
        with turn_token_scope(_user()):
            assert policy.refusal("kingdee", CONFIG) is None

    def test_a_turn_with_an_expired_token_is_refused(self, minted):
        with turn_token_scope(_user(expires_in=-120)):
            message = policy.refusal("kingdee", CONFIG)
        assert message is not None
        assert "sign in again" in message          # actionable for the model
        assert "service account" in message        # states that no fallback happened

    def test_the_placeholder_binds_a_refusal_not_an_anonymous_turn(self, minted):
        """The distinction the whole module turns on: an anonymous turn may use the service
        credential; a refused one may not."""
        with turn_token_scope(TurnToken(access_token="", issuer="i", subject="s",
                                        expires_at=0)):
            assert policy.refusal("kingdee", CONFIG) is not None

    def test_a_service_scoped_server_is_never_refused(self, minted):
        with turn_token_scope(_user(expires_in=-120)):
            assert policy.refusal("k", {"auth_scope": "service"}) is None

    def test_an_unconfigured_server_is_never_refused(self):
        with turn_token_scope(_user(expires_in=-120)):
            assert policy.refusal("plain", {}) is None

    def test_a_config_error_is_reported_by_the_connect_path_not_here(self, minted):
        """Refusing here would hide the real problem behind a sign-in prompt."""
        with turn_token_scope(_user(expires_in=-120)):
            assert policy.refusal("srv", {"service_credentials": {"token_url": TOKEN_URL}}) is None


class TestRefusalForServerName:
    """``refusal_for`` is the name-only entry point the tool handler calls."""

    def _install(self, name: str, config: dict):
        from tools.mcp_tool_common import _core
        from tools.mcp_tool_scope import _resolve_server_key

        key = _resolve_server_key(name)
        _core._lazy_server_configs[key] = config
        return key

    def test_a_user_scoped_ledger_entry_without_a_token_is_refused(self, minted):
        from tools.mcp_tool_common import _core

        key = self._install("refuse-srv", CONFIG)
        try:
            with turn_token_scope(_user(expires_in=-120)):
                message = policy.refusal_for("refuse-srv")
            assert message is not None and "sign in again" in message
        finally:
            _core._lazy_server_configs.pop(key, None)

    def test_a_turn_with_a_token_proceeds(self, minted):
        from tools.mcp_tool_common import _core

        key = self._install("ok-srv", CONFIG)
        try:
            with turn_token_scope(_user()):
                assert policy.refusal_for("ok-srv") is None
        finally:
            _core._lazy_server_configs.pop(key, None)

    def test_an_unknown_server_is_never_refused(self):
        with turn_token_scope(_user(expires_in=-120)):
            assert policy.refusal_for("no-such-server") is None


# --------------------------------------------------------------------------- ledger lookup

class TestServerConfigLookup:
    """The precheck resolves a server's config from the ledger — and must not connect it.

    ``discovery._get_connected_server_for_call`` is the lazy-start entry point: asking it whether
    a server is connected *connects it*. A precheck built on that call would do the transport
    work it exists to prevent, for a call it is about to refuse.
    """

    def test_an_unknown_server_resolves_to_nothing(self):
        assert router_module._ledger_config("never-seen") == {}

    def test_a_connected_server_reports_its_rendered_config(self):
        from tools.mcp_tool_common import _core
        from tools.mcp_tool_scope import _resolve_server_key

        class _Server:
            _config = {**CONFIG, "url": "https://mcp.example/mcp"}

        key = _resolve_server_key("live-srv")
        _core._servers[key] = _Server()
        try:
            assert router_module._ledger_config("live-srv")["auth_scope"] == "user"
        finally:
            _core._servers.pop(key, None)

    def test_a_lazy_server_reports_its_cached_config(self):
        from tools.mcp_tool_common import _core
        from tools.mcp_tool_scope import _resolve_server_key

        key = _resolve_server_key("lazy-srv")
        _core._lazy_server_configs[key] = CONFIG
        try:
            assert router_module._ledger_config("lazy-srv")["auth_scope"] == "user"
        finally:
            _core._lazy_server_configs.pop(key, None)

    def test_the_lookup_never_reaches_the_connect_entry_point(self, monkeypatch):
        from tools import mcp_tool_discovery

        def _fail(*_args, **_kwargs):
            raise AssertionError("the precheck must never connect a server")

        monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", _fail)
        assert router_module._ledger_config("whatever") == {}

    @pytest.mark.parametrize("exc", [LookupError("no key"), AttributeError("torn"),
                                     RuntimeError("registry")])
    def test_an_unreadable_ledger_yields_nothing(self, monkeypatch, exc):
        """Best effort by design: skip the check rather than block a legitimate call."""
        from tools import mcp_tool_scope

        def _boom(*_args, **_kwargs):
            raise exc

        monkeypatch.setattr(mcp_tool_scope, "_resolve_server_key", _boom)
        assert router_module._ledger_config("srv") == {}


# --------------------------------------------------------------------------- service mint

class TestServiceCredential:
    """The machine credential: minted on demand, cached, refreshed ahead of expiry."""

    def _spec(self):
        return service.parse_service_credentials("k", CONFIG)

    def test_it_mints_once_and_caches(self, minted):
        minter = service.minter_for(self._spec())
        assert asyncio.run(minter.async_token()) == asyncio.run(minter.async_token())
        assert minted[0] == 1

    def test_it_re_mints_after_expiry(self, minted):
        minter = service.minter_for(self._spec())
        first = asyncio.run(minter.async_token())
        minter._expires_at = time.time() - 1
        assert asyncio.run(minter.async_token()) != first

    def test_one_minter_per_spec(self, minted):
        assert service.minter_for(self._spec()) is service.minter_for(self._spec())

    def test_a_failed_mint_yields_none(self, minted, monkeypatch):
        """A failed mint must not break the connection: the request goes out bare so the
        server's own 401 is the error the operator sees."""
        monkeypatch.setattr(service, "_post_form",
                            lambda url, data: (_ for _ in ()).throw(OSError("idp down")))
        assert asyncio.run(service.minter_for(self._spec()).async_token()) is None

    def test_a_response_without_an_access_token_is_an_error(self, minted, monkeypatch):
        monkeypatch.setattr(service, "_post_form", lambda url, data: {"expires_in": 60})
        assert asyncio.run(service.minter_for(self._spec()).async_token()) is None

    def test_the_secret_comes_from_the_named_variable(self, minted, monkeypatch):
        seen: dict = {}

        def _post(url, data):
            seen.update(data)
            return {"access_token": "at", "expires_in": 60}

        monkeypatch.setattr(service, "_post_form", _post)
        monkeypatch.setenv("HERMES_MCP_CLIENT_SECRET", "s3cret")
        asyncio.run(service.minter_for(self._spec()).async_token())
        assert seen["grant_type"] == "client_credentials"
        assert seen["client_id"] == "hermes-mcp-client"
        assert seen["client_secret"] == "s3cret"

    def test_at_most_one_mint_under_a_concurrent_burst(self, minted):
        import concurrent.futures

        minter = service.minter_for(self._spec())

        def _drive(_):
            return asyncio.run(minter.async_token())

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            tokens = list(pool.map(_drive, range(8)))
        assert minted[0] == 1
        assert len(set(tokens)) == 1

    def test_a_missing_secret_warns_and_names_the_variable(self, minted, monkeypatch, caplog):
        """An empty secret is an ``unauthorized_client`` from the IdP, which reads like a
        permissions problem; the log must name the missing .env entry instead."""
        import logging

        monkeypatch.setattr(service, "_read_secret", lambda _name: "")
        with caplog.at_level(logging.WARNING):
            asyncio.run(service.minter_for(self._spec()).async_token())
        assert any("HERMES_MCP_CLIENT_SECRET" in r.message for r in caplog.records)

    def test_the_secret_is_read_from_dotenv_not_only_the_bound_scope(self, monkeypatch):
        """Reading the bound profile scope alone misses a secret kept in ``.env``, and the mint
        then goes out with no secret and Keycloak answers 401."""
        monkeypatch.delenv("HERMES_MCP_CLIENT_SECRET", raising=False)
        monkeypatch.setattr("hermes_cli.config.get_env_value",
                            lambda key: "from-dotenv" if key == "HERMES_MCP_CLIENT_SECRET" else None)
        assert service._read_secret("HERMES_MCP_CLIENT_SECRET") == "from-dotenv"

    def test_no_env_name_means_no_secret(self):
        assert service._read_secret("") == ""


# --------------------------------------------------------------------------- collaborators

class TestPolicyCollaborators:
    """The collaborators are injectable — the reason this is a class, not a bag of functions."""

    def test_a_replaced_user_token_source_is_honoured(self, minted):
        custom = CredentialPolicy(user_token=lambda: "EXTERNAL-AT", in_user_turn=lambda: True)
        with turn_token_scope(_user()):   # the built-in reader would say otherwise
            assert custom.refusal("k", CONFIG) is None
        plan = custom.plan_for("k", CONFIG)
        assert plan is not None and plan.service is not None   # the service path still works

    def test_a_policy_never_in_a_turn_never_refuses(self, minted):
        custom = CredentialPolicy(in_user_turn=lambda: False, user_token=lambda: None)
        assert custom.refusal("k", CONFIG) is None

    def test_a_replaced_config_reader_removes_the_ledger_dependency(self, minted, monkeypatch):
        from tools import mcp_tool_scope

        def _fail(*_args, **_kwargs):
            raise AssertionError("the injected reader must be used instead of the ledger")

        monkeypatch.setattr(mcp_tool_scope, "_resolve_server_key", _fail)
        custom = CredentialPolicy(server_config=lambda _name: CONFIG)
        with turn_token_scope(_user(expires_in=-120)):
            assert custom.refusal_for("anything") is not None

    def test_an_injected_minter_bypasses_the_real_token_endpoint(self, minted):
        class _FakeMinter:
            async def async_token(self):
                return "FAKE-SERVICE-AT"

        custom = CredentialPolicy(minter_for=lambda _spec: _FakeMinter(),
                                  in_user_turn=lambda: False)
        plan = custom.plan_for("k", CONFIG)
        assert plan.service is not None
        assert asyncio.run(custom.async_token_for(plan)) == "FAKE-SERVICE-AT"


# --------------------------------------------------------------------------- end to end

class TestEndToEnd:
    """The invariant the whole feature exists for: a turn's token reaches an MCP POST header."""

    def test_a_turn_token_reaches_the_request_header(self, minted, captured):
        build, seen = captured
        with turn_token_scope(_user("AT-end-to-end")):
            _send(build(CONFIG), "/mcp", {"method": "tools/call"})
        assert seen["/mcp"] == "Bearer AT-end-to-end"

    def test_an_exported_plan_shape_is_stable(self, minted):
        """``CredentialPlan`` is what the hook captures; keep its public shape pinned."""
        plan = policy.plan_for("k", CONFIG)
        assert isinstance(plan, CredentialPlan)
        assert plan.server_name == "k"
        assert plan.auth_scope == "user"