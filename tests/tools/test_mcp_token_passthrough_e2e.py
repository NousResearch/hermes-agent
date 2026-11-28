"""End-to-end: a dashboard login becomes a bearer on an MCP request, and stops working when
it should.

The unit tests cover each hop; this file drives the whole chain so a break in the middle shows
up as one failing journey instead of six green hops:

    login -> store -> session note -> turn bind -> per-request hook -> Authorization header

and the three ways the chain must *not* produce a header.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time

import pytest

from agent.turn_token import (
    TurnToken,
    bind_session_token,
    clear_sessions,
    clear_store,
    current_turn_token,
    note_login,
    remember,
    reset_turn_token,
    turn_token_scope,
)
from hermes_cli.dashboard_auth.base import Session
from tools.mcp_credential_router import credential_router

ISSUER = "https://sso.example/realms/ebmedical"

CONFIG = {
    "auth_scope": "user",
    "service_credentials": {
        "token_url": "https://idp.example/realms/r/protocol/openid-connect/token",
        "client_id": "hermes-mcp-client",
        "client_secret_env": "HERMES_MCP_CLIENT_SECRET",
    },
}


@pytest.fixture(autouse=True)
def _clean():
    from agent.turn_token import bind_turn_token

    # The binding is a ContextVar that outlives a test in the same thread, so it is cleared on
    # both sides rather than only before.
    handle = bind_turn_token(None)
    clear_store()
    clear_sessions()
    yield
    clear_store()
    clear_sessions()
    reset_turn_token(handle)


@pytest.fixture
def service_mint(monkeypatch):
    """Stub the client_credentials endpoint so plumbing requests have a credential."""
    from tools import mcp_service_credentials as service

    service.clear_minters()

    def _post(url, data):
        return {"access_token": "SERVICE-AT", "expires_in": 600}

    monkeypatch.setattr(service, "_post_form", _post)
    yield
    service.clear_minters()


@pytest.fixture
def mcp_request(monkeypatch):
    """An SDK-httpx client that records the Authorization header each request carried."""
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    seen: dict[str, str | None] = {}

    async def handler(request):
        seen[request.url.path] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    def _post(path: str = "/mcp", config: dict = CONFIG, payload: dict | None = None):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                   auth=credential_router("kingdee", config))

        async def _drive():
            async with client:
                await client.post(f"https://k3.example{path}",
                                  json=payload or {"method": "tools/call"})

        asyncio.run(_drive())
        return seen.get(path)

    return _post


def _login(subject: str = "zhangsan", token: str = "AT-zhangsan", *, expires_in: int = 600,
           refresh_token: str = "RT") -> Session:
    return Session(user_id=subject, email="zhangsan@example.com", display_name="Zhang",
                   org_id="", provider="keycloak", expires_at=int(time.time()) + expires_in,
                   access_token=token, refresh_token=refresh_token, issuer=ISSUER)


# --------------------------------------------------------------------------- the happy path

class TestTheChainWorks:
    def test_a_browser_login_becomes_a_bearer_on_the_mcp_request(self, service_mint, mcp_request):
        """The whole feature, in one journey."""
        remember(_login())                                    # login -> store
        note_login("browser-session", {"user_id": "zhangsan", "provider": "keycloak",
                                       "issuer": ISSUER})      # which login drives the session
        handle = bind_session_token("browser-session")         # turn start
        try:
            assert mcp_request() == "Bearer AT-zhangsan"       # per-request hook
        finally:
            reset_turn_token(handle)

    def test_plumbing_still_uses_the_service_account(self, service_mint, mcp_request):
        """Connection traffic is not made on anyone's behalf."""
        remember(_login())
        assert mcp_request(payload={"method": "initialize"}) == "Bearer SERVICE-AT"

    def test_the_token_follows_the_request_not_the_connection(self, service_mint, mcp_request):
        """One client, two callers: the header must change with the turn."""
        remember(_login("alice", "AT-alice"))
        remember(_login("bob", "AT-bob"))

        with turn_token_scope(TurnToken("AT-alice", ISSUER, "alice", int(time.time()) + 600)):
            in_alice = mcp_request("/alice")
        with turn_token_scope(TurnToken("AT-bob", ISSUER, "bob", int(time.time()) + 600)):
            in_bob = mcp_request("/bob")

        assert (in_alice, in_bob) == ("Bearer AT-alice", "Bearer AT-bob")


# --------------------------------------------------------------------------- the chain refuses

class TestTheChainRefuses:
    def test_no_login_means_no_user_credential(self, service_mint, mcp_request):
        """CLI/cron/messaging: there is nobody to act as, so plumbing semantics apply."""
        from tools.mcp_credential_router import policy

        assert policy.refusal("kingdee", CONFIG) is None
        assert mcp_request() == "Bearer SERVICE-AT"

    def test_an_expired_turn_token_refuses_the_call(self, service_mint):
        from tools.mcp_credential_router import policy

        with turn_token_scope(TurnToken("AT-dead", ISSUER, "zhangsan", int(time.time()) - 900)):
            message = policy.refusal("kingdee", CONFIG)
        assert message is not None and "sign in again" in message

    def test_an_expired_token_never_falls_back_to_the_service_account(self, service_mint,
                                                                    mcp_request):
        """The core safety property: a refused call must not become an integration-account call."""
        with turn_token_scope(TurnToken("AT-dead", ISSUER, "zhangsan", int(time.time()) - 900)):
            assert mcp_request() is None      # bare, so the server's own 401 is the error

    def test_a_service_scoped_server_never_sees_a_user_token(self, service_mint, mcp_request):
        """``auth_scope: service`` asked to be the integration account; a running turn must not
        change that."""
        remember(_login())
        with turn_token_scope(TurnToken("AT-zhangsan", ISSUER, "zhangsan",
                                        int(time.time()) + 600)):
            header = mcp_request(config={**CONFIG, "auth_scope": "service"})
        assert header == "Bearer SERVICE-AT"


# --------------------------------------------------------------------------- the idle return

class TestTheIdleReturn:
    """A browser left alone for hours has had no request to trigger the gate's refresh, so the
    turn is the first chance to renew — and refusing it would demand a re-login the user does
    not owe."""

    def test_a_stale_login_is_renewed_and_the_new_token_reaches_mcp(self, monkeypatch, mcp_request):
        import agent.turn_token as store

        remember(_login(token="AT-dead", expires_in=-900, refresh_token="RT-good"))
        note_login("browser-session", {"user_id": "zhangsan", "provider": "keycloak",
                                       "issuer": ISSUER})
        monkeypatch.setattr(store, "_refresh_via_provider",
                            lambda rt, p: _login(token="AT-fresh", refresh_token="RT-2"))

        handle = bind_session_token("browser-session")
        try:
            assert mcp_request() == "Bearer AT-fresh"
        finally:
            reset_turn_token(handle)

    def test_a_dead_refresh_token_still_refuses(self, monkeypatch, service_mint, mcp_request):
        """A dead RT is a re-login, never a silent downgrade."""
        import agent.turn_token as store
        from tools.mcp_credential_router import policy

        remember(_login(token="AT-dead", expires_in=-900, refresh_token="RT-dead"))
        note_login("browser-session", {"user_id": "zhangsan", "provider": "keycloak",
                                       "issuer": ISSUER})
        monkeypatch.setattr(store, "_refresh_via_provider", lambda rt, p: None)

        handle = bind_session_token("browser-session")
        try:
            assert policy.refusal("kingdee", CONFIG) is not None
            assert mcp_request() is None
        finally:
            reset_turn_token(handle)


# --------------------------------------------------------------------------- isolation

class TestIsolationAcrossTurns:
    def test_a_worker_thread_sees_the_turn_it_was_started_under(self, service_mint):
        """Tools run on executor workers whose context is copied from the turn's."""
        remember(_login())
        seen: dict = {}

        with turn_token_scope(TurnToken("AT-turn", ISSUER, "zhangsan", int(time.time()) + 600)):
            captured = contextvars.copy_context()

        def _worker():
            seen["token"] = current_turn_token().access_token if current_turn_token() else None

        thread = threading.Thread(target=lambda: captured.run(_worker))
        thread.start()
        thread.join()
        assert seen["token"] == "AT-turn"

    def test_the_drain_does_not_carry_one_turn_into_the_next(self):
        """The drain is one long-lived task, so ContextVars persist: without the reset, a
        post-delivery callback would run with the previous turn's token."""
        remember(_login("alice", "AT-alice"))
        remember(_login("bob", "AT-bob"))
        note_login("sess", {"user_id": "alice", "provider": "keycloak", "issuer": ISSUER})

        handle = bind_session_token("sess")
        assert current_turn_token().access_token == "AT-alice"
        reset_turn_token(handle)                        # the turn ends

        note_login("sess", {"user_id": "bob", "provider": "keycloak", "issuer": ISSUER})
        handle = bind_session_token("sess")
        try:
            assert current_turn_token().access_token == "AT-bob"
        finally:
            reset_turn_token(handle)

    def test_a_forwarded_desktop_login_resolves_the_browser_token(self, service_mint, mcp_request):
        """A private/native grant forwarding a dashboard login must land on the same key."""
        remember(_login())
        forwarded = "auth:v1:" + json.dumps(["keycloak", ISSUER, "zhangsan"])
        note_login("desktop-session", {"user_id": forwarded, "provider": "local"})

        handle = bind_session_token("desktop-session")
        try:
            assert mcp_request() == "Bearer AT-zhangsan"
        finally:
            reset_turn_token(handle)


# --------------------------------------------------------------------------- the serve task

class TestTheConnectionOutlivesTheTurn:
    """A server first connected during someone's turn must not keep that person's token.

    The serve task copies the context it is created in, and a lazy server connects on first use —
    i.e. inside a turn. Without the fix, that task's keepalive pings, dynamic tool refreshes and
    every later reconnect carried the token of whoever happened to trigger the connect, and kept
    carrying it after they signed out.
    """

    @staticmethod
    def _stub():
        from tools.mcp_tool_server_run import MCPServerRunMixin

        class _Stub(MCPServerRunMixin):
            def __init__(self):
                self._ready = asyncio.Event()
                self._error = None
                self._task = None

            async def run(self, config):
                self._ready.set()
                return current_turn_token()

        return _Stub()

    def test_the_serve_task_does_not_inherit_the_connecting_turn(self):
        stub = self._stub()

        async def _go():
            with turn_token_scope(TurnToken("AT-zhangsan", ISSUER, "zhangsan",
                                            int(time.time()) + 600)):
                await stub.start({})
            return await stub._task

        assert asyncio.run(_go()) is None

    def test_the_caller_keeps_its_own_token(self):
        """Clearing it for the task must not clear it for the turn that is running."""
        stub = self._stub()

        async def _go():
            with turn_token_scope(TurnToken("AT-zhangsan", ISSUER, "zhangsan",
                                            int(time.time()) + 600)):
                await stub.start({})
                return current_turn_token()

        assert asyncio.run(_go()).access_token == "AT-zhangsan"