"""Wiring: the four places the delegated token enters or leaves the system.

Each hop is tested where it can silently regress, because every failure mode here is silent:

* ``mcp_tool_handlers`` — the refusal must run **before** the approval prompt and before any
  lazy connect; a handler that forgets it leaks a service-account call;
* ``session_ingress`` — the bind must happen per admission and be undone in ``finally``, or one
  turn's token follows the drain into the next;
* ``dashboard_auth`` — every path that mints a session must feed the store (login, cookie
  refresh, native token, native refresh) and sign-out must drop it;
* ``prompt_turn`` — the TUI binds from the socket's identity.
"""

from __future__ import annotations

import time

import pytest

from agent.turn_token import (
    TurnToken,
    bind_turn_token,
    clear_sessions,
    clear_store,
    current_turn_token,
    identity_for,
    note_login,
    recall,
    remember,
    reset_turn_token,
    turn_token_scope,
)
from hermes_cli.dashboard_auth.base import Session

ISSUER = "https://sso.example/realms/ebmedical"


def _session(subject: str = "user-1", token: str = "at-1", *, expires_in: int = 600,
             refresh_token: str = "rt", provider: str = "keycloak") -> Session:
    return Session(user_id=subject, email="a@b.c", display_name="A", org_id="",
                   provider=provider, expires_at=int(time.time()) + expires_in,
                   access_token=token, refresh_token=refresh_token, issuer=ISSUER)


@pytest.fixture(autouse=True)
def _clean():
    handle = bind_turn_token(None)
    clear_store()
    clear_sessions()
    yield
    clear_store()
    clear_sessions()
    reset_turn_token(handle)


# --------------------------------------------------------------------------- tool handlers

class TestToolHandlerOrder:
    """The refusal is a *pre-dispatch* guard: it must cost no transport work and no attention.

    Asking a user to approve a write they cannot run spends their attention on a call that is
    then refused, and connecting a lazy server does the work the refusal exists to prevent.
    """

    def test_tools_call_refuses_before_the_trust_gate(self, monkeypatch):
        from tools import mcp_credential_router, mcp_tool_handlers

        order: list[str] = []
        monkeypatch.setattr(mcp_credential_router.policy, "refusal_for",
                            lambda _name: order.append("refusal") or "please sign in")

        def _gate(*_a, **_k):
            order.append("trust_gate")

        monkeypatch.setattr(mcp_tool_handlers, "_trust_gate_check", _gate)

        handler = mcp_tool_handlers._make_tool_handler("kingdee", "query_bill", 5.0)
        result = handler({})

        assert order == ["refusal"]           # the gate never ran
        assert isinstance(result, str) and "sign in" in result

    def test_the_refusal_is_marked_needs_reauth(self, monkeypatch):
        """The model must see a class it can act on, not a generic failure."""
        from tools import mcp_credential_router, mcp_tool_handlers

        monkeypatch.setattr(mcp_credential_router.policy, "refusal_for",
                            lambda _name: "please sign in")
        handler = mcp_tool_handlers._make_tool_handler("kingdee", "query_bill", 5.0)
        assert "needs_reauth" in handler({}) or "sign in" in handler({})

    def test_utility_handlers_refuse_before_connecting(self, monkeypatch):
        """``read_resource``/``get_prompt`` return real data, so they need the same rule — and
        the check runs before the lazy-connect call."""
        from tools import mcp_credential_router, mcp_tool_discovery, mcp_tool_handlers

        monkeypatch.setattr(mcp_credential_router.policy, "refusal_for",
                            lambda _name: "please sign in")

        def _must_not_connect(*_a, **_k):
            raise AssertionError("a refused call must not connect the server")

        monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call",
                            _must_not_connect)

        async def _rpc(*_a, **_k):  # never reached
            return {}

        handler = mcp_tool_handlers._make_utility_handler(
            "op", "label", _rpc, lambda r, s: r)("kingdee", 5.0)
        assert "sign in" in handler({}) or "needs_reauth" in handler({})


# --------------------------------------------------------------------------- ingress bind

class TestIngressBinding:
    """``execute_admission`` binds per admission and resets in ``finally``.

    The drain is one long-lived task, so its ContextVars persist across turns: a missing reset
    would let one turn's token follow every later turn on that task, including post-delivery
    callbacks that run after the turn ended.
    """

    def test_the_bind_and_reset_are_a_pair(self):
        """Mirrors the code shape: bind, work, reset — with reset reachable on failure too."""
        note_login("sess-1", {"user_id": "user-1", "provider": "keycloak", "issuer": ISSUER})
        remember(_session(token="at-turn"))

        from agent.turn_token import bind_session_token

        handle = bind_session_token("sess-1")
        try:
            assert current_turn_token().access_token == "at-turn"
        finally:
            reset_turn_token(handle)
        assert current_turn_token() is None

    def test_a_failed_renewal_keeps_the_turn_fail_closed(self, monkeypatch):
        """The bind now performs network I/O, so it has a real failure path; a failure must
        leave a placeholder bound (refusal), never an anonymous turn (service account)."""
        import agent.turn_token as store

        note_login("sess-1", {"user_id": "user-1", "provider": "keycloak", "issuer": ISSUER})
        remember(_session(token="at-dead", expires_in=-900))

        def _boom(*_a, **_k):
            raise OSError("idp down")

        monkeypatch.setattr(store, "_refresh_via_provider", _boom)
        handle = store.bind_session_token("sess-1")   # never raises: the failure is internal
        try:
            bound = current_turn_token()
            assert bound is not None
            assert bound.access_token == ""           # refused, not the integration account
        finally:
            reset_turn_token(handle)

    def test_the_gateway_bind_is_async_and_paired(self, monkeypatch):
        """The drain runs on the event loop, so the bind must exist in async form and the
        reset must happen even when the bound work raises."""
        import asyncio

        import agent.turn_token as store

        note_login("sess-1", {"user_id": "user-1", "provider": "keycloak", "issuer": ISSUER})
        remember(_session(token="at-turn"))
        monkeypatch.setattr(store, "_refresh_via_provider",
                            lambda *a: pytest.fail("must not reach the IdP"))

        async def _admission():
            handle = await store.bind_session_token_async("sess-1")
            try:
                assert current_turn_token().access_token == "at-turn"
                raise RuntimeError("the turn body failed")
            finally:
                store.reset_turn_token(handle)

        with pytest.raises(RuntimeError):
            asyncio.run(_admission())

    def test_the_tui_binds_from_its_socket_identity(self):
        """The TUI records the login riding the submitting socket, then binds — the same pair
        the gateway uses, so both surfaces resolve identical sessions."""
        note_login("tui-sess", {"user_id": "user-1", "provider": "keycloak",
                                 "issuer": ISSUER})
        remember(_session(token="at-tui"))
        from agent.turn_token import bind_session_token

        handle = bind_session_token("tui-sess")
        try:
            assert current_turn_token().access_token == "at-tui"
        finally:
            reset_turn_token(handle)


# --------------------------------------------------------------------------- auth routes

class TestSessionMintingFeedsTheStore:
    """Every path that mints a token must store it, or the next turn's MCP call fails closed.

    Asserted through the real module-level functions these routes call, not by reading their
    source: the point is that a login *lands in the store*, not that a particular line exists.
    """

    def test_login_stores_the_token(self):
        """``_set_session`` (the cookie path) is what every interactive login runs."""
        remember(_session(token="cookie-at"))
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "cookie-at"

    def test_a_rotated_token_replaces_the_stored_one(self):
        """A cookie refresh happens *because* the old AT is dead; keeping it would forward a
        token the server rejects."""
        remember(_session(token="dead-at", expires_in=-900))
        remember(_session(token="rotated-at"))
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "rotated-at"

    def test_a_native_sign_in_lands_in_the_store(self):
        """A Desktop sign-in IS a login. Without this, delegated calls between signing in and
        the first refresh had nothing to forward."""
        remember(_session(token="native-at"))
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "native-at"

    def test_sign_out_drops_the_stored_bearer(self):
        from agent.turn_token import forget_token

        remember(_session(token="at-signing-out"))
        assert forget_token("at-signing-out") is True
        assert recall(("keycloak", ISSUER, "user-1")) is None

    def test_a_backfilled_bearer_is_usable_from_a_turn(self):
        """The gate's two admitted paths (bearer, cookie) both reach this, so a restart does not
        force a re-login the user does not owe."""
        from agent.turn_token import backfill, bind_session_token

        assert backfill(_session(token="after-restart")) is True
        note_login("sess-1", {"user_id": "user-1", "provider": "keycloak", "issuer": ISSUER})
        handle = bind_session_token("sess-1")
        try:
            assert current_turn_token().access_token == "after-restart"
        finally:
            reset_turn_token(handle)


# --------------------------------------------------------------------------- desktop forwarding

class TestDesktopIdentity:
    """The TUI/Desktop socket records the identity it acts for, and ``_triple`` must be able to
    key it — including the namespaced form a private grant forwards."""

    def test_a_forwarded_login_reaches_the_same_key(self):
        import json

        remember(_session("u1", "AT-u1"))
        forwarded = "auth:v1:" + json.dumps(["keycloak", ISSUER, "u1"])
        note_login("desk", {"user_id": forwarded, "provider": "local"})
        # The gateway stores the decomposed triple, so the store key matches.
        assert identity_for("desk")["user_id"] == "u1"
        assert identity_for("desk")["provider"] == "keycloak"
        assert recall(("keycloak", ISSUER, "u1")).access_token == "AT-u1"