"""End-to-end integration test for the headless ``client_credentials`` grant.

The unit tests in ``test_mcp_oauth_client_credentials.py`` cover construction
and dispatch. This module covers the runtime path: the **real** MCP SDK
provider built by ``tools.mcp_oauth.build_oauth_auth``, driven by a **real**
``AsyncClient`` from the SDK's own HTTP stack, against a **real** HTTP server
on loopback. Nothing is mocked — no ``MockTransport``, no patched SDK
internals.

The sequence exercised is the one the feature promises:

    GET  /mcp                      -> 401 + WWW-Authenticate: resource_metadata
    GET  <that metadata URL>       -> authorization_servers
    GET  .well-known/oauth-authorization-server -> token_endpoint
    POST /token                    (grant_type=client_credentials)
    GET  /mcp                      (Authorization: Bearer <minted>) -> 200

The protected-resource metadata is served **only** at an unguessable path
advertised in ``WWW-Authenticate``, so the RFC 9728 hint is load-bearing: if
the client ignored it, the SDK's well-known fallbacks would 404 and discovery
would fail. The fake authorization server also verifies the client credentials
it is sent, so "the flow completed" means the request was actually acceptable.

Hermetic: an ephemeral port on 127.0.0.1, torn down with the context manager.
No outbound network, no sleeps, no wall-clock dependence.
"""

from __future__ import annotations

import base64
import http.server
import json
import socketserver
import sys
import threading
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import pytest

from tools import mcp_oauth as mo
from tools.mcp_tool import sdk_httpx

# The provider is an ``httpx.Auth`` subclass built by the MCP SDK, and mcp 2.0
# moved that stack from ``httpx`` to ``httpx2``. A client from the other
# distribution rejects it outright ("Invalid \"auth\" argument"), so this
# harness has to drive it with the same flavour the SDK hands its own
# transports — which is exactly what tools/mcp_tool.py does at runtime.
# Resolving it here keeps the test honest on either SDK generation.
httpx = sdk_httpx()

CLIENT_ID = "test-client"
CLIENT_SECRET = "s3cr3t"
# Deliberately unguessable: reachable only via the WWW-Authenticate hint.
PRM_PATH = "/oauth-metadata/9f3c1a/resource"


class _ASState:
    """Mutable state shared with the request handler (per-server instance)."""

    def __init__(self) -> None:
        self.port: int = 0
        self.requests: list[tuple[str, str]] = []   # (method, path)
        self.token_bodies: list[dict] = []          # parsed /token form bodies
        self.token_raw_bodies: list[str] = []       # unparsed, to spot repeats
        self.token_auth_headers: list[str | None] = []
        self.resource_auth_headers: list[str | None] = []
        self.issued: list[str] = []                 # access tokens handed out
        self.revoked: set[str] = set()
        self.mint_count = 0
        self.reject_credentials = False

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def valid(self, token: str | None) -> bool:
        return bool(token) and token in self.issued and token not in self.revoked

    def credentials_ok(self, form: dict, auth_header: str | None) -> bool:
        """Accept either RFC 6749 client authentication method, and only those."""
        if self.reject_credentials:
            return False
        if auth_header and auth_header.startswith("Basic "):
            expected = base64.b64encode(
                f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
            ).decode()
            return auth_header == f"Basic {expected}"
        # client_secret_post: BOTH fields are required to identify the client.
        return (
            form.get("client_id") == CLIENT_ID
            and form.get("client_secret") == CLIENT_SECRET
        )


def _handler_cls(state: _ASState):
    class _H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence stderr noise in test output
            pass

        # -- helpers ------------------------------------------------------
        def _json(self, status: int, payload: dict, headers: dict | None = None):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _bearer(self) -> str | None:
            raw = self.headers.get("Authorization")
            if raw and raw.startswith("Bearer "):
                return raw[len("Bearer "):]
            return None

        # -- routes -------------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            state.requests.append(("GET", path))

            if path == "/mcp":
                state.resource_auth_headers.append(self.headers.get("Authorization"))
                if state.valid(self._bearer()):
                    return self._json(200, {"ok": True})
                return self._json(
                    401,
                    {"error": "unauthorized"},
                    {"WWW-Authenticate": (
                        f'Bearer resource_metadata="{state.base}{PRM_PATH}"'
                    )},
                )

            if path == PRM_PATH:
                return self._json(200, {
                    # Deliberately NOT the canonical server URL: asserting this
                    # value in the token request proves the resource came from
                    # the discovered document, not from the configured URL.
                    "resource": f"{state.base}/",
                    "authorization_servers": [state.base],
                })

            if path == "/.well-known/oauth-authorization-server":
                return self._json(200, {
                    "issuer": state.base,
                    "authorization_endpoint": f"{state.base}/authorize",
                    "token_endpoint": f"{state.base}/token",
                    "grant_types_supported": ["client_credentials"],
                    "response_types_supported": ["code"],
                })

            return self._json(404, {"error": "not_found"})

        def do_POST(self):
            path = urlparse(self.path).path
            state.requests.append(("POST", path))

            if path == "/token":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode()
                form = {k: v[0] for k, v in parse_qs(raw).items()}
                auth_header = self.headers.get("Authorization")
                state.token_raw_bodies.append(raw)
                state.token_bodies.append(form)
                state.token_auth_headers.append(auth_header)

                # This grant issues no refresh_token, so the client must never
                # try to redeem one.
                if form.get("grant_type") != "client_credentials":
                    return self._json(400, {"error": "unsupported_grant_type"})
                if not state.credentials_ok(form, auth_header):
                    return self._json(401, {"error": "invalid_client"})

                state.mint_count += 1
                token = f"access-token-{state.mint_count}"
                state.issued.append(token)
                return self._json(200, {
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                })

            # Dynamic client registration must never be reached: the grant
            # supplies client_id/client_secret up front.
            return self._json(500, {"error": "unexpected_endpoint"})

    return _H


@contextmanager
def _auth_server():
    """Run the fake resource + authorization server on loopback."""
    state = _ASState()
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _handler_cls(state))
    state.port = httpd.server_address[1]
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _provider(state: _ASState, monkeypatch, **overrides):
    """Build the real provider through Hermes' own config path."""
    # Belt and braces: if the M2M branch ever regressed, the interactive
    # provider would try to open a browser and block on a callback server.
    # Forcing non-interactive turns that into a loud failure, not a hang.
    monkeypatch.setattr(mo, "_is_interactive", lambda: False)
    cfg = {
        "grant": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        **overrides,
    }
    auth = mo.build_oauth_auth("gw", f"{state.base}/mcp", cfg)
    assert auth is not None, "SDK client_credentials extension unavailable"
    return auth


def _paths(state: _ASState) -> list[str]:
    return [p for _, p in state.requests]


def _assert_subsequence(actual: list[str], expected: list[str]) -> None:
    """Assert *expected* appears in order within *actual*.

    A subsequence rather than equality: the load-bearing hops must all happen
    in order, but the SDK is free to try additional discovery fallbacks
    without that counting as a Hermes regression.
    """
    remaining = list(expected)
    for path in actual:
        if remaining and path == remaining[0]:
            remaining.pop(0)
    assert not remaining, (
        f"missing {remaining} in order; actual sequence was {actual}"
    )


@pytest.mark.asyncio
async def test_401_discovery_token_exchange_and_authenticated_retry(
    tmp_path, monkeypatch
):
    """The full runtime path, asserted request by request."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _auth_server() as state:
        auth = _provider(state, monkeypatch)

        async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
            response = await client.get(f"{state.base}/mcp")

        assert response.status_code == 200
        assert response.json() == {"ok": True}

        # 1. Unauthenticated probe, both discovery hops, exchange, retry.
        _assert_subsequence(_paths(state), [
            "/mcp",
            PRM_PATH,
            "/.well-known/oauth-authorization-server",
            "/token",
            "/mcp",
        ])

        # 2. Dynamic client registration is skipped entirely.
        assert "/register" not in _paths(state)

        # 3. A real client_credentials exchange, accepted by the server.
        assert len(state.token_bodies) == 1
        body = state.token_bodies[0]
        assert body["grant_type"] == "client_credentials"
        # Credentials travel in the Authorization header, not the body.
        assert "client_secret" not in body
        expected = base64.b64encode(
            f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
        ).decode()
        assert state.token_auth_headers[0] == f"Basic {expected}"
        # The resource comes from the discovered PRM document, not from the
        # configured server URL (the fake advertises a different value).
        assert body["resource"] == f"{state.base}/"

        # 4. First call carried no credentials; the retry carries the minted
        #    token — this is what proves the retry re-authenticated.
        assert state.resource_auth_headers[0] is None
        assert state.resource_auth_headers[1] == f"Bearer {state.issued[0]}"


@pytest.mark.asyncio
async def test_configured_scope_reaches_the_token_request(tmp_path, monkeypatch):
    """``oauth.scope`` must survive the SDK's scope-selection strategy.

    The flow assigns its own scope choice (WWW-Authenticate scope, else the
    metadata's ``scopes_supported``, else nothing) over ``client_metadata``
    before every authorization — so a scope set at construction is otherwise
    dropped, and the documented ``--scope`` flag would do nothing. Here the
    fake advertises no scopes at all, so anything in the body can only have
    come from the configuration.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _auth_server() as state:
        auth = _provider(state, monkeypatch, scope="gateway.read gateway.write")

        async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
            response = await client.get(f"{state.base}/mcp")

        assert response.status_code == 200
        assert state.token_bodies[0]["scope"] == "gateway.read gateway.write"


@pytest.mark.asyncio
async def test_client_secret_post_sends_both_credentials_in_the_body(
    tmp_path, monkeypatch
):
    """``client_secret_post`` puts BOTH credentials in the form body.

    RFC 6749 §2.3.1: a client authenticating this way must send ``client_id``
    *and* ``client_secret`` in the request body. The 1.x SDK sent only the
    secret, leaving the authorization server unable to identify the caller
    (modelcontextprotocol/python-sdk#2128); 2.x fixes it in
    ``prepare_token_auth`` (#2185), which is why Hermes no longer carries a
    repair of its own. This is the guard on that removal: the fake rejects a
    body without ``client_id``, so an SDK that regressed here would fail the
    test loudly instead of reaching an authorization server as
    ``invalid_client``.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _auth_server() as state:
        auth = _provider(
            state, monkeypatch, token_endpoint_auth_method="client_secret_post"
        )

        async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
            response = await client.get(f"{state.base}/mcp")

        assert response.status_code == 200
        body = state.token_bodies[0]
        assert body["client_id"] == CLIENT_ID
        assert body["client_secret"] == CLIENT_SECRET
        assert body["grant_type"] == "client_credentials"
        # Credentials are in the body for this method, never duplicated as Basic.
        assert state.token_auth_headers[0] is None
        # The rebuilt body must not corrupt the rest of the exchange.
        assert body["resource"] == f"{state.base}/"
        assert state.resource_auth_headers[-1] == f"Bearer {state.issued[0]}"


@pytest.mark.asyncio
async def test_reserved_characters_survive_the_form_encoding(tmp_path, monkeypatch):
    """Form-encoding the token body must not corrupt the credentials.

    A secret containing characters that are meaningful in a form-encoded
    payload (``&``, ``=``, ``+``, ``%``, spaces, non-ASCII) is where encoding
    goes wrong. The fake server compares the decoded values byte for byte and
    answers ``invalid_client`` on any mismatch, so corruption fails the test
    rather than going unnoticed. Hermes reads real secrets from ``${VAR}``, so
    an operator can perfectly well hold one of these.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    nasty = "a+b=c%20d&e f/g?h#ié中~_-.*"
    # The fake server validates against this module-level constant.
    monkeypatch.setattr(sys.modules[__name__], "CLIENT_SECRET", nasty)

    with _auth_server() as state:
        auth = _provider(
            state,
            monkeypatch,
            client_secret=nasty,
            token_endpoint_auth_method="client_secret_post",
        )

        async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
            response = await client.get(f"{state.base}/mcp")

        assert response.status_code == 200          # server accepted the creds
        body = state.token_bodies[0]
        assert body["client_secret"] == nasty
        assert body["client_id"] == CLIENT_ID
        assert body["grant_type"] == "client_credentials"


@pytest.mark.asyncio
async def test_rejected_token_is_reminted_without_a_refresh_token(
    tmp_path, monkeypatch
):
    """A rejected token triggers a fresh mint — never a refresh_token grant.

    ``client_credentials`` responses carry no ``refresh_token``, so recovery
    can only be a new exchange. The fake answers 400 to any other grant_type,
    so an attempted refresh would fail the run rather than pass silently.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _auth_server() as state:
        auth = _provider(state, monkeypatch)

        async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
            first = await client.get(f"{state.base}/mcp")
            assert first.status_code == 200

            # The server revokes it (rotated secret, restarted gateway, …).
            state.revoked.add(state.issued[0])

            second = await client.get(f"{state.base}/mcp")

        assert second.status_code == 200
        assert state.mint_count == 2
        assert state.issued[0] != state.issued[1]
        assert state.resource_auth_headers[-1] == f"Bearer {state.issued[1]}"
        assert all(
            b["grant_type"] == "client_credentials" for b in state.token_bodies
        )


@pytest.mark.asyncio
async def test_rejected_credentials_surface_as_an_error(tmp_path, monkeypatch):
    """Bad credentials fail loudly instead of hanging or looping.

    This is the failure the M2M auth-failure message exists for: the exchange
    itself is refused, so there is nothing to retry.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _auth_server() as state:
        state.reject_credentials = True
        auth = _provider(state, monkeypatch)

        with pytest.raises(Exception) as excinfo:
            async with httpx.AsyncClient(auth=auth, timeout=10.0) as client:
                await client.get(f"{state.base}/mcp")

        assert "invalid_client" in str(excinfo.value)
        assert state.mint_count == 0
        assert "/register" not in _paths(state)
