"""Tests for ``_connect_failure_reason``: the one stable token ``get_mcp_status()`` reports per
failed MCP server, so status readers stop re-grepping the connect-error prose.

DNS / CONNECTION_REFUSED / TLS / TIMEOUT / HTTP_5XX / OTHER follow the structured
unreachability reasons an MCP control plane reports; the remaining classes are the failures
Hermes itself raises on connect (a missing stdio command, an invalid URL, a rejected
credential, a non-auth 4xx, a handshake that never completed).
"""

import errno

import pytest

from tools.mcp_tool_errors import (
    MCP_CONNECT_FAILURE_REASONS, InvalidMcpUrlError, NonMcpEndpointError, _connect_failure_reason,
    _format_connect_error)

_URL = "https://mcp.internal.example/mcp"

# ``(recorded connect-error text, reason)``. The strings are what ``_format_connect_error``
# stores for the real failure: its own wording for a missing executable, httpx's outer
# ``ConnectError`` joined with its cause, the SDK's message-less leaves named by type.
_CASES = [
    # ── missing stdio executable ──
    ("missing executable 'npx' (ensure Node.js is installed and PATH includes its bin directory, "
     "or set mcp_servers.docs.command to an absolute path)", "MISSING_COMMAND"),
    ("[Errno 2] No such file or directory: 'docker'", "MISSING_COMMAND"),
    ("spawn npx ENOENT", "MISSING_COMMAND"),
    ("'npx' is not recognized as an internal or external command,\r\noperable program or batch file.",
     "MISSING_COMMAND"),
    # ── config ──
    ("Invalid MCP URL for 'docs': scheme must be http or https, got 'tcp' ('tcp://host:8080')",
     "INVALID_URL"),
    ("Invalid MCP URL for 'docs': missing hostname ('http://:8080')", "INVALID_URL"),
    # ── rejected credential ──
    (f"Server error '401 Unauthorized' for url '{_URL}'", "AUTH"),
    (f"HTTP 401 from POST {_URL}: {{\"error\":\"invalid_token\"}}", "AUTH"),
    (f"HTTP 403 from POST {_URL}", "AUTH"),
    ("OAuthTokenError: token is expired or revoked", "AUTH"),
    ("UnauthorizedError: invalid api key", "AUTH"),
    ("OAuth skipped for MCP server 'docs'. Run `hermes mcp login docs` later to authenticate", "AUTH"),
    # ── HTTP status ──
    (f"HTTP 503 from POST {_URL}: upstream connect error or disconnect", "HTTP_5XX"),
    (f"Server error '502 Bad Gateway' for url '{_URL}'", "HTTP_5XX"),
    (f"HTTP 504 from POST {_URL}", "HTTP_5XX"),
    ("Error code: 500 - {'message': 'Internal Server Error'}", "HTTP_5XX"),
    ("unexpected status=503 Service Unavailable from the upstream proxy", "HTTP_5XX"),
    ("ReadError: read 512 bytes then EOF", "OTHER"),  # a byte count is not a status
    (f"HTTP 404 from POST {_URL}", "HTTP_4XX"),
    (f"Client error '405 Method Not Allowed' for url '{_URL}'", "HTTP_4XX"),
    (f"HTTP 400 from POST {_URL}: {{\"code\":-32020,\"message\":\"Unsupported MCP-Protocol-Version\"}}",
     "HTTP_4XX"),
    (f"HTTP 429 from POST {_URL}: rate limit exceeded", "HTTP_4XX"),
    # ── name resolution ──
    ("ConnectError: All connection attempts failed; [Errno -2] Name or service not known", "DNS"),
    ("ConnectError: All connection attempts failed; [Errno 11001] getaddrinfo failed", "DNS"),
    ("ConnectError: All connection attempts failed; [Errno -3] Temporary failure in name resolution",
     "DNS"),
    ("gaierror: [Errno 8] nodename nor servname provided, or not known", "DNS"),
    # ── nothing listening ──
    ("ConnectError: All connection attempts failed; [Errno 111] Connection refused", "CONNECTION_REFUSED"),
    ("ConnectError: All connection attempts failed; [Errno 146] Connection refused", "CONNECTION_REFUSED"),
    ("ConnectError: All connection attempts failed; [WinError 10061] No connection could be made "
     "because the target machine actively refused it", "CONNECTION_REFUSED"),
    # ── handshake never completed ──
    ("Connection closed", "HANDSHAKE_FAILED"),
    ("ClosedResourceError", "HANDSHAKE_FAILED"),
    ("EndOfStream", "HANDSHAKE_FAILED"),
    ("BrokenResourceError", "HANDSHAKE_FAILED"),
    (f"MCP server 'docs' at {_URL} returned Content-Type 'text/html', not an MCP response "
     "(expected one of: text/event-stream, application/json).", "HANDSHAKE_FAILED"),
    ("Unsupported protocol version from the server: 2025-06-18", "HANDSHAKE_FAILED"),
    ("MCP server 'docs' rejected initialize: method not found (-32601)", "HANDSHAKE_FAILED"),
    # ── TLS ──
    ("ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
     "local issuer certificate (_ssl.c:1028)", "TLS"),
    ("ConnectError: [SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1032)", "TLS"),
    ("ConnectError: [SSL: TLSV1_ALERT_INTERNAL_ERROR] tlsv1 alert internal error (_ssl.c:1045)", "TLS"),
    # ── timeout ──
    ("TimeoutError", "TIMEOUT"),
    ("ConnectTimeout: timed out", "TIMEOUT"),
    ("ReadTimeout: timed out", "TIMEOUT"),
    ("[Errno 110] Connection timed out", "TIMEOUT"),
    ("[WinError 10060] A connection attempt failed because the connected party did not properly "
     "respond after a period of time", "TIMEOUT"),
    # ── nothing specific ──
    ("", "OTHER"),
    (None, "OTHER"),
    ("All connection attempts failed", "OTHER"),
    ("something odd", "OTHER"),
]


class TestConnectFailureReason:
    @pytest.mark.parametrize("error, reason", _CASES)
    def test_recorded_message_maps_to_one_reason(self, error, reason):
        assert _connect_failure_reason(error) == reason

    def test_every_reason_is_reached_by_a_recorded_message(self):
        reasons = [reason for _, reason in _CASES]
        assert len({error for error, _ in _CASES}) == len(_CASES)  # every recorded message distinct
        assert set(reasons) == set(MCP_CONNECT_FAILURE_REASONS)  # no constant without a real message

    @pytest.mark.parametrize("error, expected", [
        # A credential rejection is AUTH, not "some 4xx".
        (f"HTTP 401 from POST {_URL}", "AUTH"),
        (f"Client error '403 Forbidden' for url '{_URL}'", "AUTH"),
        # A 504 is the server's gateway timing out, not this client's connect timeout.
        ("Server error '504 Gateway Timeout' for url 'https://mcp.internal.example/mcp'", "HTTP_5XX"),
        # A timed-out TLS handshake is still a timeout.
        ("ConnectError: SSL handshake timed out", "TIMEOUT"),
        # The OAuth flow itself timing out is a credential problem to fix with a login.
        ("OAuthFlowError: authorization request timed out", "AUTH"),
    ])
    def test_the_more_actionable_class_wins(self, error, expected):
        assert _connect_failure_reason(error) == expected

    @pytest.mark.parametrize("error", [
        "upstream request abc-401-def failed",  # an id, not a status code
        f"POST {_URL}/oauth-authorization-server",  # a URL path, not an OAuth failure
        "client_cert not found at 'C:\\certs\\client.pem'",
    ])
    def test_lookalikes_are_not_misread(self, error):
        assert _connect_failure_reason(error) == "OTHER"

    def test_always_returns_a_member_for_unexpected_input(self):
        for error in (object(), b"\xff\xfe", "x" * 10_000, 404, {"error": 500}):
            assert _connect_failure_reason(error) in MCP_CONNECT_FAILURE_REASONS


class TestReasonOfTheFormattedMessage:
    """The classifier reads what ``_note_connect_failure`` stores: ``_format_connect_error`` of the
    raised exception, anyio TaskGroup wrappers included."""

    @pytest.mark.parametrize("exc, reason", [
        (ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [
            ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")]), "CONNECTION_REFUSED"),
        (ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [
            FileNotFoundError(errno.ENOENT, "No such file or directory", "npx")]), "MISSING_COMMAND"),
        (NonMcpEndpointError(
            f"MCP server 'docs' at {_URL} returned Content-Type 'text/html', not an MCP response "
            "(expected one of: text/event-stream, application/json)."), "HANDSHAKE_FAILED"),
        (InvalidMcpUrlError(
            "Invalid MCP URL for 'docs': scheme must be http or https, got 'ftp' ('ftp://host')"),
         "INVALID_URL"),
    ])
    def test_real_exception_through_the_real_formatter(self, exc, reason):
        assert _connect_failure_reason(_format_connect_error(exc)) == reason


class TestStatusEntryReason:
    """``get_mcp_status()`` attaches the token to every failed entry."""

    def test_status_reports_one_structured_reason_per_failed_server(self, monkeypatch):
        """``reason`` rides along with ``error``, so a reader branches on a token instead of
        re-grepping the message. A parked server records no connect error, only its revival
        reason, and still gets one; a non-failed entry gets none."""
        from types import SimpleNamespace

        from tools import mcp_tool
        from tools import mcp_tool_config as _mcp_config
        from tools.mcp_tool_discovery import get_mcp_status

        monkeypatch.setattr(
            _mcp_config, "_load_mcp_config",
            lambda: {
                "refused": {"url": "https://mcp.internal.example/mcp"},
                "parked": {"command": "slow-mcp"},
                "needs_login": {"url": "https://mcp.example.test/mcp", "auth": "oauth"},
                "configured": {"command": "new-mcp"},
            },
        )
        monkeypatch.setitem(mcp_tool._servers, "parked", SimpleNamespace(
            session=None, _park_reason="after initial connection failures", _auth_park=None))
        monkeypatch.setitem(mcp_tool._servers, "needs_login", SimpleNamespace(
            session=None, _park_reason="after permanent initial failure", _auth_park=object()))
        with mcp_tool._lock:
            saved_errors = dict(mcp_tool._server_connect_errors)
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connect_errors["refused"] = "[Errno 111] Connection refused"

        try:
            statuses = {entry["name"]: entry for entry in get_mcp_status()}
        finally:
            with mcp_tool._lock:
                mcp_tool._server_connect_errors.clear()
                mcp_tool._server_connect_errors.update(saved_errors)

        assert statuses["refused"]["status"] == "failed"
        assert statuses["refused"]["reason"] == "CONNECTION_REFUSED"
        assert statuses["parked"]["status"] == "failed"
        assert statuses["parked"]["reason"] == "OTHER"
        assert statuses["needs_login"]["status"] == "failed"
        assert statuses["needs_login"]["reason"] == "AUTH"
        assert statuses["needs_login"]["needs_auth"] is True
        assert "reason" not in statuses["configured"]
