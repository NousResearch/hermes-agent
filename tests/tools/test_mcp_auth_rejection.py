"""A 401/403 on the MCP connect is an authorization refusal, not a transport mismatch (#119232).

mcp >= 2.0 folds a non-2xx it cannot parse as JSON-RPC into the opaque ``-32603 Server returned an
error response``, which is also the shape an SSE-only server's rejection takes. A 401 therefore
entered the Streamable-HTTP -> SSE fallback, the retry was refused the same way, and the connect
ended on the generic timeout message — indistinguishable from an unreachable host on every surface
(the Desktop MCP card among them). The status the recorder observed keeps the two apart.
"""

import httpx
import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import (
    McpAuthRequiredError, _is_http_auth_rejection, _is_streamable_http_rejection)


def _status_error(status):
    request = httpx.Request("POST", "http://127.0.0.1:1/mcp")
    return httpx.HTTPStatusError("err", request=request, response=httpx.Response(status, request=request))


class _SdkInternalError(Exception):
    """Shape of mcp.shared.exceptions.MCPError for an opaque initialize rejection."""

    def __init__(self):
        super().__init__("Server returned an error response")
        self.error = type("E", (), {"code": -32603})()


def _rejection(status):
    return {"status": status, "method": "POST", "url": "http://127.0.0.1:1/mcp", "body": '{"error":"unauthorized"}'}


class TestIsHttpAuthRejection:
    @pytest.mark.parametrize("status", [401, 403])
    def test_status_on_the_exception(self, status):
        assert _is_http_auth_rejection(_status_error(status)) is True

    @pytest.mark.parametrize("status", [401, 403])
    def test_status_only_in_the_recorded_rejection(self, status):
        """The usual case: the SDK's opaque error carries no status at all."""
        assert _is_http_auth_rejection(ExceptionGroup("g", [_SdkInternalError()]), _rejection(status)) is True

    @pytest.mark.parametrize("status", [400, 405, 500])
    def test_other_statuses_are_not_auth(self, status):
        assert _is_http_auth_rejection(_status_error(status)) is False
        assert _is_http_auth_rejection(_SdkInternalError(), _rejection(status)) is False

    def test_sdk_typed_auth_failure_counts(self):
        """An OAuth error the SDK names directly is the same refusal."""
        pytest.importorskip("mcp.client.auth")
        from mcp.client.auth import OAuthFlowError

        assert _is_http_auth_rejection(ExceptionGroup("g", [OAuthFlowError("token expired")])) is True

    def test_no_status_anywhere(self):
        assert _is_http_auth_rejection(TimeoutError("timed out"), None) is False
        assert _is_http_auth_rejection(TimeoutError("timed out"), {}) is False


class TestStreamableRejectionExcludesAuth:
    def test_recorded_401_is_not_a_transport_mismatch(self):
        """What the docstring already promised: 'auth errors never qualify'."""
        exc = ExceptionGroup("g", [_SdkInternalError()])
        assert _is_streamable_http_rejection(exc) is True  # opaque -32603 alone still looks like one
        assert _is_streamable_http_rejection(exc, _rejection(401)) is False

    def test_recorded_400_family_still_qualifies(self):
        exc = ExceptionGroup("g", [_SdkInternalError()])
        assert _is_streamable_http_rejection(exc, _rejection(400)) is True

    def test_direct_400_status_error_still_qualifies(self):
        assert _is_streamable_http_rejection(_status_error(405), None) is True


def _task(monkeypatch, http_exc, rejection):
    """MCPServerTask whose HTTP transport raises *http_exc* after the recorder saw *rejection*."""
    task = MCPServerTask("inspo")
    task._config = {}
    calls = []

    async def fake_serve(self, cm, label, timeout):
        calls.append(label)
        if label != "SSE":
            self._http_rejection.update(rejection)
            raise http_exc
        self._ever_connected = True
        return "shutdown"

    monkeypatch.setattr(MCPServerTask, "_serve_transport", fake_serve)
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_sse_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_build_oauth_auth", lambda self, *a: None)
    return task, calls


_CONFIG = {"url": "http://127.0.0.1:1/mcp", "connect_timeout": 1}


@pytest.mark.asyncio
async def test_401_reports_authentication_and_never_retries_over_sse(monkeypatch):
    task, calls = _task(monkeypatch, ExceptionGroup("g", [_SdkInternalError()]), _rejection(401))

    with pytest.raises(McpAuthRequiredError) as info:
        await task._run_http(dict(_CONFIG))

    assert calls == ["HTTP"]  # the SSE retry is refused the same way; it only burns a second timeout
    message = str(info.value)
    assert "requires authentication" in message
    assert "HTTP 401 from POST http://127.0.0.1:1/mcp" in message  # what the server actually said
    assert "hermes mcp login" in message


@pytest.mark.asyncio
async def test_400_family_rejection_still_falls_back_to_sse(monkeypatch):
    task, calls = _task(monkeypatch, ExceptionGroup("g", [_SdkInternalError()]), _rejection(400))

    assert await task._run_http(dict(_CONFIG)) == "shutdown"
    assert calls == ["HTTP", "SSE"]


class TestTheRefusalStaysAnAuthFailureDownstream:
    """``_classify_mcp_failure`` promises 'permanent ... auth 401/403': a refusal must park for
    credentials, not burn the reconnect ladder against a server that cannot recover without them."""

    def test_classified_permanent(self):
        from tools.mcp_tool_errors import _classify_mcp_failure, _is_auth_error

        exc = McpAuthRequiredError("MCP server 'inspo': the endpoint requires authentication (HTTP 401 ...)")
        assert _is_auth_error(exc) is True  # also selects the park's 'hermes mcp login' wording
        assert _classify_mcp_failure(exc) == "permanent"
        assert _classify_mcp_failure(ExceptionGroup("g", [exc])) == "permanent"

    def test_a_timeout_is_still_transient(self):
        from tools.mcp_tool_errors import _classify_mcp_failure

        assert _classify_mcp_failure(TimeoutError("timed out after 30s")) == "transient"
