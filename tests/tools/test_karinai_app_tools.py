from __future__ import annotations

import io
import json
import urllib.error

from gateway.session_context import clear_session_vars, set_session_vars
from karinai.runtime.session_bridge import bind_karinai_run_context
from tools import karinai_app_tools


def _bind_gateway(url: str, token: str, expires_at: str = "") -> list:
    """Bind an api_server session with run-scoped app-tool gateway creds, as
    the api_server's _bind_api_server_session does: base session vars via
    set_session_vars, KarinAI vars via bind_karinai_run_context, one token list."""
    tokens = set_session_vars(platform="api_server", async_delivery=False)
    tokens += bind_karinai_run_context(
        app_tool_gateway={"url": url, "token": token, "expires_at": expires_at}
    )
    return tokens


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_app_tools_list_reports_unavailable_without_run_gateway() -> None:
    tokens = set_session_vars(platform="api_server", async_delivery=False)
    try:
        result = json.loads(karinai_app_tools._handle_list({}))
    finally:
        clear_session_vars(tokens)

    assert result["available"] is False


def test_app_tool_execute_posts_to_run_scoped_gateway(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": request.data,
                "timeout": timeout,
            }
        )
        return _FakeResponse({"successful": True, "data": {"messages": []}})

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway("http://backend.internal/internal/app-tools", "kat_secret")
    try:
        result = json.loads(
            karinai_app_tools._handle_execute(
                {"tool_slug": "GOOGLESUPER_FETCH_EMAILS", "arguments": {"max_results": 2}}
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["available"] is True
    assert result["successful"] is True
    assert calls[0]["url"] == "http://backend.internal/internal/app-tools/execute"
    assert calls[0]["headers"]["Authorization"] == "Bearer kat_secret"
    assert json.loads(calls[0]["body"].decode("utf-8")) == {
        "tool_slug": "GOOGLESUPER_FETCH_EMAILS",
        "arguments": {"max_results": 2},
    }


def test_api_server_bind_exposes_and_clears_app_tool_gateway() -> None:
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import get_session_env

    tokens = APIServerAdapter._bind_api_server_session(
        session_id="cnv_1",
        app_tool_gateway={
            "url": "http://backend.internal",
            "token": "kat_secret",
            "expires_at": "2026-07-05T12:00:00.000000Z",
        },
    )
    try:
        assert get_session_env("KARINAI_APP_TOOL_GATEWAY_URL") == "http://backend.internal"
        assert get_session_env("KARINAI_APP_TOOL_GATEWAY_TOKEN") == "kat_secret"
        assert get_session_env("KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT") == "2026-07-05T12:00:00.000000Z"
    finally:
        clear_session_vars(tokens)

    assert get_session_env("KARINAI_APP_TOOL_GATEWAY_TOKEN") == ""


# ---------------------------------------------------------------------------
# Review-fix coverage: never leak the gateway token / internal host to the
# model, and enforce token expiry agent-side before any HTTP call.
# ---------------------------------------------------------------------------

GATEWAY_URL = "http://backend.internal/internal/app-tools"
GATEWAY_HOST = "backend.internal"
SECRET_TOKEN = "kat_secret_topsecret_value"


def test_gateway_http_error_does_not_leak_token_or_url(monkeypatch) -> None:
    """A 401/403 from the gateway must not echo the token or internal host."""

    def fake_urlopen(request, timeout):
        # Empty body -> no structured detail -> static fallback message. The URL
        # and reason carry the internal host to prove the branch never echoes it.
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg=f"Forbidden for {GATEWAY_URL}",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN)
    try:
        raw_result = karinai_app_tools._handle_execute(
            {"tool_slug": "GOOGLESUPER_FETCH_EMAILS", "arguments": {}}
        )
    finally:
        clear_session_vars(tokens)

    result = json.loads(raw_result)
    assert result["available"] is False
    assert result["status"] == 403
    # Model-visible result must contain NEITHER the token NOR the gateway URL/host.
    assert SECRET_TOKEN not in raw_result
    assert "kat_secret" not in raw_result
    assert GATEWAY_HOST not in raw_result


def test_generic_gateway_error_returns_static_message(monkeypatch) -> None:
    """DNS/socket errors (str(exc) carries the host) collapse to a static message."""

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError(f"[Errno -2] Name or service not known: {GATEWAY_HOST}")

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN)
    try:
        raw_result = karinai_app_tools._handle_list({})
    finally:
        clear_session_vars(tokens)

    result = json.loads(raw_result)
    assert result["available"] is False
    assert result["error"] == "The KarinAI app tool gateway is unavailable."
    assert GATEWAY_HOST not in raw_result
    assert SECRET_TOKEN not in raw_result


def test_expired_gateway_returns_clean_message_without_http_call(monkeypatch) -> None:
    """A past expires_at yields the clean expired message and makes NO HTTP call."""
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _FakeResponse({"successful": True})

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN, expires_at="2000-01-01T00:00:00.000000Z")
    try:
        exec_result = json.loads(
            karinai_app_tools._handle_execute({"tool_slug": "X", "arguments": {}})
        )
        list_result = json.loads(karinai_app_tools._handle_list({}))
    finally:
        clear_session_vars(tokens)

    # No HTTP call was attempted for either tool.
    assert calls == []
    for result in (exec_result, list_result):
        assert result["available"] is False
        assert "expired" in result["error"].lower()
        assert "reconnect" in result["error"].lower()


def test_future_expiry_still_calls_gateway(monkeypatch) -> None:
    """A not-yet-elapsed expiry does not block the call (backend still enforces)."""
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _FakeResponse({"successful": True, "data": {}})

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN, expires_at="2999-01-01T00:00:00.000000Z")
    try:
        result = json.loads(
            karinai_app_tools._handle_execute({"tool_slug": "X", "arguments": {}})
        )
    finally:
        clear_session_vars(tokens)

    assert len(calls) == 1
    assert result["available"] is True


def test_token_never_appears_in_serialized_result(monkeypatch) -> None:
    """The raw token string must never appear in the tool result handed to the model."""

    def fake_urlopen(request, timeout):
        return _FakeResponse({"successful": True, "data": {"messages": []}})

    monkeypatch.setattr(karinai_app_tools.urllib.request, "urlopen", fake_urlopen)
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN)
    try:
        exec_result = karinai_app_tools._handle_execute(
            {"tool_slug": "X", "arguments": {"q": 1}}
        )
        list_result = karinai_app_tools._handle_list({})
    finally:
        clear_session_vars(tokens)

    for raw in (exec_result, list_result):
        assert SECRET_TOKEN not in raw
        assert "kat_secret" not in raw


def test_app_tools_advertised_only_when_token_bound() -> None:
    """Fix 3: the tools are advertised iff a run-scoped token is bound, and the
    definition caches never strip them from the token-bound (/v1/runs) path."""
    import model_tools
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    model_tools._clear_tool_defs_cache()
    app_tools = {"karinai_app_tools_list", "karinai_app_tool"}

    def advertised() -> set:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["karinai_app_integrations"], quiet_mode=True
        )
        return {d["function"]["name"] for d in defs} & app_tools

    # Unbound (CLI / messaging / /v1/responses): tools stay out of the list.
    tokens = set_session_vars(platform="api_server", async_delivery=False)
    try:
        assert advertised() == set()
    finally:
        clear_session_vars(tokens)

    # Bound (as /v1/runs delivers): tools appear — even right after a tokenless
    # call, proving the memo/check caches don't mask the token path.
    tokens = _bind_gateway(GATEWAY_URL, SECRET_TOKEN)
    try:
        assert advertised() == app_tools
    finally:
        clear_session_vars(tokens)
