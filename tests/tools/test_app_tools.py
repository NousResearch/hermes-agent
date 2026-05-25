"""Unit tests for tools/app_tools.py — the Nous tool gateway integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from tools.managed_tool_gateway import ManagedToolGatewayConfig


_FAKE_GATEWAY = ManagedToolGatewayConfig(
    vendor="tools",
    gateway_origin="https://tools-gateway.example.com",
    nous_user_token="test-token-abc123",
    managed_mode=True,
)


@pytest.fixture(autouse=True)
def _reset_http_client_cache():
    """Clear the module-level cached httpx client between tests."""
    import tools.app_tools as mod
    mod._http_client = None
    mod._http_client_origin = None
    yield
    mod._http_client = None
    mod._http_client_origin = None


@pytest.fixture()
def gateway_post(monkeypatch):
    """Patch the gateway and httpx.Client.post; return a dict capturing the request."""
    monkeypatch.setattr(
        "tools.app_tools.resolve_managed_tool_gateway", lambda v: _FAKE_GATEWAY
    )
    monkeypatch.setattr(
        "tools.app_tools._get_current_model_name", lambda: None
    )
    captured = {}
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": {}, "error": None}
    resp.text = json.dumps({"data": {}, "error": None})

    def fake_post(self, url, *, json=None, headers=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return resp

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return captured


# ---------------------------------------------------------------------------
# check_fn gating
# ---------------------------------------------------------------------------

class TestAppToolsAvailability:
    def test_returns_false_when_gateway_not_ready(self, monkeypatch):
        monkeypatch.setattr("tools.app_tools.is_managed_tool_gateway_ready", lambda vendor: False)
        monkeypatch.setattr("tools.app_tools._read_portal_app_tools_enabled", lambda: True)
        from tools.app_tools import _app_tools_available
        assert _app_tools_available() is False

    def test_returns_true_when_gateway_ready_and_config_on(self, monkeypatch):
        monkeypatch.setattr("tools.app_tools.is_managed_tool_gateway_ready", lambda vendor: True)
        monkeypatch.setattr("tools.app_tools._read_portal_app_tools_enabled", lambda: True)
        from tools.app_tools import _app_tools_available
        assert _app_tools_available() is True

    def test_returns_false_when_config_off(self, monkeypatch):
        monkeypatch.setattr("tools.app_tools.is_managed_tool_gateway_ready", lambda vendor: True)
        monkeypatch.setattr("tools.app_tools._read_portal_app_tools_enabled", lambda: False)
        from tools.app_tools import _app_tools_available
        assert _app_tools_available() is False


# ---------------------------------------------------------------------------
# URL + auth header
# ---------------------------------------------------------------------------

class TestSearchPostsCorrectUrlAndAuth:
    def test_posts_to_v1_search_with_bearer_token(self, monkeypatch, gateway_post):
        monkeypatch.setattr("tools.app_tools._get_current_model_name", lambda: "test-model")
        from tools.app_tools import handle_app_search_tools
        handle_app_search_tools({"queries": [{"use_case": "send email"}]})

        assert gateway_post["url"] == "https://tools-gateway.example.com/v1/search"
        assert gateway_post["headers"]["Authorization"] == "Bearer test-token-abc123"
        assert gateway_post["headers"]["Content-Type"] == "application/json"
        assert gateway_post["json"]["queries"] == [{"use_case": "send email"}]
        assert gateway_post["json"]["model"] == "test-model"


# ---------------------------------------------------------------------------
# Model auto-injection
# ---------------------------------------------------------------------------

class TestModelAutoInjection:
    def test_injects_model_from_config(self, monkeypatch, gateway_post):
        monkeypatch.setattr("tools.app_tools._get_current_model_name", lambda: "claude-sonnet-4")
        from tools.app_tools import handle_app_search_tools
        handle_app_search_tools({"queries": [{"use_case": "test"}]})
        assert gateway_post["json"]["model"] == "claude-sonnet-4"

    def test_omits_model_when_unresolvable(self, gateway_post):
        from tools.app_tools import handle_app_search_tools
        handle_app_search_tools({"queries": [{"use_case": "test"}]})
        assert "model" not in gateway_post["json"]


# ---------------------------------------------------------------------------
# Gateway-internal param stripping (allowlist approach)
# ---------------------------------------------------------------------------

class TestExecuteStripsInternalParams:
    def test_strips_sync_response_thought_step_metric(self, gateway_post):
        from tools.app_tools import handle_app_execute_tools
        handle_app_execute_tools({
            "tools": [{"tool_slug": "TEST", "arguments": {}}],
            "sync_response_to_workbench": True,
            "thought": "testing",
            "current_step": "TESTING",
            "current_step_metric": "1/1 tests",
        })
        body = gateway_post["json"]
        for key in ("sync_response_to_workbench", "thought", "current_step", "current_step_metric"):
            assert key not in body
        assert body["tools"] == [{"tool_slug": "TEST", "arguments": {}}]


# ---------------------------------------------------------------------------
# HTTP error → tool result (not exception)
# ---------------------------------------------------------------------------

class TestHttpErrorReturnedAsToolResult:
    @pytest.mark.parametrize("status_code", [402, 403, 422, 500])
    def test_returns_error_json_not_exception(self, monkeypatch, status_code):
        monkeypatch.setattr("tools.app_tools.resolve_managed_tool_gateway", lambda v: _FAKE_GATEWAY)
        error_body = {"error": {"code": "TEST_ERROR", "message": "fail"}}
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = error_body
        resp.text = json.dumps(error_body)
        monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: resp)

        from tools.app_tools import handle_app_search_tools
        result = json.loads(handle_app_search_tools({"queries": [{"use_case": "test"}]}))
        assert result["error"]["code"] == "TEST_ERROR"


# ---------------------------------------------------------------------------
# Network failure → tool result
# ---------------------------------------------------------------------------

class TestNetworkFailureReturnedAsToolResult:
    def test_connect_error_returns_gateway_unreachable(self, monkeypatch):
        monkeypatch.setattr("tools.app_tools.resolve_managed_tool_gateway", lambda v: _FAKE_GATEWAY)

        def raise_connect(self, url, **kw):
            raise httpx.ConnectError("Connection refused")
        monkeypatch.setattr(httpx.Client, "post", raise_connect)

        from tools.app_tools import handle_app_search_tools
        result = json.loads(handle_app_search_tools({"queries": [{"use_case": "test"}]}))
        assert result["error"]["code"] == "GATEWAY_UNREACHABLE"

    def test_timeout_returns_gateway_timeout(self, monkeypatch):
        monkeypatch.setattr("tools.app_tools.resolve_managed_tool_gateway", lambda v: _FAKE_GATEWAY)

        def raise_timeout(self, url, **kw):
            raise httpx.ReadTimeout("timed out")
        monkeypatch.setattr(httpx.Client, "post", raise_timeout)

        from tools.app_tools import handle_app_search_tools
        result = json.loads(handle_app_search_tools({"queries": [{"use_case": "test"}]}))
        assert result["error"]["code"] == "GATEWAY_TIMEOUT"


# ---------------------------------------------------------------------------
# Endpoint routing + payload forwarding
# ---------------------------------------------------------------------------

class TestEndpointRouting:
    def test_manage_connections_forwards_toolkits(self, gateway_post):
        from tools.app_tools import handle_app_manage_connections
        handle_app_manage_connections({"toolkits": ["gmail", "slack"], "reinitiate_all": True})
        assert gateway_post["url"].endswith("/v1/connections")
        assert gateway_post["json"]["toolkits"] == ["gmail", "slack"]
        assert gateway_post["json"]["reinitiate_all"] is True

    def test_tool_schemas_forwards_slugs(self, gateway_post):
        from tools.app_tools import handle_app_tool_schemas
        handle_app_tool_schemas({"tool_slugs": ["GMAIL_SEND_EMAIL"], "include": ["input_schema", "output_schema"]})
        assert gateway_post["url"].endswith("/v1/schemas")
        assert gateway_post["json"]["tool_slugs"] == ["GMAIL_SEND_EMAIL"]
        assert gateway_post["json"]["include"] == ["input_schema", "output_schema"]


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------

class TestRegistryEntries:
    def test_all_four_tools_registered_under_app_tools(self):
        from tools.registry import registry
        import tools.app_tools  # noqa: F401
        expected = {"app_search_tools", "app_tool_schemas", "app_execute_tools", "app_manage_connections"}
        for name in expected:
            entry = registry._tools.get(name)
            assert entry is not None, f"{name} not registered"
            assert entry.toolset == "app_tools"


# ---------------------------------------------------------------------------
# session (object) vs session_id (string) asymmetry
# ---------------------------------------------------------------------------

class TestSessionHandling:
    def test_search_uses_session_object(self, gateway_post):
        from tools.app_tools import handle_app_search_tools
        handle_app_search_tools({"queries": [{"use_case": "test"}], "session": {"generate_id": True}})
        assert isinstance(gateway_post["json"]["session"], dict)
        assert "session_id" not in gateway_post["json"]

    def test_schemas_uses_session_id_string(self, gateway_post):
        from tools.app_tools import handle_app_tool_schemas
        handle_app_tool_schemas({"tool_slugs": ["TEST"], "session_id": "sess-123"})
        assert gateway_post["json"]["session_id"] == "sess-123"
        assert "session" not in gateway_post["json"]

    def test_execute_uses_session_id_string(self, gateway_post):
        from tools.app_tools import handle_app_execute_tools
        handle_app_execute_tools({"tools": [{"tool_slug": "TEST", "arguments": {}}], "session_id": "sess-456"})
        assert gateway_post["json"]["session_id"] == "sess-456"
        assert "session" not in gateway_post["json"]

    def test_connections_uses_session_id_string(self, gateway_post):
        from tools.app_tools import handle_app_manage_connections
        handle_app_manage_connections({"toolkits": ["gmail"], "session_id": "sess-789"})
        assert gateway_post["json"]["session_id"] == "sess-789"
        assert "session" not in gateway_post["json"]
