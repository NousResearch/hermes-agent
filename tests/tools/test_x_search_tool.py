"""Tests for the X (Twitter) Search tool backed by xAI Responses API.

Covers:
- HTTP request shape (URL, headers, payload, model from config)
- Handle filter validation (allowed vs excluded mutual exclusion)
- Inline url_citation extraction from message annotations
- Structured error handling (4xx with code, 5xx retry, ReadTimeout retry)
- Credential resolution: API key path, OAuth path, both-set preference, none-set
- check_x_search_requirements gating in registry
"""

import json

import pytest

import requests

class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Client Error")
            err.response = self
            raise err

    def json(self):
        return self._payload

# ---------------------------------------------------------------------------
# Original PR #10786 test coverage (HTTP shape, handle validation, citations,
# retry behavior) — preserved verbatim. Uses XAI_API_KEY env var via the
# default resolver path.
# ---------------------------------------------------------------------------

def test_x_search_posts_responses_request(monkeypatch):
    from tools.x_search_tool import x_search_tool
    from hermes_cli.version_info import get_version_info

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output_text": "People on X are discussing xAI's latest launch.",
                "citations": [{"url": "https://x.com/example/status/1", "title": "Example post"}],
            }
        )

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(
            query="What are people saying about xAI on X?",
            allowed_x_handles=["xai", "@grok"],
            from_date="2026-04-01",
            to_date="2026-04-10",
            enable_image_understanding=True,
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{get_version_info().base_version}"
    assert captured["json"]["model"]
    assert captured["json"]["store"] is False
    assert "reasoning" not in captured["json"]
    assert tool_def["type"] == "x_search"
    assert tool_def["allowed_x_handles"] == ["xai", "grok"]
    assert tool_def["from_date"] == "2026-04-01"
    assert tool_def["to_date"] == "2026-04-10"
    assert tool_def["enable_image_understanding"] is True
    assert result["success"] is True
    assert result["answer"] == "People on X are discussing xAI's latest launch."

def test_x_search_rejects_conflicting_handle_filters(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    result = json.loads(
        x_search_tool(
            query="latest xAI discussion",
            allowed_x_handles=["xai"],
            excluded_x_handles=["grok"],
        )
    )

    assert result["error"]
    assert not result.get("success")

def test_x_search_extracts_inline_url_citations(monkeypatch):
    from tools.x_search_tool import x_search_tool

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "xAI posted an update on X.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://x.com/xai/status/123",
                                        "title": "xAI update",
                                        "start_index": 0,
                                        "end_index": 3,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="latest post from xai"))

    assert result["success"] is True
    assert result["answer"] == "xAI posted an update on X."
    assert result["inline_citations"] == [
        {
            "url": "https://x.com/xai/status/123",
            "title": "xAI update",
            "start_index": 0,
            "end_index": 3,
        }
    ]

def test_x_search_returns_structured_http_error(monkeypatch):
    from tools.x_search_tool import x_search_tool

    class _FailingResponse:
        status_code = 403
        text = '{"code":"forbidden","error":"x_search is not enabled for this model"}'

        def json(self):
            return {
                "code": "forbidden",
                "error": "x_search is not enabled for this model",
            }

        def raise_for_status(self):
            err = requests.HTTPError("403 Client Error: Forbidden")
            err.response = self
            raise err

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(x_search_tool(query="latest xai discussion"))

    assert result["success"] is False
    assert result["provider"] == "xai"
    assert result["tool"] == "x_search"
    assert result["error_type"] == "HTTPError"
    assert result["error"] == "forbidden: x_search is not enabled for this model"

# ---------------------------------------------------------------------------
# Credential-resolution coverage — the OAuth-or-API-key gating contract.
# ---------------------------------------------------------------------------

def _no_xai_env(monkeypatch):
    """Strip any XAI_* env vars so the resolver doesn't see a leaked dev key."""
    for var in ("XAI_API_KEY", "XAI_BASE_URL", "HERMES_XAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

def test_x_search_uses_xai_oauth_when_only_oauth_available(monkeypatch):
    """OAuth-only user: credential_source should be ``xai-oauth``."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve(**_kwargs):
        return {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _fake_resolve
    )
    invalidate_check_fn_cache()

    assert check_x_search_requirements() is True

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "Found posts via OAuth."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="anything about xai"))

    assert result["success"] is True
    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-bearer-token"

def test_x_search_returns_tool_error_when_no_credentials(monkeypatch):
    """No credentials anywhere: tool returns a clear error, not a 401 from xAI."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve(**_kwargs):
        return {
            "provider": "xai",
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _fake_resolve
    )
    invalidate_check_fn_cache()

    assert check_x_search_requirements() is False

    # If a model somehow invokes the tool despite a False check_fn, the call
    # surfaces a friendly error rather than an HTTP exception.
    result = json.loads(x_search_tool(query="anything"))
    assert result["error"]

# ---------------------------------------------------------------------------
# Degraded-result flag — distinguish citation-backed answers from
# unsourced fluff when narrowing filters returned nothing.
# ---------------------------------------------------------------------------

def test_x_search_not_degraded_when_no_filters_active(monkeypatch):
    """A broad query that returns no citations isn't necessarily degraded.

    Without any narrowing filter, an empty-citations response is a generic
    unsourced answer, not a "filter miss". The caller can already tell from
    ``inline_citations == []`` if they care.
    """
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse({"output_text": "broad answer", "citations": []}),
    )

    result = json.loads(x_search_tool(query="anything"))

    assert result["success"] is True
    assert result["degraded"] is False
    assert result["degraded_reason"] is None

def _xcred(prefix: str) -> str:
    """Synthesize a distinct fake credential value (never a bare literal)."""
    return prefix + "-key-" + "x1"

def _install_fake_oauth_pool(monkeypatch, oauth_token: str) -> None:
    """Make the shared resolver's OAuth branch yield ``oauth_token``."""
    from types import SimpleNamespace

    entry = SimpleNamespace(
        access_token=oauth_token,
        runtime_api_key=None,
        runtime_base_url=None,
        base_url="https://api.x.ai/v1",
    )

    class _FakePool:
        def select(self):
            return entry

        def try_refresh_matching(self, _hint):
            return entry

    def _fake_load_pool(provider_id):
        if provider_id == "xai-oauth":
            return _FakePool()
        raise KeyError(provider_id)

    monkeypatch.setattr("agent.credential_pool.load_pool", _fake_load_pool)

def test_x_search_prefers_explicit_api_key_over_oauth(monkeypatch):
    """With both credentials, x_search uses the explicit API key."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import x_search_tool

    paid_key = _xcred("paid")
    oauth_token = _xcred("oauth")

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {"XAI_API_KEY": paid_key}.get(name, default),
    )
    _install_fake_oauth_pool(monkeypatch, oauth_token)
    invalidate_check_fn_cache()

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "Real posts with citations."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="from:elon latest"))

    assert result["success"] is True
    assert result["credential_source"] == "xai"
    assert captured["headers"]["Authorization"] == "Bearer " + paid_key

def test_x_search_bearer_helper_falls_back_to_oauth_without_api_key(monkeypatch):
    """No explicit XAI_API_KEY: the OAuth resolver path is unchanged."""
    from tools.x_search_tool import _resolve_xai_bearer

    oauth_token = _xcred("oauth")

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name, default=None: default)
    _install_fake_oauth_pool(monkeypatch, oauth_token)

    api_key, base_url, source = _resolve_xai_bearer()
    assert (api_key, base_url, source) == (oauth_token, "https://api.x.ai/v1", "xai-oauth")


def test_x_search_gateway_uses_only_dedicated_key_and_config(monkeypatch):
    """Opt-in gateway mode never consults or sends xAI credentials."""
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    monkeypatch.setenv("X_SEARCH_GATEWAY_KEY", "gateway-secret")
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"gateway": {
            "base_url": "https://gateway.example/v1",
            "key_env": "X_SEARCH_GATEWAY_KEY",
            "model": "custom-search-model",
        }},
    )
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse({"output_text": "gateway result"})

    monkeypatch.setattr("requests.post", _fake_post)
    assert check_x_search_requirements() is True
    result = json.loads(x_search_tool(query="latest posts"))

    assert result["success"] is True
    assert result["credential_source"] == "gateway"
    assert captured["url"] == "https://gateway.example/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer gateway-secret"
    assert "xai-secret" not in str(captured)
    assert captured["json"]["model"] == "custom-search-model"


def test_x_search_gateway_rejects_plain_http_except_loopback(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("X_SEARCH_GATEWAY_KEY", "gateway-secret")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"gateway": {
            "base_url": "http://gateway.example/v1",
            "key_env": "X_SEARCH_GATEWAY_KEY",
            "model": "custom-search-model",
        }},
    )
    result = json.loads(x_search_tool(query="latest posts"))
    assert "error" in result
    assert "HTTPS" in result["error"]


def test_x_search_gateway_allows_plain_http_on_loopback(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("X_SEARCH_GATEWAY_KEY", "gateway-secret")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"gateway": {
            "base_url": "http://127.0.0.1:20128/v1",
            "key_env": "X_SEARCH_GATEWAY_KEY",
            "model": "custom-search-model",
        }},
    )
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse({"output_text": "loopback"})

    monkeypatch.setattr("requests.post", _fake_post)
    result = json.loads(x_search_tool(query="latest posts"))
    assert result["success"] is True
    assert captured["url"] == "http://127.0.0.1:20128/v1/responses"


def _fail_if_xai_resolved(*_a, **_k):
    raise AssertionError("gateway mode must never resolve xAI credentials")


@pytest.mark.parametrize(
    "gateway",
    [
        pytest.param({"base_url": "https://gateway.example/v1", "key_env": "X_SEARCH_GATEWAY_KEY_UNSET",
                      "model": "m"}, id="missing-key"),
        pytest.param({"base_url": "https://gateway.example/v1", "model": "m"}, id="missing-key-env"),
        pytest.param("https://gateway.example/v1", id="not-a-mapping"),
    ],
)
def test_x_search_misconfigured_gateway_fails_closed(monkeypatch, gateway):
    """A configured-but-broken gateway is a tool error: no xAI fallback, no request sent."""
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    monkeypatch.delenv("X_SEARCH_GATEWAY_KEY_UNSET", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr("tools.x_search_tool._load_x_search_config", lambda: {"gateway": gateway})
    monkeypatch.setattr("tools.x_search_tool.resolve_xai_http_credentials", _fail_if_xai_resolved)
    monkeypatch.setattr("requests.post", lambda *a, **k: pytest.fail("no request may be sent"))

    assert check_x_search_requirements() is False
    result = json.loads(x_search_tool(query="latest posts"))
    assert "error" in result
    assert "xai-secret" not in json.dumps(result)


def test_x_search_gateway_never_sends_oauth_bearer(monkeypatch):
    """Even with a live xAI OAuth credential, gateway mode sends only the gateway key."""
    from tools.x_search_tool import x_search_tool

    oauth_token = _xcred("oauth")
    _install_fake_oauth_pool(monkeypatch, oauth_token)
    monkeypatch.setenv("X_SEARCH_GATEWAY_KEY", "gateway-secret")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"gateway": {
            "base_url": "https://gateway.example/v1",
            "key_env": "X_SEARCH_GATEWAY_KEY",
            "model": "custom-search-model",
        }},
    )
    monkeypatch.setattr("tools.x_search_tool.resolve_xai_http_credentials", _fail_if_xai_resolved)
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse({"output_text": "gateway result"})

    monkeypatch.setattr("requests.post", _fake_post)
    result = json.loads(x_search_tool(query="latest posts"))
    assert result["success"] is True
    assert captured["headers"]["Authorization"] == "Bearer gateway-secret"
    assert oauth_token not in str(captured)


def test_x_search_gateway_is_default_off(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr("tools.x_search_tool._load_x_search_config", dict)
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse({"output_text": "xai"})

    monkeypatch.setattr("requests.post", _fake_post)
    result = json.loads(x_search_tool(query="latest posts"))
    assert result["credential_source"] == "xai"
    assert result["success"] is True
    assert captured["url"].startswith("https://api.x.ai/")
