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
    from hermes_cli import __version__

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
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{__version__}"
    assert captured["json"]["model"] == "grok-4.5"
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

    assert result["error"] == "allowed_x_handles and excluded_x_handles cannot be used together"


def test_x_search_schema_is_read_only_without_cross_tool_names():
    """Static schema must state read-only scope without naming other surfaces.

    AGENTS.md forbids hardcoding cross-tool/skill names in tool schemas because
    those surfaces may be unavailable. Keep out-of-scope guidance generic here;
    xurl routing lives in the skill and feature docs.
    """
    from tools.x_search_tool import X_SEARCH_SCHEMA

    description = X_SEARCH_SCHEMA["description"]
    lowered = description.lower()

    assert "read-only" in lowered
    assert "public x" in lowered
    for action in ("post", "reply", "like", "dm", "upload media", "delete"):
        assert action in lowered
    assert "authenticated" in lowered
    # No static cross-surface names in the model-facing schema.
    assert "xurl" not in lowered
    assert "web_search" not in lowered


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
    result = x_search_tool(query="anything")
    assert "No xAI credentials available" in result
    assert "hermes auth add xai-oauth" in result


# ---------------------------------------------------------------------------
# Date validation — fail fast before burning an API call on a window that
# cannot possibly return X posts. xAI itself happily 200s with a fluff
# answer when the range is malformed or pure-future, which is hard for
# callers to distinguish from a real result.
# ---------------------------------------------------------------------------

def _no_post_allowed(monkeypatch):
    """Guard: any test that should fail before HTTP can hit this fence."""
    def _fail(*_, **__):
        raise AssertionError("requests.post must not be called — validation should reject first")

    monkeypatch.setattr("requests.post", _fail)


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
    """Make the shared resolver's OAuth branch yield ``oauth_token``.

    Patches ``agent.credential_pool.load_pool`` (the seam
    ``resolve_xai_http_credentials`` imports at call time). Any pool key
    other than ``xai-oauth`` raises so ``resolve_provider_secret``'s
    credential-pool fallback can't accidentally surface the OAuth bearer
    as an "explicit key".
    """
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
    """#88040: with a paid XAI_API_KEY configured alongside subscription
    OAuth, x_search must use the API key — the OAuth path authorizes but
    answers /v1/responses in a degraded Grok explanatory mode with no
    citations. Same prefer-API-key root cause as the TTS fix (#87045/#87081);
    the precedence lives in the shared resolver behind ``prefer_api_key``."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import x_search_tool

    paid_key = _xcred("paid")
    oauth_token = _xcred("oauth")

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tools.xai_http.get_env_value",
        lambda name, default=None: {
            "XAI_API_KEY": paid_key,
        }.get(name, default),
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
    monkeypatch.setattr(
        "tools.xai_http.get_env_value",
        lambda name, default=None: default,
    )
    _install_fake_oauth_pool(monkeypatch, oauth_token)

    api_key, base_url, source = _resolve_xai_bearer()
    assert (api_key, base_url, source) == (
        oauth_token,
        "https://api.x.ai/v1",
        "xai-oauth",
    )


def test_x_search_bearer_requests_prefer_api_key_from_shared_resolver(monkeypatch):
    """The x_search call site must opt into the shared resolver's
    ``prefer_api_key`` precedence rather than re-implementing it inline."""
    from tools.x_search_tool import _resolve_xai_bearer

    captured = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "xai",
            "api_key": _xcred("paid"),
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _fake_resolve
    )

    _api_key, _base_url, source = _resolve_xai_bearer()
    assert captured.get("prefer_api_key") is True
    assert source == "xai"

