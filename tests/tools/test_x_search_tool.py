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
    assert captured["json"]["model"] == "grok-4.20-reasoning"
    assert captured["json"]["store"] is False
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


def test_x_search_retries_read_timeout_then_succeeds(monkeypatch):
    from tools.x_search_tool import x_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return _FakeResponse(
            {
                "output_text": "Recovered after retry.",
                "citations": [],
            }
        )

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="grok xai"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after retry."


def test_x_search_retries_5xx_then_succeeds(monkeypatch):
    from tools.x_search_tool import x_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(
                {"code": "Internal error", "error": "Service temporarily unavailable."},
                status_code=500,
            )
        return _FakeResponse({"output_text": "Recovered after 5xx retry."})

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="grok xai"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after 5xx retry."


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

    def _fake_resolve():
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


def test_x_search_uses_api_key_when_only_xai_api_key_set(monkeypatch):
    """API-key-only user: credential_source should be ``xai``."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve():
        # Real ``resolve_xai_http_credentials`` returns ``"xai"`` when it
        # falls through to the XAI_API_KEY env var path.
        return {
            "provider": "xai",
            "api_key": "raw-api-key",
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
        return _FakeResponse({"output_text": "Found posts via API key."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="anything"))

    assert result["success"] is True
    assert result["credential_source"] == "xai"
    assert captured["headers"]["Authorization"] == "Bearer raw-api-key"


def test_x_search_prefers_oauth_when_both_available(monkeypatch):
    """Both credentials present: OAuth wins (matches Teknium's billing preference).

    The real ordering is implemented in ``tools.xai_http.resolve_xai_http_credentials``
    — OAuth runtime first, fallback OAuth resolver second, ``XAI_API_KEY`` third.
    This test exercises the contract by having the resolver return the OAuth
    bearer (the ``xai-oauth`` ``provider`` tag is the marker).
    """
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "raw-api-key")

    # Mimic xai_http's preference: OAuth wins, so we return the OAuth tuple
    # even though XAI_API_KEY is also set.
    def _fake_resolve():
        return {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _fake_resolve
    )
    invalidate_check_fn_cache()

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "OAuth preferred."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="anything"))

    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-bearer-token"


def test_x_search_returns_tool_error_when_no_credentials(monkeypatch):
    """No credentials anywhere: tool returns a clear error, not a 401 from xAI."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements, x_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve():
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


def test_x_search_check_fn_false_when_resolver_raises(monkeypatch):
    """Resolver exceptions (e.g. expired token + failed refresh) gate the tool out."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import check_x_search_requirements

    _no_xai_env(monkeypatch)

    def _boom():
        raise RuntimeError("token revoked and refresh failed")

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _boom
    )
    invalidate_check_fn_cache()

    assert check_x_search_requirements() is False


def test_x_search_honors_config_model_and_timeout(monkeypatch, tmp_path):
    """``x_search.model`` and ``x_search.timeout_seconds`` override the defaults."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    # Patch the in-module config loader so tests don't touch ~/.hermes/config.yaml.
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"model": "grok-custom-test", "timeout_seconds": 45, "retries": 0},
    )

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        return _FakeResponse({"output_text": "Custom model OK."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="anything"))

    assert result["success"] is True
    assert captured["model"] == "grok-custom-test"
    assert captured["timeout"] == 45


def test_x_search_registered_in_registry_with_check_fn():
    """The tool is registered under the x_search toolset with the gating check_fn."""
    import tools.x_search_tool  # noqa: F401 — ensures registration runs
    from tools.registry import registry

    entry = registry.get_entry("x_search")
    assert entry is not None
    assert entry.toolset == "x_search"
    assert entry.check_fn is not None
    assert entry.check_fn.__name__ == "check_x_search_requirements"
    assert "XAI_API_KEY" in entry.requires_env
    assert entry.emoji == "🐦"


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


def test_x_search_rejects_malformed_from_date(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    _no_post_allowed(monkeypatch)

    result = json.loads(x_search_tool(query="anything", from_date="not-a-date"))

    assert "from_date must be YYYY-MM-DD" in result["error"]


def test_x_search_rejects_malformed_to_date(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    _no_post_allowed(monkeypatch)

    result = json.loads(x_search_tool(query="anything", to_date="2026/05/01"))

    assert "to_date must be YYYY-MM-DD" in result["error"]


def test_x_search_rejects_inverted_date_range(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    _no_post_allowed(monkeypatch)

    result = json.loads(
        x_search_tool(
            query="anything",
            from_date="2026-05-10",
            to_date="2026-05-01",
        )
    )

    assert "from_date (2026-05-10) must be on or before to_date (2026-05-01)" in result["error"]


def test_x_search_rejects_future_from_date(monkeypatch):
    """``from_date`` in the future can never match any post → reject."""
    import datetime as _dt

    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    _no_post_allowed(monkeypatch)

    class _FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 21, 12, 0, 0, tzinfo=tz or _dt.timezone.utc)

    monkeypatch.setattr("tools.x_search_tool.datetime", _FrozenDateTime)

    result = json.loads(x_search_tool(query="anything", from_date="2030-01-01"))

    assert "from_date (2030-01-01) is in the future" in result["error"]


def test_x_search_allows_future_to_date(monkeypatch):
    """``to_date`` in the future is fine — caller may want posts as they arrive."""
    import datetime as _dt

    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    class _FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 21, 12, 0, 0, tzinfo=tz or _dt.timezone.utc)

    monkeypatch.setattr("tools.x_search_tool.datetime", _FrozenDateTime)

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {"output_text": "future to_date is allowed", "citations": []}
        )

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(
            query="anything",
            from_date="2026-05-20",
            to_date="2030-01-01",
        )
    )

    assert result["success"] is True
    assert result["answer"] == "future to_date is allowed"


def test_x_search_accepts_today_as_from_date(monkeypatch):
    """``from_date == today UTC`` is a valid edge case (today is past + present)."""
    import datetime as _dt

    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    class _FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 21, 12, 0, 0, tzinfo=tz or _dt.timezone.utc)

    monkeypatch.setattr("tools.x_search_tool.datetime", _FrozenDateTime)
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse({"output_text": "ok", "citations": []}),
    )

    result = json.loads(x_search_tool(query="anything", from_date="2026-05-21"))

    assert result["success"] is True


# ---------------------------------------------------------------------------
# Degraded-result flag — distinguish citation-backed answers from
# unsourced fluff when narrowing filters returned nothing.
# ---------------------------------------------------------------------------

def test_x_search_marks_degraded_when_handle_filter_returns_no_citations(monkeypatch):
    """allowed_x_handles set + zero citations → degraded=True."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse(
            {"output_text": "Generic encyclopedic answer with no citations.", "citations": []}
        ),
    )

    result = json.loads(
        x_search_tool(query="what has @ghostuser posted", allowed_x_handles=["ghostuser"])
    )

    assert result["success"] is True
    assert result["degraded"] is True
    assert "allowed_x_handles" in result["degraded_reason"]


def test_x_search_marks_degraded_when_excluded_handles_and_no_citations(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse({"output_text": "fluff", "citations": []}),
    )

    result = json.loads(
        x_search_tool(query="anything", excluded_x_handles=["someuser"])
    )

    assert result["degraded"] is True
    assert "excluded_x_handles" in result["degraded_reason"]


def test_x_search_marks_degraded_when_date_range_and_no_citations(monkeypatch):
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse({"output_text": "fluff", "citations": []}),
    )

    result = json.loads(
        x_search_tool(
            query="anything",
            from_date="2026-04-01",
            to_date="2026-04-02",
        )
    )

    assert result["degraded"] is True
    assert "from_date" in result["degraded_reason"]
    assert "to_date" in result["degraded_reason"]


def test_x_search_not_degraded_when_filter_returns_inline_citations(monkeypatch):
    """A real citation from the inline annotations clears the degraded flag."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Real post from xai.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://x.com/xai/status/1",
                                        "title": "xAI post",
                                        "start_index": 0,
                                        "end_index": 4,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
    )

    result = json.loads(
        x_search_tool(query="latest xAI post", allowed_x_handles=["xai"])
    )

    assert result["success"] is True
    assert result["degraded"] is False
    assert result["degraded_reason"] is None
    assert len(result["inline_citations"]) == 1


def test_x_search_not_degraded_when_filter_returns_top_level_citations(monkeypatch):
    """A real citation from xAI's top-level ``citations`` array also clears the flag."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse(
            {
                "output_text": "Found discussion.",
                "citations": [{"url": "https://x.com/example/status/1", "title": "Example"}],
            }
        ),
    )

    result = json.loads(
        x_search_tool(query="anything", allowed_x_handles=["xai"])
    )

    assert result["degraded"] is False
    assert result["degraded_reason"] is None


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

"""Additional tests for x_search_tool covering edge cases, error paths, and branch coverage.

Expands coverage to >= 70% for tools/x_search_tool.py by targeting:
- Error-handling branches (empty query, config exceptions, HTTP error message paths)
- Handle normalization edge cases (empty strings, too many handles)
- Response extraction edge cases (non-message output types, non-citation annotations)
- Retry exhaustion paths (ReadTimeout, ConnectionError, 5xx)
- Handler wrapper and image+video understanding flags
- Non-dict JSON and missing-response HTTPError scenarios
"""

import json
import time
from typing import Any, Dict

import pytest
import requests


class _FakeResponse:
    """Reusable fake response helper, kept local to avoid cross-file coupling."""
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
# Config edge cases — exercise _load_x_search_config directly and its
# exception-handling branches along with _get_x_search_timeout_seconds and
# _get_x_search_retries error paths.
# ---------------------------------------------------------------------------

def test_load_x_search_config_returns_empty_on_exception(monkeypatch):
    """_load_x_search_config returns {} when load_config itself raises.

    Covers the ``except Exception: return {}`` path (lines 74-75).
    """
    from tools.x_search_tool import _load_x_search_config

    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("config file is corrupt")))

    result = _load_x_search_config()
    assert result == {}


def test_get_x_search_timeout_seconds_default_on_invalid_value(monkeypatch):
    """_get_x_search_timeout_seconds falls back to default when config has a bad value."""
    from tools.x_search_tool import _get_x_search_timeout_seconds

    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"timeout_seconds": "not-a-number"},
    )

    assert _get_x_search_timeout_seconds() == 180


def test_get_x_search_retries_default_on_invalid_value(monkeypatch):
    """_get_x_search_retries falls back to default when config has a bad value."""
    from tools.x_search_tool import _get_x_search_retries

    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"retries": "not-a-number"},
    )

    assert _get_x_search_retries() == 2


# ---------------------------------------------------------------------------
# _normalize_handles edge cases
# ---------------------------------------------------------------------------

def test_normalize_handles_rejects_excessive_handles():
    """_normalize_handles raises when more than MAX_HANDLES handles are given."""
    from tools.x_search_tool import _normalize_handles

    many_handles = [f"user{i}" for i in range(11)]

    with pytest.raises(ValueError, match="supports at most 10 handles"):
        _normalize_handles(many_handles, "allowed_x_handles")


def test_normalize_handles_filters_empty_and_none_entries():
    """_normalize_handles skips empty and None handle entries gracefully."""
    from tools.x_search_tool import _normalize_handles

    result = _normalize_handles(["good", "", None, "@also-good", " "], "allowed_x_handles")
    assert result == ["good", "also-good"]


# ---------------------------------------------------------------------------
# _extract_response_text — non-message output types and empty text content
# ---------------------------------------------------------------------------

def test_extract_response_text_skips_non_message_types():
    """_extract_response_text skips output items whose type is not 'message'."""
    from tools.x_search_tool import _extract_response_text

    payload: Dict[str, Any] = {
        "output": [
            {"type": "not_message", "content": [{"type": "output_text", "text": "should skip"}]},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "only this gets through"},
                ],
            },
        ]
    }
    result = _extract_response_text(payload)
    assert result == "only this gets through"


def test_extract_response_text_ignores_empty_content_text():
    """_extract_response_text skips content entries with empty text."""
    from tools.x_search_tool import _extract_response_text

    payload: Dict[str, Any] = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "output_text", "text": "second content"},
                ],
            }
        ]
    }
    result = _extract_response_text(payload)
    assert result == "second content"


def test_extract_response_text_returns_empty_when_no_text_found():
    """_extract_response_text returns empty string when no text is extractable."""
    from tools.x_search_tool import _extract_response_text

    payload: Dict[str, Any] = {"output": [{"type": "message", "content": []}]}
    result = _extract_response_text(payload)
    assert result == ""


def test_extract_response_text_handles_missing_output_key():
    """_extract_response_text handles a payload with no 'output' key at all."""
    from tools.x_search_tool import _extract_response_text

    result = _extract_response_text({})
    assert result == ""


# ---------------------------------------------------------------------------
# _extract_inline_citations — non-message output, non-url_citation annotations
# ---------------------------------------------------------------------------

def test_extract_inline_citations_skips_non_message_items():
    """_extract_inline_citations skips non-message output items."""
    from tools.x_search_tool import _extract_inline_citations

    payload: Dict[str, Any] = {
        "output": [
            {"type": "not_message", "content": [{"annotations": []}]},
            {
                "type": "message",
                "content": [
                    {
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://x.com/user/status/1",
                                "title": "Real post",
                                "start_index": 0,
                                "end_index": 5,
                            }
                        ]
                    }
                ],
            },
        ]
    }
    result = _extract_inline_citations(payload)
    assert len(result) == 1
    assert result[0]["url"] == "https://x.com/user/status/1"


def test_extract_inline_citations_skips_non_url_citation_annotations():
    """_extract_inline_citations skips annotations whose type is not url_citation."""
    from tools.x_search_tool import _extract_inline_citations

    payload: Dict[str, Any] = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "annotations": [
                            {"type": "mention", "user_id": "123"},
                            {
                                "type": "url_citation",
                                "url": "https://x.com/user/status/2",
                                "title": "Real citation",
                                "start_index": 0,
                                "end_index": 3,
                            },
                        ]
                    }
                ],
            }
        ]
    }
    result = _extract_inline_citations(payload)
    assert len(result) == 1
    assert result[0]["url"] == "https://x.com/user/status/2"


def test_extract_inline_citations_empty_with_no_output():
    """_extract_inline_citations returns [] when payload has no output key."""
    from tools.x_search_tool import _extract_inline_citations

    assert _extract_inline_citations({}) == []
    assert _extract_inline_citations({"output": None}) == []


# ---------------------------------------------------------------------------
# _http_error_message — response-less errors, non-dict JSON, response text
# ---------------------------------------------------------------------------

def test_http_error_message_without_response():
    """_http_error_message returns str(exc) when the error has no response object."""
    from tools.x_search_tool import _http_error_message

    exc = requests.HTTPError("raw error with no response")
    assert _http_error_message(exc) == "raw error with no response"


def test_http_error_message_with_non_json_response():
    """_http_error_message handles a response whose body is not valid JSON."""
    from tools.x_search_tool import _http_error_message

    class _NonJsonResponse:
        status_code = 502
        text = "Bad Gateway: upstream is down"

        def json(self):
            raise ValueError("not JSON")

    exc = requests.HTTPError("502 Server Error")
    exc.response = _NonJsonResponse()

    msg = _http_error_message(exc)
    assert msg == "Bad Gateway: upstream is down"


def test_http_error_message_with_non_dict_json():
    """_http_error_message handles a response whose JSON body is an array, not a dict."""
    from tools.x_search_tool import _http_error_message

    class _ArrayResponse:
        status_code = 500
        text = '["error", "details"]'

        def json(self):
            return ["error", "details"]

    exc = requests.HTTPError("500 Server Error")
    exc.response = _ArrayResponse()

    msg = _http_error_message(exc)
    # Falls through to the response.text path since payload is not a dict
    assert "error" in msg


def test_http_error_message_with_empty_text_fallback():
    """_http_error_message falls back to str(exc) when response has no text."""
    from tools.x_search_tool import _http_error_message

    class _EmptyTextResponse:
        status_code = 500
        text = ""

        def json(self):
            return "not-a-dict"

    exc = requests.HTTPError("500 Server Error")
    exc.response = _EmptyTextResponse()

    msg = _http_error_message(exc)
    assert msg == "500 Server Error"


# ---------------------------------------------------------------------------
# x_search_tool — query validation, empty query guard
# ---------------------------------------------------------------------------

def test_x_search_returns_error_for_empty_query():
    """x_search_tool returns a tool_error when query is empty or whitespace."""
    from tools.x_search_tool import x_search_tool

    result = x_search_tool(query="")
    assert "query is required" in result

    result = x_search_tool(query="   ")
    assert "query is required" in result


# ---------------------------------------------------------------------------
# x_search_tool — enable_video_understanding and both understanding flags
# ---------------------------------------------------------------------------

def test_x_search_sends_video_understanding_flag(monkeypatch):
    """enable_video_understanding=True is forwarded in the tool definition."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "video understanding enabled", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(
            query="search with video",
            enable_video_understanding=True,
        )
    )

    assert result["success"] is True
    assert captured["tool_def"]["enable_video_understanding"] is True


def test_x_search_sends_both_image_and_video_flags(monkeypatch):
    """Both image and video understanding flags can be enabled simultaneously."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "both understanding flags", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(
            query="search with media",
            enable_image_understanding=True,
            enable_video_understanding=True,
        )
    )

    assert result["success"] is True
    assert captured["tool_def"]["enable_image_understanding"] is True
    assert captured["tool_def"]["enable_video_understanding"] is True


# ---------------------------------------------------------------------------
# x_search_tool — excluded_x_handles (alone, not combined with allowed)
# ---------------------------------------------------------------------------

def test_x_search_works_with_excluded_handles_only(monkeypatch):
    """excluded_x_handles alone (no allowed) is valid."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "excluded handles ok", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(query="search terms", excluded_x_handles=["spamaccount"])
    )

    assert result["success"] is True
    assert captured["tool_def"]["excluded_x_handles"] == ["spamaccount"]
    assert "allowed_x_handles" not in captured["tool_def"]


# ---------------------------------------------------------------------------
# x_search_tool — from_date only, to_date only (individual date filters)
# ---------------------------------------------------------------------------

def test_x_search_with_from_date_only(monkeypatch):
    """x_search_tool works with only from_date set."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "from_date only", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="test", from_date="2026-05-01"))

    assert result["success"] is True
    assert captured["tool_def"]["from_date"] == "2026-05-01"
    assert "to_date" not in captured["tool_def"]


def test_x_search_with_to_date_only(monkeypatch):
    """x_search_tool works with only to_date set."""
    import datetime as _dt

    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    class _FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 21, 12, 0, 0, tzinfo=tz or _dt.timezone.utc)

    monkeypatch.setattr("tools.x_search_tool.datetime", _FrozenDateTime)

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "to_date only", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="test", to_date="2026-05-21"))

    assert result["success"] is True
    assert "from_date" not in captured["tool_def"]
    assert captured["tool_def"]["to_date"] == "2026-05-21"


# ---------------------------------------------------------------------------
# x_search_tool — ReadTimeout exhaustion (retries=0, all attempts time out)
# This covers lines 358 (raise inside ReadTimeout handler) and
# 429-443 (the outer except requests.ReadTimeout handler).
# ---------------------------------------------------------------------------

def test_x_search_exhausts_read_timeout_retries(monkeypatch):
    """x_search_tool returns a timeout error when all retries are exhausted on ReadTimeout."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"model": "test-model", "timeout_seconds": 30, "retries": 0},
    )
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ReadTimeout("Connection timed out")),
    )
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="timeout test"))

    assert result["success"] is False
    assert result["error_type"] == "ReadTimeout"
    assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# x_search_tool — 5xx exhaustion (all retries exhausted on 5xx errors)
# This covers the HTTPError re-raise path when attempt >= max_retries.
# ---------------------------------------------------------------------------

def test_x_search_exhausts_5xx_retries(monkeypatch):
    """x_search_tool returns an HTTPError when all retries are exhausted on 5xx."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"model": "test-model", "timeout_seconds": 30, "retries": 0},
    )
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResponse(
            {"code": "service_unavailable", "error": "Service temporarily unavailable."},
            status_code=503,
        ),
    )
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="5xx exhaust"))

    assert result["success"] is False
    assert result["error_type"] == "HTTPError"
    assert "service_unavailable" in result["error"]


# ---------------------------------------------------------------------------
# x_search_tool — ConnectionError retry and succeed
# ---------------------------------------------------------------------------

def test_x_search_retries_connection_error_then_succeeds(monkeypatch):
    """x_search_tool retries a ConnectionError and succeeds on the second attempt."""
    from tools.x_search_tool import x_search_tool

    calls: Dict[str, int] = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.ConnectionError("Connection refused")
        return _FakeResponse({"output_text": "Recovered after connection errors."})

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"model": "test-model", "timeout_seconds": 30, "retries": 2},
    )
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="connection test"))

    assert calls["count"] == 3
    assert result["success"] is True
    assert result["answer"] == "Recovered after connection errors."


# ---------------------------------------------------------------------------
# x_search_tool — ConnectionError exhaustion
# ---------------------------------------------------------------------------

def test_x_search_exhausts_connection_error_retries(monkeypatch):
    """x_search_tool returns a ConnectionError when all ConnectionError retries are exhausted.

    Unlike ReadTimeout, ConnectionError is NOT caught by the outer
    ``except requests.ReadTimeout`` handler — it falls through to the
    generic ``except Exception`` handler.
    """
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.x_search_tool._load_x_search_config",
        lambda: {"model": "test-model", "timeout_seconds": 30, "retries": 0},
    )
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("Connection refused")),
    )
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="connection exhaust"))

    assert result["success"] is False
    assert result["error_type"] == "ConnectionError"
    assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# x_search_tool — custom base_url from credentials
# ---------------------------------------------------------------------------

def test_x_search_uses_custom_base_url_from_credentials(monkeypatch):
    """A custom base_url from resolve_xai_http_credentials is honored in the request URL."""
    from tools.registry import invalidate_check_fn_cache
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _custom_resolve():
        return {
            "provider": "xai",
            "api_key": "custom-key",
            "base_url": "https://custom.x.ai/v2",
        }

    monkeypatch.setattr(
        "tools.x_search_tool.resolve_xai_http_credentials", _custom_resolve
    )
    invalidate_check_fn_cache()

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse({"output_text": "custom base url ok", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="custom base test"))

    assert result["success"] is True
    assert captured["url"] == "https://custom.x.ai/v2/responses"


# ---------------------------------------------------------------------------
# _handle_x_search — the handler wrapper
# ---------------------------------------------------------------------------

def test_handle_x_search_passes_args_to_x_search_tool(monkeypatch):
    """_handle_x_search correctly forwards arguments from the handler dict."""
    from tools.x_search_tool import x_search_tool, _handle_x_search

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        captured["model"] = json["model"]
        return _FakeResponse({"output_text": "from handler", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        _handle_x_search(
            {
                "query": "handler test",
                "allowed_x_handles": ["user1"],
                "from_date": "2026-06-01",
                "to_date": "2026-06-10",
                "enable_image_understanding": True,
            }
        )
    )

    assert result["success"] is True
    assert result["answer"] == "from handler"
    assert captured["tool_def"]["allowed_x_handles"] == ["user1"]
    assert captured["tool_def"]["from_date"] == "2026-06-01"
    assert captured["tool_def"]["to_date"] == "2026-06-10"
    assert captured["tool_def"]["enable_image_understanding"] is True


def test_handle_x_search_with_minimal_args(monkeypatch):
    """_handle_x_search works with just a query (no optional args)."""
    from tools.x_search_tool import _handle_x_search

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"output_text": "minimal handler ok", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(_handle_x_search({"query": "minimal test"}))

    assert result["success"] is True
    assert result["answer"] == "minimal handler ok"


# ---------------------------------------------------------------------------
# x_search_tool — response text extraction via "text" content type (not "output_text")
# ---------------------------------------------------------------------------

def test_x_search_extracts_text_content_type(monkeypatch):
    """x_search_tool extracts text from content entries with type 'text' (not 'output_text')."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "text", "text": "Response via text type content."}
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="test text type"))

    assert result["success"] is True
    assert result["answer"] == "Response via text type content."


# ---------------------------------------------------------------------------
# x_search_tool — HTTPError with no response body (no status_code or response)
# This covers the case where HTTPError has no .response at all.
# ---------------------------------------------------------------------------

def test_x_search_handles_http_error_with_no_response(monkeypatch):
    """x_search_tool handles an HTTPError exception with no attached response."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    def _raise_no_response(*a, **k):
        exc = requests.HTTPError("503 upstream unavailable")
        # Deliberately do NOT set exc.response
        raise exc

    monkeypatch.setattr("requests.post", _raise_no_response)

    result = json.loads(x_search_tool(query="test"))

    assert result["success"] is False
    assert result["error_type"] == "HTTPError"
    # Should return str(exc) since there's no response to extract from
    assert result["error"] == "503 upstream unavailable"


# ---------------------------------------------------------------------------
# x_search_tool — correct from_date for today test with properly trimmed whitespace
# ---------------------------------------------------------------------------

def test_x_search_with_whitespace_query(monkeypatch):
    """x_search_tool handles queries with surrounding whitespace."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["query"] = json["input"][0]["content"]
        return _FakeResponse({"output_text": "whitespace ok", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(x_search_tool(query="   hello world   "))

    assert result["success"] is True
    assert result["query"] == "hello world"
    assert captured["query"] == "hello world"


# ---------------------------------------------------------------------------
# x_search_tool — date range with no dates (empty strings) should work
# ---------------------------------------------------------------------------

def test_x_search_with_empty_dates(monkeypatch):
    """x_search_tool works when from_date and to_date are empty strings."""
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    captured: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "empty dates ok", "citations": []})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        x_search_tool(query="test", from_date="", to_date="")
    )

    assert result["success"] is True
    assert "from_date" not in captured["tool_def"]
    assert "to_date" not in captured["tool_def"]


# ---------------------------------------------------------------------------
# x_search_tool — response is None after loop (line 368)
#
# This path is exercised by making _get_x_search_retries return a negative
# value, which causes range(0) to produce an empty loop body so `response`
# stays None.
# ---------------------------------------------------------------------------

def test_x_search_response_none_after_loop(monkeypatch):
    """Cover line 368: raise RuntimeError when response is None after the retry loop.

    Force _get_x_search_retries to return a negative value so ``range(-1+1) = range(0)``
    produces an empty iterator and the ``for`` body never executes.
    """
    from tools.x_search_tool import x_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.x_search_tool._get_x_search_retries",
        lambda: -1,
    )

    def _never_called(*a, **k):
        raise AssertionError("requests.post must not be called — retry loop is empty")

    monkeypatch.setattr("requests.post", _never_called)
    monkeypatch.setattr("tools.x_search_tool.time.sleep", lambda *_: None)

    result = json.loads(x_search_tool(query="response none test"))

    assert result["success"] is False
    assert "did not return a response" in result.get("error", "")
