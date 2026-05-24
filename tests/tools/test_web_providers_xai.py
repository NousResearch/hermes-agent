"""Tests for the xAI Web Search provider (plugins/web/xai/).

Covers:
- XAIWebSearchProvider.is_available() — cheap probe (env var + auth.json)
- search() — JSON happy path, annotation fallback, citations fallback, empty results
- search() error paths — HTTP error, request error, missing creds, mutually-exclusive domain filters,
  200-OK error envelope
- Request payload shape — model, tools list, allowed_domains/excluded_domains filters
- OAuth credential resolution end-to-end through tools.xai_http
- _is_backend_available("xai") integration with tools.web_tools
- _get_backend() accepts "xai" as a configured backend
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _creds(api_key: str = "xai-test-key", base_url: str = "https://api.x.ai/v1") -> dict:
    return {"provider": "xai", "api_key": api_key, "base_url": base_url}


def _mock_resp(json_data, status_code: int = 200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


def _responses_payload(text: str, annotations=None, citations=None) -> dict:
    """Build a minimal Responses-API reply with one message + output_text block."""
    chunk: dict = {"type": "output_text", "text": text}
    if annotations is not None:
        chunk["annotations"] = annotations
    payload: dict = {
        "output": [
            {
                "type": "message",
                "content": [chunk],
            }
        ],
    }
    if citations is not None:
        payload["citations"] = citations
    return payload


# ---------------------------------------------------------------------------
# Provider identity / availability
# ---------------------------------------------------------------------------


class TestXAIProviderIdentity:
    def test_provider_name(self):
        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().name == "xai"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.xai.provider import XAIWebSearchProvider
        assert issubclass(XAIWebSearchProvider, WebSearchProvider)

    def test_supports_search_only(self):
        from plugins.web.xai.provider import XAIWebSearchProvider
        p = XAIWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is False

    def test_display_name(self):
        from plugins.web.xai.provider import XAIWebSearchProvider
        assert "Grok" in XAIWebSearchProvider().display_name


class TestXAIProviderIsAvailable:
    """``is_available()`` MUST be cheap — no network, no token refresh, no
    auth-store lock. It runs on every ``hermes tools`` repaint and at
    tool-registration time, so any I/O regression here would surface as
    visible CLI latency.
    """

    def test_available_via_env_var(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().is_available() is True

    def test_available_via_auth_store(self, monkeypatch, tmp_path):
        """Cheap probe should detect xai-oauth tokens in ~/.hermes/auth.json
        without invoking the resolver (which can trigger refresh)."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {
                "xai-oauth": {"tokens": {"access_token": "ya29.fake-access-token"}},
            },
        }))

        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().is_available() is True

    def test_unavailable_when_no_env_and_no_auth_store(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # No auth.json written.
        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().is_available() is False

    def test_unavailable_when_auth_store_has_empty_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {"xai-oauth": {"tokens": {"access_token": ""}}},
        }))

        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().is_available() is False

    def test_unavailable_when_auth_store_corrupted(self, monkeypatch, tmp_path):
        """A malformed auth.json must not crash availability scans."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text("not json at all }{")

        from plugins.web.xai.provider import XAIWebSearchProvider
        assert XAIWebSearchProvider().is_available() is False

    def test_is_available_does_not_call_resolver(self, monkeypatch):
        """Regression guard: ``is_available()`` must NEVER touch the resolver,
        because the OAuth resolver can trigger a network refresh."""
        monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
        from plugins.web.xai import provider as xai_provider

        with patch.object(
            xai_provider, "resolve_xai_http_credentials",
            side_effect=AssertionError("is_available must not call the resolver"),
        ):
            assert xai_provider.XAIWebSearchProvider().is_available() is True


# ---------------------------------------------------------------------------
# search() happy + parse paths
# ---------------------------------------------------------------------------


class TestXAIProviderSearchJSONPath:
    _GROK_JSON = json.dumps({
        "results": [
            {"title": "xAI", "url": "https://x.ai", "description": "The company."},
            {"title": "Grok docs", "url": "https://docs.x.ai", "description": "API reference."},
            {"title": "Grokipedia", "url": "https://grokipedia.com", "description": "Wiki."},
        ]
    })

    def test_happy_path_normalizes_results(self):
        from plugins.web.xai import provider as xai_provider

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(_responses_payload(self._GROK_JSON))):
            result = xai_provider.XAIWebSearchProvider().search("what is xai", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "xAI",
            "url": "https://x.ai",
            "description": "The company.",
            "position": 1,
        }
        assert web[2]["position"] == 3

    def test_limit_truncates_json_results(self):
        from plugins.web.xai import provider as xai_provider

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(_responses_payload(self._GROK_JSON))):
            result = xai_provider.XAIWebSearchProvider().search("x", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_parses_json_with_leading_prose(self):
        """Reasoning models sometimes narrate before the JSON block; we tolerate it."""
        from plugins.web.xai import provider as xai_provider

        text = "Here are the results:\n" + self._GROK_JSON
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(_responses_payload(text))):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_drops_rows_without_url(self):
        from plugins.web.xai import provider as xai_provider

        bad_json = json.dumps({
            "results": [
                {"title": "no url", "description": "skip me"},
                {"title": "good", "url": "https://ok.com", "description": "keep"},
            ]
        })
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(_responses_payload(bad_json))):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 1
        assert web[0]["url"] == "https://ok.com"
        assert web[0]["position"] == 1


class TestXAIProviderSearchFallbacks:
    def test_falls_back_to_annotations_when_json_missing(self):
        """If Grok ignores the JSON instruction, derive results from url_citation annotations."""
        from plugins.web.xai import provider as xai_provider

        body = "xAI is an AI company founded in 2023. They make Grok."
        annotations = [
            {
                "type": "url_citation",
                "url": "https://x.ai/about",
                "title": "1",
                "start_index": 4,
                "end_index": 9,
            },
            {
                "type": "url_citation",
                "url": "https://docs.x.ai",
                "title": "2",
                "start_index": 47,
                "end_index": 52,
            },
        ]
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(_responses_payload(body, annotations=annotations))):
            result = xai_provider.XAIWebSearchProvider().search("xai", limit=5)

        assert result["success"] is True
        urls = [r["url"] for r in result["data"]["web"]]
        assert urls == ["https://x.ai/about", "https://docs.x.ai"]
        assert result["data"]["web"][0]["position"] == 1
        assert result["data"]["web"][1]["position"] == 2

    def test_falls_back_to_citations_list(self):
        """If no JSON and no annotations, derive from top-level citations list."""
        from plugins.web.xai import provider as xai_provider

        payload = _responses_payload("free-form narration", citations=["https://a.com", "https://b.com"])
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(payload)):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        urls = [r["url"] for r in result["data"]["web"]]
        assert urls == ["https://a.com", "https://b.com"]

    def test_annotations_without_url_citations_fall_through_to_citations(self):
        """When annotations exist but none are url_citation type (e.g. future
        annotation types xAI may add), the citations list MUST still be
        consulted — otherwise we'd silently report success-with-no-rows
        and mask real data the API provided.
        """
        from plugins.web.xai import provider as xai_provider

        body = "Some narration about xAI."
        # Non-url_citation annotations only — the fallback shouldn't extract
        # any URLs from them, and must defer to the citations list below.
        annotations = [
            {"type": "future_citation_type", "url": "https://ignored.example", "title": "x"},
        ]
        payload = _responses_payload(
            body,
            annotations=annotations,
            citations=["https://real-fallback.com"],
        )
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(payload)):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        urls = [r["url"] for r in result["data"]["web"]]
        assert urls == ["https://real-fallback.com"]

    def test_empty_response_returns_empty_success(self):
        from plugins.web.xai import provider as xai_provider

        payload = _responses_payload("", citations=[])
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(payload)):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []


# ---------------------------------------------------------------------------
# Request payload shape
# ---------------------------------------------------------------------------


class TestXAIProviderRequestShape:
    def test_posts_to_responses_endpoint_with_bearer_token(self):
        from plugins.web.xai import provider as xai_provider

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds("secret-key")), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.x.ai/v1/responses"
        assert captured["headers"].get("Authorization") == "Bearer secret-key"
        body = captured["json"]
        # Assert against the module constant rather than the literal value,
        # so renaming DEFAULT_MODEL (when xAI deprecates grok-4.3) doesn't
        # turn this into a change-detector failure.
        assert body["model"] == xai_provider.DEFAULT_MODEL
        assert body["tools"] == [{"type": "web_search"}]
        assert body["input"][0]["role"] == "user"
        # No-inline-citations is opt-in via `include` per xAI Responses docs.
        assert "no_inline_citations" in body.get("include", [])

    def test_honors_configured_model(self):
        from plugins.web.xai import provider as xai_provider

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={"model": "grok-4.3-fast"}), \
             patch("httpx.post", side_effect=fake_post):
            xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert captured["json"]["model"] == "grok-4.3-fast"

    def test_allowed_domains_passes_through_as_filters(self):
        from plugins.web.xai import provider as xai_provider

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        cfg = {"allowed_domains": ["x.ai", "grokipedia.com"]}
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value=cfg), \
             patch("httpx.post", side_effect=fake_post):
            xai_provider.XAIWebSearchProvider().search("q", limit=5)

        tools = captured["json"]["tools"]
        assert tools == [{
            "type": "web_search",
            "filters": {"allowed_domains": ["x.ai", "grokipedia.com"]},
        }]

    def test_excluded_domains_passes_through_as_filters(self):
        from plugins.web.xai import provider as xai_provider

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        cfg = {"excluded_domains": ["spam.com"]}
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value=cfg), \
             patch("httpx.post", side_effect=fake_post):
            xai_provider.XAIWebSearchProvider().search("q", limit=5)

        tools = captured["json"]["tools"]
        assert tools == [{
            "type": "web_search",
            "filters": {"excluded_domains": ["spam.com"]},
        }]

    def test_allowed_domains_capped_at_five(self):
        """xAI caps domain filters at 5; we trim silently to avoid 400s."""
        from plugins.web.xai import provider as xai_provider

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        cfg = {"allowed_domains": [f"d{i}.com" for i in range(10)]}
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value=cfg), \
             patch("httpx.post", side_effect=fake_post):
            xai_provider.XAIWebSearchProvider().search("q", limit=5)

        domains = captured["json"]["tools"][0]["filters"]["allowed_domains"]
        assert len(domains) == 5


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestXAIProviderSearchErrors:
    def test_missing_creds_returns_failure(self):
        from plugins.web.xai import provider as xai_provider

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds("")):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "xAI" in result["error"]

    def test_mutually_exclusive_domain_filters_rejected_locally(self):
        from plugins.web.xai import provider as xai_provider

        cfg = {"allowed_domains": ["a.com"], "excluded_domains": ["b.com"]}
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value=cfg), \
             patch("httpx.post") as posted:
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "cannot both be set" in result["error"]
        posted.assert_not_called()

    def test_http_error_returns_failure(self):
        import httpx
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 429
        bad.text = "rate limited"
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=bad)

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=err):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "429" in result["error"]

    def test_request_error_returns_failure(self):
        import httpx
        from plugins.web.xai import provider as xai_provider

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "boom" in result["error"] or "xAI" in result["error"]

    def test_bad_json_response_returns_failure(self):
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 200
        bad.raise_for_status = MagicMock()
        bad.json.side_effect = ValueError("not json")

        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=bad):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "JSON" in result["error"]

    def test_401_on_oauth_path_triggers_force_refresh_and_retry(self):
        """OAuth credentials → 401 must force-refresh and retry once.

        Closes the two-gap scenario the resolver's JWT-exp shortcut doesn't
        cover: opaque (non-JWT) tokens and mid-window revocation. We expect
        ``httpx.post`` to be called twice with two different Bearer tokens.
        """
        import httpx
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 401
        bad.text = "Unauthorized"
        unauthorized = httpx.HTTPStatusError("401", request=MagicMock(), response=bad)

        calls = {"posts": [], "refresh_count": 0}

        def fake_post(url, **kwargs):
            calls["posts"].append(kwargs.get("headers", {}).get("Authorization"))
            if len(calls["posts"]) == 1:
                raise unauthorized
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        def fake_resolve(*, force_refresh=False):
            if force_refresh:
                calls["refresh_count"] += 1
                return {
                    "provider": "xai-oauth",
                    "api_key": "fresh-after-refresh",
                    "base_url": "https://api.x.ai/v1",
                }
            return {
                "provider": "xai-oauth",
                "api_key": "stale-token",
                "base_url": "https://api.x.ai/v1",
            }

        with patch.object(xai_provider, "resolve_xai_http_credentials", side_effect=fake_resolve), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert calls["refresh_count"] == 1
        assert calls["posts"] == ["Bearer stale-token", "Bearer fresh-after-refresh"]

    def test_401_on_env_var_path_does_not_retry(self):
        """Env-var (XAI_API_KEY) creds can't be refreshed — must not retry."""
        import httpx
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 401
        bad.text = "Unauthorized"
        unauthorized = httpx.HTTPStatusError("401", request=MagicMock(), response=bad)

        calls = {"posts": 0, "refreshed": False}

        def fake_post(url, **kwargs):
            calls["posts"] += 1
            raise unauthorized

        def fake_resolve(*, force_refresh=False):
            if force_refresh:
                calls["refreshed"] = True
            # provider=="xai" signals env-var path; retry must be skipped.
            return {"provider": "xai", "api_key": "sk-env-var-key", "base_url": "https://api.x.ai/v1"}

        with patch.object(xai_provider, "resolve_xai_http_credentials", side_effect=fake_resolve), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "401" in result["error"]
        assert calls["posts"] == 1
        assert calls["refreshed"] is False

    def test_401_retry_gives_up_when_refresh_returns_same_token(self):
        """If the force-refresh returns the same token (refresh-token also
        dead), don't loop — surface the 401 to the caller."""
        import httpx
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 401
        bad.text = "Unauthorized"
        unauthorized = httpx.HTTPStatusError("401", request=MagicMock(), response=bad)

        calls = {"posts": 0, "refresh_count": 0}

        def fake_post(url, **kwargs):
            calls["posts"] += 1
            raise unauthorized

        def fake_resolve(*, force_refresh=False):
            if force_refresh:
                calls["refresh_count"] += 1
            return {
                "provider": "xai-oauth",
                "api_key": "same-dead-token",
                "base_url": "https://api.x.ai/v1",
            }

        with patch.object(xai_provider, "resolve_xai_http_credentials", side_effect=fake_resolve), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "401" in result["error"]
        # One post, one force-refresh attempt, no second post.
        assert calls["posts"] == 1
        assert calls["refresh_count"] == 1

    def test_non_401_http_error_is_not_retried(self):
        """Only 401 is retryable — 429 / 500 / 503 must fail fast so the
        agent (or upstream rate-limiter) decides what to do."""
        import httpx
        from plugins.web.xai import provider as xai_provider

        bad = MagicMock()
        bad.status_code = 500
        bad.text = "internal error"
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=bad)

        calls = {"posts": 0, "refreshed": False}

        def fake_post(url, **kwargs):
            calls["posts"] += 1
            raise err

        def fake_resolve(*, force_refresh=False):
            if force_refresh:
                calls["refreshed"] = True
            return {"provider": "xai-oauth", "api_key": "tok", "base_url": "https://api.x.ai/v1"}

        with patch.object(xai_provider, "resolve_xai_http_credentials", side_effect=fake_resolve), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "500" in result["error"]
        assert calls["posts"] == 1
        assert calls["refreshed"] is False

    def test_http_200_with_error_envelope_surfaces_failure(self):
        """xAI sometimes returns 200 with ``{"error": {...}}`` (model
        overloaded, refusal, etc.). Must be surfaced as a failure rather
        than silently masked as success-with-empty-results.
        """
        from plugins.web.xai import provider as xai_provider

        payload = {"error": {"message": "model overloaded", "type": "server_error"}}
        with patch.object(xai_provider, "resolve_xai_http_credentials", return_value=_creds()), \
             patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", return_value=_mock_resp(payload)):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "model overloaded" in result["error"]


# ---------------------------------------------------------------------------
# Integration with tools/web_tools.py backend wiring
# ---------------------------------------------------------------------------


class TestXAIBackendWiring:
    def test_is_backend_available_true_via_env_var(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
        assert web_tools._is_backend_available("xai") is True

    def test_is_backend_available_false_when_no_creds(self, monkeypatch, tmp_path):
        from tools import web_tools

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert web_tools._is_backend_available("xai") is False

    def test_is_backend_available_does_not_call_resolver(self, monkeypatch):
        """Regression guard — `_is_backend_available` runs on every web_search
        dispatch and every `hermes tools` repaint. It must not invoke the
        OAuth resolver (which can trigger a network refresh)."""
        from tools import web_tools

        monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
        with patch(
            "tools.xai_http.resolve_xai_http_credentials",
            side_effect=AssertionError("must not call resolver"),
        ):
            assert web_tools._is_backend_available("xai") is True

    def test_configured_backend_xai_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "xai"})
        assert web_tools._get_backend() == "xai"

    def test_xai_not_in_legacy_backend_candidate_chain(self, monkeypatch):
        """The hardcoded ``backend_candidates`` tuple in ``_get_backend()``
        does not include xAI — by design, since the no-config legacy
        chain is for users who set env vars but never ran ``hermes tools``,
        and we don't want a stray ``XAI_API_KEY`` (perhaps set for chat
        inference) to silently re-route web_search through Grok.

        Note: this does NOT prevent the registry's single-provider
        shortcut (``agent.web_search_registry._resolve``) from selecting
        xAI when it's the only available web provider. That path is the
        normal "pick the one provider the user actually configured"
        behavior shared by every other backend.
        """
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "TAVILY_API_KEY", "EXA_API_KEY", "LLMLAYER_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() != "xai"


# ---------------------------------------------------------------------------
# OAuth credential resolution (end-to-end through tools.xai_http)
# ---------------------------------------------------------------------------


class TestXAIProviderOAuthPath:
    """Verifies the provider works when credentials come from the OAuth
    runtime resolver (``hermes auth`` sign-in) rather than an env-var key.
    Patches at the ``hermes_cli.runtime_provider.resolve_runtime_provider``
    boundary so the full ``tools.xai_http.resolve_xai_http_credentials``
    chain is exercised end-to-end.
    """

    def test_search_uses_oauth_bearer_token_and_base_url(self, monkeypatch):
        from plugins.web.xai import provider as xai_provider

        # Force the env-var fallback to fail so resolution must go via OAuth.
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        oauth_runtime = {
            "provider": "xai-oauth",
            "api_mode": "codex_responses",
            "base_url": "https://api.x.ai/v1",
            "api_key": "ya29.fake-oauth-access-token",
            "source": "hermes-auth-store",
        }

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return _mock_resp(_responses_payload(json.dumps({"results": []})))

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=oauth_runtime,
        ), patch.object(xai_provider, "_load_xai_web_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            result = xai_provider.XAIWebSearchProvider().search("q", limit=3)

        assert result["success"] is True
        assert captured["url"] == "https://api.x.ai/v1/responses"
        assert captured["headers"].get("Authorization") == "Bearer ya29.fake-oauth-access-token"
