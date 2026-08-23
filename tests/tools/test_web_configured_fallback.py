"""Request-level fallback from a failed primary web provider to a configured secondary."""

import json
from unittest.mock import patch

import pytest

import tools.web_result_cache as web_result_cache
import tools.web_tools as web_tools
import tools.web_tools_rescue as web_rescue
import tools.web_tools_extract as web_extract
from hermes_cli.config_defaults import DEFAULT_CONFIG
from plugins.web import keyless_mcp


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep memo and disk cache state local to each fallback test."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(web_result_cache, "search_memo", web_result_cache.SearchMemo())
    monkeypatch.setattr(web_result_cache, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(web_result_cache, "_web_config", lambda: {})


def test_fallback_keys_are_recognized_configuration():
    web = DEFAULT_CONFIG["web"]
    assert web["fallback_backend"] == ""
    assert web["search_fallback_backend"] == ""
    assert web["extract_fallback_backend"] == ""


class _Provider:
    def __init__(
        self, name, *, search_result=None, extract_result=None, available=True
    ):
        self.name = name
        self.display_name = name.title()
        self.search_result = search_result
        self.extract_result = extract_result
        self.available = available
        self.search_calls = 0
        self.extract_calls = 0
        self.extract_urls = []

    def supports_search(self):
        return self.search_result is not None

    def supports_extract(self):
        return self.extract_result is not None

    def is_available(self):
        return self.available

    def search(self, query, limit=5):
        self.search_calls += 1
        return self.search_result

    def extract(self, urls, **kwargs):
        self.extract_calls += 1
        self.extract_urls.append(list(urls))
        return self.extract_result


class _RaisingSearchProvider(_Provider):
    def search(self, query, limit=5):
        self.search_calls += 1
        raise RuntimeError("primary connection reset")


def _search_ok(vendor):
    return {
        "success": True,
        "data": {"web": [{"url": f"https://{vendor}.example", "title": vendor}]},
    }


def _extract_ok(url):
    return [
        {
            "url": url,
            "title": "Fallback",
            "content": "fallback content",
            "raw_content": "fallback content",
            "metadata": {"sourceURL": url},
        }
    ]


def _registry(monkeypatch, providers):
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider",
        lambda name: providers.get(name),
    )


class TestKeyedSearchFallback:
    def test_failed_primary_uses_configured_secondary_before_keyless(self, monkeypatch):
        primary = _Provider(
            "exa",
            search_result={"success": False, "error": "HTTP 402 credits exhausted"},
        )
        secondary = _Provider("tavily", search_result=_search_ok("tavily"))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "search_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        first = json.loads(web_tools.web_search_tool("q", limit=2))
        second = json.loads(web_tools.web_search_tool("q", limit=2))

        for result in (first, second):
            assert result["success"] is True
            assert result["data"]["served_by"] == "tavily"
            assert result["data"]["fallback_from"] == "exa"
            assert "402" in result["data"]["backend_error"]
        assert primary.search_calls == 2
        assert secondary.search_calls == 2

    def test_capability_specific_fallback_overrides_shared_fallback(self, monkeypatch):
        primary = _Provider(
            "exa",
            search_result={"success": False, "error": "HTTP 402 credits exhausted"},
        )
        tavily = _Provider("tavily", search_result=_search_ok("tavily"))
        parallel = _Provider("parallel", search_result=_search_ok("parallel"))
        _registry(
            monkeypatch,
            {"exa": primary, "tavily": tavily, "parallel": parallel},
        )
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "search_backend": "exa",
                "fallback_backend": "parallel",
                "search_fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is True
        assert result["data"]["served_by"] == "tavily"
        assert tavily.search_calls == 1
        assert parallel.search_calls == 0

    def test_primary_exception_uses_configured_secondary(self, monkeypatch):
        primary = _RaisingSearchProvider("exa", search_result={"unused": True})
        secondary = _Provider("tavily", search_result=_search_ok("tavily"))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "search_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is True
        assert result["data"]["served_by"] == "tavily"
        assert "connection reset" in result["data"]["backend_error"]

    def test_failed_secondary_continues_to_keyless_rescue(self, monkeypatch):
        primary = _Provider(
            "exa", search_result={"success": False, "error": "HTTP 402 exhausted"}
        )
        secondary = _Provider(
            "tavily", search_result={"success": False, "error": "HTTP 503 unavailable"}
        )
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"search_backend": "exa", "fallback_backend": "tavily"},
        )
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "exa-key" if name == "EXA_API_KEY" else "",
        )
        monkeypatch.setattr(
            "agent.web_search_registry._keyless_tier_enabled", lambda: True
        )
        monkeypatch.setattr(
            keyless_mcp, "search_with_failover", lambda *args: _search_ok("free")
        )

        result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is True
        assert result["data"]["rescued_from"] == "exa"
        assert "tavily" in result["data"]["backend_error"]
        assert primary.search_calls == 1
        assert secondary.search_calls == 1

    def test_free_tier_secondary_is_skipped(self, monkeypatch):
        primary = _Provider(
            "exa", search_result={"success": False, "error": "HTTP 402 exhausted"}
        )
        secondary = _Provider("tavily", search_result=_search_ok("tavily"))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"search_backend": "exa", "fallback_backend": "tavily"},
        )
        monkeypatch.setattr(
            keyless_mcp,
            "provider_tier",
            lambda name: "free" if name == "tavily" else None,
        )
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "exa-key" if name == "EXA_API_KEY" else "tavily-key",
        )
        monkeypatch.setattr(
            "agent.web_search_registry._keyless_tier_enabled", lambda: True
        )

        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_search_ok("free")
        ) as keyless:
            result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is True
        assert secondary.search_calls == 0
        keyless.assert_called_once()

    def test_unavailable_secondary_is_skipped_instead_of_using_keyless(self, monkeypatch):
        primary = _Provider(
            "exa", search_result={"success": False, "error": "HTTP 402 exhausted"}
        )
        secondary = _Provider(
            "tavily", search_result=_search_ok("tavily"), available=False
        )
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"search_backend": "exa", "fallback_backend": "tavily"},
        )
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "exa-key" if name == "EXA_API_KEY" else "",
        )
        monkeypatch.setattr(
            "agent.web_search_registry._keyless_tier_enabled", lambda: True
        )

        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_search_ok("free")
        ) as keyless:
            result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is True
        assert secondary.search_calls == 0
        keyless.assert_called_once_with("exa", "q", 10)

    def test_same_provider_is_not_retried_as_its_own_fallback(self, monkeypatch):
        primary = _Provider(
            "exa", search_result={"success": False, "error": "HTTP 402 exhausted"}
        )
        _registry(monkeypatch, {"exa": primary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "search_backend": "exa",
                "fallback_backend": "exa",
                "keyless_fallback": False,
            },
        )

        result = json.loads(web_tools.web_search_tool("q"))

        assert result["success"] is False
        assert primary.search_calls == 1


class TestKeyedExtractFallback:
    @pytest.mark.asyncio
    async def test_failed_batch_uses_configured_secondary_before_keyless(
        self, monkeypatch
    ):
        url = "https://example.com/page"
        primary = _Provider(
            "exa",
            extract_result=[
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "HTTP 402 credits exhausted",
                }
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(url))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        first = json.loads(await web_tools.web_extract_tool([url]))
        second = json.loads(await web_tools.web_extract_tool([url]))

        for result in (first, second):
            rows = result["results"] if isinstance(result, dict) else result
            assert rows[0]["content"] == "fallback content"
        assert primary.extract_calls == 2
        assert secondary.extract_calls == 2

    @pytest.mark.asyncio
    async def test_partial_primary_failure_does_not_retry_secondary(self, monkeypatch):
        urls = ["https://example.com/a", "https://example.com/b"]
        primary = _Provider(
            "exa",
            extract_result=[
                {"url": urls[0], "title": "A", "content": "ok", "raw_content": "ok"},
                {"url": urls[1], "title": "", "content": "", "error": "404"},
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(urls[1]))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        await web_tools.web_extract_tool(urls)

        assert primary.extract_calls == 1
        assert secondary.extract_calls == 0

    @pytest.mark.asyncio
    async def test_policy_block_is_not_sent_to_secondary(self, monkeypatch):
        url = "https://blocked.example"
        primary = _Provider(
            "exa",
            extract_result=[
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "Blocked by website policy",
                    "blocked_by_policy": True,
                }
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(url))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        await web_tools.web_extract_tool([url])

        assert primary.extract_calls == 1
        assert secondary.extract_calls == 0

    @pytest.mark.asyncio
    async def test_global_policy_block_never_reaches_any_provider(self, monkeypatch):
        url = "https://blocked.example/private"
        primary = _Provider(
            "exa",
            extract_result=[
                {"url": url, "title": "", "content": "", "error": "primary outage"}
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(url))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        blocked = {
            "host": "blocked.example",
            "rule": "blocked.example",
            "source": "config",
            "message": "Blocked by website policy",
        }
        with patch("tools.website_policy.check_website_access", return_value=blocked):
            result = json.loads(await web_tools.web_extract_tool([url]))

        rows = result["results"] if isinstance(result, dict) else result
        assert rows[0]["blocked_by_policy"] is True
        assert "Blocked by website policy" in rows[0]["error"]
        assert primary.extract_calls == 0
        assert secondary.extract_calls == 0

    @pytest.mark.asyncio
    async def test_mixed_policy_batch_sends_only_allowed_urls(self, monkeypatch):
        blocked_url = "https://blocked.example/private"
        allowed_url = "https://allowed.example/article"
        primary = _Provider(
            "exa",
            extract_result=[
                {
                    "url": allowed_url,
                    "title": "",
                    "content": "",
                    "error": "primary outage",
                }
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(allowed_url))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        blocked = {
            "host": "blocked.example",
            "rule": "blocked.example",
            "source": "config",
            "message": "Blocked by website policy",
        }

        def _policy(url):
            return blocked if url == blocked_url else None

        with patch("tools.website_policy.check_website_access", side_effect=_policy):
            result = json.loads(
                await web_tools.web_extract_tool([blocked_url, allowed_url])
            )

        rows = result["results"] if isinstance(result, dict) else result
        assert rows[0]["blocked_by_policy"] is True
        assert rows[1]["content"] == "fallback content"
        assert primary.extract_urls == [[allowed_url]]
        assert secondary.extract_urls == [[allowed_url]]

    @pytest.mark.asyncio
    async def test_empty_primary_batch_retries_configured_secondary(self, monkeypatch):
        url = "https://example.com/a"
        primary = _Provider("exa", extract_result=[])
        secondary = _Provider("tavily", extract_result=_extract_ok(url))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        result = json.loads(await web_tools.web_extract_tool([url]))

        rows = result["results"] if isinstance(result, dict) else result
        assert rows[0]["content"] == "fallback content"
        assert primary.extract_calls == 1
        assert secondary.extract_calls == 1

    @pytest.mark.asyncio
    async def test_malformed_partial_policy_result_suppresses_fallback(self, monkeypatch):
        urls = ["https://blocked.example", "https://missing.example"]
        primary = _Provider(
            "exa",
            extract_result=[
                {
                    "url": urls[0],
                    "title": "",
                    "content": "",
                    "error": "Blocked by website policy",
                    "blocked_by_policy": True,
                }
            ],
        )
        secondary = _Provider("tavily", extract_result=_extract_ok(urls[1]))
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "extract_backend": "exa",
                "fallback_backend": "tavily",
                "keyless_fallback": False,
            },
        )

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        await web_tools.web_extract_tool(urls)

        assert primary.extract_calls == 1
        assert secondary.extract_calls == 0

    @pytest.mark.asyncio
    async def test_secondary_policy_block_is_preserved_for_keyless_rescue(
        self, monkeypatch
    ):
        url = "https://redirects-to-blocked.example"
        primary_results = [
            {"url": url, "title": "", "content": "", "error": "primary outage"}
        ]
        secondary_results = [
            {
                "url": url,
                "title": "",
                "content": "",
                "error": "Blocked by website policy after redirect",
                "blocked_by_policy": True,
            }
        ]
        secondary = _Provider("tavily", extract_result=secondary_results)
        _registry(monkeypatch, {"tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "exa", "fallback_backend": "tavily"},
        )

        fallback_results, fallback_error = await web_rescue._try_fallback_extract(
            "exa", [url], primary_results
        )

        assert fallback_error == ""
        assert fallback_results == secondary_results
        with patch.object(keyless_mcp, "extract_with_failover") as keyless:
            returned = web_rescue._rescue_extract("exa", [url], fallback_results)
        assert returned == secondary_results
        keyless.assert_not_called()

    @pytest.mark.asyncio
    async def test_secondary_policy_block_preserved_while_other_failure_is_rescued(
        self, monkeypatch
    ):
        urls = [
            "https://redirects-to-blocked.example",
            "https://temporary-outage.example",
        ]
        primary = _Provider(
            "exa",
            extract_result=[
                {"url": urls[0], "title": "", "content": "", "error": "primary outage"},
                {"url": urls[1], "title": "", "content": "", "error": "primary outage"},
            ],
        )
        secondary = _Provider(
            "tavily",
            extract_result=[
                {
                    "url": urls[0],
                    "title": "",
                    "content": "",
                    "error": "Blocked by website policy after redirect",
                    "blocked_by_policy": True,
                },
                {
                    "url": urls[1],
                    "title": "",
                    "content": "",
                    "error": "secondary outage",
                },
            ],
        )
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "exa", "fallback_backend": "tavily"},
        )
        monkeypatch.setattr(web_extract, "_rescue_eligible", lambda provider: True)

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        rescued = _extract_ok(urls[1])
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=rescued
        ) as keyless:
            result = json.loads(await web_tools.web_extract_tool(urls))

        rows = result["results"] if isinstance(result, dict) else result
        assert rows[0]["blocked_by_policy"] is True
        assert rows[1]["content"] == "fallback content"
        keyless.assert_called_once_with("exa", [urls[1]])

    @pytest.mark.asyncio
    async def test_secondary_policy_block_with_other_success_does_not_rescue(
        self, monkeypatch
    ):
        urls = [
            "https://redirects-to-blocked.example",
            "https://secondary-success.example",
        ]
        primary = _Provider(
            "exa",
            extract_result=[
                {"url": url, "title": "", "content": "", "error": "primary outage"}
                for url in urls
            ],
        )
        secondary = _Provider(
            "tavily",
            extract_result=[
                {
                    "url": urls[0],
                    "title": "",
                    "content": "",
                    "error": "Blocked by website policy after redirect",
                    "blocked_by_policy": True,
                },
                _extract_ok(urls[1])[0],
            ],
        )
        _registry(monkeypatch, {"exa": primary, "tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "exa", "fallback_backend": "tavily"},
        )
        monkeypatch.setattr(web_extract, "_rescue_eligible", lambda provider: True)

        async def _allow_all(candidate, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        with patch.object(keyless_mcp, "extract_with_failover") as keyless:
            result = json.loads(await web_tools.web_extract_tool(urls))

        rows = result["results"] if isinstance(result, dict) else result
        assert rows[0]["blocked_by_policy"] is True
        assert rows[1]["content"] == "fallback content"
        keyless.assert_not_called()

    def test_policy_partition_preserved_when_rescue_returns_empty(self):
        urls = ["https://blocked.example", "https://outage.example"]
        results = [
            {
                "url": urls[0],
                "error": "Blocked by website policy after redirect",
                "blocked_by_policy": True,
            },
            {"url": urls[1], "error": "secondary outage"},
        ]

        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=[]
        ) as keyless:
            returned = web_rescue._rescue_extract("exa", urls, results)

        assert returned == results
        keyless.assert_called_once_with("exa", [urls[1]])

    def test_policy_partition_preserved_when_rescue_returns_short_batch(self):
        urls = [
            "https://blocked.example",
            "https://rescued.example",
            "https://still-failed.example",
        ]
        results = [
            {
                "url": urls[0],
                "error": "Blocked by website policy after redirect",
                "blocked_by_policy": True,
            },
            {"url": urls[1], "error": "secondary outage"},
            {"url": urls[2], "error": "secondary outage"},
        ]
        short_rescue = _extract_ok(urls[1])

        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=short_rescue
        ) as keyless:
            returned = web_rescue._rescue_extract("exa", urls, results)

        assert len(returned) == len(urls)
        assert returned[0] == results[0]
        assert returned[1]["content"] == "fallback content"
        assert returned[2] == results[2]
        keyless.assert_called_once_with("exa", urls[1:])

    @pytest.mark.asyncio
    async def test_short_secondary_policy_batch_preserves_unreturned_failures(
        self, monkeypatch
    ):
        urls = ["https://blocked.example", "https://outage.example"]
        primary_results = [
            {"url": urls[0], "error": "primary outage"},
            {"url": urls[1], "error": "primary outage"},
        ]
        secondary_results = [
            {
                "url": urls[0],
                "error": "Blocked by website policy after redirect",
                "blocked_by_policy": True,
            }
        ]
        secondary = _Provider("tavily", extract_result=secondary_results)
        _registry(monkeypatch, {"tavily": secondary})
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "exa", "fallback_backend": "tavily"},
        )

        returned, fallback_error = await web_rescue._try_fallback_extract(
            "exa", urls, primary_results
        )

        assert fallback_error == ""
        assert len(returned) == len(urls)
        assert returned[0] == secondary_results[0]
        assert returned[1] == primary_results[1]

    @pytest.mark.parametrize("malformed", [[None], {"results": []}])
    def test_malformed_keyless_rescue_rows_preserve_original_batch(self, malformed):
        urls = ["https://first.example", "https://second.example"]
        results = [
            {"url": urls[0], "error": "secondary outage"},
            {"url": urls[1], "error": "secondary outage"},
        ]

        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=malformed
        ):
            returned = web_rescue._rescue_extract("exa", urls, results)

        assert returned == results

    def test_short_keyless_rescue_row_is_merged_by_url_not_position(self):
        urls = [
            "https://blocked.example",
            "https://first-failure.example",
            "https://later-rescued.example",
        ]
        results = [
            {
                "url": urls[0],
                "error": "Blocked by website policy",
                "blocked_by_policy": True,
            },
            {"url": urls[1], "error": "secondary outage"},
            {"url": urls[2], "error": "secondary outage"},
        ]
        later_only = _extract_ok(urls[2])

        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=later_only
        ):
            returned = web_rescue._rescue_extract("exa", urls, results)

        assert returned[0] == results[0]
        assert returned[1] == results[1]
        assert returned[2]["url"] == urls[2]
        assert returned[2]["content"] == "fallback content"

    def test_keyless_rescue_also_suppresses_malformed_partial_policy_result(self):
        urls = ["https://blocked.example", "https://missing.example"]
        results = [
            {
                "url": urls[0],
                "error": "Blocked by website policy",
                "blocked_by_policy": True,
            }
        ]

        with patch.object(keyless_mcp, "extract_with_failover") as keyless:
            returned = web_rescue._rescue_extract("exa", urls, results)

        assert returned == results
        keyless.assert_not_called()
