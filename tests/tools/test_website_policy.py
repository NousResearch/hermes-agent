import json
from pathlib import Path

import pytest
import yaml

from tests.tools.conftest import register_all_web_providers

from tools.website_policy import WebsitePolicyError, check_website_access, load_website_blocklist


def test_load_website_blocklist_merges_config_and_shared_file(tmp_path):
    shared = tmp_path / "community-blocklist.txt"
    shared.write_text("# comment\nexample.org\nsub.bad.net\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["example.com", "https://www.evil.test/path"],
                        "shared_files": [str(shared)],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    policy = load_website_blocklist(config_path)

    assert policy["enabled"] is True
    assert {rule["pattern"] for rule in policy["rules"]} == {
        "example.com",
        "evil.test",
        "example.org",
        "sub.bad.net",
    }


def test_check_website_access_matches_parent_domain_subdomains(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["example.com"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    blocked = check_website_access("https://docs.example.com/page", config_path=config_path)

    assert blocked is not None
    assert blocked["host"] == "docs.example.com"
    assert blocked["rule"] == "example.com"


def test_default_config_exposes_website_blocklist_shape():
    from hermes_cli.config import DEFAULT_CONFIG

    website_blocklist = DEFAULT_CONFIG["security"]["website_blocklist"]
    assert website_blocklist["enabled"] is False
    assert website_blocklist["domains"] == []
    assert website_blocklist["shared_files"] == []


def test_load_website_blocklist_wraps_shared_file_read_errors(tmp_path, monkeypatch):
    shared = tmp_path / "community-blocklist.txt"
    shared.write_text("example.org\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": [str(shared)],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def failing_read_text(self, *args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr(Path, "read_text", failing_read_text)

    # Unreadable shared files are now warned and skipped (not raised),
    # so the blocklist loads successfully but without those rules.
    result = load_website_blocklist(config_path)
    assert result["enabled"] is True
    assert result["rules"] == []  # shared file rules skipped


def test_check_website_access_blocks_scheme_less_urls(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["blocked.test"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    blocked = check_website_access("www.blocked.test/path", config_path=config_path)

    assert blocked is not None
    assert blocked["host"] == "www.blocked.test"
    assert blocked["rule"] == "blocked.test"


def test_browser_navigate_returns_policy_block(monkeypatch):
    from tools import browser_tool

    # Allow SSRF check to pass so the policy check is reached
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(
        browser_tool,
        "check_website_access",
        lambda url: {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        },
    )
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *args, **kwargs: pytest.fail("browser command should not run for blocked URL"),
    )

    result = json.loads(browser_tool.browser_navigate("https://blocked.test"))

    assert result["success"] is False
    assert result["blocked_by_policy"]["rule"] == "blocked.test"


def test_browser_navigate_allows_when_shared_file_missing(monkeypatch, tmp_path):
    """Missing shared blocklist files are warned and skipped, not fatal."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": ["missing-blocklist.txt"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # check_website_access should return None (allow) — missing file is skipped
    result = check_website_access("https://allowed.test", config_path=config_path)
    assert result is None


class TestWebToolPolicy:
    """Tests that exercise web_extract_tool with website-policy gates.

    These tests need the bundled web providers to be registered in the
    agent.web_search_registry so the tool dispatchers can find an active
    provider.  Without registration, the tools return an error dict that
    lacks a ``results`` key, causing ``KeyError``.
    """

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    @pytest.mark.asyncio
    async def test_web_extract_short_circuits_blocked_url(self, monkeypatch):
        from tools import web_tools
        from plugins.web.firecrawl import provider as firecrawl_provider

        # Allow test URLs past SSRF check so website policy is what gets tested
        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        # The per-URL website-policy gate moved into the firecrawl plugin's
        # extract() during the web-provider migration. Patch it at the new
        # location.
        monkeypatch.setattr(
            firecrawl_provider,
            "check_website_access",
            lambda url: {
                "host": "blocked.test",
                "rule": "blocked.test",
                "source": "config",
                "message": "Blocked by website policy",
            },
        )
        monkeypatch.setattr(
            firecrawl_provider,
            "_get_firecrawl_client",
            lambda: pytest.fail("firecrawl should not run for blocked URL"),
        )
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        # Force the firecrawl plugin to be the active extract provider.
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        result = json.loads(await web_tools.web_extract_tool(["https://blocked.test"]))

        assert result["results"][0]["url"] == "https://blocked.test"
        assert "Blocked by website policy" in result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_web_extract_blocks_redirected_final_url(self, monkeypatch):
        from tools import web_tools
        from plugins.web.firecrawl import provider as firecrawl_provider

        # Allow test URLs past SSRF check so website policy is what gets tested
        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda url: True)

        def fake_check(url):
            if url == "https://allowed.test":
                return None
            if url == "https://blocked.test/final":
                return {
                    "host": "blocked.test",
                    "rule": "blocked.test",
                    "source": "config",
                    "message": "Blocked by website policy",
                }
            pytest.fail(f"unexpected URL checked: {url}")

        class FakeFirecrawlClient:
            def scrape(self, url, formats):
                return {
                    "markdown": "secret content",
                    "metadata": {
                        "title": "Redirected",
                        "sourceURL": "https://blocked.test/final",
                    },
                }

        # After the web-provider migration, the per-URL gate + firecrawl client
        # live in the plugin. Patch both at the plugin location.
        monkeypatch.setattr(firecrawl_provider, "check_website_access", fake_check)
        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: FakeFirecrawlClient())
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        result = json.loads(await web_tools.web_extract_tool(["https://allowed.test"]))

        assert result["results"][0]["url"] == "https://blocked.test/final"
        assert result["results"][0]["content"] == ""
        assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"


def test_check_website_access_fails_open_on_malformed_config(tmp_path, monkeypatch):
    """Malformed config with default path should fail open (return None), not crash."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("security: [oops\n", encoding="utf-8")

    # With explicit config_path (test mode), errors propagate
    with pytest.raises(WebsitePolicyError):
        check_website_access("https://example.com", config_path=config_path)

    # Simulate default path by pointing HERMES_HOME to tmp_path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import website_policy
    website_policy.invalidate_cache()

    # With default path, errors are caught and fail open
    result = check_website_access("https://example.com")
    assert result is None  # allowed, not crashed


# ─── Keyless rescue must never re-fetch policy-blocked URLs ──────────────────


def test_rescue_extract_skips_policy_blocked_results():
    """A whole-batch failure that is actually a policy refusal must NOT be
    routed through the keyless rescue ring — that would fetch content the
    user explicitly blocked. Regression for CI reds on main (Aug 2026)."""
    from tools import web_tools

    urls = ["https://blocked.test"]
    results = [{
        "url": "https://blocked.test",
        "title": "",
        "content": "",
        "error": "Blocked by website policy (rule: blocked.test)",
        "blocked_by_policy": {"rule": "blocked.test"},
    }]

    def _boom(*a, **kw):
        raise AssertionError("keyless ring must not be called for policy blocks")

    import plugins.web.keyless_mcp as ring
    orig = ring.extract_with_failover
    ring.extract_with_failover = _boom
    try:
        out = web_tools._rescue_extract("firecrawl", urls, results)
    finally:
        ring.extract_with_failover = orig

    assert out == results  # blocked results preserved verbatim


def test_rescue_extract_mixed_batch_only_rescues_real_failures(monkeypatch):
    """Mixed batch: the policy-blocked entry is preserved; only the genuine
    backend failure goes through the ring, and order is preserved."""
    from tools import web_tools

    urls = ["https://blocked.test", "https://down.test"]
    results = [
        {"url": "https://blocked.test", "title": "", "content": "",
         "error": "Blocked by website policy", "blocked_by_policy": {"rule": "blocked.test"}},
        {"url": "https://down.test", "title": "", "content": "",
         "error": "backend 500"},
    ]

    seen = {}

    def fake_failover(provider_name, rescue_urls):
        seen["urls"] = list(rescue_urls)
        return [{"url": u, "title": "ok", "content": "rescued", "error": ""} for u in rescue_urls]

    monkeypatch.setattr(
        "plugins.web.keyless_mcp.extract_with_failover", fake_failover
    )

    out = web_tools._rescue_extract("firecrawl", urls, results)

    assert seen["urls"] == ["https://down.test"]
    assert out[0]["error"] == "Blocked by website policy"  # untouched
    assert out[1]["content"] == "rescued"
    assert out[1]["metadata"]["rescued_from"] == "firecrawl"
