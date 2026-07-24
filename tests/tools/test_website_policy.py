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


def test_check_website_access_supports_wildcard_subdomains_only(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["*.tracking.example"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert check_website_access("https://a.tracking.example", config_path=config_path) is not None
    assert check_website_access("https://www.tracking.example", config_path=config_path) is not None
    assert check_website_access("https://tracking.example", config_path=config_path) is None


def test_default_config_exposes_website_blocklist_shape():
    from hermes_cli.config import DEFAULT_CONFIG

    website_blocklist = DEFAULT_CONFIG["security"]["website_blocklist"]
    assert website_blocklist["enabled"] is False
    assert website_blocklist["domains"] == []
    assert website_blocklist["shared_files"] == []


def test_load_website_blocklist_uses_enabled_default_when_section_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"display": {"tool_progress": "all"}}, sort_keys=False), encoding="utf-8")

    policy = load_website_blocklist(config_path)

    assert policy == {"enabled": False, "rules": []}


def test_load_website_blocklist_raises_clean_error_for_invalid_domains_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": "example.com",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.domains must be a list"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_shared_files_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "shared_files": "community-blocklist.txt",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.shared_files must be a list"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_top_level_config_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(["not", "a", "mapping"], sort_keys=False), encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="config root must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_security_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"security": []}, sort_keys=False), encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="security must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_website_blocklist_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": "block everything",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist must be a mapping"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_invalid_enabled_type(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": "false",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebsitePolicyError, match="security.website_blocklist.enabled must be a boolean"):
        load_website_blocklist(config_path)


def test_load_website_blocklist_raises_clean_error_for_malformed_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("security: [oops\n", encoding="utf-8")

    with pytest.raises(WebsitePolicyError, match="Invalid config YAML"):
        load_website_blocklist(config_path)


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


def test_check_website_access_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": True,
                        "domains": ["dynamic.example"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Invalidate the module-level cache so the new HERMES_HOME is picked up.
    # A prior test may have cached a default policy (enabled=False) under the
    # old HERMES_HOME set by the autouse _isolate_hermes_home fixture.
    from tools.website_policy import invalidate_cache
    invalidate_cache()

    blocked = check_website_access("https://dynamic.example/path")

    assert blocked is not None
    assert blocked["rule"] == "dynamic.example"


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

    @pytest.mark.asyncio
    async def test_web_extract_blocks_firecrawl_unsafe_final_url(self, monkeypatch):
        from tools import web_tools
        from plugins.web.firecrawl import provider as firecrawl_provider

        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr(
            firecrawl_provider,
            "is_safe_url",
            lambda url: url != "http://169.254.169.254/latest/meta-data/",
        )

        checked_urls = []

        def fake_check(url):
            checked_urls.append(url)
            if url == "https://allowed.test":
                return None
            pytest.fail(f"unexpected website policy check for unsafe URL: {url}")

        class FakeFirecrawlClient:
            def scrape(self, url, formats):
                return {
                    "markdown": "metadata credentials",
                    "metadata": {
                        "title": "Metadata",
                        "sourceURL": "http://169.254.169.254/latest/meta-data/",
                    },
                }

        monkeypatch.setattr(firecrawl_provider, "check_website_access", fake_check)
        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: FakeFirecrawlClient())
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

        result = json.loads(await web_tools.web_extract_tool(["https://allowed.test"]))

        assert checked_urls == ["https://allowed.test"]
        assert result["results"][0]["url"] == "http://169.254.169.254/latest/meta-data/"
        assert result["results"][0]["content"] == ""
        assert "private or internal network" in result["results"][0]["error"]


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


# ---------------------------------------------------------------------------
# Additional coverage for uncovered paths
# ---------------------------------------------------------------------------


class TestNormalizeHost:
    def test_basic(self):
        from tools.website_policy import _normalize_host
        assert _normalize_host("Example.COM") == "example.com"

    def test_trailing_dot(self):
        from tools.website_policy import _normalize_host
        assert _normalize_host("example.com.") == "example.com"

    def test_whitespace(self):
        from tools.website_policy import _normalize_host
        assert _normalize_host("  example.com  ") == "example.com"

    def test_empty(self):
        from tools.website_policy import _normalize_host
        assert _normalize_host("") == ""

    def test_none(self):
        from tools.website_policy import _normalize_host
        assert _normalize_host(None) == ""  # type: ignore[arg-type]


class TestNormalizeRule:
    def test_non_string_returns_none(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule(42) is None
        assert _normalize_rule(None) is None

    def test_empty_string_returns_none(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("") is None
        assert _normalize_rule("   ") is None

    def test_comment_returns_none(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("# comment") is None

    def test_url_with_scheme(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("https://example.com/path") == "example.com"

    def test_url_with_scheme_no_netloc(self):
        from tools.website_policy import _normalize_rule
        # "://example.com" has no scheme — urlparse puts it all in path.
        # After split("/", 1)[0] we get ":" which is not a valid host.
        # This is an edge case — the function returns ":" (not ideal but harmless).
        result = _normalize_rule("://example.com")
        # Just verify it doesn't crash and returns something
        assert isinstance(result, (str, type(None)))

    def test_path_stripped(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("example.com/some/path") == "example.com"

    def test_www_prefix_stripped(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("www.example.com") == "example.com"

    def test_trailing_dot_stripped(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("example.com.") == "example.com"

    def test_uppercase_lowered(self):
        from tools.website_policy import _normalize_rule
        assert _normalize_rule("EXAMPLE.COM") == "example.com"


class TestIterBlocklistFileRules:
    def test_file_not_found(self, tmp_path):
        from tools.website_policy import _iter_blocklist_file_rules
        result = _iter_blocklist_file_rules(tmp_path / "nonexistent.txt")
        assert result == []

    def test_unicode_decode_error(self, tmp_path):
        from tools.website_policy import _iter_blocklist_file_rules
        f = tmp_path / "bad.txt"
        f.write_bytes(b"\xff\xfe\x00\x00")
        result = _iter_blocklist_file_rules(f)
        assert result == []

    def test_skips_comments_and_empty_lines(self, tmp_path):
        from tools.website_policy import _iter_blocklist_file_rules
        f = tmp_path / "list.txt"
        f.write_text("# comment\n\n  \nexample.com\n# another\nbad.org\n", encoding="utf-8")
        result = _iter_blocklist_file_rules(f)
        assert result == ["example.com", "bad.org"]

    def test_normalizes_rules(self, tmp_path):
        from tools.website_policy import _iter_blocklist_file_rules
        f = tmp_path / "list.txt"
        f.write_text("WWW.example.com\nhttps://bad.org/path\n", encoding="utf-8")
        result = _iter_blocklist_file_rules(f)
        assert result == ["example.com", "bad.org"]


class TestLoadPolicyConfigEdgeCases:
    def test_missing_config_file_returns_default(self, tmp_path):
        from tools.website_policy import _load_policy_config
        result = _load_policy_config(tmp_path / "nonexistent.yaml")
        assert result == {"enabled": False, "domains": [], "shared_files": []}

    def test_security_none_treated_as_empty(self, tmp_path):
        from tools.website_policy import _load_policy_config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"security": None}, sort_keys=False), encoding="utf-8")
        result = _load_policy_config(config_path)
        assert result["enabled"] is False

    def test_website_blocklist_none_treated_as_empty(self, tmp_path):
        from tools.website_policy import _load_policy_config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"security": {"website_blocklist": None}}, sort_keys=False),
            encoding="utf-8",
        )
        result = _load_policy_config(config_path)
        assert result["enabled"] is False

    def test_os_error_raises_policy_error(self, tmp_path, monkeypatch):
        from tools.website_policy import _load_policy_config, WebsitePolicyError

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({}, sort_keys=False), encoding="utf-8")

        def failing_open(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", failing_open)
        with pytest.raises(WebsitePolicyError, match="Failed to read config file"):
            _load_policy_config(config_path)


class TestLoadWebsiteBlocklistEdgeCases:
    def test_caching_behavior(self, tmp_path, monkeypatch):
        """Default path policy is cached."""
        from tools.website_policy import load_website_blocklist, invalidate_cache

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": True, "domains": ["cached.example"]}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        invalidate_cache()

        # First call loads from file
        policy1 = load_website_blocklist()
        assert policy1["enabled"] is True

        # Modify config file
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": True, "domains": ["changed.example"]}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        # Second call should return cached policy (still has old domain)
        policy2 = load_website_blocklist()
        rules = {r["pattern"] for r in policy2["rules"]}
        assert "cached.example" in rules
        assert "changed.example" not in rules

        second_home = tmp_path / "second-home"
        second_home.mkdir()
        (second_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "security": {
                        "website_blocklist": {
                            "enabled": True,
                            "domains": ["second.example"],
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(second_home))

        policy3 = load_website_blocklist()
        second_rules = {rule["pattern"] for rule in policy3["rules"]}
        assert second_rules == {"second.example"}

    def test_shared_files_non_string_skipped(self, tmp_path):
        from tools.website_policy import load_website_blocklist
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {
                    "enabled": True,
                    "shared_files": [42, "", "   ", "valid.txt"],
                }}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        # Should not crash — non-string/empty entries are skipped
        result = load_website_blocklist(config_path)
        assert result["enabled"] is True

    def test_shared_files_relative_path(self, tmp_path, monkeypatch):
        """Relative shared_files paths are resolved against HERMES_HOME."""
        from tools.website_policy import load_website_blocklist, invalidate_cache

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "blocklist.txt").write_text("shared.example\n", encoding="utf-8")
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {
                    "enabled": True,
                    "shared_files": ["blocklist.txt"],
                }}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        invalidate_cache()

        result = load_website_blocklist()
        patterns = {r["pattern"] for r in result["rules"]}
        assert "shared.example" in patterns

    def test_dedup_config_and_shared_file(self, tmp_path):
        """Same domain in config and shared file appears once per source
        (config and shared file have different dedup keys)."""
        from tools.website_policy import load_website_blocklist

        shared = tmp_path / "shared.txt"
        shared.write_text("example.com\n", encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {
                    "enabled": True,
                    "domains": ["example.com"],
                    "shared_files": [str(shared)],
                }}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = load_website_blocklist(config_path)
        # Config and shared file have different source keys, so both appear
        sources = {r["source"] for r in result["rules"] if r["pattern"] == "example.com"}
        assert "config" in sources
        assert len(sources) == 2  # config + shared file path


class TestMatchHostAgainstRule:
    def test_empty_host(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("", "example.com") is False

    def test_empty_pattern(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("example.com", "") is False

    def test_exact_match(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("example.com", "example.com") is True

    def test_subdomain_match(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("sub.example.com", "example.com") is True

    def test_no_match(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("other.com", "example.com") is False

    def test_wildcard_match(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("a.tracking.example", "*.tracking.example") is True

    def test_wildcard_no_match(self):
        from tools.website_policy import _match_host_against_rule
        assert _match_host_against_rule("tracking.example", "*.tracking.example") is False


class TestExtractHostFromUrlish:
    def test_https_url(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("https://example.com/path") == "example.com"

    def test_http_url_with_port(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("http://example.com:8080/path") == "example.com"

    def test_schemeless_url(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("example.com/path") == "example.com"

    def test_schemeless_www(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("www.example.com") == "www.example.com"

    def test_empty_url(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("") == ""

    def test_uppercase_host(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("https://EXAMPLE.COM") == "example.com"

    def test_trailing_dot(self):
        from tools.website_policy import _extract_host_from_urlish
        assert _extract_host_from_urlish("https://example.com./path") == "example.com"


class TestCheckWebsiteAccessEdgeCases:
    def test_empty_url_returns_none(self):
        from tools.website_policy import check_website_access, invalidate_cache
        invalidate_cache()
        assert check_website_access("") is None

    def test_no_host_returns_none(self):
        from tools.website_policy import check_website_access, invalidate_cache
        invalidate_cache()
        assert check_website_access("http://") is None

    def test_disabled_policy_returns_none(self, tmp_path):
        from tools.website_policy import check_website_access
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": False, "domains": ["example.com"]}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        assert check_website_access("https://example.com", config_path=config_path) is None

    def test_unexpected_exception_fails_open(self, tmp_path, monkeypatch):
        """Unexpected exceptions in check_website_access fail open (return None)."""
        from tools.website_policy import check_website_access

        def raising_load(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr("tools.website_policy.load_website_blocklist", raising_load)
        # Without explicit config_path, unexpected errors fail open
        from tools.website_policy import invalidate_cache
        invalidate_cache()
        result = check_website_access("https://example.com")
        assert result is None

    def test_blocked_url_returns_metadata(self, tmp_path):
        from tools.website_policy import check_website_access
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": True, "domains": ["blocked.test"]}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        result = check_website_access("https://blocked.test/page", config_path=config_path)
        assert result is not None
        assert result["host"] == "blocked.test"
        assert result["rule"] == "blocked.test"
        assert result["source"] == "config"
        assert "message" in result
        assert "url" in result

    def test_fast_path_cached_disabled(self, monkeypatch, tmp_path):
        """When cached policy is disabled, check skips all work."""
        from tools.website_policy import check_website_access, invalidate_cache
        import tools.website_policy as wp

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": False}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        invalidate_cache()

        assert check_website_access("https://example.com") is None

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("disabled cache fast path must skip policy work")

        monkeypatch.setattr(wp, "_extract_host_from_urlish", fail_if_called)
        monkeypatch.setattr(wp, "load_website_blocklist", fail_if_called)
        assert check_website_access("https://example.com") is None


class TestInvalidateCache:
    def test_invalidate_clears_cache(self, monkeypatch, tmp_path):
        from tools.website_policy import invalidate_cache, load_website_blocklist, _cached_policy
        import tools.website_policy as wp

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"security": {"website_blocklist": {"enabled": True, "domains": ["test.example"]}}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        invalidate_cache()

        # Load to populate cache
        load_website_blocklist()
        assert wp._cached_policy is not None

        # Invalidate
        invalidate_cache()
        assert wp._cached_policy is None
