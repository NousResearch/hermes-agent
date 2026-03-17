import json
from pathlib import Path

import pytest
import yaml

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
    from tools import browser_tool

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


@pytest.mark.asyncio
async def test_web_extract_short_circuits_blocked_url(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(
        web_tools,
        "check_website_access",
        lambda url: {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        },
    )
    monkeypatch.setattr(
        web_tools,
        "_get_firecrawl_client",
        lambda: pytest.fail("firecrawl should not run for blocked URL"),
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = json.loads(await web_tools.web_extract_tool(["https://blocked.test"], use_llm_processing=False))

    assert result["results"][0]["url"] == "https://blocked.test"
    assert "Blocked by website policy" in result["results"][0]["error"]


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


@pytest.mark.asyncio
async def test_web_extract_blocks_redirected_final_url(monkeypatch):
    from tools import web_tools

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

    monkeypatch.setattr(web_tools, "check_website_access", fake_check)
    monkeypatch.setattr(web_tools, "_get_firecrawl_client", lambda: FakeFirecrawlClient())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = json.loads(await web_tools.web_extract_tool(["https://allowed.test"], use_llm_processing=False))

    assert result["results"][0]["url"] == "https://blocked.test/final"
    assert result["results"][0]["content"] == ""
    assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"


@pytest.mark.asyncio
async def test_web_crawl_short_circuits_blocked_url(monkeypatch):
    from tools import web_tools

    # web_crawl_tool checks for Firecrawl env before website policy
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    monkeypatch.setattr(
        web_tools,
        "check_website_access",
        lambda url: {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        },
    )
    monkeypatch.setattr(
        web_tools,
        "_get_firecrawl_client",
        lambda: pytest.fail("firecrawl should not run for blocked crawl URL"),
    )
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = json.loads(await web_tools.web_crawl_tool("https://blocked.test", use_llm_processing=False))

    assert result["results"][0]["url"] == "https://blocked.test"
    assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"


@pytest.mark.asyncio
async def test_web_crawl_blocks_redirected_final_url(monkeypatch):
    from tools import web_tools

    # web_crawl_tool checks for Firecrawl env before website policy
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

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

    class FakeCrawlClient:
        def crawl(self, url, **kwargs):
            return {
                "data": [
                    {
                        "markdown": "secret crawl content",
                        "metadata": {
                            "title": "Redirected crawl page",
                            "sourceURL": "https://blocked.test/final",
                        },
                    }
                ]
            }

    monkeypatch.setattr(web_tools, "check_website_access", fake_check)
    monkeypatch.setattr(web_tools, "_get_firecrawl_client", lambda: FakeCrawlClient())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = json.loads(await web_tools.web_crawl_tool("https://allowed.test", use_llm_processing=False))

    assert result["results"][0]["content"] == ""
    assert result["results"][0]["error"] == "Blocked by website policy"
    assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"
