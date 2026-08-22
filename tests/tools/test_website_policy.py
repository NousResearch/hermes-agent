"""Stable unit coverage for the website-policy parser and decision helper."""

import builtins
from pathlib import Path

import pytest
import yaml

import tools.website_policy as website_policy


@pytest.fixture(autouse=True)
def reset_website_policy_cache():
    website_policy.invalidate_cache()
    yield
    website_policy.invalidate_cache()


def write_config(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Example.COM. ", "example.com"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_host(value, expected):
    assert website_policy._normalize_host(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, None),
        (None, None),
        ("", None),
        (" # comment", None),
        ("https://www.Example.com/path", "example.com"),
        ("example.com/path", "example.com"),
        ("EXAMPLE.COM.", "example.com"),
    ],
)
def test_normalize_rule(value, expected):
    assert website_policy._normalize_rule(value) == expected


def test_iter_blocklist_file_rules_skips_comments_and_normalizes(tmp_path):
    path = tmp_path / "blocklist.txt"
    path.write_text(
        "# comment\n\nwww.example.com\nhttps://bad.test/path\n",
        encoding="utf-8",
    )

    assert website_policy._iter_blocklist_file_rules(path) == [
        "example.com",
        "bad.test",
    ]


@pytest.mark.parametrize("path_bytes", [b"\xff\xfe\x00", None])
def test_iter_blocklist_file_rules_skips_unreadable_files(tmp_path, monkeypatch, path_bytes):
    path = tmp_path / "blocklist.txt"
    if path_bytes is None:
        def fail_read(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_text", fail_read)
    else:
        path.write_bytes(path_bytes)

    assert website_policy._iter_blocklist_file_rules(path) == []


def test_iter_blocklist_file_rules_skips_missing_file(tmp_path, caplog):
    path = tmp_path / "missing.txt"

    assert website_policy._iter_blocklist_file_rules(path) == []
    assert "not found" in caplog.text


def test_iter_blocklist_file_rules_skips_rules_that_normalize_to_empty(tmp_path, monkeypatch):
    path = tmp_path / "blocklist.txt"
    path.write_text("skip.test\nvalid.test\n", encoding="utf-8")

    original_normalize_rule = website_policy._normalize_rule

    def skip_one_rule(rule):
        if rule == "skip.test":
            return None
        return original_normalize_rule(rule)

    monkeypatch.setattr(website_policy, "_normalize_rule", skip_one_rule)

    assert website_policy._iter_blocklist_file_rules(path) == ["valid.test"]


def test_load_policy_config_returns_defaults_for_missing_file(tmp_path):
    assert website_policy._load_policy_config(tmp_path / "missing.yaml") == {
        "enabled": False,
        "domains": [],
        "shared_files": [],
    }


@pytest.mark.parametrize(
    "payload",
    [
        ["invalid root"],
        {"security": ["invalid security"]},
        {"security": {"website_blocklist": ["invalid blocklist"]}},
    ],
)
def test_load_policy_config_rejects_non_mapping_sections(tmp_path, payload):
    path = tmp_path / "config.yaml"
    write_config(path, payload)

    with pytest.raises(website_policy.WebsitePolicyError, match="mapping"):
        website_policy._load_policy_config(path)


def test_load_policy_config_treats_none_sections_as_empty(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path, {"security": None})

    assert website_policy._load_policy_config(path)["enabled"] is False


def test_load_policy_config_reports_yaml_and_io_errors(tmp_path, monkeypatch):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("security: [oops\n", encoding="utf-8")
    with pytest.raises(website_policy.WebsitePolicyError, match="Invalid config YAML"):
        website_policy._load_policy_config(malformed)

    readable = tmp_path / "readable.yaml"
    readable.write_text("{}\n", encoding="utf-8")

    def fail_open(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(website_policy.WebsitePolicyError, match="Failed to read config"):
        website_policy._load_policy_config(readable)


def test_load_policy_config_defaults_when_yaml_is_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    real_import = builtins.__import__

    def fail_yaml_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("PyYAML unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_yaml_import)

    assert website_policy._load_policy_config(path) == {
        "enabled": False,
        "domains": [],
        "shared_files": [],
    }


def test_load_website_blocklist_merges_and_deduplicates_sources(tmp_path):
    shared = tmp_path / "shared.txt"
    shared.write_text("example.com\nwww.shared.test\nexample.com\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    write_config(
        config,
        {
            "security": {
                "website_blocklist": {
                    "enabled": True,
                    "domains": ["example.com", "https://www.config.test/path", 42],
                    "shared_files": [str(shared), str(shared), "", 42],
                }
            }
        },
    )

    result = website_policy.load_website_blocklist(config)
    assert result["enabled"] is True
    assert {rule["pattern"] for rule in result["rules"]} == {
        "example.com",
        "config.test",
        "shared.test",
    }
    example_sources = {
        rule["source"] for rule in result["rules"] if rule["pattern"] == "example.com"
    }
    assert example_sources == {"config", str(shared)}


def test_load_website_blocklist_treats_none_blocklist_as_defaults(tmp_path):
    config = tmp_path / "config.yaml"
    write_config(config, {"security": {"website_blocklist": None}})

    assert website_policy.load_website_blocklist(config) == {
        "enabled": False,
        "rules": [],
    }


def test_load_website_blocklist_resolves_relative_shared_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "shared.txt").write_text("shared.test\n", encoding="utf-8")
    write_config(
        home / "config.yaml",
        {"security": {"website_blocklist": {"enabled": True, "shared_files": ["shared.txt"]}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = website_policy.load_website_blocklist()
    assert result["rules"] == [{"pattern": "shared.test", "source": str(home / "shared.txt")}]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("domains", {"invalid": True}, "domains must be a list"),
        ("shared_files", {"invalid": True}, "shared_files must be a list"),
        ("enabled", "yes", "enabled must be a boolean"),
    ],
)
def test_load_website_blocklist_validates_policy_shape(tmp_path, key, value, message):
    path = tmp_path / "config.yaml"
    write_config(path, {"security": {"website_blocklist": {key: value}}})

    with pytest.raises(website_policy.WebsitePolicyError, match=message):
        website_policy.load_website_blocklist(path)


def test_default_policy_cache_is_profile_aware_and_invalidation_reloads(tmp_path, monkeypatch):
    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    first_home.mkdir()
    second_home.mkdir()
    write_config(
        first_home / "config.yaml",
        {"security": {"website_blocklist": {"enabled": True, "domains": ["first.test"]}}},
    )
    write_config(
        second_home / "config.yaml",
        {"security": {"website_blocklist": {"enabled": True, "domains": ["second.test"]}}},
    )

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    first = website_policy.load_website_blocklist()
    monkeypatch.setenv("HERMES_HOME", str(second_home))
    second = website_policy.load_website_blocklist()

    assert first["rules"][0]["pattern"] == "first.test"
    assert second["rules"][0]["pattern"] == "second.test"

    write_config(
        second_home / "config.yaml",
        {"security": {"website_blocklist": {"enabled": True, "domains": ["reloaded.test"]}}},
    )
    website_policy.invalidate_cache()
    assert website_policy.load_website_blocklist()["rules"][0]["pattern"] == "reloaded.test"


def test_default_policy_cache_returns_fresh_cached_result_without_reload(tmp_path, monkeypatch):
    write_config(
        tmp_path / "config.yaml",
        {"security": {"website_blocklist": {"enabled": True, "domains": ["cached.test"]}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = website_policy.load_website_blocklist()

    monkeypatch.setattr(
        website_policy,
        "_load_policy_config",
        lambda *_args, **_kwargs: pytest.fail("fresh default policy should come from cache"),
    )

    second = website_policy.load_website_blocklist()
    assert second is first


@pytest.mark.parametrize(
    ("host", "pattern", "expected"),
    [
        ("", "example.com", False),
        ("example.com", "", False),
        ("example.com", "example.com", True),
        ("www.example.com", "example.com", True),
        ("other.test", "example.com", False),
        ("a.tracking.example", "*.tracking.example", True),
        ("tracking.example", "*.tracking.example", False),
    ],
)
def test_match_host_against_rule(host, pattern, expected):
    assert website_policy._match_host_against_rule(host, pattern) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://EXAMPLE.COM:8443/path", "example.com"),
        ("example.com/path", "example.com"),
        ("www.example.com", "www.example.com"),
        ("", ""),
        ("http://", ""),
    ],
)
def test_extract_host_from_urlish(url, expected):
    assert website_policy._extract_host_from_urlish(url) == expected


def test_check_website_access_returns_block_metadata(tmp_path):
    config = tmp_path / "config.yaml"
    write_config(
        config,
        {"security": {"website_blocklist": {"enabled": True, "domains": ["blocked.test"]}}},
    )

    result = website_policy.check_website_access("https://blocked.test/page", config_path=config)

    assert result == {
        "url": "https://blocked.test/page",
        "host": "blocked.test",
        "rule": "blocked.test",
        "source": "config",
        "message": "Blocked by website policy: 'blocked.test' matched rule 'blocked.test' from config",
    }


def test_check_website_access_allows_disabled_and_unmatched_urls(tmp_path):
    disabled = tmp_path / "disabled.yaml"
    write_config(
        disabled,
        {"security": {"website_blocklist": {"enabled": False, "domains": ["blocked.test"]}}},
    )
    enabled = tmp_path / "enabled.yaml"
    write_config(
        enabled,
        {"security": {"website_blocklist": {"enabled": True, "domains": ["blocked.test"]}}},
    )

    assert website_policy.check_website_access("https://blocked.test", config_path=disabled) is None
    assert website_policy.check_website_access("https://allowed.test", config_path=enabled) is None
    assert website_policy.check_website_access("") is None
    assert website_policy.check_website_access("http://") is None


def test_check_website_access_propagates_explicit_policy_errors(tmp_path):
    config = tmp_path / "malformed.yaml"
    config.write_text("security: [oops\n", encoding="utf-8")

    with pytest.raises(website_policy.WebsitePolicyError):
        website_policy.check_website_access("https://example.test", config_path=config)


def test_check_website_access_fails_open_on_default_policy_errors(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("security: [oops\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert website_policy.check_website_access("https://example.test") is None


def test_disabled_cached_policy_short_circuits_host_and_loader(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    write_config(
        home / "config.yaml",
        {"security": {"website_blocklist": {"enabled": False}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert website_policy.check_website_access("https://example.test") is None

    monkeypatch.setattr(
        website_policy,
        "_extract_host_from_urlish",
        lambda _: pytest.fail("cached disabled policy must skip host extraction"),
    )
    monkeypatch.setattr(
        website_policy,
        "load_website_blocklist",
        lambda *_args, **_kwargs: pytest.fail("cached disabled policy must skip loading"),
    )

    assert website_policy.check_website_access("https://example.test") is None


def test_default_policy_loader_unexpected_errors_fail_open(monkeypatch):
    def fail_load(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(website_policy, "load_website_blocklist", fail_load)
    assert website_policy.check_website_access("https://example.test") is None
