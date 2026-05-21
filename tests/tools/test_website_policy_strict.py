from pathlib import Path

from tools.website_policy import check_website_access, invalidate_cache


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_unattended_strict_blocks_when_policy_disabled(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """
security:
  website_blocklist:
    enabled: false
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://example.com", config_path=cfg)

    assert result is not None
    assert result["rule"] == "policy-disabled"


def test_unattended_strict_blocks_on_allowlist_miss(tmp_path, monkeypatch):
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("allowed.example\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: allowlist
    allowlist_files:
      - {allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://blocked.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-miss"


def test_unattended_strict_allows_allowlisted_host(tmp_path, monkeypatch):
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("allowed.example\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: allowlist
    allowlist_files:
      - {allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://allowed.example/path", config_path=cfg)

    assert result is None


def test_unattended_strict_reports_empty_allowlist(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    mode: allowlist
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://blocked.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_unattended_strict_fails_closed_on_empty_configured_allowlist_file(tmp_path, monkeypatch):
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# no active rules\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: blocklist
    allowlist_files:
      - {allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://blocked.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_unattended_strict_fails_closed_on_missing_configured_allowlist_file(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: blocklist
    allowlist_files:
      - {tmp_path / "missing-allow.txt"}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://blocked.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_unattended_strict_fails_closed_without_configured_allowlist(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    mode: blocklist
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://unlisted.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_unattended_strict_fails_closed_on_malformed_allowlist_file_entry(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    mode: blocklist
    allowlist_files:
      - 123
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://blocked.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_config_strict_enforces_allowlist_files_in_blocklist_mode(tmp_path, monkeypatch):
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("allowed.example\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    strict: true
    mode: blocklist
    allowlist_files:
      - {allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.delenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", raising=False)

    allowed = check_website_access("https://allowed.example", config_path=cfg)
    blocked = check_website_access("https://blocked.example", config_path=cfg)

    assert allowed is None
    assert blocked is not None
    assert blocked["rule"] == "allowlist-miss"


def test_unattended_strict_fails_closed_when_one_of_multiple_allowlist_files_is_empty(tmp_path, monkeypatch):
    good_allowlist = tmp_path / "good-allow.txt"
    empty_allowlist = tmp_path / "empty-allow.txt"
    good_allowlist.write_text("allowed.example\n")
    empty_allowlist.write_text("# no active rules\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: blocklist
    allowlist_files:
      - {good_allowlist}
      - {empty_allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://allowed.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_unattended_strict_fails_closed_when_one_of_multiple_allowlist_files_is_missing(tmp_path, monkeypatch):
    good_allowlist = tmp_path / "good-allow.txt"
    missing_allowlist = tmp_path / "missing-allow.txt"
    good_allowlist.write_text("allowed.example\n")
    cfg = _write_config(
        tmp_path / "config.yaml",
        f"""
security:
  website_blocklist:
    enabled: true
    mode: blocklist
    allowlist_files:
      - {good_allowlist}
      - {missing_allowlist}
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://allowed.example", config_path=cfg)

    assert result is not None
    assert result["rule"] == "allowlist-empty"


def test_config_strict_blocks_when_policy_disabled(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """
security:
  website_blocklist:
    enabled: false
    strict: true
""".strip()
        + "\n",
    )
    monkeypatch.delenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", raising=False)

    result = check_website_access("https://example.com", config_path=cfg)

    assert result is not None
    assert result["rule"] == "policy-disabled"


def test_config_strict_bypasses_cached_disabled_policy(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes-disabled-cache"
    hermes_home.mkdir()
    cfg = hermes_home / "config.yaml"
    _write_config(
        cfg,
        """
security:
  website_blocklist:
    enabled: false
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", raising=False)
    invalidate_cache()

    first = check_website_access("https://unlisted.example")

    allowlist = hermes_home / "allow.txt"
    allowlist.write_text("allowed.example\n")
    _write_config(
        cfg,
        """
security:
  website_blocklist:
    enabled: true
    strict: true
    mode: blocklist
    allowlist_files:
      - allow.txt
""".strip()
        + "\n",
    )
    second = check_website_access("https://unlisted.example")

    assert first is None
    assert second is not None
    assert second["rule"] == "allowlist-miss"


def test_unattended_strict_bypasses_cached_allowlist_when_source_becomes_empty(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes-cache"
    hermes_home.mkdir()
    allowlist = hermes_home / "allow.txt"
    allowlist.write_text("allowed.example\n")
    _write_config(
        hermes_home / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    mode: allowlist
    allowlist_files:
      - allow.txt
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")
    invalidate_cache()

    first = check_website_access("https://allowed.example")
    allowlist.write_text("# source emptied after first load\n")
    second = check_website_access("https://allowed.example")

    assert first is None
    assert second is not None
    assert second["rule"] == "allowlist-empty"


def test_config_strict_fails_closed_on_default_path_malformed_strict_value(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes-malformed-strict"
    hermes_home.mkdir()
    _write_config(
        hermes_home / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    strict: "true"
    mode: invalid-mode
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", raising=False)
    invalidate_cache()

    result = check_website_access("https://example.com")

    assert result is not None
    assert result["rule"] == "policy-error"


def test_config_strict_fails_closed_on_default_path_policy_error(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes-strict"
    hermes_home.mkdir()
    _write_config(
        hermes_home / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    strict: true
    mode: invalid-mode
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", raising=False)

    result = check_website_access("https://example.com")

    assert result is not None
    assert result["rule"] == "policy-error"


def test_unattended_strict_fails_closed_on_default_path_policy_error(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_config(
        hermes_home / "config.yaml",
        """
security:
  website_blocklist:
    enabled: true
    mode: invalid-mode
""".strip()
        + "\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_UNATTENDED_STRICT_WEBSITE_POLICY", "1")

    result = check_website_access("https://example.com")

    assert result is not None
    assert result["rule"] == "policy-error"
