from pathlib import Path

from tools.website_policy import check_website_access


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
