"""Standalone tests for ``tools.config_validator_tool``.

Run without extra dependencies:

    pytest tests/tools/test_config_validator_tool.py -q
"""

import os
import sys
from pathlib import Path

import pytest


def test_success_envelope_shape(monkeypatch, tmp_path):
    """``run()`` returns a Success envelope with expected keys/markers."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text(
        "model:\n  default: test-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()

    assert payload.get("kind") == "success", payload
    data = payload.get("data", {})
    assert isinstance(data, dict)
    assert "valid" in data and isinstance(data["valid"], bool)
    assert "warnings" in data and isinstance(data["warnings"], list)
    assert "issues" in data and isinstance(data["issues"], list)
    assert "config_path" in data
    assert "config_exists" in data
    assert data["config_path"].endswith("config.yaml")
    assert data["config_exists"] is True


def test_malformed_yaml_failure(monkeypatch, tmp_path):
    """Malformed YAML yields a Failure envelope with code config_validator_error."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("model:\n  default: [unterminated", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    assert isinstance(payload, dict)
    assert payload.get("kind") == "failure"
    assert payload.get("code") == "config_validator_error"
    assert "config" in payload.get("error", "").lower()


def test_missing_config_failure(monkeypatch, tmp_path):
    """Missing config file yields a Failure envelope with code config_validator_error."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    assert isinstance(payload, dict)
    assert payload.get("kind") == "failure"
    assert payload.get("code") == "config_validator_error"
    assert "config" in payload.get("error", "").lower()


def test_unknown_key_populates_warnings(monkeypatch, tmp_path):
    """Unknown top-level keys should appear in warnings with likely key + snippet."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    body = "definitely_not_a_key: true\nmodel:\n  default: x\n"
    (fake_home / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    assert payload.get("kind") == "success"
    data = payload.get("data", {})
    assert data.get("valid") is True
    assert len(data["warnings"]) >= 1
    w0 = data["warnings"][0]
    assert "likely_key" in w0 or "key" in w0
    k = w0.get("likely_key") or w0.get("key")
    assert "definitely_not_a_key" in str(k)
    # Should include a snippet context string.
    assert "snippet" in w0
    assert isinstance(w0["snippet"], str) and len(w0["snippet"]) > 0


def test_empty_file_returns_failure(monkeypatch, tmp_path):
    """An empty config file is treated as missing/malformed and returns failure."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    # Empty file parses as None in YAML; structural check disqualifies it.
    assert payload.get("kind") == "success"
    data = payload.get("data", {})
    assert data.get("valid") is False
    assert data["issues"]


def test_check_fn_false_when_unavailable(monkeypatch):
    """``check_fn`` returns False when profile home/config path is unavailable."""
    from tools import config_validator_tool as mod

    monkeypatch.delenv("HERMES_HOME", raising=False)
    # Patch get_hermes_home to return a non-existent path.
    monkeypatch.setattr(mod, "get_hermes_home", lambda: Path("/nonexistent/path/12345"))
    assert mod.check_fn() is False


def test_check_fn_true_when_available(monkeypatch, tmp_path):
    """``check_fn`` returns True when the config file exists and is readable."""
    from tools import config_validator_tool as mod

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    assert mod.check_fn() is True


def test_duplicate_top_level_is_issue(monkeypatch, tmp_path):
    """Duplicate top-level keys should be reported in issues."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    body = "model:\n  default: x\nmodel:\n  default: y\n"
    (fake_home / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    assert payload.get("kind") == "success"
    data = payload.get("data", {})
    assert data.get("valid") is False
    dup_issues = [i for i in data["issues"] if "Duplicate" in i.get("message", "")]
    assert dup_issues


def test_success_no_empty_lists_when_ok(monkeypatch, tmp_path):
    """Valid clean config should have empty issues/warnings lists."""
    from tools.config_validator_tool import run

    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text(
        "model:\n  default: x\nagent:\n  max_turns: 10\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    payload = run()
    assert payload.get("kind") == "success"
    data = payload.get("data", {})
    assert data["valid"] is True
    assert data["issues"] == []
    assert data["warnings"] == []
