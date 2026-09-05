"""Tests for MCP config hidden-whitespace warnings.

Inspired by Claude Code v2.1.219: warn when MCP config values carry hidden
leading/trailing whitespace (pasted tokens with trailing newlines, URLs with
leading spaces), which otherwise surfaces as opaque auth/connect failures.
"""

import logging

import pytest

from tools import mcp_tool
from tools import mcp_tool_config as _mcp_config
from tools.mcp_tool_config import _warn_hidden_whitespace


@pytest.fixture(autouse=True)
def _reset_dedupe():
    _mcp_config._whitespace_warned.clear()
    yield
    _mcp_config._whitespace_warned.clear()


def test_clean_config_no_warnings(caplog):
    config = {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer abc123"},
        "args": ["--flag", "value"],
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_hidden_whitespace("clean", config)
    assert flagged == []
    assert not [r for r in caplog.records if "hidden" in r.getMessage()]


def test_trailing_newline_in_header_flagged(caplog):
    config = {"url": "https://example.com/mcp",
              "headers": {"Authorization": "Bearer abc123\n"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_hidden_whitespace("srv", config)
    assert flagged == ["headers.Authorization"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("srv" in m and "headers.Authorization" in m for m in messages)
    # The secret value itself must never appear in the log.
    assert not any("abc123" in m for m in messages)


def test_leading_space_in_url_flagged():
    flagged = _warn_hidden_whitespace("srv", {"url": " https://example.com"})
    assert flagged == ["url"]


def test_whitespace_in_list_item_flagged_with_index():
    flagged = _warn_hidden_whitespace(
        "srv", {"command": "npx", "args": ["-y", "some-pkg "]}
    )
    assert flagged == ["args[1]"]


def test_nested_env_dict_flagged():
    flagged = _warn_hidden_whitespace(
        "srv", {"command": "npx", "env": {"API_KEY": "secret\t"}}
    )
    assert flagged == ["env.API_KEY"]


def test_multiple_flags_all_reported():
    flagged = _warn_hidden_whitespace(
        "srv", {"url": "https://x.com ", "headers": {"X-Key": " k"}}
    )
    assert set(flagged) == {"url", "headers.X-Key"}


def test_non_string_values_ignored():
    flagged = _warn_hidden_whitespace(
        "srv", {"timeout": 30, "enabled": True, "retries": None}
    )
    assert flagged == []


def test_values_never_mutated():
    config = {"headers": {"Authorization": "Bearer tok\n"}}
    _warn_hidden_whitespace("srv", config)
    assert config["headers"]["Authorization"] == "Bearer tok\n"


def test_warning_deduped_per_process(caplog):
    config = {"url": "https://x.com "}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        first = _warn_hidden_whitespace("srv", config)
        second = _warn_hidden_whitespace("srv", config)
    # Both calls still report the flagged path (return value is for callers)...
    assert first == second == ["url"]
    # ...but only one warning record is emitted.
    warn_records = [r for r in caplog.records if "hidden" in r.getMessage()]
    assert len(warn_records) == 1


def test_distinct_servers_warned_separately(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        _warn_hidden_whitespace("a", {"url": "https://x.com "})
        _warn_hidden_whitespace("b", {"url": "https://x.com "})
    warn_records = [r for r in caplog.records if "hidden" in r.getMessage()]
    assert len(warn_records) == 2


def test_load_mcp_config_emits_warning(tmp_path, monkeypatch, caplog):
    """E2E through _load_mcp_config with a real config load path."""
    from unittest.mock import patch as mock_patch

    servers = {
        "pasted": {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer tok\n"},
        }
    }
    with mock_patch("hermes_cli.config.load_config",
                    return_value={"mcp_servers": servers}), \
         caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        result = _mcp_config._load_mcp_config()

    assert "pasted" in result
    # Value passes through unmutated.
    assert result["pasted"]["headers"]["Authorization"] == "Bearer tok\n"
    messages = [r.getMessage() for r in caplog.records]
    assert any("headers.Authorization" in m for m in messages)
