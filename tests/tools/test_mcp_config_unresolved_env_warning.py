"""Tests for MCP config unresolved ``${VAR}`` env-ref warnings.

An unset ``${VAR}`` placeholder survives interpolation as a literal string and
is then sent to the server as-is (e.g. ``Authorization: ${MY_TOKEN}``), which
surfaces downstream only as an opaque 401/connect failure. The load path warns
once per (server, key path), naming the missing variables so the fix is a
``.env``/secrets-source entry instead of a log spelunk.
"""

import logging

import pytest

from tools import mcp_tool_config as _mcp_config
from tools.mcp_tool_config import _warn_unresolved_env_refs


@pytest.fixture(autouse=True)
def _reset_dedupe():
    _mcp_config._unresolved_env_warned.clear()
    yield
    _mcp_config._unresolved_env_warned.clear()


def test_fully_resolved_config_no_warnings(caplog):
    config = {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer abc123"},
        "args": ["--flag", "value"],
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("clean", config)
    assert flagged == []
    assert not [r for r in caplog.records if "not set" in r.getMessage()]


def test_literal_ref_in_header_named(caplog):
    config = {"url": "https://example.com/mcp",
              "headers": {"Authorization": "${MY_MCP_TOKEN}"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["headers.Authorization"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("srv" in m and "MY_MCP_TOKEN" in m and "headers.Authorization" in m
               for m in messages)


def test_cursor_env_prefix_reports_bare_name(caplog):
    config = {"url": "https://example.com/mcp",
              "headers": {"X-Auth": "${env:PG_TOKEN}"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["headers.X-Auth"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("'PG_TOKEN'" in m for m in messages)
    assert not any("env:" in m for m in messages)


def test_ref_inside_compound_string_flagged(caplog):
    config = {"args": ["--home", "${UNSET_HOME_DIR}/sub"]}
    flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["args[1]"]


def test_multiple_refs_in_one_value_all_named(caplog):
    config = {"url": "${MISSING_HOST}:${MISSING_PORT}/mcp"}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["url"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("MISSING_HOST" in m and "MISSING_PORT" in m for m in messages)


def test_placeholder_value_itself_not_logged(caplog):
    config = {"headers": {"Authorization": "Bearer ${TOKEN_X} suffix"}}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        _warn_unresolved_env_refs("srv", config)
    messages = [r.getMessage() for r in caplog.records]
    assert any("TOKEN_X" in m for m in messages)
    assert not any("Bearer ${TOKEN_X} suffix" in m for m in messages)


def test_values_never_mutated():
    config = {"headers": {"Authorization": "${STILL_UNSET}"}}
    _warn_unresolved_env_refs("srv", config)
    assert config["headers"]["Authorization"] == "${STILL_UNSET}"


def test_warning_deduped_per_process(caplog):
    config = {"url": "${UNSET_URL_VAR}"}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        first = _warn_unresolved_env_refs("srv", config)
        second = _warn_unresolved_env_refs("srv", config)
    assert first == second == ["url"]
    warn_records = [r for r in caplog.records if "not set" in r.getMessage()]
    assert len(warn_records) == 1


def test_distinct_servers_warned_separately(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        _warn_unresolved_env_refs("a", {"url": "${MISSING_A}"})
        _warn_unresolved_env_refs("b", {"url": "${MISSING_B}"})
    warn_records = [r.getMessage() for r in caplog.records]
    assert any("MISSING_A" in m for m in warn_records)
    assert any("MISSING_B" in m for m in warn_records)


def test_load_mcp_config_names_unset_vars(tmp_path, monkeypatch, caplog):
    """E2E through _load_mcp_config with a real config load path."""
    from unittest.mock import patch as mock_patch

    monkeypatch.delenv("LIBRARIAN_MCP_AUTH_TEST", raising=False)
    servers = {
        "brain": {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "${LIBRARIAN_MCP_AUTH_TEST}"},
        }
    }
    with mock_patch("hermes_cli.config.load_config",
                    return_value={"mcp_servers": servers}), \
         caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        result = _mcp_config._load_mcp_config()

    assert "brain" in result
    # Placeholder still passes through unmutated (literal preserved for send).
    assert result["brain"]["headers"]["Authorization"] == "${LIBRARIAN_MCP_AUTH_TEST}"
    messages = [r.getMessage() for r in caplog.records]
    assert any("LIBRARIAN_MCP_AUTH_TEST" in m for m in messages)


def test_load_mcp_config_set_var_stays_quiet(monkeypatch, caplog):
    from unittest.mock import patch as mock_patch

    monkeypatch.setenv("RESOLVED_MCP_AUTH_TEST", "tok-resolved")
    servers = {
        "brain": {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "${RESOLVED_MCP_AUTH_TEST}"},
        }
    }
    with mock_patch("hermes_cli.config.load_config",
                    return_value={"mcp_servers": servers}), \
         caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        result = _mcp_config._load_mcp_config()

    assert result["brain"]["headers"]["Authorization"] == "tok-resolved"
    assert not [r for r in caplog.records if "not set" in r.getMessage()]
