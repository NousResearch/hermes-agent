"""Tests for MCP diagnostics helpers."""

import types

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    return tmp_path


def _seed_config(tmp_path, servers):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": servers, "_config_version": 9}),
        encoding="utf-8",
    )


def test_diagnostics_reports_configured_disconnected_server(tmp_path):
    _seed_config(tmp_path, {
        "firecrawl": {"url": "https://api.example.com/mcp?api_key=secret-token"},
    })
    from tools.mcp_tool import get_mcp_diagnostics

    result = get_mcp_diagnostics()

    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "firecrawl"
    assert entry["configured"] is True
    assert entry["attempted"] is False
    assert entry["connected"] is False
    assert entry["tool_count"] == 0
    assert "secret-token" not in entry["transport"]
    assert "[REDACTED]" in entry["transport"]
    assert "hermes mcp doctor firecrawl --refresh" in entry["next_action"]


def test_diagnostics_hides_stdio_args_that_may_contain_secrets(tmp_path):
    _seed_config(tmp_path, {
        "secret_stdio": {
            "command": "npx",
            "args": ["server", "--token", "plain-secret-value", "--project", "public-id"],
        },
    })
    from tools.mcp_tool import get_mcp_diagnostics

    entry = get_mcp_diagnostics(name="secret_stdio")[0]
    assert "plain-secret-value" not in entry["transport"]
    assert "public-id" not in entry["transport"]
    assert "--token" not in entry["transport"]
    assert "5 args hidden" in entry["transport"]


def test_diagnostics_redacts_sensitive_flag_values_in_errors(tmp_path, monkeypatch):
    _seed_config(tmp_path, {"remote": {"url": "https://example.com/mcp"}})
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)

    def _raise_secret_arg(coro, *args, **kwargs):
        coro.close()
        raise RuntimeError("process failed --token plain-secret-value")

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _raise_secret_arg)

    entry = mcp_tool.get_mcp_diagnostics(name="remote", refresh=True)[0]
    assert "plain-secret-value" not in entry["last_error"]
    assert "[REDACTED]" in entry["last_error"]


def test_diagnostics_reports_active_connected_server(tmp_path, monkeypatch):
    _seed_config(tmp_path, {"obsidian": {"command": "uvx", "args": ["server"]}})
    from tools import mcp_tool

    fake_server = types.SimpleNamespace(
        session=object(),
        _registered_tool_names=["mcp_obsidian_read", "mcp_obsidian_write"],
        _tools=[object(), object()],
        _sampling=None,
        _error=None,
    )
    with mcp_tool._lock:
        mcp_tool._servers["obsidian"] = fake_server
    try:
        result = mcp_tool.get_mcp_diagnostics()
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop("obsidian", None)

    assert result[0]["connected"] is True
    assert result[0]["tool_count"] == 2
    assert result[0]["next_action"] == "No action needed."


def test_diagnostics_classifies_auth_failure_on_refresh(tmp_path, monkeypatch):
    _seed_config(tmp_path, {"remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}}})
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
    def _raise_auth(coro, *args, **kwargs):
        coro.close()
        raise RuntimeError("401 Unauthorized token=super-secret")

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _raise_auth)

    result = mcp_tool.get_mcp_diagnostics(name="remote", refresh=True)
    entry = result[0]
    assert entry["attempted"] is True
    assert entry["needs_auth"] is True
    assert entry["process_or_http_failure"] is False
    assert "super-secret" not in entry["last_error"]
    assert "[REDACTED]" in entry["last_error"]


def test_diagnostics_unknown_name():
    from tools.mcp_tool import get_mcp_diagnostics

    result = get_mcp_diagnostics(name="missing")

    assert result[0]["configured"] is False
    assert result[0]["connected"] is False
    assert "not found" in result[0]["last_error"]
