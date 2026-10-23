"""#137825: a flat ``not connected`` for an OAuth server with a token file on disk must carry an
auth hint, so the model waits for the self-healing reconnect instead of sending the user into a
browser OAuth flow that was never needed."""

import json

import pytest

from tools.mcp_oauth import HermesTokenStorage


def _flat_acquire(monkeypatch, name="zoho_projects"):
    """``_acquire_call_server`` with no registered server (the flat-error path) and a inert breaker."""
    from tools import mcp_tool, mcp_tool_discovery, mcp_tool_handlers
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda _name: None)
    monkeypatch.setattr(mcp_tool, "_bump_server_error", lambda _name, **_kw: None)
    return mcp_tool_handlers._acquire_call_server(name, 0)


def _configure(monkeypatch, auth):
    from tools import mcp_tool_config
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config",
                        lambda: {"zoho_projects": {"url": "https://mcp.example/mcp", "auth": auth}})


def test_oauth_token_on_disk_upgrades_flat_error(monkeypatch, tmp_path):
    from tools import mcp_oauth
    monkeypatch.setattr(mcp_oauth, "_get_token_dir", lambda _home=None: tmp_path)
    _configure(monkeypatch, "oauth")
    token_file = HermesTokenStorage("zoho_projects")._tokens_path()
    token_file.write_text('{"access_token": "x"}')

    server, error = _flat_acquire(monkeypatch)
    payload = json.loads(error)
    assert server is None
    assert payload["server"] == "zoho_projects"
    assert "OAuth token file is on disk" in payload["error"]
    assert "retry this tool" in payload["error"].lower()
    # The hint is the point: it steers the model away from a needless user OAuth round-trip.
    assert payload["user_action"] == ("Wait briefly and retry the tool; the connection self-heals. "
                                      "Do not start an OAuth flow for this server.")
    assert payload["retry"] == "after_user_action"


def test_non_oauth_server_keeps_flat_error(monkeypatch, tmp_path):
    from tools import mcp_oauth
    monkeypatch.setattr(mcp_oauth, "_get_token_dir", lambda _home=None: tmp_path)
    HermesTokenStorage("zoho_projects")._tokens_path().write_text('{"access_token": "x"}')
    _configure(monkeypatch, "")

    _server, error = _flat_acquire(monkeypatch)
    payload = json.loads(error)
    assert payload == {"error": "MCP server 'zoho_projects' is not connected"}


def test_oauth_without_token_file_keeps_flat_error(monkeypatch, tmp_path):
    from tools import mcp_oauth
    monkeypatch.setattr(mcp_oauth, "_get_token_dir", lambda _home=None: tmp_path)
    _configure(monkeypatch, "oauth")

    _server, error = _flat_acquire(monkeypatch)
    payload = json.loads(error)
    assert payload == {"error": "MCP server 'zoho_projects' is not connected"}


def test_token_stat_failure_keeps_flat_error(monkeypatch, tmp_path):
    from tools import mcp_oauth
    monkeypatch.setattr(mcp_oauth, "_get_token_dir", lambda _home=None: tmp_path)
    _configure(monkeypatch, "oauth")
    monkeypatch.setattr(HermesTokenStorage, "has_cached_tokens",
                        lambda self: (_ for _ in ()).throw(OSError("stat failed")))

    _server, error = _flat_acquire(monkeypatch)
    payload = json.loads(error)
    assert payload == {"error": "MCP server 'zoho_projects' is not connected"}
