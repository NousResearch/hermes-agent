"""Regression coverage for MCP credential rotation during gateway hot reload."""

from unittest.mock import patch


def test_hot_reload_prefers_fresh_profile_secret(tmp_path, monkeypatch):
    """A rotated MCP bearer on disk must beat a stale process value."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ROTATING_MCP_TOKEN", "revoked-old-token")
    (tmp_path / ".env").write_text("ROTATING_MCP_TOKEN=fresh-token\n")
    servers = {
        "remote": {
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer ${ROTATING_MCP_TOKEN}"},
        }
    }

    with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}):
        from tools.mcp_tool import _load_mcp_config

        result = _load_mcp_config()

    assert result["remote"]["headers"]["Authorization"] == "Bearer fresh-token"


def test_hot_reload_ignores_request_local_home_override(tmp_path, monkeypatch):
    """Process-global MCP reload must use the control-plane process home."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    process_home = tmp_path / "process"
    request_home = tmp_path / "request"
    process_home.mkdir()
    request_home.mkdir()
    (process_home / ".env").write_text("ROTATING_MCP_TOKEN=process-fresh-token\n")
    (request_home / ".env").write_text("ROTATING_MCP_TOKEN=request-local-token\n")
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    servers = {
        "remote": {
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer ${ROTATING_MCP_TOKEN}"},
        }
    }

    override = set_hermes_home_override(request_home)
    try:
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}):
            from tools.mcp_tool import _load_mcp_config

            result = _load_mcp_config()
    finally:
        reset_hermes_home_override(override)

    assert result["remote"]["headers"]["Authorization"] == "Bearer process-fresh-token"
