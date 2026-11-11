"""Stdio MCP child env: a secret source must not forward a credential Hermes itself consumes."""

import os
from pathlib import Path
from unittest.mock import patch


def test_secret_source_injected_vars_are_passed(monkeypatch):
    """Vars tagged by an external secret source (Bitwarden/1Password) are
    deliberately allowed for MCP stdio servers."""
    from hermes_cli import env_loader
    from tools.mcp_tool_config import _build_safe_env

    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ALPACA_API_KEY", "bitwarden")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "NOTION_TOKEN", "onepassword")
    fake_env = {
        "PATH": "/usr/bin",
        "ALPACA_API_KEY": "from-bws-key",
        "NOTION_TOKEN": "from-op",
        "UNTRACKED_SECRET_KEY": "still-filtered",
        "HERMES_HOME": os.environ["HERMES_HOME"],
    }
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert result["PATH"] == "/usr/bin"
    assert result["ALPACA_API_KEY"] == "from-bws-key"
    assert result["NOTION_TOKEN"] == "from-op"
    assert "UNTRACKED_SECRET_KEY" not in result


def test_secret_source_vars_resolve_through_active_profile_scope(monkeypatch):
    """Under multiplex the stdio child gets the ROUTED profile's value for a source-tagged name,
    never the launch profile's os.environ copy; a name the profile lacks is omitted."""
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from hermes_cli import env_loader
    from tools.mcp_tool_config import _build_safe_env

    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ALPACA_API_KEY", "bitwarden")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "NOTION_TOKEN", "onepassword")
    fake_env = {"PATH": "/usr/bin", "ALPACA_API_KEY": "default-profile", "NOTION_TOKEN": "default-notion",
                "HERMES_HOME": os.environ["HERMES_HOME"]}
    set_multiplex_active(True)
    token = set_secret_scope({"ALPACA_API_KEY": "profile-b"})
    try:
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    assert result["PATH"] == "/usr/bin"
    assert result["ALPACA_API_KEY"] == "profile-b"
    assert "NOTION_TOKEN" not in result


def test_secret_source_never_forwards_hermes_credentials(monkeypatch):
    """Where a Hermes-managed credential is stored (``.env`` or a secret source) must not change
    whether a stdio child receives it; a server that needs one declares it in its own ``env``."""
    from hermes_cli import env_loader
    from tools.mcp_tool_config import _build_safe_env

    hermes_creds = {"OPENAI_API_KEY": "sk-MARKER-provider", "TELEGRAM_BOT_TOKEN": "MARKER-bot"}
    fake_env = {"PATH": "/usr/bin", "ALPACA_API_KEY": "MARKER-third-party", **hermes_creds,
                "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        from_dotenv = _build_safe_env(None)
        for name in [*hermes_creds, "ALPACA_API_KEY"]:
            monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "bitwarden")
        from_source = _build_safe_env(None)
        declared = _build_safe_env({"OPENAI_API_KEY": "sk-MARKER-provider"})

    assert {k: from_source.get(k) for k in hermes_creds} == {k: from_dotenv.get(k) for k in hermes_creds}
    assert not set(hermes_creds) & set(from_source)
    assert from_source["ALPACA_API_KEY"] == "MARKER-third-party"
    assert declared["OPENAI_API_KEY"] == "sk-MARKER-provider"


def test_secret_source_never_forwards_profile_platform_plugin_secret(monkeypatch):
    """A secret the bound profile's user-installed platform plugin declares is that adapter's
    credential, so a secret-source copy of it is withheld like a bundled adapter's token."""
    from hermes_cli import env_loader
    from tools.mcp_tool_config import _build_safe_env

    plugin = Path(os.environ["HERMES_HOME"]) / "plugins" / "platforms" / "chatx"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: chatx\nkind: platform\nrequires_env:\n  - name: CHATX_SIGNING_SECRET\n    password: true\n",
        encoding="utf-8")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "CHATX_SIGNING_SECRET", "bitwarden")
    fake_env = {"PATH": "/usr/bin", "CHATX_SIGNING_SECRET": "MARKER-adapter",
                "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert "CHATX_SIGNING_SECRET" not in result
