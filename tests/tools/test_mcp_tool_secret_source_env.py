"""Stdio MCP child env: a secret source must not forward a credential Hermes itself consumes."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_secret_source_injected_vars_are_passed(monkeypatch):
    """Vars tagged by an external secret source (Bitwarden/1Password) are
    deliberately allowed for MCP stdio servers."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

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
    from hermes_agent.agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

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
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

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
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

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


def test_secret_source_never_forwards_config_declared_provider_keys(monkeypatch):
    """A name the profile's config.yaml points a provider credential at (``key_env`` /
    ``api_key_env`` on the main model, a custom/named provider, an auxiliary task, a fallback
    route) is a Hermes credential under that name, so a vault copy of it is withheld too."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n  key_env: MY_LLM_KEY\n"
        "custom_providers:\n  - name: lab\n    base_url: http://127.0.0.1:9/v1\n    key_env: LAB_KEY\n"
        "providers:\n  edge:\n    base_url: http://127.0.0.1:9/v1\n    api_key_env: EDGE_KEY\n"
        "auxiliary:\n  vision:\n    provider: custom\n    key_env: VISION_KEY\n"
        "fallback_providers:\n  - provider: custom\n    model: m\n    base_url: http://127.0.0.1:9/v1\n"
        "    key_env: FALLBACK_KEY\n"
        "hooks:\n  outbound:\n    - url: https://example.invalid/hook\n      secret_env: HOOK_SECRET\n",
        encoding="utf-8")
    declared = ["MY_LLM_KEY", "LAB_KEY", "EDGE_KEY", "VISION_KEY", "FALLBACK_KEY", "HOOK_SECRET"]
    for name in [*declared, "ALPACA_API_KEY"]:
        monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "bitwarden")
    fake_env = {"PATH": "/usr/bin", "ALPACA_API_KEY": "MARKER-third-party",
                **{name: f"MARKER-{name}" for name in declared}, "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert not set(declared) & set(result)
    assert result["ALPACA_API_KEY"] == "MARKER-third-party"


def test_secret_source_never_forwards_config_env_ref_credentials(monkeypatch):
    """A credential field written as a whole-value ``${VAR}`` / ``${env:VAR}`` ref
    (``delegation.api_key: ${DELEGATION_KEY}``) makes VAR a Hermes credential, so its vault copy
    is withheld even though the merged config already holds the expanded value. An
    ``mcp_servers.*.env`` ref is the server's declared env and is still forwarded."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "delegation:\n  base_url: http://127.0.0.1:9/v1\n  api_key: ${DELEGATION_KEY}\n"
        "auxiliary:\n  vision:\n    base_url: http://127.0.0.1:9/v1\n    api_key: '${env:LAB_VISION}'\n"
        "custom_providers:\n  - name: lab\n    base_url: http://127.0.0.1:9/v1\n    api_key: ${LAB_LLM}\n"
        "platforms:\n  api_server:\n    extra:\n      key: ${INGRESS_CREDENTIAL}\n"
        "mcp_servers:\n  other:\n    command: x\n    env:\n      SERVER_TOKEN: ${SERVER_TOKEN}\n",
        encoding="utf-8")
    referenced = ["DELEGATION_KEY", "LAB_VISION", "LAB_LLM", "INGRESS_CREDENTIAL"]
    for name in [*referenced, "SERVER_TOKEN"]:
        monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "command")
    fake_env = {"PATH": "/usr/bin", "SERVER_TOKEN": "MARKER-server",
                **{name: f"MARKER-{name}" for name in referenced}, "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert not set(referenced) & set(result)
    assert result["SERVER_TOKEN"] == "MARKER-server"


def test_secret_source_never_forwards_model_api_alias_ref(monkeypatch):
    """``model.api`` is read as the main model's API key (an alias of ``model.api_key``), so a
    ``${VAR}`` there makes VAR a Hermes credential whose vault copy is withheld."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n  api: ${LAB_LLM}\n", encoding="utf-8")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "LAB_LLM", "command")
    fake_env = {"PATH": "/usr/bin", "LAB_LLM": "MARKER-llm", "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert "LAB_LLM" not in result


def test_secret_source_never_forwards_managed_config_credential_refs(monkeypatch, tmp_path):
    """The managed-scope config.yaml is expanded and merged over the profile's, so a credential
    ``${VAR}`` it declares is a Hermes credential exactly like one in the profile file; its own
    ``mcp_servers.*.env`` refs stay the server's declared env."""
    from hermes_agent.hermes_cli import env_loader, managed_scope
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "custom_providers:\n  - name: lab\n    base_url: http://127.0.0.1:9/v1\n    api_key: ${LAB_LLM}\n"
        "mcp_servers:\n  other:\n    command: x\n    env:\n      SERVER_TOKEN: ${SERVER_TOKEN}\n",
        encoding="utf-8")
    for name in ("LAB_LLM", "SERVER_TOKEN"):
        monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "command")
    fake_env = {"PATH": "/usr/bin", "LAB_LLM": "MARKER-llm", "SERVER_TOKEN": "MARKER-server",
                "HERMES_MANAGED_DIR": str(managed), "HERMES_HOME": os.environ["HERMES_HOME"]}
    managed_scope.invalidate_managed_cache()
    try:
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)
    finally:
        managed_scope.invalidate_managed_cache()

    assert "LAB_LLM" not in result
    assert result["SERVER_TOKEN"] == "MARKER-server"


@pytest.mark.parametrize("template", ["prefix-${LAB_FRAGMENT}", "${LAB_FRAGMENT}-suffix",
                                      "${env:LAB_FRAGMENT}${ALPACA_API_KEY}"])
def test_secret_source_never_forwards_embedded_credential_ref(monkeypatch, template):
    """The loader expands every ``${VAR}`` inside a credential string, not only a whole-value one,
    so a name embedded in one is a Hermes credential too; a third-party name stays forwarded."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        f'model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n  api_key: "{template}"\n',
        encoding="utf-8")
    for name in ("LAB_FRAGMENT", "ALPACA_API_KEY"):
        monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "command")
    fake_env = {"PATH": "/usr/bin", "LAB_FRAGMENT": "MARKER-fragment", "ALPACA_API_KEY": "MARKER-third-party",
                "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert "LAB_FRAGMENT" not in result
    if "ALPACA_API_KEY" not in template:
        assert result["ALPACA_API_KEY"] == "MARKER-third-party"


def test_secret_source_withholds_last_good_credential_while_config_is_broken(monkeypatch):
    """A config.yaml edited into broken YAML keeps its last-known-good credentials in use, so the
    names they were read from stay withheld instead of being released by the failed raw read."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.hermes_cli.config import load_config_readonly
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    config = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config.write_text("model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n"
                      "  api_key: ${LAB_FRAGMENT}\n", encoding="utf-8")
    for name in ("LAB_FRAGMENT", "ALPACA_API_KEY"):
        monkeypatch.setitem(env_loader._SECRET_SOURCES, name, "command")
    fake_env = {"PATH": "/usr/bin", "LAB_FRAGMENT": "MARKER-fragment", "ALPACA_API_KEY": "MARKER-third-party",
                "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        assert load_config_readonly()["model"]["api_key"] == "MARKER-fragment"
        config.write_text("model: [broken\n", encoding="utf-8")
        assert load_config_readonly()["model"]["api_key"] == "MARKER-fragment"
        result = _build_safe_env(None)
        explicit = _build_safe_env({"LAB_FRAGMENT": "explicit"})

    assert "LAB_FRAGMENT" not in result
    assert result["ALPACA_API_KEY"] == "MARKER-third-party"
    assert explicit["LAB_FRAGMENT"] == "explicit"


def test_unreadable_backups_dir_does_not_stop_stdio_env_while_config_is_broken(monkeypatch):
    """Broken YAML plus an unreadable ``backups/config`` must not turn the last-good ref lookup into
    an exception: the stdio server still starts (the env is built), as it did before the lookup."""
    from hermes_agent.hermes_cli import config_backups, env_loader
    from hermes_agent.hermes_cli.config import load_config_readonly
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    config = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config.write_text("model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n"
                      "  api_key: ${LAB_FRAGMENT}\n", encoding="utf-8")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ALPACA_API_KEY", "command")
    fake_env = {"PATH": "/usr/bin", "LAB_FRAGMENT": "MARKER-fragment", "ALPACA_API_KEY": "MARKER-third-party",
                "HERMES_HOME": os.environ["HERMES_HOME"]}

    def unreadable(*_args, **_kwargs):
        raise PermissionError("backups/config")

    with patch.dict("os.environ", fake_env, clear=True):
        load_config_readonly()
        config.write_text("model: [broken\n", encoding="utf-8")
        monkeypatch.setattr(config_backups, "list_config_backups", unreadable)
        result = _build_safe_env(None)

    assert result["ALPACA_API_KEY"] == "MARKER-third-party"


def test_secret_source_mcp_server_env_pointer_does_not_block_vendor_token(monkeypatch):
    """An MCP server env entry is not a Hermes credential declaration, even if its
    arbitrary name matches the *_key_env pointer convention elsewhere in config."""
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.tools.mcp_tool_config import _build_safe_env

    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n  key_env: MY_LLM_KEY\n"
        "mcp_servers:\n  helper:\n    command: x\n    env:\n      api_key_env: VENDOR_TOKEN\n",
        encoding="utf-8")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "MY_LLM_KEY", "command")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "VENDOR_TOKEN", "command")
    fake_env = {"PATH": "/usr/bin", "MY_LLM_KEY": "MARKER-hermes",
                "VENDOR_TOKEN": "MARKER-vendor", "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)

    assert "MY_LLM_KEY" not in result
    assert result["VENDOR_TOKEN"] == "MARKER-vendor"
