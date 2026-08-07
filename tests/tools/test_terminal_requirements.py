import importlib
import logging

import pytest


terminal_tool_module = importlib.import_module("tools.terminal_tool")


def _clear_terminal_env(monkeypatch):
    """Remove terminal env vars that could affect requirements checks."""
    keys = [
        "TERMINAL_ENV",
        "TERMINAL_CONTAINER_CPU",
        "TERMINAL_CONTAINER_DISK",
        "TERMINAL_CONTAINER_MEMORY",
        "TERMINAL_DOCKER_FORWARD_ENV",
        "TERMINAL_DOCKER_VOLUMES",
        "TERMINAL_LIFETIME_SECONDS",
        "TERMINAL_MODAL_MODE",
        "TERMINAL_SSH_HOST",
        "TERMINAL_SSH_PORT",
        "TERMINAL_SSH_USER",
        "TERMINAL_TIMEOUT",
        "TERMINAL_VERCEL_RUNTIME",
        "TERMINAL_BLAXEL_IMAGE",
        "TERMINAL_BLAXEL_TTL",
        "BL_WORKSPACE",
        "BL_API_KEY",
        "BL_REGION",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "VERCEL_OIDC_TOKEN",
        "VERCEL_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
        "HOME",
        "USERPROFILE",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    # Default: no Nous subscription — patch both the terminal_tool local
    # binding and tool_backend_helpers (used by resolve_modal_backend_state).
    monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: False)
    import tools.tool_backend_helpers as _tbh
    monkeypatch.setattr(_tbh, "managed_nous_tools_enabled", lambda: False)


def test_local_terminal_requirements(monkeypatch, caplog):
    """Local backend uses Hermes' own LocalEnvironment wrapper."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is True
    assert "Terminal requirements check failed" not in caplog.text


def test_unknown_terminal_env_logs_error_and_returns_false(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "unknown-backend")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Unknown TERMINAL_ENV 'unknown-backend'" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_managed_mode_without_feature_flag_logs_clear_error(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Nous Tool Gateway access is not currently available" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_backend_without_sdk_logs_specific_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: None)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "vercel is required for the Vercel Sandbox terminal backend" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_backend_without_auth_logs_specific_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "no supported auth configuration was found" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_backend_accepts_oidc_auth(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_vercel_backend_accepts_token_tuple_auth(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("VERCEL_TOKEN", "token")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "project")
    monkeypatch.setenv("VERCEL_TEAM_ID", "team")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


@pytest.mark.parametrize("runtime", ["node24", "node22", "python3.13"])
def test_vercel_backend_accepts_supported_runtimes(monkeypatch, runtime):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", runtime)
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_vercel_backend_accepts_blank_runtime(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", "   ")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_vercel_backend_rejects_unsupported_runtime(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", "node20")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Vercel Sandbox runtime 'node20' is not supported" in record.getMessage()
        and "node24, node22, python3.13" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_backend_rejects_nondefault_disk(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_CONTAINER_DISK", "8192")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "does not support custom TERMINAL_CONTAINER_DISK=8192" in record.getMessage()
        for record in caplog.records
    )


def test_vercel_backend_rejects_malformed_disk_without_raising(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_CONTAINER_DISK", "large")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Invalid value for TERMINAL_CONTAINER_DISK" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Blaxel backend
# ---------------------------------------------------------------------------


def _isolate_blaxel_cli_config(monkeypatch, tmp_path, workspaces=None):
    """Point the Blaxel CLI-config reader at a temp file.

    Without this, these tests read the developer's real ~/.blaxel/config.yaml
    and pass or fail depending on whose machine they run on.
    """
    import yaml

    from hermes_cli import blaxel_auth

    path = tmp_path / "blaxel-config.yaml"
    if workspaces is None:
        path.write_text("{}", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(workspaces), encoding="utf-8")
    monkeypatch.setattr(blaxel_auth, "_BLAXEL_CLI_CONFIG", path)
    return path


def test_blaxel_backend_without_any_workspace_logs_specific_error(
    monkeypatch, tmp_path, caplog
):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    _isolate_blaxel_cli_config(monkeypatch, tmp_path)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any("no workspace resolved" in r.getMessage() for r in caplog.records)


def test_blaxel_backend_with_workspace_but_no_credentials_logs_specific_error(
    monkeypatch, tmp_path, caplog
):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_WORKSPACE", "ws")
    _isolate_blaxel_cli_config(monkeypatch, tmp_path)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any("no credentials found" in r.getMessage() for r in caplog.records)


def test_blaxel_backend_accepts_cli_login_without_an_api_key(monkeypatch, tmp_path):
    """`bl login` is the documented path; requiring BL_API_KEY would reject it.

    The Blaxel SDK authenticates from the CLI's stored credentials, so the
    requirements gate must not be stricter than the SDK.
    """
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_WORKSPACE", "ws")
    _isolate_blaxel_cli_config(
        monkeypatch,
        tmp_path,
        {"workspaces": [{"name": "ws", "credentials": {"access_token": "tok"}}]},
    )
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_blaxel_backend_falls_back_to_the_cli_selected_workspace(monkeypatch, tmp_path):
    """No BL_WORKSPACE: use whatever workspace the CLI has selected."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    _isolate_blaxel_cli_config(
        monkeypatch,
        tmp_path,
        {
            "context": {"workspace": "ws"},
            "workspaces": [{"name": "ws", "credentials": {"apiKey": "k"}}],
        },
    )
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_blaxel_backend_without_sdk_reports_the_documented_install(
    monkeypatch, tmp_path, caplog
):
    """The failure message must match the install the extra actually provides."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_WORKSPACE", "ws")
    monkeypatch.setenv("BL_API_KEY", "key")
    _isolate_blaxel_cli_config(monkeypatch, tmp_path)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "hermes-agent[blaxel]" in r.getMessage() for r in caplog.records
    ), "install guidance must point at the extra, not a bare SDK pin"


def test_blaxel_backend_with_credentials_and_sdk_passes(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_WORKSPACE", "ws")
    monkeypatch.setenv("BL_API_KEY", "key")
    _isolate_blaxel_cli_config(monkeypatch, tmp_path)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_blaxel_defaults_do_not_inherit_the_shared_container_sizes(monkeypatch):
    """Blaxel memory/volume defaults must come from one place only.

    Guards the drift the review flagged on #20809: the env path defaulted to
    4096 MB while the persisted config defaulted to 5120 MB.
    """
    from hermes_constants import (
        BLAXEL_DEFAULT_CWD,
        BLAXEL_DEFAULT_MEMORY_MB,
        BLAXEL_DEFAULT_VOLUME_SIZE_MB,
    )

    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setattr(terminal_tool_module, "_terminal_config_bridge_attempted", True)

    config = terminal_tool_module._get_env_config()

    assert config["env_type"] == "blaxel"
    assert config["cwd"] == BLAXEL_DEFAULT_CWD
    assert config["container_memory"] == BLAXEL_DEFAULT_MEMORY_MB
    assert config["container_disk"] == BLAXEL_DEFAULT_VOLUME_SIZE_MB


def test_blaxel_sdk_pin_matches_across_every_declaration_site():
    """The pin is declared in three files; a mismatch is a silent install bug.

    This mirrors the sibling-site contract style already used elsewhere in the
    suite rather than sharing a constant, because lazy_deps and pyproject are
    intentionally literal.
    """
    import re
    from pathlib import Path

    from hermes_constants import BLAXEL_SDK_DEPENDENCY
    from tools.lazy_deps import LAZY_DEPS

    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    extra = re.search(r'^blaxel = \["([^"]+)"\]', pyproject, re.MULTILINE)

    assert extra, "pyproject.toml must declare a [blaxel] extra"
    assert LAZY_DEPS["terminal.blaxel"] == (BLAXEL_SDK_DEPENDENCY,)
    assert extra.group(1) == BLAXEL_SDK_DEPENDENCY
