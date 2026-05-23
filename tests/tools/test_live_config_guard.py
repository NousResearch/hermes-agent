"""Regression tests for guarded writes to live Hermes config/secrets files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.file_operations import ShellFileOperations


class _LocalShellEnv:
    def __init__(self, cwd: Path):
        self.cwd = str(cwd)

    def execute(self, command: str, cwd: str | None = None, timeout: int | None = None,
                stdin_data: str | None = None):
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.cwd,
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "output": completed.stdout + completed.stderr,
            "returncode": completed.returncode,
        }


@pytest.fixture()
def hermes_home(tmp_path: Path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.fixture()
def file_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ops = ShellFileOperations(_LocalShellEnv(tmp_path))
    monkeypatch.setattr(ops, "_snapshot_lsp_baseline", lambda path: None)
    monkeypatch.setattr(ops, "_maybe_lsp_diagnostics", lambda *args, **kwargs: "")
    return ops


def test_config_write_allows_safe_edit_with_backup_and_redacted_diff(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = "model:\n  api_key: sk-live-secret\n  default: old-model\n"
    new = "model:\n  api_key: sk-live-secret\n  default: new-model\n"
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error is None
    assert config_path.read_text() == new
    assert result.backup_path
    assert Path(result.backup_path).read_text() == old
    assert result.redacted_diff
    assert "default: new-model" in result.redacted_diff
    assert "sk-live-secret" not in result.redacted_diff
    assert "[REDACTED]" in result.redacted_diff


def test_config_write_redacts_multiline_private_key_blocks(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = """ssh:
  private_key: |
    -----BEGIN PRIVATE KEY-----
    live-secret-key-material
    -----END PRIVATE KEY-----
  host: old.example.test
"""
    new = old.replace("host: old.example.test", "host: new.example.test")
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error is None
    assert result.redacted_diff
    assert "host: new.example.test" in result.redacted_diff
    assert "BEGIN PRIVATE KEY" not in result.redacted_diff
    assert "live-secret-key-material" not in result.redacted_diff
    assert "END PRIVATE KEY" not in result.redacted_diff


def test_config_write_refuses_deleting_existing_api_key(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = "custom_providers:\n  yuna:\n    api_key: sk-provider-secret\n    base_url: https://example.test/v1\n"
    new = "custom_providers:\n  yuna:\n    base_url: https://example.test/v1\n"
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error
    assert "credential" in result.error.lower()
    assert config_path.read_text() == old
    assert not result.backup_path
    assert result.redacted_diff
    assert "sk-provider-secret" not in result.redacted_diff


def test_env_write_allows_non_secret_edit_with_backup(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-env-secret\nLOG_LEVEL=info\n"
    new = "OPENAI_API_KEY=sk-env-secret\nLOG_LEVEL=debug\n"
    env_path.write_text(old)

    result = file_ops.write_file(str(env_path), new)

    assert result.error is None
    assert env_path.read_text() == new
    assert result.backup_path
    assert Path(result.backup_path).read_text() == old
    assert result.redacted_diff
    assert "LOG_LEVEL=debug" in result.redacted_diff
    assert "sk-env-secret" not in result.redacted_diff


def test_env_write_refuses_replacing_real_secret_with_placeholder(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-env-secret\nLOG_LEVEL=info\n"
    new = "OPENAI_API_KEY=YOUR_API_KEY_HERE\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    result = file_ops.write_file(str(env_path), new)

    assert result.error
    assert "placeholder" in result.error.lower()
    assert env_path.read_text() == old
    assert "sk-env-secret" not in result.redacted_diff
    assert "YOUR_API_KEY_HERE" not in result.redacted_diff


def test_patch_replace_on_env_uses_live_config_guard(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-env-secret\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    result = file_ops.patch_replace(str(env_path), "LOG_LEVEL=info", "LOG_LEVEL=debug")

    assert result.success is True
    assert env_path.read_text() == "OPENAI_API_KEY=sk-env-secret\nLOG_LEVEL=debug\n"
    assert result.backup_path
    assert result.redacted_diff
    assert "sk-env-secret" not in result.redacted_diff
