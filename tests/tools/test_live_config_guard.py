"""Regression tests for guarded writes to live Hermes config/secrets files."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest
import yaml

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
    old = "OPENAI_API_KEY=***\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    result = file_ops.patch_replace(str(env_path), "LOG_LEVEL=info", "LOG_LEVEL=debug")

    assert result.success is True
    assert env_path.read_text() == "OPENAI_API_KEY=***\nLOG_LEVEL=debug\n"
    assert result.backup_path
    assert result.redacted_diff
    assert "***" not in result.redacted_diff


def test_config_write_redacts_flow_style_secret_values(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = "model: {api_key: FLOWVALUE12345, default: old}\n"
    new = "model: {api_key: FLOWVALUE12345, default: new}\n"
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error is None
    assert result.backup_path
    assert result.redacted_diff
    assert "default: new" in result.redacted_diff
    assert "FLOWVALUE12345" not in result.redacted_diff
    assert "[REDACTED]" in result.redacted_diff


def test_patch_v4a_on_env_uses_live_config_guard(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=LIVEVALUE12345\nLOG_LEVEL=info\n"
    env_path.write_text(old)
    patch = f"""*** Begin Patch
*** Update File: {env_path}
@@ LOG_LEVEL=info @@
 OPENAI_API_KEY=LIVEVALUE12345
-LOG_LEVEL=info
+LOG_LEVEL=debug
*** End Patch
"""

    result = file_ops.patch_v4a(patch)

    assert result.success is True
    assert env_path.read_text() == "OPENAI_API_KEY=LIVEVALUE12345\nLOG_LEVEL=debug\n"
    assert result.backup_path
    assert Path(result.backup_path).read_text() == old
    assert result.redacted_diff
    assert "LIVEVALUE12345" not in result.diff
    assert "LIVEVALUE12345" not in result.redacted_diff


def test_atomic_yaml_write_refuses_live_config_secret_deletion(
    hermes_home: Path,
):
    from utils import atomic_yaml_write

    config_path = hermes_home / "config.yaml"
    old = "custom_providers:\n  yuna:\n    api_key: sk-live-value\n    base_url: https://example.test/v1\n"
    config_path.write_text(old)

    with pytest.raises(ValueError, match="credential"):
        atomic_yaml_write(
            config_path,
            {"custom_providers": {"yuna": {"base_url": "https://example.test/v1"}}},
            sort_keys=False,
        )

    assert config_path.read_text() == old
    assert list(hermes_home.glob("config.yaml.bak.*")) == []


def test_live_config_guard_does_not_treat_token_count_fields_as_secrets(
    hermes_home: Path,
):
    from utils import atomic_yaml_write

    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n  default: old-model\n  max_tokens: 4096\n  tokenizer: cl100k_base\n"
    )

    atomic_yaml_write(
        config_path,
        {"model": {"default": "new-model"}},
        sort_keys=False,
    )

    assert yaml.safe_load(config_path.read_text()) == {"model": {"default": "new-model"}}


def test_live_config_guard_does_not_treat_camelcase_token_count_fields_as_secrets(
    hermes_home: Path,
):
    from utils import atomic_yaml_write

    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  default: old-model\n"
        "  maxTokens: 4096\n"
        "  contextTokens: 8192\n"
        "  promptTokenCount: 128\n"
        "  completionTokens: 256\n"
    )

    atomic_yaml_write(
        config_path,
        {"model": {"default": "new-model"}},
        sort_keys=False,
    )

    assert yaml.safe_load(config_path.read_text()) == {"model": {"default": "new-model"}}


def test_live_config_guard_treats_camelcase_token_fields_as_secrets(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = """channels:
  telegram:
    botToken: REALBOTTOKEN123
  slack:
    appToken: REALAPPTOKEN123
  session:
    refreshToken: REALREFRESHTOKEN123
    sessionToken: REALSESSIONTOKEN123
"""
    new = """channels:
  telegram: {}
  slack: {}
  session: {}
"""
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error
    assert "channels.telegram.botToken" in result.error
    assert "channels.slack.appToken" in result.error
    assert "channels.session.refreshToken" in result.error
    assert "channels.session.sessionToken" in result.error
    assert config_path.read_text() == old


def test_live_config_guard_line_fallback_treats_camelcase_token_fields_as_secrets(
    hermes_home: Path,
    file_ops: ShellFileOperations,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = hermes_home / "config.yaml"
    old = """channels:
  telegram:
    botToken: REALBOTTOKEN123
model:
  maxTokens: 4096
  promptTokenCount: 128
"""
    new = """channels:
  telegram: {}
model: {}
"""
    config_path.write_text(old)
    monkeypatch.setitem(sys.modules, "yaml", None)

    result = file_ops.write_file(str(config_path), new)

    assert result.error
    assert "line:3" in result.error
    assert "line:5" not in result.error
    assert "line:6" not in result.error
    assert config_path.read_text() == old
    assert "REALBOTTOKEN123" not in result.redacted_diff


def test_live_config_guard_refuses_deleting_duplicate_secret_even_if_value_remains(
    hermes_home: Path,
    file_ops: ShellFileOperations,
):
    config_path = hermes_home / "config.yaml"
    old = """providers:
  one:
    api_key: SHAREDREALVALUE123
  two:
    api_key: SHAREDREALVALUE123
"""
    new = """providers:
  two:
    api_key: SHAREDREALVALUE123
"""
    config_path.write_text(old)

    result = file_ops.write_file(str(config_path), new)

    assert result.error
    assert "providers.one.api_key" in result.error
    assert config_path.read_text() == old


def test_atomic_yaml_write_backs_up_live_config_safe_edit(
    hermes_home: Path,
):
    from utils import atomic_yaml_write

    config_path = hermes_home / "config.yaml"
    old_data = {"model": {"api_key": "sk-live-value", "default": "old-model"}}
    config_path.write_text(yaml.safe_dump(old_data, sort_keys=False))

    atomic_yaml_write(
        config_path,
        {"model": {"api_key": "sk-live-value", "default": "new-model"}},
        sort_keys=False,
    )

    assert yaml.safe_load(config_path.read_text())["model"]["default"] == "new-model"
    backups = list(hermes_home.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    assert yaml.safe_load(backups[0].read_text()) == old_data


def test_save_env_value_refuses_overwriting_real_secret_with_placeholder(
    hermes_home: Path,
):
    from hermes_cli.config import save_env_value

    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-live-value\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    with pytest.raises(ValueError, match="placeholder"):
        save_env_value("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

    assert env_path.read_text() == old
    assert list(hermes_home.glob(".env.bak.*")) == []


def test_save_env_value_backs_up_live_env_safe_edit(
    hermes_home: Path,
):
    from hermes_cli.config import save_env_value

    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-live-value\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    save_env_value("LOG_LEVEL", "debug")

    assert env_path.read_text() == "OPENAI_API_KEY=sk-live-value\nLOG_LEVEL=debug\n"
    backups = list(hermes_home.glob(".env.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == old


def test_remove_env_value_refuses_deleting_real_secret(
    hermes_home: Path,
):
    from hermes_cli.config import remove_env_value

    env_path = hermes_home / ".env"
    old = "OPENAI_API_KEY=sk-live-value\nLOG_LEVEL=info\n"
    env_path.write_text(old)

    with pytest.raises(ValueError, match="remove existing credential"):
        remove_env_value("OPENAI_API_KEY")

    assert env_path.read_text() == old
