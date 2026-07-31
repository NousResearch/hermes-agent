from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hermes_cli.config import DEFAULT_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker_config_migrate.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("docker_config_migrate_test_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_migration(hermes_home: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_SKIP_CHMOD": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_docker_config_migrate_backs_up_and_migrates_legacy_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 12,
                "model_catalog": {"ttl_hours": 24},
                "delegation": {"max_async_children": 8},
            }
        ),
        encoding="utf-8",
    )
    env_path.write_text("OPENROUTER_API_KEY=test\n", encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema 12 ->" in proc.stdout
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    # v24→25 lowers the old default model_catalog TTL; v32→33 folds
    # max_async_children into max_concurrent_children.
    assert raw["model_catalog"]["ttl_hours"] == 1
    assert raw["delegation"] == {"max_concurrent_children": 8}
    assert list(tmp_path.glob("config.yaml.bak-*"))
    assert list(tmp_path.glob(".env.bak-*"))


def test_docker_config_migrate_skips_below_floor_config_untouched(tmp_path: Path) -> None:
    """Configs below the v12 auto-migration support floor are refused with a
    warning: no migration, no backup, no rewrite — and the boot continues."""
    config_path = tmp_path / "config.yaml"
    original = (
        yaml.safe_dump(
            {
                "_config_version": 11,
                "custom_providers": [
                    {
                        "name": "Local API",
                        "base_url": "http://localhost:8080/v1",
                        "api_key": "test-key",
                    }
                ],
            }
        )
    )
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema" not in proc.stdout
    assert "can no longer be auto-migrated" in proc.stderr
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_skips_unversioned_config_untouched(tmp_path: Path) -> None:
    """Unversioned configs coerce to version 0 — below the floor, so refused."""
    config_path = tmp_path / "config.yaml"
    original = yaml.safe_dump({"model": {"default": "m", "provider": "openrouter"}})
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema" not in proc.stdout
    assert "can no longer be auto-migrated" in proc.stderr
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_does_not_rewrite_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    original = "model: [unterminated\n"
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema" not in proc.stdout
    assert "hermes config:" in proc.stderr
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_skip_env_leaves_config_unchanged(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    original = yaml.safe_dump({"_config_version": 11})
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path, HERMES_SKIP_CONFIG_MIGRATION="1")

    assert proc.returncode == 0, proc.stderr
    assert "skipping config migration" in proc.stdout
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_restores_backups_after_failed_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    original_config = yaml.safe_dump({"_config_version": 12, "gateway": {"provider": "telegram"}})
    original_env = "TELEGRAM_BOT_TOKEN=test-token\n"
    config_path.write_text(original_config, encoding="utf-8")
    env_path.write_text(original_env, encoding="utf-8")

    monkeypatch.setattr(module, "check_config_version", lambda: (12, DEFAULT_CONFIG["_config_version"]))
    monkeypatch.setattr(module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(module, "get_env_path", lambda: env_path)

    def _failing_migrate(*, interactive: bool, quiet: bool):
        config_path.write_text("gateway: {}\n", encoding="utf-8")
        env_path.write_text("", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "migrate_config", _failing_migrate)

    with pytest.raises(RuntimeError, match="boom"):
        module.main()

    assert config_path.read_text(encoding="utf-8") == original_config
    assert env_path.read_text(encoding="utf-8") == original_env
    assert list(tmp_path.glob("config.yaml.bak-*"))
    assert list(tmp_path.glob(".env.bak-*"))


def test_docker_config_migrate_restores_backups_when_version_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    original_config = yaml.safe_dump({"_config_version": 12, "gateway": {"provider": "telegram"}})
    original_env = "TELEGRAM_BOT_TOKEN=test-token\n"
    config_path.write_text(original_config, encoding="utf-8")
    env_path.write_text(original_env, encoding="utf-8")

    calls = iter([(12, DEFAULT_CONFIG["_config_version"]), (12, DEFAULT_CONFIG["_config_version"])])
    monkeypatch.setattr(module, "check_config_version", lambda: next(calls))
    monkeypatch.setattr(module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(module, "get_env_path", lambda: env_path)

    def _non_advancing_migrate(*, interactive: bool, quiet: bool):
        config_path.write_text("gateway: {}\n", encoding="utf-8")
        env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "migrate_config", _non_advancing_migrate)

    with pytest.raises(RuntimeError, match="did not advance config version"):
        module.main()

    assert config_path.read_text(encoding="utf-8") == original_config
    assert env_path.read_text(encoding="utf-8") == original_env


def test_docker_config_migrate_second_boot_preserves_env_byte_for_byte(tmp_path: Path) -> None:
    """Regression for #51579: booting ``gateway run`` twice (i.e. a host
    reboot under ``--restart unless-stopped``) must not strip or rewrite
    ``$HERMES_HOME/.env``. The first boot migrates the stale config and bumps
    ``_config_version``; the second boot must be a no-op that leaves ``.env``
    byte-identical to what the user supplied.

    This exercises the real script + real ``migrate_config`` + real file I/O
    via subprocess — not mocks — so it covers the actual Docker boot path,
    not just the failure-rollback shapes above.
    """
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 12,
                "gateway": {"provider": "telegram"},
            }
        ),
        encoding="utf-8",
    )
    original_env = (
        "TELEGRAM_BOT_TOKEN=secret-bot-token\n"
        "TELEGRAM_ALLOWED_USERS=123456789\n"
        "OPENROUTER_API_KEY=sk-test-provider-key\n"
    )
    env_path.write_text(original_env, encoding="utf-8")
    env_bytes_before = env_path.read_bytes()

    # ── First boot: stale config migrates, version advances. ──
    first = _run_migration(tmp_path)
    assert first.returncode == 0, first.stderr
    assert "Migrating config schema 12 ->" in first.stdout
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    # The token (and every other credential) must survive the migration.
    assert env_path.exists(), ".env must never be deleted by the boot migration"
    assert env_path.read_bytes() == env_bytes_before

    config_after_first = config_path.read_bytes()
    first_boot_backups = sorted(tmp_path.glob("config.yaml.bak-*"))

    # ── Second boot (host reboot): version is current, must be a no-op. ──
    second = _run_migration(tmp_path)
    assert second.returncode == 0, second.stderr
    assert "Migrating config schema" not in second.stdout
    # .env is still present and byte-for-byte identical to the original.
    assert env_path.exists()
    assert env_path.read_bytes() == env_bytes_before
    # config.yaml is untouched by the second boot, and no new backup is made.
    assert config_path.read_bytes() == config_after_first
    assert sorted(tmp_path.glob("config.yaml.bak-*")) == first_boot_backups
