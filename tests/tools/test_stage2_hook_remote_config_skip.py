"""stage2 skips ``docker_config_migrate.py`` in remote config mode (config-config P4, D12): the
agent migrates the fetched document in memory, so a leftover local config.yaml must not be
migrated. The mode comes from the container env or from ``$HERMES_HOME/.env``."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STAGE2_HOOK = Path(__file__).resolve().parents[2] / "docker" / "stage2-hook.sh"


def _block() -> str:
    text = STAGE2_HOOK.read_text()
    start = text.index("_remote_config_mode=0")
    end = text.index("\nfi\n", text.index("docker_config_migrate.py failed", start)) + len("\nfi\n")
    return text[start:end]


def _run(home: Path, backend_env: str | None) -> str:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    env_line = "unset HERMES_CONFIG_BACKEND\n" if backend_env is None else f"HERMES_CONFIG_BACKEND='{backend_env}'\n"
    bin_dir = home.parent / "bin"  # s6-setuidgid is not a valid sh function name: stub it on PATH
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "s6-setuidgid"
    stub.write_text("#!/bin/sh\necho MIGRATE_RAN\n")
    stub.chmod(0o755)
    script = (
        "set -eu\n" + env_line
        + f'PATH="{bin_dir}:$PATH"\nHERMES_HOME="{home}"\nINSTALL_DIR=/nonexistent\n'
        + _block()
    )
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30,
                          stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n")
    return home


def test_file_mode_migrates(home):
    assert "MIGRATE_RAN" in _run(home, None)


def test_remote_from_container_env_skips(home):
    out = _run(home, "remote")
    assert "MIGRATE_RAN" not in out and "skipping docker_config_migrate.py" in out


@pytest.mark.parametrize("line", ["HERMES_CONFIG_BACKEND=remote", 'HERMES_CONFIG_BACKEND="remote"',
                                  "export HERMES_CONFIG_BACKEND='remote'"])
def test_remote_from_dotenv_skips(home, line):
    (home / ".env").write_text(f"OTHER=1\n{line}\n")
    assert "MIGRATE_RAN" not in _run(home, None)


def test_other_dotenv_values_still_migrate(home):
    (home / ".env").write_text("HERMES_CONFIG_BACKEND=file\n# HERMES_CONFIG_BACKEND=remote\n")
    assert "MIGRATE_RAN" in _run(home, None)
