"""stage2 skips ``docker_config_migrate.py`` in remote config mode (config-config P4, D12): the
agent migrates the fetched document in memory, so a leftover local config.yaml must not be
migrated. stage2 decides with the agent's own backend selection, so every dotenv form the runtime
accepts (comments, spaces around ``=``, quotes, ``export``) and the runtime's precedence (the home's
``.env`` over the container env) decide the same way at both places."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO / "docker" / "stage2-hook.sh"


def _block() -> str:
    text = STAGE2_HOOK.read_text()
    start = text.index("# --- Migrate persisted config schema ---")
    end = text.index("\nfi\n", text.index("docker_config_migrate.py failed", start)) + len("\nfi\n")
    return text[start:end]


def _env(root: Path, backend_env: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "HERMES_CONFIG_BACKEND"}
    env.update(PYTHONPATH=str(REPO), HERMES_MANAGED_DIR=str(root / "managed"))
    if backend_env is not None:
        env["HERMES_CONFIG_BACKEND"] = backend_env
    return env


def _run(home: Path, backend_env: str | None) -> str:
    """The checked-in stage2 block, with only the migration replaced by a marker."""
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    root = home.parent
    bin_dir = root / "bin"  # s6-setuidgid is not a valid sh function name: stub it on PATH
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "s6-setuidgid"
    stub.write_text('#!/bin/sh\nshift\ncase "$*" in *docker_config_migrate.py*) echo MIGRATE_RAN ;; *) exec "$@" ;; esac\n')
    stub.chmod(0o755)
    venv_bin = root / "install" / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    python = venv_bin / "python"  # a wrapper, not a symlink: a moved venv symlink loses its site-packages
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    script = (
        "set -eu\n"
        + f'PATH="{bin_dir}:$PATH"\nHERMES_HOME="{home}"\nINSTALL_DIR="{root / "install"}"\n'
        + _block()
    )
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL, env=_env(root, backend_env))
    assert proc.returncode == 0, proc.stderr
    assert "could not determine the config backend" not in proc.stdout, proc.stdout
    return proc.stdout


def _runtime_selects(home: Path, backend_env: str | None) -> str:
    """The backend a real Hermes start selects: the first config read (``hermes_cli.config`` reads
    at import time) or ``load_hermes_dotenv``'s boot, whichever selects first, reports the backend
    and ends the process right there (there is no plane in this test)."""
    code = ("import os, sys\n"
            "from hermes_cli.config_backend import FileBackend\n"
            "from plugins.config_backends.remote.backend import RemoteBackend\n"
            "def report(name):\n"
            "    def stop(self, *args, **kwargs):\n"
            "        print(name, flush=True)\n"
            "        os._exit(0)\n"
            "    return stop\n"
            "RemoteBackend._state = report('remote')\n"
            "FileBackend.boot = report('file')\n"
            "from hermes_cli import env_loader\n"
            "env_loader.load_hermes_dotenv(hermes_home=sys.argv[1], load_external_secrets=False)\n"
            "sys.exit('no backend was selected')\n")
    env = _env(home.parent, backend_env)
    env["HERMES_HOME"] = str(home)
    proc = subprocess.run([sys.executable, "-c", code, str(home)], capture_output=True, text=True,
                          timeout=60, env=env, cwd=str(home.parent))
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n")
    (tmp_path / "managed").mkdir()
    return home


def test_file_mode_migrates(home):
    assert "MIGRATE_RAN" in _run(home, None)


def test_remote_from_container_env_skips(home):
    out = _run(home, "remote")
    assert "MIGRATE_RAN" not in out and "skipping docker_config_migrate.py" in out


@pytest.mark.parametrize("dotenv, container", [
    ("HERMES_CONFIG_BACKEND=remote", None),
    ('HERMES_CONFIG_BACKEND="remote"', None),
    ("export HERMES_CONFIG_BACKEND='remote'", None),
    ("HERMES_CONFIG_BACKEND=remote  # fleet-managed", None),
    ("HERMES_CONFIG_BACKEND = remote", None),
    ("HERMES_CONFIG_BACKEND=file\n# HERMES_CONFIG_BACKEND=remote", None),
    ("HERMES_CONFIG_BACKEND=file", "remote"),
    ("HERMES_CONFIG_BACKEND=remote", "file"),
])
def test_stage2_decides_like_the_runtime(home, dotenv, container):
    (home / ".env").write_text(f"OTHER=1\n{dotenv}\n")
    runtime = _runtime_selects(home, container)
    migrated = "MIGRATE_RAN" in _run(home, container)
    assert migrated == (runtime == "file"), (dotenv, container, runtime)


def test_remote_from_managed_dotenv_skips(home):
    (home.parent / "managed" / ".env").write_text("HERMES_CONFIG_BACKEND=remote\n")
    assert _runtime_selects(home, None) == "remote"
    assert "MIGRATE_RAN" not in _run(home, None)
