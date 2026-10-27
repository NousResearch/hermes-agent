"""The stage2 hook must not call ``s6-setuidgid`` directly when started non-root.

``docker run --user 10000:10000`` (or a derived image with ``USER 10000``) is a
supported start (see tests/docker/test_user_flag_guard.py). In that case PID 1
has no capabilities, and ``s6-setuidgid hermes`` fails with
``s6-applyuidgid: fatal: unable to set supplementary group list: Operation not
permitted``. ``as_hermes`` exists for exactly this reason: it drops privileges
only when running as root and runs the command directly otherwise.

Three call sites used ``s6-setuidgid hermes`` directly (config migration, Nous
session reseed, dependency-generation refresh), so on a non-root start each was
silently skipped (the hook prints a warning and continues): configs stayed on
the old schema and opt-in dependency generations were never re-resolved after
an image update.

The tests run the real hook blocks with stubbed ``id`` / ``s6-setuidgid`` /
python on PATH, for both a root and a non-root start.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text(encoding="utf-8")


def _as_hermes_definition(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("as_hermes() {"))


def _block_around(text: str, marker: str) -> str:
    """The top-level ``if ... fi`` block of the hook that contains *marker*."""
    idx = text.index(marker)
    start = text.rindex("\nif ", 0, idx) + 1
    end = text.index("\nfi\n", idx) + len("\nfi\n")
    return text[start:end]


def _write_exe(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_block(tmp_path: Path, stage2_text: str, marker: str, uid: int,
               extra_env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    # `id -u` reports the simulated start uid.
    _write_exe(bin_dir / "id", 'echo "$FAKE_UID"\n')
    # s6-setuidgid behaves like the real one: only root may set the group list.
    _write_exe(bin_dir / "s6-setuidgid", (
        'if [ "$FAKE_UID" != 0 ]; then\n'
        '  echo "s6-applyuidgid: fatal: unable to set supplementary group list: '
        'Operation not permitted" >&2\n'
        '  exit 111\n'
        'fi\n'
        f'echo "s6-setuidgid $1" >> "{calls}"\n'
        'shift\n'
        'exec "$@"\n'
    ))
    install_dir = tmp_path / "install"
    (install_dir / ".venv" / "bin").mkdir(parents=True)
    _write_exe(install_dir / ".venv" / "bin" / "python", f'echo "python $*" >> "{calls}"\n')

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("_config_version: 1\n", encoding="utf-8")
    (home / "auth.json").write_text("{}\n", encoding="utf-8")

    script = (
        "set -eu\n"
        f'HERMES_HOME="{home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "refuse_symlinked_path() { return 1; }\n"
        f"{_as_hermes_definition(stage2_text)}\n"
        f"{_block_around(stage2_text, marker)}"
    )
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_UID": str(uid)}
    env.pop("HERMES_SKIP_CONFIG_MIGRATION", None)
    env.update(extra_env or {})
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env, timeout=30)
    return proc, calls.read_text(encoding="utf-8") if calls.exists() else ""


MIGRATE = "scripts/docker_config_migrate.py"
REBOOTSTRAP = "scripts/docker_rebootstrap_nous_session.py"


@pytest.mark.parametrize("uid", [10000, 0], ids=["non-root-start", "root-start"])
@pytest.mark.parametrize("marker, extra_env", [
    (MIGRATE, None),
    (REBOOTSTRAP, {"HERMES_AUTH_JSON_REBOOTSTRAP": "1"}),
])
def test_block_runs_as_hermes_for_either_start_uid(tmp_path, stage2_text, marker, extra_env, uid):
    """Non-root start: run directly (no setgroups). Root start: still drop via s6-setuidgid."""
    proc, calls = _run_block(tmp_path, stage2_text, marker, uid=uid, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stdout, proc.stdout + proc.stderr
    assert marker in calls
    if uid == 0:
        assert calls.splitlines()[0] == "s6-setuidgid hermes"
    else:
        assert "s6-setuidgid" not in calls


def test_no_direct_s6_setuidgid_hermes_outside_as_hermes(stage2_text):
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(stage2_text.splitlines(), 1)
        if "s6-setuidgid hermes" in line
        and not line.lstrip().startswith("#")
        and not line.startswith("as_hermes() {")
    ]
    assert offenders == [], "use as_hermes so a non-root start still works:\n" + "\n".join(offenders)
