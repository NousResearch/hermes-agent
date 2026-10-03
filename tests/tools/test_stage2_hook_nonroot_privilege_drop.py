"""The stage2 hook must not call ``s6-setuidgid`` directly when started non-root.

``docker run --user 10000:10000`` (or a derived image with ``USER 10000``) is a
supported start (see tests/docker/test_user_flag_guard.py). In that case PID 1
has no capabilities, and ``s6-setuidgid hermes`` fails with
``s6-applyuidgid: fatal: unable to set supplementary group list: Operation not
permitted``. ``as_hermes`` exists for exactly this reason: it drops privileges
only when running as root and runs the command directly otherwise.

Regression for #118265: the boot-time config migration and the Nous session
reseed called ``s6-setuidgid hermes`` directly, so on a non-root start both
subprocesses failed, the hook's ``|| echo ... failed; continuing`` degraded the
failure to a warning, and the boot proceeded without the config migration —
leaving ``config.yaml`` on the old schema after every image update.

The tests run the real hook blocks with a stubbed ``id`` / ``s6-setuidgid`` /
venv python on PATH, for both a root and a non-root start.
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

HERMES_UID = 10000


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    # Normalize CRLF from a Windows checkout; sh cannot parse \r.
    return STAGE2_HOOK.read_text().replace("\r\n", "\n")


def _as_hermes_definition(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("as_hermes() {"))


def _block_around(text: str, marker: str) -> str:
    """The top-level ``if ... fi`` block of the hook that contains *marker*."""
    idx = text.index(marker)
    start = text.rindex("\nif ", 0, idx) + 1
    end = text.index("\nfi\n", idx) + len("\nfi\n")
    return text[start:end]


def _write_exe(path: Path, body: str) -> None:
    # newline="\n": sh (unlike cmd) cannot parse CRLF scripts, and a Windows
    # checkout must not leak \r into the stubs or the extracted hook block.
    path.write_text("#!/bin/sh\n" + body, newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_block(
    tmp_path: Path,
    stage2_text: str,
    marker: str,
    uid: int,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run one real hook block under ``set -eu`` with the start uid simulated.

    ``id`` is shadowed with a shell function (function definitions beat PATH
    lookup on every platform — an MSYS2/Git-Bash ``sh`` re-injects ``/usr/bin``
    ahead of a Windows caller's PATH, so a stubbed ``id`` binary loses there).
    The ``s6-setuidgid`` stub on PATH mirrors the real binary on a non-root
    start: setting the supplementary group list needs capabilities the process
    does not have, so it dies with the exact message from the report, exit 111.
    """
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    _write_exe(bin_dir / "s6-setuidgid", (
        'if [ "$FAKE_UID" != 0 ]; then\n'
        '  echo "s6-applyuidgid: fatal: unable to set supplementary group list: '
        'Operation not permitted" >&2\n'
        "  exit 111\n"
        "fi\n"
        f'echo "s6-setuidgid $1" >> "{calls}"\n'
        "shift\n"
        'exec "$@"\n'
    ))
    install_dir = tmp_path / "install"
    (install_dir / ".venv" / "bin").mkdir(parents=True)
    # The hook invokes python by absolute path; PATH never plays a role here.
    _write_exe(install_dir / ".venv" / "bin" / "python", f'echo "python $*" >> "{calls}"\n')

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("_config_version: 1\n")
    (home / "auth.json").write_text("{}\n")

    script = (
        "set -eu\n"
        f'HERMES_HOME="{home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "refuse_symlinked_path() { return 1; }\n"
        'id() { printf \'%s\\n\' "$FAKE_UID"; }\n'
        f"{_as_hermes_definition(stage2_text)}\n"
        f"{_block_around(stage2_text, marker)}"
    )
    env = {**os.environ,
           "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_UID": str(uid)}
    env.pop("HERMES_SKIP_CONFIG_MIGRATION", None)
    env.update(extra_env or {})
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True,
                          env=env, timeout=30)
    return proc, calls.read_text() if calls.exists() else ""


MIGRATE = "scripts/docker_config_migrate.py"
REBOOTSTRAP = "scripts/docker_rebootstrap_nous_session.py"
BLOCKS = [
    (MIGRATE, None),
    (REBOOTSTRAP, {"HERMES_AUTH_JSON_REBOOTSTRAP": "1"}),
]


@pytest.mark.parametrize("marker, extra_env", BLOCKS)
def test_runs_directly_when_started_non_root(tmp_path, stage2_text, marker, extra_env):
    """A non-root start must run the payload directly — no doomed setuid drop,
    and no swallowed warning on the boot output."""
    proc, calls = _run_block(tmp_path, stage2_text, marker, uid=HERMES_UID, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stdout, proc.stdout + proc.stderr
    assert marker in calls
    assert "s6-setuidgid" not in calls


@pytest.mark.parametrize("marker, extra_env", BLOCKS)
def test_drops_to_hermes_when_started_as_root(tmp_path, stage2_text, marker, extra_env):
    """A root start must still drop to the hermes user before the payload.

    Fully stubbed (fake `id`, fake `s6-setuidgid`), so no host privilege is
    involved and the case is host-independent.
    """
    proc, calls = _run_block(tmp_path, stage2_text, marker, uid=0, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stdout, proc.stdout + proc.stderr
    assert calls.splitlines()[0] == "s6-setuidgid hermes"
    assert marker in calls


def test_no_direct_s6_setuidgid_hermes_outside_as_hermes(stage2_text):
    """The hook has one privilege-drop helper; every payload call site must go
    through it so a non-root start runs the payload directly instead of dying
    on the (capability-less) setuid drop and getting swallowed as a warning."""
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(stage2_text.splitlines(), 1)
        if "s6-setuidgid hermes" in line
        and not line.lstrip().startswith("#")
        and not line.startswith("as_hermes() {")
    ]
    assert offenders == [], (
        "call as_hermes instead of s6-setuidgid directly, so a non-root start "
        "still works:\n" + "\n".join(offenders)
    )
