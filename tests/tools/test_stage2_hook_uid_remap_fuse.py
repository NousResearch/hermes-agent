"""Integration tests: stage2 UID remap must not walk $HERMES_HOME (#77072).

``usermod -u`` recursively chowns the user's passwd home. When that home is
the data volume and contains FUSE mounts (rclone), the chown can hang forever
and stall s6 boot. These tests execute the real ``docker/stage2-hook.sh`` under
a stubbed PATH (recording ``usermod`` / ``chown``) and verify:

1. passwd-home is staged away before the UID change and restored after
2. the hermes UID is remapped
3. targeted ownership repair still runs after remap
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

_INITIAL_UID = "10000"
_TARGET_UID = "1000"
_STAGING_HOME = "/tmp/hermes-uid-remap"


def _require_sh() -> str:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    if not STAGE2_HOOK.is_file():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return shell


def _mktemp_dir(shell: str) -> str:
    """Create a directory under /tmp (colon-free ??? safe for PATH and passwd)."""
    proc = subprocess.run(
        [shell, "-c", "mktemp -d /tmp/hermes-stage2-XXXXXX"],
        capture_output=True,
        text=True,
        check=False,
    )
    path = (proc.stdout or "").strip()
    if proc.returncode != 0 or not path or ":" in path:
        pytest.skip(f"mktemp under /tmp failed: rc={proc.returncode} out={path!r} err={proc.stderr!r}")
    return path


def _rm_rf(shell: str, path: str) -> None:
    subprocess.run(
        [shell, "-c", 'rm -rf "$1"', "_", path],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_stub(bin_dir: str, name: str, body: str, shell: str) -> None:
    proc = subprocess.run(
        [shell, "-c", 'cat > "$1" && chmod 755 "$1"', "_", f"{bin_dir}/{name}"],
        input="#!/bin/sh\n" + body.lstrip("\n") + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"failed to write stub {name}: {proc.stderr}"


def _build_stub_bin(shell: str, bin_dir: str) -> None:
    _write_stub(
        bin_dir,
        "id",
        r"""
STATE_DIR="${HERMES_STAGE2_TEST_STATE:?}"
uid="$(cat "$STATE_DIR/hermes_uid")"
case "$1" in
  -u)
    if [ "${2:-}" = "hermes" ]; then
      printf '%s\n' "$uid"
    else
      printf '0\n'
    fi
    ;;
  -g)
    if [ "${2:-}" = "hermes" ]; then
      printf '10000\n'
    else
      printf '0\n'
    fi
    ;;
  -G)
    printf '10000\n'
    ;;
  *)
    echo "id stub: unexpected args: $*" >&2
    exit 1
    ;;
esac
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "getent",
        r"""
STATE_DIR="${HERMES_STAGE2_TEST_STATE:?}"
case "$1" in
  passwd)
    if [ "$2" = "hermes" ]; then
      uid="$(cat "$STATE_DIR/hermes_uid")"
      home="$(cat "$STATE_DIR/hermes_home")"
      printf 'hermes:x:%s:10000::%s:/bin/sh\n' "$uid" "$home"
      exit 0
    fi
    exit 2
    ;;
  group)
    exit 2
    ;;
  *)
    exit 2
    ;;
esac
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "usermod",
        r"""
STATE_DIR="${HERMES_STAGE2_TEST_STATE:?}"
printf '%s\n' "$*" >> "$STATE_DIR/usermod.log"
case "$1" in
  -d)
    printf '%s' "$2" > "$STATE_DIR/hermes_home"
    ;;
  -u)
    if [ -f "$STATE_DIR/fail_usermod_u" ]; then
      exit 1
    fi
    printf '%s' "$2" > "$STATE_DIR/hermes_uid"
    ;;
  -aG)
    ;;
  *)
    echo "usermod stub: unexpected args: $*" >&2
    exit 1
    ;;
esac
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "chown",
        r"""
STATE_DIR="${HERMES_STAGE2_TEST_STATE:?}"
printf '%s\n' "$*" >> "$STATE_DIR/chown.log"
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "stat",
        r"""
# stage2 probes: stat -c %u "$HERMES_HOME" and (optionally) stat -c %g socket
case "$1" in
  -c)
    case "$2" in
      %u|%g)
        # Report root-owned so the targeted ownership-repair block runs.
        printf '0\n'
        ;;
      *)
        echo "stat stub: unexpected format: $2" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "stat stub: unexpected args: $*" >&2
    exit 1
    ;;
esac
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "find",
        r"""
# tree_has_non_hermes_owner: find TARGET \( ! -user hermes ... \) -print -quit
# Claim every probed tree has a non-hermes owner so targeted chown runs.
target="$1"
for arg in "$@"; do
  if [ "$arg" = "hermes" ]; then
    printf '%s\n' "$target"
    exit 0
  fi
done
exit 0
""",
        shell,
    )
    _write_stub(
        bin_dir,
        "s6-setuidgid",
        r"""
# Drop-in: ignore the username and run the remaining command as-is.
shift
exec "$@"
""",
        shell,
    )
    _write_stub(bin_dir, "groupmod", "exit 0\n", shell)
    _write_stub(bin_dir, "groupadd", "exit 0\n", shell)


def _run_stage2(
    shell: str,
    *,
    hermes_home: str,
    state_dir: str,
    bin_dir: str,
    fail_usermod_u: bool = False,
) -> subprocess.CompletedProcess[str]:
    subprocess.run(
        [
            shell,
            "-c",
            'printf "%s" "$2" > "$1/hermes_uid" && '
            'printf "%s" "$3" > "$1/hermes_home" && '
            'rm -f "$1/usermod.log" "$1/chown.log" "$1/fail_usermod_u" && '
            'if [ "$4" = "1" ]; then printf 1 > "$1/fail_usermod_u"; fi && '
            'mkdir -p "$3/cron"',
            "_",
            state_dir,
            _INITIAL_UID,
            hermes_home,
            "1" if fail_usermod_u else "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Resolve the hook to a path the shell can exec. Prefer a /tmp copy on
    # Windows so we never put a drive-letter path in argv quirks; on Linux
    # the repo path is already colon-free.
    hook_path = STAGE2_HOOK.as_posix()
    if ":" in hook_path:
        hook_copy = f"{state_dir}/stage2-hook.sh"
        subprocess.run(
            [shell, "-c", 'cp "$1" "$2" && chmod 755 "$2"', "_", hook_path, hook_copy],
            check=True,
            capture_output=True,
            text=True,
        )
        hook_path = hook_copy

    env = os.environ.copy()
    # Keep optional remap/migrate/bootstrap paths quiet.
    for key in (
        "HERMES_GID",
        "PUID",
        "PGID",
        "HERMES_AUTH_JSON_BOOTSTRAP",
        "HERMES_AUTH_JSON_REBOOTSTRAP",
        "HERMES_GATEWAY_BOOTSTRAP_STATE",
        "PLAYWRIGHT_BROWSERS_PATH",
        "AGENT_BROWSER_EXECUTABLE_PATH",
        # Do not inject a mixed Windows/Unix PATH from Python ??? MSYS path
        # conversion breaks stub lookup. Prepend bin_dir inside the shell.
        "HERMES_HOME",
        "HERMES_UID",
        "HERMES_STAGE2_TEST_STATE",
    ):
        env.pop(key, None)

    # Prepend stub bin inside sh so PATH stays in the shell's native form.
    return subprocess.run(
        [
            shell,
            "-c",
            'PATH="$1:$PATH" '
            'HERMES_HOME="$2" '
            'HERMES_UID="$3" '
            'HERMES_STAGE2_TEST_STATE="$4" '
            "export PATH HERMES_HOME HERMES_UID HERMES_STAGE2_TEST_STATE; "
            'exec "$5"',
            "_",
            bin_dir,
            hermes_home,
            _TARGET_UID,
            state_dir,
            hook_path,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def _read_state(shell: str, path: str) -> str:
    proc = subprocess.run(
        [shell, "-c", 'cat "$1"', "_", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _remap_usermod_calls(usermod_log: list[str]) -> list[str]:
    """Return UID-remap usermod lines, ignoring optional docker-socket -aG adds.

    Hosts with ``/var/run/docker.sock`` (or ``/run/docker.sock``) still enter
    stage2's DooD group block after remap; that appends ``usermod -aG …`` and
    must not break the remap sequence assertion.
    """
    return [line for line in usermod_log if not line.startswith("-aG ")]


@pytest.fixture
def stage2_env():
    shell = _require_sh()
    hermes_home = _mktemp_dir(shell)
    bin_dir = _mktemp_dir(shell)
    state_dir = _mktemp_dir(shell)
    _build_stub_bin(shell, bin_dir)
    try:
        yield shell, hermes_home, state_dir, bin_dir
    finally:
        _rm_rf(shell, hermes_home)
        _rm_rf(shell, bin_dir)
        _rm_rf(shell, state_dir)


def test_stage2_uid_remap_isolates_home_and_repairs_ownership(stage2_env) -> None:
    shell, hermes_home, state_dir, bin_dir = stage2_env

    proc = _run_stage2(
        shell,
        hermes_home=hermes_home,
        state_dir=state_dir,
        bin_dir=bin_dir,
    )

    assert proc.returncode == 0, (
        f"stage2 failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    usermod_log = _read_state(shell, f"{state_dir}/usermod.log").splitlines()
    assert _remap_usermod_calls(usermod_log) == [
        f"-d {_STAGING_HOME} hermes",
        f"-u {_TARGET_UID} hermes",
        f"-d {hermes_home} hermes",
    ], usermod_log

    # Passwd home restored; UID remapped.
    assert _read_state(shell, f"{state_dir}/hermes_home") == hermes_home
    assert _read_state(shell, f"{state_dir}/hermes_uid") == _TARGET_UID

    chown_log = _read_state(shell, f"{state_dir}/chown.log").splitlines()
    assert f"hermes:hermes {hermes_home}" in chown_log
    assert f"-R hermes:hermes {hermes_home}/cron" in chown_log

    assert f"Changing hermes UID to {_TARGET_UID}" in proc.stdout
    assert (
        f"Fixing ownership of {hermes_home} (targeted) to hermes ({_TARGET_UID})"
        in proc.stdout
    )
    assert "Setup complete" in proc.stdout


def test_stage2_uid_remap_restores_home_when_usermod_u_fails(stage2_env) -> None:
    shell, hermes_home, state_dir, bin_dir = stage2_env

    proc = _run_stage2(
        shell,
        hermes_home=hermes_home,
        state_dir=state_dir,
        bin_dir=bin_dir,
        fail_usermod_u=True,
    )

    # set -e aborts stage2 after remap returns non-zero, but home must already
    # have been restored so a partial boot never leaves passwd home on staging.
    assert proc.returncode != 0

    usermod_log = _read_state(shell, f"{state_dir}/usermod.log").splitlines()
    assert _remap_usermod_calls(usermod_log) == [
        f"-d {_STAGING_HOME} hermes",
        f"-u {_TARGET_UID} hermes",
        f"-d {hermes_home} hermes",
    ], usermod_log
    assert _read_state(shell, f"{state_dir}/hermes_home") == hermes_home
    # UID change failed — passwd UID must remain the original build UID.
    assert _read_state(shell, f"{state_dir}/hermes_uid") == _INITIAL_UID
