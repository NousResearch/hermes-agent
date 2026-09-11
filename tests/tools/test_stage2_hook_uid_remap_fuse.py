"""Integration tests: stage2 UID remap must not walk $HERMES_HOME (#77072).

``usermod -u`` recursively chowns the user's passwd home. When that home is
the data volume (``/opt/data``) and contains FUSE mounts (rclone), the chown
can hang forever and stall s6 boot. These tests execute the real
``docker/stage2-hook.sh`` under a stubbed PATH (recording ``usermod`` /
``chown``) and verify:

1. passwd-home is staged away before the UID change and restored after
2. the hermes UID is remapped
3. targeted ownership repair still runs after remap
4. passwd-home is restored even when ``usermod -u`` fails
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
_ORIGINAL_HOME = "/opt/data"
_PREDICTABLE_STAGING = "/tmp/hermes-uid-remap"
_STAGING_PREFIX = "/tmp/hermes-uid-remap."


def _require_sh() -> str:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    if not STAGE2_HOOK.is_file():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return shell


def _mktemp_dir(shell: str) -> str:
    """Create a directory under /tmp (colon-free — safe for PATH and passwd)."""
    proc = subprocess.run(
        [shell, "-c", "mktemp -d /tmp/hermes-stage2-XXXXXX"],
        capture_output=True,
        text=True,
        check=False,
    )
    path = (proc.stdout or "").strip()
    if proc.returncode != 0 or not path or ":" in path:
        pytest.skip(
            f"mktemp under /tmp failed: rc={proc.returncode} out={path!r} err={proc.stderr!r}"
        )
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
# Record argv and the passwd home *before* mutating, so tests can prove
# whether usermod -u ran while the account still pointed at /opt/data.
printf '%s\n' "$*" >> "$STATE_DIR/usermod.log"
printf 'HOME=%s ARGS=%s\n' "$(cat "$STATE_DIR/hermes_home")" "$*" >> "$STATE_DIR/usermod.home.log"
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
    passwd_home: str = _ORIGINAL_HOME,
) -> subprocess.CompletedProcess[str]:
    subprocess.run(
        [
            shell,
            "-c",
            'printf "%s" "$2" > "$1/hermes_uid" && '
            'printf "%s" "$3" > "$1/hermes_home" && '
            'rm -f "$1/usermod.log" "$1/usermod.home.log" "$1/chown.log" "$1/fail_usermod_u" && '
            'if [ "$4" = "1" ]; then printf 1 > "$1/fail_usermod_u"; fi && '
            'mkdir -p "$5/cron"',
            "_",
            state_dir,
            _INITIAL_UID,
            passwd_home,
            "1" if fail_usermod_u else "0",
            hermes_home,
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


def _home_at_uid_change(home_log: list[str]) -> list[str]:
    """Passwd home recorded at each ``usermod -u`` invocation."""
    homes = []
    for line in home_log:
        if " ARGS=-u " not in line:
            continue
        prefix, _, _rest = line.partition(" ARGS=")
        assert prefix.startswith("HOME="), line
        homes.append(prefix[len("HOME=") :])
    return homes


def _staging_from_usermod(usermod_log: list[str]) -> str:
    """First remap ``usermod -d`` target — the mktemp staging directory."""
    remap = _remap_usermod_calls(usermod_log)
    assert remap, usermod_log
    first = remap[0]
    assert first.startswith("-d ") and first.endswith(" hermes"), first
    return first[len("-d ") : -len(" hermes")]


def _assert_unique_private_staging(staging: str) -> None:
    """Staging must be mktemp-unique, never the predictable path or data home."""
    assert staging.startswith(_STAGING_PREFIX), staging
    assert staging != _PREDICTABLE_STAGING, staging
    assert staging != _ORIGINAL_HOME, staging
    suffix = staging[len(_STAGING_PREFIX) :]
    assert suffix, staging
    assert "/" not in suffix, staging


def _assert_remap_sequence(usermod_log: list[str], staging: str) -> None:
    assert _remap_usermod_calls(usermod_log) == [
        f"-d {staging} hermes",
        f"-u {_TARGET_UID} hermes",
        f"-d {_ORIGINAL_HOME} hermes",
    ], usermod_log


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


def test_stage2_uid_remap_isolates_opt_data_home_and_repairs_ownership(stage2_env) -> None:
    shell, hermes_home, state_dir, bin_dir = stage2_env

    proc = _run_stage2(
        shell,
        hermes_home=hermes_home,
        state_dir=state_dir,
        bin_dir=bin_dir,
        passwd_home=_ORIGINAL_HOME,
    )

    assert proc.returncode == 0, (
        f"stage2 failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    usermod_log = _read_state(shell, f"{state_dir}/usermod.log").splitlines()
    staging = _staging_from_usermod(usermod_log)
    _assert_unique_private_staging(staging)
    _assert_remap_sequence(usermod_log, staging)

    home_log = _read_state(shell, f"{state_dir}/usermod.home.log").splitlines()
    assert _home_at_uid_change(home_log) == [staging], home_log

    # Unique scratch is removed after home restore.
    gone = subprocess.run(
        [shell, "-c", 'if [ -e "$1" ]; then exit 0; else exit 1; fi', "_", staging],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gone.returncode != 0, f"staging directory leaked: {staging}"

    # Passwd home restored to /opt/data; UID remapped.
    assert _read_state(shell, f"{state_dir}/hermes_home") == _ORIGINAL_HOME
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
        passwd_home=_ORIGINAL_HOME,
    )

    # set -e aborts stage2 after remap returns non-zero, but home must already
    # have been restored so a partial boot never leaves passwd home on staging.
    assert proc.returncode != 0

    usermod_log = _read_state(shell, f"{state_dir}/usermod.log").splitlines()
    staging = _staging_from_usermod(usermod_log)
    _assert_unique_private_staging(staging)
    _assert_remap_sequence(usermod_log, staging)
    assert _read_state(shell, f"{state_dir}/hermes_home") == _ORIGINAL_HOME
    # UID change failed — passwd UID must remain the original build UID.
    assert _read_state(shell, f"{state_dir}/hermes_uid") == _INITIAL_UID


def test_stage2_uid_remap_target_uid_is_applied(stage2_env) -> None:
    shell, hermes_home, state_dir, bin_dir = stage2_env

    proc = _run_stage2(
        shell,
        hermes_home=hermes_home,
        state_dir=state_dir,
        bin_dir=bin_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert _read_state(shell, f"{state_dir}/hermes_uid") == _TARGET_UID


def test_stage2_uid_remap_ignores_preexisting_predictable_staging_symlink(stage2_env) -> None:
    """Pre-existing ``/tmp/hermes-uid-remap`` symlink must not become usermod home.

    Worst case: that predictable path already points at the FUSE/data home.
    Unique mktemp staging must be used instead so UID remap never follows it.
    """
    shell, hermes_home, state_dir, bin_dir = stage2_env

    # Plant a sentinel in the data volume, then make the predictable staging
    # path a symlink to it. Git Bash on Windows copies unless native
    # winsymlinks are requested; Linux ln -s is unaffected.
    planted = subprocess.run(
        [
            shell,
            "-c",
            'printf "poison\n" > "$1/.fuse-sentinel" && '
            'if [ -L "$2" ]; then rm -f "$2"; else rm -rf "$2"; fi && '
            'MSYS=winsymlinks:nativestrict ln -sfn "$1" "$2" && test -L "$2"',
            "_",
            hermes_home,
            _PREDICTABLE_STAGING,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert planted.returncode == 0, (
        f"failed to create poison symlink at {_PREDICTABLE_STAGING} -> {hermes_home}: "
        f"rc={planted.returncode} out={planted.stdout!r} err={planted.stderr!r}"
    )
    try:
        proc = _run_stage2(
            shell,
            hermes_home=hermes_home,
            state_dir=state_dir,
            bin_dir=bin_dir,
            passwd_home=_ORIGINAL_HOME,
        )
        assert proc.returncode == 0, (
            f"stage2 failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

        usermod_log = _read_state(shell, f"{state_dir}/usermod.log").splitlines()
        staging = _staging_from_usermod(usermod_log)
        _assert_unique_private_staging(staging)
        _assert_remap_sequence(usermod_log, staging)
        assert staging != hermes_home, staging
        assert all(
            line != f"-d {_PREDICTABLE_STAGING} hermes"
            for line in _remap_usermod_calls(usermod_log)
        ), usermod_log

        home_log = _read_state(shell, f"{state_dir}/usermod.home.log").splitlines()
        assert _home_at_uid_change(home_log) == [staging], home_log
        assert _PREDICTABLE_STAGING not in _home_at_uid_change(home_log)
        assert hermes_home not in _home_at_uid_change(home_log)

        # Symlink was irrelevant: still a symlink, target sentinel untouched,
        # unique staging used then removed.
        link_check = subprocess.run(
            [
                shell,
                "-c",
                'test -L "$1" && test -f "$2/.fuse-sentinel" && '
                'grep -qx poison "$2/.fuse-sentinel" && '
                'if [ -e "$3" ]; then exit 2; fi',
                "_",
                _PREDICTABLE_STAGING,
                hermes_home,
                staging,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert link_check.returncode == 0, (
            f"symlink/sentinel/cleanup check failed rc={link_check.returncode} "
            f"err={link_check.stderr!r} staging={staging}"
        )
        assert _read_state(shell, f"{state_dir}/hermes_home") == _ORIGINAL_HOME
        assert _read_state(shell, f"{state_dir}/hermes_uid") == _TARGET_UID
    finally:
        subprocess.run(
            [
                shell,
                "-c",
                'if [ -L "$1" ]; then rm -f "$1"; else rm -rf "$1"; fi',
                "_",
                _PREDICTABLE_STAGING,
            ],
            capture_output=True,
            text=True,
            check=False,
        )


def _docker_available() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    proc = subprocess.run(
        [docker, "info"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return proc.returncode == 0


# Bypass: the volume path contains "hermes-agent" and the inner script
# runs `apt-get update`; the live-system guard treats that as `hermes update`.
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_docker_real_usermod_restores_opt_data_home() -> None:
    """E2E: real usermod/passwd in Debian. Not a FUSE proof.

    Wraps the image ``usermod`` with a logger that records passwd-home at
    each invocation, then runs the actual ``stage2-hook.sh`` as root.
    """
    docker = shutil.which("docker")
    assert docker is not None
    hook_src = STAGE2_HOOK.resolve()
    script = r"""
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq passwd adduser >/dev/null
mkdir -p /opt/data/cron /usr/local/sbin /opt/hermes
useradd -u 10000 -d /opt/data -M -s /bin/sh hermes
# Real usermod lives in /usr/sbin. Interpose a logger on PATH.
cat > /usr/local/sbin/usermod <<'WRAP'
#!/bin/sh
home=$(getent passwd hermes | cut -d: -f6)
printf 'HOME=%s ARGS=%s\n' "$home" "$*" >> /tmp/usermod.home.log
exec /usr/sbin/usermod "$@"
WRAP
chmod 755 /usr/local/sbin/usermod
# s6-setuidgid is not in Debian slim; run the rest as the caller (root).
cat > /usr/local/sbin/s6-setuidgid <<'WRAP'
#!/bin/sh
shift
exec "$@"
WRAP
chmod 755 /usr/local/sbin/s6-setuidgid
export PATH="/usr/local/sbin:/usr/sbin:/usr/bin:/bin"
export HERMES_HOME=/opt/data
export HERMES_UID=1000
cp /hook/stage2-hook.sh /tmp/stage2-hook.sh
sed -i 's/\r$//' /tmp/stage2-hook.sh
chmod 755 /tmp/stage2-hook.sh
set +e
/tmp/stage2-hook.sh
rc=$?
set -e
echo "STAGE2_RC=$rc"
echo "PASSWD=$(getent passwd hermes)"
echo "UID=$(id -u hermes)"
echo "HOME_LOG<<"
cat /tmp/usermod.home.log
echo ">>HOME_LOG"
exit 0
"""
    proc = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "-v",
            f"{hook_src}:/hook/stage2-hook.sh:ro",
            "debian:bookworm-slim",
            "sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert proc.returncode == 0, combined[-4000:]
    assert "PASSWD=hermes:x:1000:" in combined, combined[-4000:]
    assert ":/opt/data:" in combined, combined[-4000:]
    assert "UID=1000" in combined, combined[-4000:]
    # usermod -u must not have run while passwd home was still /opt/data.
    uid_homes = []
    in_log = False
    for line in combined.splitlines():
        if line.strip() == "HOME_LOG<<":
            in_log = True
            continue
        if line.strip() == ">>HOME_LOG":
            in_log = False
            continue
        if in_log and " ARGS=-u " in line:
            home = line.partition(" ARGS=")[0][len("HOME=") :]
            uid_homes.append(home)
    assert uid_homes, combined[-4000:]
    assert "/opt/data" not in uid_homes, uid_homes
    assert all(h != "/opt/data" for h in uid_homes), uid_homes
    assert all(h != "/tmp/hermes-uid-remap" for h in uid_homes), uid_homes
    assert all(h.startswith("/tmp/hermes-uid-remap.") for h in uid_homes), uid_homes


def test_docker_fuse_not_claimed_from_this_harness() -> None:
    """Honesty marker: PATH stubs and bind mounts are not a FUSE proof."""
    pytest.skip("real FUSE validation: NOT EXECUTED on this host")
