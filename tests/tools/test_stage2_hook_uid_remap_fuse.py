"""Regression tests: UID remap must not walk $HERMES_HOME (#77072).

``usermod -u`` recursively chowns the user's home directory. When that home
is the data volume and contains FUSE mounts (rclone), the chown can hang
forever and stall s6 boot. stage2 must isolate home to a staging dir for
the UID change, then restore the original home — ownership repair stays
with the targeted chown block.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _remap_hermes_uid_function(text: str) -> str:
    start = text.index("remap_hermes_uid() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def test_stage2_defines_uid_remap_helper(stage2_text: str) -> None:
    assert "remap_hermes_uid() {" in stage2_text
    assert 'remap_hermes_uid "$HERMES_UID"' in stage2_text
    # Must not call bare usermod -u against the live hermes home (FUSE hang).
    assert "\n    usermod -u \"$HERMES_UID\" hermes\n" not in stage2_text
    assert "/tmp/hermes-uid-remap" in stage2_text


def test_remap_hermes_uid_isolates_home_during_uid_change(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")

    log_path = tmp_path / "usermod.log"
    # Use a colon-free POSIX home — getent/passwd fields are colon-delimited,
    # matching the real container path (/opt/data).
    original_home = "/opt/data"
    staging = "/tmp/hermes-uid-remap"

    script = (
        "set -eu\n"
        f'HERMES_HOME="{original_home}"\n'
        f'getent() {{ printf "hermes:x:10000:10000::{original_home}:/bin/sh\\n"; }}\n'
        "mkdir() { :; }\n"
        f'usermod() {{ printf "%s\\n" "$*" >> "{log_path.as_posix()}"; }}\n'
        f"{_remap_hermes_uid_function(stage2_text)}\n"
        'remap_hermes_uid "1000"\n'
    )
    proc = subprocess.run([shell, "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().splitlines() == [
        f"-d {staging} hermes",
        "-u 1000 hermes",
        f"-d {original_home} hermes",
    ]


def test_remap_hermes_uid_restores_home_when_uid_change_fails(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")

    log_path = tmp_path / "usermod.log"
    original_home = "/opt/data"

    script = (
        "set -eu\n"
        f'HERMES_HOME="{original_home}"\n'
        f'getent() {{ printf "hermes:x:10000:10000::{original_home}:/bin/sh\\n"; }}\n'
        "mkdir() { :; }\n"
        "usermod() {\n"
        f'  printf "%s\\n" "$*" >> "{log_path.as_posix()}"\n'
        '  case "$*" in\n'
        '    -u\\ *) return 1 ;;\n'
        "  esac\n"
        "}\n"
        f"{_remap_hermes_uid_function(stage2_text)}\n"
        "if remap_hermes_uid 1000; then exit 2; fi\n"
    )
    proc = subprocess.run([shell, "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().splitlines() == [
        "-d /tmp/hermes-uid-remap hermes",
        "-u 1000 hermes",
        f"-d {original_home} hermes",
    ]
