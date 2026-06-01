"""Contract test: the s6-overlay stage2 hook accepts PUID/PGID as aliases for
HERMES_UID/HERMES_GID.

Regression guard for #15290.  NAS platforms (UGOS, Synology, unRAID) bind-mount
/opt/data from a host directory owned by the user's own UID and expect the
LinuxServer.io PUID/PGID convention.  Without the alias those vars are silently
ignored, the s6-setuidgid drop lands on UID 10000, and the runtime cannot read
the volume.  HERMES_UID/HERMES_GID must still take precedence when both are
set.

The s6-overlay rework moved bootstrap from docker/entrypoint.sh (now a shim)
to docker/stage2-hook.sh, which is installed as /etc/cont-init.d/01-hermes-setup
by the Dockerfile.  This test targets the post-rework location.
"""
from __future__ import annotations

import os
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


def _alias_lines(text: str) -> list[str]:
    """The stage2 hook lines that resolve HERMES_UID/HERMES_GID from aliases."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(("HERMES_UID=", "HERMES_GID="))
    ]


def test_stage2_hook_resolves_puid_pgid_aliases(stage2_text: str) -> None:
    alias_lines = _alias_lines(stage2_text)
    assert any("PUID" in line for line in alias_lines), (
        "docker/stage2-hook.sh must resolve HERMES_UID from a PUID alias; see #15290"
    )
    assert any("PGID" in line for line in alias_lines), (
        "docker/stage2-hook.sh must resolve HERMES_GID from a PGID alias; see #15290"
    )


def _resolve(stage2_text: str, env: dict[str, str]) -> str:
    """Run the stage2 hook's alias-resolution lines in isolation and report the
    resolved ``HERMES_UID:HERMES_GID`` pair."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = "\n".join(_alias_lines(stage2_text))
    script += '\necho "${HERMES_UID:-}:${HERMES_GID:-}"\n'
    proc = subprocess.run(
        [bash, "-ec", script],
        env={"PATH": os.environ.get("PATH", "")} | env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_puid_pgid_populate_hermes_uid_gid(stage2_text: str) -> None:
    assert _resolve(stage2_text, {"PUID": "1000", "PGID": "10"}) == "1000:10"


def test_hermes_uid_gid_take_precedence_over_aliases(stage2_text: str) -> None:
    resolved = _resolve(
        stage2_text,
        {"HERMES_UID": "2000", "HERMES_GID": "2001", "PUID": "1000", "PGID": "10"},
    )
    assert resolved == "2000:2001"


def test_no_uid_vars_leaves_values_empty(stage2_text: str) -> None:
    # An empty resolution means the stage2 hook keeps the default hermes user.
    assert _resolve(stage2_text, {}) == ":"


def test_stage2_hook_creates_s6_envdir_before_writing_browser_path(stage2_text: str) -> None:
    """Regression guard for browser-path export on runtimes where the
    s6 container_environment directory is absent when the cont-init hook runs.
    """
    mkdir_line = "mkdir -p /run/s6/container_environment"
    write_line = (
        "printf '%s' \"$browser_bin\" > "
        "/run/s6/container_environment/AGENT_BROWSER_EXECUTABLE_PATH"
    )

    assert mkdir_line in stage2_text
    assert write_line in stage2_text
    assert stage2_text.index(mkdir_line) < stage2_text.index(write_line)
