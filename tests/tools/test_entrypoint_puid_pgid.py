"""Contract test: the container entrypoint accepts PUID/PGID as aliases for
HERMES_UID/HERMES_GID.

Regression guard for #15290.  NAS platforms (UGOS, Synology, unRAID) bind-mount
/opt/data from a host directory owned by the user's own UID and expect the
LinuxServer.io PUID/PGID convention.  Without the alias those vars are silently
ignored, the gosu drop lands on UID 10000, and the runtime cannot read the
volume.  HERMES_UID/HERMES_GID must still take precedence when both are set.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"


@pytest.fixture(scope="module")
def entrypoint_text() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("docker/entrypoint.sh not present in this checkout")
    return ENTRYPOINT.read_text()


def _alias_lines(text: str) -> list[str]:
    """The entrypoint lines that resolve HERMES_UID/HERMES_GID from their aliases."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(("HERMES_UID=", "HERMES_GID="))
    ]


def test_entrypoint_resolves_puid_pgid_aliases(entrypoint_text: str) -> None:
    alias_lines = _alias_lines(entrypoint_text)
    assert any("PUID" in line for line in alias_lines), (
        "docker/entrypoint.sh must resolve HERMES_UID from a PUID alias; see #15290"
    )
    assert any("PGID" in line for line in alias_lines), (
        "docker/entrypoint.sh must resolve HERMES_GID from a PGID alias; see #15290"
    )


def _resolve(entrypoint_text: str, env: dict[str, str]) -> str:
    """Run the entrypoint's alias-resolution lines in isolation and report the
    resolved ``HERMES_UID:HERMES_GID`` pair."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = "\n".join(_alias_lines(entrypoint_text))
    script += '\necho "${HERMES_UID:-}:${HERMES_GID:-}"\n'
    proc = subprocess.run(
        [bash, "-ec", script],
        env={"PATH": os.environ.get("PATH", "")} | env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_puid_pgid_populate_hermes_uid_gid(entrypoint_text: str) -> None:
    assert _resolve(entrypoint_text, {"PUID": "1000", "PGID": "10"}) == "1000:10"


def test_hermes_uid_gid_take_precedence_over_aliases(entrypoint_text: str) -> None:
    resolved = _resolve(
        entrypoint_text,
        {"HERMES_UID": "2000", "HERMES_GID": "2001", "PUID": "1000", "PGID": "10"},
    )
    assert resolved == "2000:2001"


def test_no_uid_vars_leaves_values_empty(entrypoint_text: str) -> None:
    # An empty resolution means the entrypoint keeps the default hermes user.
    assert _resolve(entrypoint_text, {}) == ":"
