"""Regression for #126950: the Docker auth.json seed is private from its first instant.

The stage2 hook wrote ``HERMES_AUTH_JSON_BOOTSTRAP`` (refresh tokens) with a root redirect under
the container umask (022 -> 0644) and only then ran ``chmod 600``; under ``set -eu`` a failing
chmod left the credential world-readable for good.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

STAGE2_HOOK = Path(__file__).resolve().parents[2] / "docker" / "stage2-hook.sh"
SEED_START = 'if [ ! -f "$HERMES_HOME/auth.json" ] && [ -n "${HERMES_AUTH_JSON_BOOTSTRAP:-}" ]; then'


def _auth_seed_block(text: str) -> str:
    start = text.index(SEED_START)
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


@pytest.mark.platforms("linux", "macos")
def test_auth_json_seed_is_never_group_or_world_readable(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash, "bash is required for the stage2 hook harness"
    home = tmp_path / "data"
    home.mkdir()
    script = (
        "set -eu\n"
        "umask 022\n"
        f'HERMES_HOME="{home}"\n'
        "HERMES_AUTH_JSON_BOOTSTRAP='{\"providers\":{\"nous\":{\"refresh_token\":\"rt-canary\"}}}'\n"
        "refuse_symlinked_path() { return 1; }\n"
        "chown() { :; }\n"
        # The durable failure case: chmod fails (e.g. EPERM on a host bind mount).
        f'chmod() {{ ls -l "$2" | cut -c1-10 > "{tmp_path}/mode_at_chmod"; return 1; }}\n'
        f"{_auth_seed_block(STAGE2_HOOK.read_text())}"
    )
    harness = tmp_path / "harness.sh"
    harness.write_text(script)
    subprocess.run([bash, str(harness)], capture_output=True, text=True, timeout=30)
    seeded = home / "auth.json"
    assert seeded.read_text().endswith('"rt-canary"}}}')
    assert stat.S_IMODE(os.stat(seeded).st_mode) & 0o077 == 0
    mode_file = tmp_path / "mode_at_chmod"
    if mode_file.exists():
        assert mode_file.read_text().strip() == "-rw-------"
