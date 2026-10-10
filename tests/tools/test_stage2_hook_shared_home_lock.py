"""Concurrent stage2 hooks must not split API_SERVER_KEY or drop routing overrides.

Shipped compose files start gateway + dashboard over one HERMES_HOME. Both
run the stage2 hook; the lock around key seeding and rewrite_env_var is what
makes that safe (#117937). mkdir is the exclusive lock so Docker Desktop
bind-mounts cannot false-succeed via flock.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

KEY_LINE_RE = re.compile(r"^API_SERVER_KEY=[0-9a-f]{64}$", re.MULTILINE)
PORTAL = "https://portal.staging-nousresearch.com"
INFERENCE = "https://stg-inference-api.nousresearch.com/v1"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def _mutation_block(text: str) -> str:
    start = text.index("# --- Ensure a gateway api_server key exists")
    end = text.index("# .env holds API keys and secrets", start)
    return text[start:end]


def _mutation_script(stage2_text: str, home: Path, extra_env: str = "") -> str:
    return (
        "set -eu\n"
        "unset API_SERVER_KEY\n"
        "unset HERMES_PORTAL_BASE_URL NOUS_PORTAL_BASE_URL NOUS_INFERENCE_BASE_URL\n"
        f"{extra_env}"
        f'HERMES_HOME="{home}"\n'
        'as_hermes() { "$@"; }\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_mutation_block(stage2_text)}\n"
    )


def _run_mutation(
    stage2_text: str,
    home: Path,
    extra_env: str = "",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    return subprocess.run(
        ["sh", "-c", _mutation_script(stage2_text, home, extra_env)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def _spawn_mutation(
    stage2_text: str,
    home: Path,
    extra_env: str = "",
) -> subprocess.Popen[str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    return subprocess.Popen(
        ["sh", "-c", _mutation_script(stage2_text, home, extra_env)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_concurrent_keygens_write_one_key(stage2_text: str, tmp_path: Path) -> None:
    """The compose-up race: both hooks see no key and both used to append."""
    for i in range(8):
        home = tmp_path / f"home-{i}"
        home.mkdir()
        first = _spawn_mutation(stage2_text, home)
        second = _spawn_mutation(stage2_text, home)
        out1 = first.communicate(timeout=30)
        out2 = second.communicate(timeout=30)
        assert first.returncode == 0, out1[1]
        assert second.returncode == 0, out2[1]
        keys = KEY_LINE_RE.findall((home / ".env").read_text())
        assert len(keys) == 1, (
            f"trial {i}: expected one API_SERVER_KEY, got {keys!r}\n"
            f"p1={out1[0]}{out1[1]}\np2={out2[0]}{out2[1]}"
        )


def test_two_concurrent_routing_syncs_keep_both_overrides(
    stage2_text: str, tmp_path: Path
) -> None:
    """rewrite_env_var is read-truncate-write; two siblings used to drop a line.

    Compose gives both containers the same env. A sibling that does not have a
    variable still removes the other's managed line — that is single-container
    semantics, not this race.
    """
    both = (
        "API_SERVER_KEY=operator-env-key-0123456789\n"
        f"HERMES_PORTAL_BASE_URL='{PORTAL}'\n"
        f"NOUS_INFERENCE_BASE_URL='{INFERENCE}'\n"
    )
    for i in range(8):
        home = tmp_path / f"route-{i}"
        home.mkdir()
        (home / ".env").write_text("OTHER=keep\n")
        first = _spawn_mutation(stage2_text, home, extra_env=both)
        second = _spawn_mutation(stage2_text, home, extra_env=both)
        out1 = first.communicate(timeout=30)
        out2 = second.communicate(timeout=30)
        assert first.returncode == 0, out1[1]
        assert second.returncode == 0, out2[1]
        text = (home / ".env").read_text()
        assert "OTHER=keep" in text
        assert f"HERMES_PORTAL_BASE_URL={PORTAL}" in text, text
        assert f"NOUS_INFERENCE_BASE_URL={INFERENCE}" in text, text
        assert len(re.findall(r"^API_SERVER_KEY=", text, re.MULTILINE)) == 0


def test_keygen_fail_opens_when_lockdir_wait_times_out(
    stage2_text: str, tmp_path: Path
) -> None:
    """A wedged mkdir lock must not abort first-boot key generation.

    Holding only flock is not enough to block: Docker Desktop can grant
    flock to both sides, so mkdir is the exclusive lock.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".stage2-home.lockdir").mkdir()
    env = os.environ.copy()
    env["STAGE2_HOME_LOCK_WAIT"] = "1"
    result = _run_mutation(stage2_text, home, env=env)
    assert result.returncode == 0, result.stderr
    assert "continuing without serialization" in (result.stdout + result.stderr)
    assert KEY_LINE_RE.search((home / ".env").read_text())


def test_stale_lockdir_is_broken_even_when_flock_exists(
    stage2_text: str, tmp_path: Path
) -> None:
    """A leftover lockdir older than two minutes must not block boot."""
    home = tmp_path / "home"
    home.mkdir()
    lockdir = home / ".stage2-home.lockdir"
    lockdir.mkdir()
    stale = time.time() - 180
    os.utime(lockdir, (stale, stale))
    result = _run_mutation(stage2_text, home)
    assert result.returncode == 0, result.stderr
    assert "breaking stale" in (result.stdout + result.stderr)
    assert KEY_LINE_RE.search((home / ".env").read_text())
    assert not lockdir.exists()
