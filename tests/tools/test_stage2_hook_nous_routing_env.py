"""Regression tests for the stage2 Nous routing-override sync.

Hosted deploys carry ``HERMES_PORTAL_BASE_URL`` / ``NOUS_INFERENCE_BASE_URL`` only in the container
environment. Under ``GATEWAY_MULTIPLEX_PROFILES`` both resolve through the profile secret scope
(#108319 / #111809), built from ``<profile>/.env`` with no ``os.environ`` fallback, so stage2 must
carry the container value into every served profile's ``.env`` — and take it out again when the
platform stops setting it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.secret_scope import load_env_file

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

PORTAL = "https://portal.staging-nousresearch.com"
INFERENCE = "https://stg-inference-api.nousresearch.com/v1"
NAMES = ("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL", "NOUS_INFERENCE_BASE_URL")


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _sync_block(text: str) -> str:
    start = text.index("# --- Sync deploy-injected Nous routing overrides")
    end = text.index("# .env holds API keys and secrets", start)
    return text[start:end]


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def _run_sync(stage2_text: str, home: Path, env: dict[str, str | None]) -> subprocess.CompletedProcess[str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    env = {name: env.get(name) for name in NAMES}
    env_setup = "".join(f"unset {k}\n" if v is None else f"{k}='{v}'\n" for k, v in env.items())
    script = (
        "set -eu\n"  # production runs the hook under set -eu
        f"{env_setup}"
        f'HERMES_HOME="{home}"\n'
        'as_hermes() { "$@"; }\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_sync_block(stage2_text)}\n"
    )
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)


def _assignments(path: Path, name: str) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.startswith(f"{name}=")]


def test_container_values_reach_home_and_every_profile_env(stage2_text: str, tmp_path: Path) -> None:
    """All three overrides land in $HERMES_HOME/.env and each profiles/*/.env (created 0600 when
    missing), unrelated secrets survive, and the runtime tokenizer reads the marked line back."""
    home = tmp_path / "home"
    (home / "profiles" / "work").mkdir(parents=True)
    (home / "profiles" / "ops").mkdir()
    (home / ".env").write_text("API_SERVER_KEY=abc\n")
    (home / "profiles" / "ops" / ".env").write_text("SLACK_BOT_TOKEN=xoxb-x\n")

    result = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL, "NOUS_PORTAL_BASE_URL": PORTAL,
                                           "NOUS_INFERENCE_BASE_URL": INFERENCE})

    assert result.returncode == 0, result.stderr
    for env_file in (home / ".env", home / "profiles" / "work" / ".env", home / "profiles" / "ops" / ".env"):
        parsed = load_env_file(env_file)
        assert parsed["HERMES_PORTAL_BASE_URL"] == parsed["NOUS_PORTAL_BASE_URL"] == PORTAL, env_file
        assert parsed["NOUS_INFERENCE_BASE_URL"] == INFERENCE, env_file
        assert len(_assignments(env_file, "HERMES_PORTAL_BASE_URL")) == 1, env_file
    assert load_env_file(home / ".env")["API_SERVER_KEY"] == "abc"
    assert load_env_file(home / "profiles" / "ops" / ".env")["SLACK_BOT_TOKEN"] == "xoxb-x"
    assert ((home / "profiles" / "work" / ".env").stat().st_mode & 0o777) == 0o600


def test_container_wins_over_stale_line_then_idempotent_then_removed_when_unset(stage2_text: str, tmp_path: Path) -> None:
    """Boot 1: a stale line is replaced (one assignment). Boot 2 (same value): the volume is not rewritten.
    Boot 3 (variable gone): the stage2-written line is removed while hand-set lines stay. A read-only file
    degrades to a warning; a symlinked .env is never written through."""
    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("HERMES_PORTAL_BASE_URL=https://portal.nousresearch.com\nNOUS_INFERENCE_BASE_URL=https://by-hand/v1\nOTHER=1\n")

    first = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL})
    assert first.returncode == 0, first.stderr
    assert load_env_file(env_file)["HERMES_PORTAL_BASE_URL"] == PORTAL
    assert len(_assignments(env_file, "HERMES_PORTAL_BASE_URL")) == 1
    assert "Synced HERMES_PORTAL_BASE_URL" in first.stdout

    before = env_file.stat().st_mtime_ns
    os.utime(env_file, ns=(before - 5_000_000_000, before - 5_000_000_000))
    stamped = env_file.stat().st_mtime_ns
    second = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL})
    assert second.returncode == 0, second.stderr
    assert env_file.stat().st_mtime_ns == stamped, "an already-correct line must not rewrite the volume"
    assert "Synced" not in second.stdout

    with env_file.open("a") as fh:
        fh.write("HERMES_PORTAL_BASE_URL=https://by-hand\n")  # operator line beside the managed one
    third = _run_sync(stage2_text, home, {})
    assert third.returncode == 0, third.stderr
    parsed = load_env_file(env_file)
    assert _assignments(env_file, "HERMES_PORTAL_BASE_URL") == ["HERMES_PORTAL_BASE_URL=https://by-hand"], \
        "only the stage2-written line goes when the platform unsets it"
    assert parsed["NOUS_INFERENCE_BASE_URL"] == "https://by-hand/v1", "a hand-set line is never touched"
    assert parsed["OTHER"] == "1"
    assert "Removed HERMES_PORTAL_BASE_URL" in third.stdout

    if os.geteuid() != 0:
        env_file.chmod(0o444)
        frozen = env_file.read_text()
        readonly = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL})
        env_file.chmod(0o600)
        assert readonly.returncode == 0, "a read-only volume degrades to a warning under set -e"
        assert "could not write HERMES_PORTAL_BASE_URL" in readonly.stdout
        assert env_file.read_text() == frozen

    outside = tmp_path / "outside.env"
    outside.write_text("KEEP=1\n")
    env_file.unlink()
    env_file.symlink_to(outside)
    fourth = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL})
    assert fourth.returncode == 0, fourth.stderr
    assert outside.read_text() == "KEEP=1\n"
    assert "refusing sync through symlinked path" in fourth.stdout
