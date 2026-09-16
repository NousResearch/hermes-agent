"""Regression tests for the stage2 Nous routing-override sync.

Hosted deploys carry ``HERMES_PORTAL_BASE_URL`` / ``NOUS_INFERENCE_BASE_URL`` only in the
container environment. Under ``GATEWAY_MULTIPLEX_PROFILES`` both are resolved through the
profile secret scope (#108319 / #111809), which is built from ``<profile>/.env`` and never
falls back to ``os.environ`` — so the staging Portal URL was dropped, the refresh token went
to the production Portal, and the login was quarantined on every boot. stage2 must carry the
container value into every served profile's ``.env``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

PORTAL = "https://portal.staging-nousresearch.com"
INFERENCE = "https://stg-inference-api.nousresearch.com/v1"


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
    env_setup = "".join(
        f"unset {k}\n" if v is None else f"{k}='{v}'\n" for k, v in env.items()
    )
    script = (
        "set -eu\n"  # production runs the hook under set -eu
        f"{env_setup}"
        f'HERMES_HOME="{home}"\n'
        'as_hermes() { "$@"; }\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_sync_block(stage2_text)}\n"
    )
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)


def _lines(path: Path, name: str) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.startswith(f"{name}=")]


def test_container_value_reaches_home_and_every_profile_env(stage2_text: str, tmp_path: Path) -> None:
    """Both overrides land in $HERMES_HOME/.env and each profiles/*/.env (created when missing)."""
    home = tmp_path / "home"
    (home / "profiles" / "work").mkdir(parents=True)
    (home / "profiles" / "ops").mkdir()
    (home / ".env").write_text("API_SERVER_KEY=abc\n")
    (home / "profiles" / "ops" / ".env").write_text("SLACK_BOT_TOKEN=xoxb-x\n")

    result = _run_sync(
        stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL, "NOUS_PORTAL_BASE_URL": None, "NOUS_INFERENCE_BASE_URL": INFERENCE}
    )

    assert result.returncode == 0, result.stderr
    for env_file in (home / ".env", home / "profiles" / "work" / ".env", home / "profiles" / "ops" / ".env"):
        assert _lines(env_file, "HERMES_PORTAL_BASE_URL") == [f"HERMES_PORTAL_BASE_URL={PORTAL}"], env_file
        assert _lines(env_file, "NOUS_INFERENCE_BASE_URL") == [f"NOUS_INFERENCE_BASE_URL={INFERENCE}"], env_file
    # Existing unrelated secrets survive the rewrite.
    assert "API_SERVER_KEY=abc" in (home / ".env").read_text()
    assert "SLACK_BOT_TOKEN=xoxb-x" in (home / "profiles" / "ops" / ".env").read_text()
    created = home / "profiles" / "work" / ".env"
    assert (created.stat().st_mode & 0o777) == 0o600


def test_stale_value_replaced_and_correct_value_left_alone(stage2_text: str, tmp_path: Path) -> None:
    """The container value wins over a stale line (one assignment, not two); a matching line is not rewritten."""
    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("HERMES_PORTAL_BASE_URL=https://portal.nousresearch.com\nOTHER=1\n")

    first = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL, "NOUS_PORTAL_BASE_URL": None, "NOUS_INFERENCE_BASE_URL": None})
    assert first.returncode == 0, first.stderr
    assert _lines(env_file, "HERMES_PORTAL_BASE_URL") == [f"HERMES_PORTAL_BASE_URL={PORTAL}"]
    assert "OTHER=1" in env_file.read_text()
    assert "Synced HERMES_PORTAL_BASE_URL" in first.stdout

    before = env_file.stat().st_mtime_ns
    os.utime(env_file, ns=(before - 5_000_000_000, before - 5_000_000_000))
    stamped = env_file.stat().st_mtime_ns
    second = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL, "NOUS_PORTAL_BASE_URL": None, "NOUS_INFERENCE_BASE_URL": None})
    assert second.returncode == 0, second.stderr
    assert env_file.stat().st_mtime_ns == stamped, "an already-correct line must not rewrite the volume"
    assert "Synced" not in second.stdout


def test_unset_override_touches_nothing_and_symlinked_env_is_refused(stage2_text: str, tmp_path: Path) -> None:
    """No container value → no .env is created or edited; a symlinked .env is never written through."""
    home = tmp_path / "home"
    (home / "profiles" / "work").mkdir(parents=True)
    result = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": None, "NOUS_PORTAL_BASE_URL": None, "NOUS_INFERENCE_BASE_URL": None})
    assert result.returncode == 0, result.stderr
    assert not (home / ".env").exists()
    assert not (home / "profiles" / "work" / ".env").exists()

    outside = tmp_path / "outside.env"
    outside.write_text("KEEP=1\n")
    (home / ".env").symlink_to(outside)
    result = _run_sync(stage2_text, home, {"HERMES_PORTAL_BASE_URL": PORTAL, "NOUS_PORTAL_BASE_URL": None, "NOUS_INFERENCE_BASE_URL": None})
    assert result.returncode == 0, result.stderr
    assert outside.read_text() == "KEEP=1\n"
    assert "refusing sync HERMES_PORTAL_BASE_URL" in result.stdout


def test_nous_portal_alias_is_synced_too(stage2_text: str, tmp_path: Path) -> None:
    """``NOUS_PORTAL_BASE_URL`` is accepted by ``_nous_portal_env_override`` and must reach the scope as well."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_sync(
        stage2_text,
        home,
        {"HERMES_PORTAL_BASE_URL": None, "NOUS_PORTAL_BASE_URL": PORTAL, "NOUS_INFERENCE_BASE_URL": None},
    )
    assert result.returncode == 0, result.stderr
    assert _lines(home / ".env", "NOUS_PORTAL_BASE_URL") == [f"NOUS_PORTAL_BASE_URL={PORTAL}"]
    assert _lines(home / ".env", "HERMES_PORTAL_BASE_URL") == []
