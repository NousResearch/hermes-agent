"""Regression tests for symlink-safe Docker stage2 ownership repair."""
from __future__ import annotations

import re
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


def _chown_hermes_tree_function(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nneeds_chown=false", start)
    return text[start:end]


def _run_helper(
    text: str,
    target: Path,
    log_path: Path,
    *,
    hermes_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    hermes_home = target if hermes_home is None else hermes_home
    script = (
        "set -eu\n"
        f'HERMES_HOME="{hermes_home}"\n'
        f"{_chown_hermes_tree_function(text)}\n"
        f'chown() {{ printf "%s\\n" "$*" >> "{log_path}"; }}\n'
        f'chown_hermes_tree "{target}"\n'
    )
    return subprocess.run([shell, "-c", script], capture_output=True, text=True)


def test_chown_helper_repairs_real_directories(stage2_text: str, tmp_path: Path) -> None:
    target = tmp_path / "home"
    target.mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_helper(stage2_text, target, log_path)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().splitlines() == [
        f"-R hermes:hermes {target}",
    ]


def test_chown_helper_refuses_symlinked_directories(stage2_text: str, tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    symlinked_home = tmp_path / "hermes-home"
    try:
        symlinked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_helper(stage2_text, symlinked_home, log_path)

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists()
    assert "refusing recursive chown through symlinked path" in proc.stdout


def test_chown_helper_refuses_target_under_symlinked_home(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    real_home = tmp_path / "real-home"
    (real_home / "cron").mkdir(parents=True)
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_helper(
        stage2_text,
        linked_home / "cron",
        log_path,
        hermes_home=linked_home,
    )

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists(), "must not chown through a symlinked HERMES_HOME"
    assert "refusing recursive chown through symlinked path" in proc.stdout


def test_chown_helper_refuses_target_with_symlinked_ancestor(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    external_platforms = tmp_path / "external-platforms"
    (external_platforms / "pairing").mkdir(parents=True)
    try:
        (home / "platforms").symlink_to(
            external_platforms,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")
    log_path = tmp_path / "chown.log"

    proc = _run_helper(
        stage2_text,
        home / "platforms" / "pairing",
        log_path,
        hermes_home=home,
    )

    assert proc.returncode == 0, proc.stderr
    assert not log_path.exists(), "must not chown through symlinked ancestors"
    assert "refusing recursive chown through symlinked path" in proc.stdout


def test_stage2_uses_symlink_safe_helper_for_hermes_home_trees(stage2_text: str) -> None:
    assert 'chown_hermes_tree "$HERMES_HOME/$sub"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/profiles"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/cron"' in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/$sub"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/profiles"' not in stage2_text
    assert 'chown -R hermes:hermes "$HERMES_HOME/cron"' not in stage2_text


def test_stage2_skips_top_level_chown_for_symlinked_hermes_home(
    stage2_text: str,
) -> None:
    assert 'refuse_symlinked_path "chown" "$HERMES_HOME"' in stage2_text


def test_stage2_skips_recursive_repairs_when_tree_is_already_owned(
    stage2_text: str,
) -> None:
    assert "tree_has_non_hermes_owner() {" in stage2_text
    assert 'if [ -e "$HERMES_HOME/$sub" ] && tree_has_non_hermes_owner "$HERMES_HOME/$sub"; then' in stage2_text
    assert 'if [ -d "$HERMES_HOME/profiles" ] && tree_has_non_hermes_owner "$HERMES_HOME/profiles"; then' in stage2_text
