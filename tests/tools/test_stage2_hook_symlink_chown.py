"""Contract tests for symlink-safe recursive ownership repair in stage2."""
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


def _chown_helper(text: str) -> str:
    match = re.search(r"(chown_hermes_tree\(\) \{\n(?:.*\n)*?\})", text)
    assert match, "stage2-hook.sh must define chown_hermes_tree"
    return match.group(1)


def test_recursive_chown_uses_symlink_safe_helper(stage2_text: str) -> None:
    helper = _chown_helper(stage2_text)
    assert '[ -L "$target" ]' in helper
    assert 'find -P "$target" ! -type l -exec chown hermes:hermes {} +' in helper
    assert 'chown -R hermes:hermes "$HERMES_HOME/$sub"' not in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/$sub"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/profiles"' in stage2_text
    assert 'chown_hermes_tree "$HERMES_HOME/cron"' in stage2_text


def test_recursive_chown_helper_skips_symlinked_home(stage2_text: str, tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    hermes_home = tmp_path / "data"
    outside_home = tmp_path / "real-home"
    hermes_home.mkdir()
    outside_home.mkdir()
    (outside_home / "ssh-key").write_text("secret", encoding="utf-8")
    (hermes_home / "home").symlink_to(outside_home, target_is_directory=True)

    log_path = tmp_path / "calls.log"
    script = (
        "set -e\n"
        f"{_chown_helper(stage2_text)}\n"
        f'find() {{ echo "find:$*" >> "{log_path}"; }}\n'
        f'chown() {{ echo "chown:$*" >> "{log_path}"; }}\n'
        f'chown_hermes_tree "{hermes_home}/home"\n'
    )

    proc = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Skipping recursive chown of symlinked path" in proc.stdout
    assert not log_path.exists()

