"""Regression tests for symlink-safe Docker stage2 first-boot seeds."""
from __future__ import annotations

import re
import shutil
import stat
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


def _seed_one_function(text: str) -> str:
    m = re.search(
        r"(seed_one\(\) \{\n(?:.*\n)*?\})\nseed_one",
        text,
    )
    assert m, "stage2-hook.sh must define seed_one before first-boot seeds"
    return m.group(1)


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def _auth_bootstrap_block(text: str) -> str:
    start = text.index('if [ ! -f "$HERMES_HOME/auth.json" ]')
    end = text.index("\n\n# auth.json: re-seed", start)
    return text[start:end]


def test_seed_one_refuses_symlinked_destinations(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    home = tmp_path / "home"
    install_dir = tmp_path / "install"
    home.mkdir()
    install_dir.mkdir()
    outside_env = tmp_path / "outside.env"
    try:
        (home / ".env").symlink_to(outside_env)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")
    (install_dir / ".env.example").write_text("SECRET=1\n")

    script = (
        "set -e\n"
        f'HERMES_HOME="{home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "as_hermes() { \"$@\"; }\n"
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_seed_one_function(stage2_text)}\n"
        'seed_one ".env" ".env.example"\n'
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not outside_env.exists()
    assert (home / ".env").is_symlink()
    assert "refusing seed through symlinked path" in proc.stdout


def test_seed_one_is_quiet_for_existing_symlinked_files(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    home = tmp_path / "home"
    install_dir = tmp_path / "install"
    home.mkdir()
    install_dir.mkdir()
    outside_env = tmp_path / "outside.env"
    outside_env.write_text("EXISTING=1\n")
    try:
        (home / ".env").symlink_to(outside_env)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")
    (install_dir / ".env.example").write_text("SECRET=1\n")

    script = (
        "set -e\n"
        f'HERMES_HOME="{home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "as_hermes() { \"$@\"; }\n"
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_seed_one_function(stage2_text)}\n"
        'seed_one ".env" ".env.example"\n'
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert outside_env.read_text() == "EXISTING=1\n"
    assert proc.stdout == ""


def test_auth_bootstrap_serializes_only_after_owner_mode_is_set(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")

    home = tmp_path / "home"
    home.mkdir()
    observed_mode = tmp_path / "observed-mode"
    auth_path = home / "auth.json"
    script = (
        "set -eu\n"
        "umask 000\n"
        f'HERMES_HOME="{home}"\n'
        "HERMES_AUTH_JSON_BOOTSTRAP='{\"access_token\":\"secret\"}'\n"
        f'MODE_LOG="{observed_mode}"\n'
        'as_hermes() { "$@"; }\n'
        "chown() { :; }\n"
        "printf() {\n"
        '    for target in "$HERMES_HOME"/auth.json "$HERMES_HOME"/.auth.json.bootstrap.*; do\n'
        '        [ -f "$target" ] || continue\n'
        '        mode="$(stat -c \'%a\' "$target" 2>/dev/null || stat -f \'%Lp\' "$target")"\n'
        '        if [ "$mode" != 600 ]; then\n'
        '            command printf \'%s\\n\' "$mode" > "$MODE_LOG"\n'
        "            return 1\n"
        "        fi\n"
        "    done\n"
        '    command printf "$@"\n'
        "}\n"
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_auth_bootstrap_block(stage2_text)}\n"
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([shell, str(script_path)], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert not observed_mode.exists()
    assert auth_path.read_text() == '{"access_token":"secret"}'
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    assert not list(home.glob(".auth.json.bootstrap.*"))
