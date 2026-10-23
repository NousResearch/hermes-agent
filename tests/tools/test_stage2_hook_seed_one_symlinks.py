"""Regression tests for symlink-safe Docker stage2 first-boot seeds.

Also guards the auth.json env-seed's permission posture (#126950): the
credential write must be wrapped in a restrictive umask so the file is
owner-only from the first instant (no 0644-then-tighten window under the
ambient umask), and the re-tightening chmod must be non-fatal so a failing
one cannot abort the boot with the credential stranded.
"""
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

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True, check=False)
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

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert outside_env.read_text() == "EXISTING=1\n"
    assert proc.stdout == ""


def _auth_seed_block(text: str) -> str:
    m = re.search(
        r'(?m)^if \[ ! -f "\$HERMES_HOME/auth\.json" \] && '
        r'\[ -n "\$\{HERMES_AUTH_JSON_BOOTSTRAP:-\}" \]; then\n'
        r"(?:.*\n)*?^fi\n",
        text,
    )
    assert m, "stage2-hook.sh must keep the auth.json first-boot seed block"
    return m.group(0)


def test_auth_seed_write_is_wrapped_in_umask_077(stage2_text: str) -> None:
    """The auth.json seed must create the credential owner-only from the first
    instant: the write runs inside a `umask 077` subshell (the .env seed's
    pattern), never as a bare root-context redirect that lands 0644 under the
    ambient umask until a later chmod catches up (#126950)."""
    block = _auth_seed_block(stage2_text)
    assert re.search(
        r"\(umask 077 && printf '%s' \"\$HERMES_AUTH_JSON_BOOTSTRAP\" "
        r'> "\$HERMES_HOME/auth\.json"\)',
        block,
    ), "auth.json seed write must be a single printf under `umask 077`"


def test_auth_seed_chmod_failure_is_non_fatal(stage2_text: str) -> None:
    """The re-tightening chmod must be silenced-or-warn, never a bare command:
    under `set -eu` a failing chmod aborted the boot with the seeded credential
    left behind — 0600 from creation now, so the failure only downgrades to a
    warning (#126950's durable case)."""
    block = _auth_seed_block(stage2_text)
    assert re.search(
        r'chmod 600 "\$HERMES_HOME/auth\.json" 2>/dev/null \\\n'
        r'\s*\|\| echo "\[stage2\] Warning: could not tighten auth\.json permissions"',
        block,
    ), "the auth.json re-tighten must warn and continue instead of aborting"
