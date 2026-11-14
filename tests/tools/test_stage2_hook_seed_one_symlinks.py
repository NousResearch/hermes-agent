"""Regression tests for symlink-safe Docker stage2 first-boot seeds.

Also covers the permission posture of the credential-bearing seeds (#126950):
auth.json and .env are owner-only from the first instant, and a failing
re-tighten chmod does not abort the boot.
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
        r"(seed_one\(\) \{\n(?:.*\n)*?\})\n(?:#.*\n)*\(?(?:umask 077 && )?seed_one",
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


def _run_auth_seed(stage2_text: str, tmp_path: Path, chmod_stub: str) -> tuple:
    """Run the real auth.json seed block under ``set -eu`` and umask 022.

    ``chmod_stub`` replaces the chmod command so the test can observe the
    mode the write itself produced (no-op) or a failing re-tighten.
    """
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("sh not available")
    home = tmp_path / "home"
    home.mkdir()
    script = (
        "set -eu\n"
        "umask 022\n"
        f'HERMES_HOME="{home}"\n'
        "HERMES_AUTH_JSON_BOOTSTRAP='{\"providers\":{}}'\n"
        f"{chmod_stub}\n"
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_auth_seed_block(stage2_text)}"
        "echo SEED_DONE\n"
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)
    proc = subprocess.run([sh, str(script_path)], capture_output=True, text=True, check=False)
    return proc, home / "auth.json"


@pytest.mark.platforms("posix")
def test_auth_seed_creates_credential_owner_only(stage2_text: str, tmp_path: Path) -> None:
    """The write itself must produce a 0600 file under a 022 ambient umask;
    with chmod neutralised, a bare redirect would leave it 0644 (#126950)."""
    proc, auth = _run_auth_seed(stage2_text, tmp_path, "chmod() { :; }")
    assert proc.returncode == 0, proc.stderr
    assert auth.read_text() == '{"providers":{}}'
    assert auth.stat().st_mode & 0o777 == 0o600


@pytest.mark.platforms("posix")
def test_auth_seed_survives_failing_chmod(stage2_text: str, tmp_path: Path) -> None:
    """A failing re-tighten warns and boot continues under ``set -eu``,
    with the credential already owner-only."""
    proc, auth = _run_auth_seed(stage2_text, tmp_path, "chmod() { return 1; }")
    assert proc.returncode == 0, proc.stderr
    assert "SEED_DONE" in proc.stdout
    assert "could not tighten auth.json permissions" in proc.stderr
    assert auth.stat().st_mode & 0o777 == 0o600


def _env_seed_line(text: str) -> str:
    m = re.search(r'(?m)^.*seed_one "\.env" "\.env\.example".*$', text)
    assert m, "stage2-hook.sh must keep the .env first-boot seed"
    return m.group(0)


@pytest.mark.platforms("posix")
def test_env_seed_is_owner_only_under_ambient_umask(stage2_text: str, tmp_path: Path) -> None:
    """The first-boot .env copy (which later receives API_SERVER_KEY) must be
    0600 straight from the seed, not 0644 until a best-effort chmod (#126950)."""
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("sh not available")
    home = tmp_path / "home"
    install_dir = tmp_path / "install"
    home.mkdir()
    install_dir.mkdir()
    example = install_dir / ".env.example"
    example.write_text("# example\n")
    example.chmod(0o644)
    script = (
        "set -eu\n"
        "umask 022\n"
        f'HERMES_HOME="{home}"\n'
        f'INSTALL_DIR="{install_dir}"\n'
        "as_hermes() { \"$@\"; }\n"
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_seed_one_function(stage2_text)}\n"
        f"{_env_seed_line(stage2_text)}\n"
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)
    proc = subprocess.run([sh, str(script_path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (home / ".env").read_text() == "# example\n"
    assert (home / ".env").stat().st_mode & 0o777 == 0o600
