"""Contract test: the s6-overlay stage2 hook resets ownership of hermes-owned
top-level state files in $HERMES_HOME — but only those, never arbitrary
host-owned files.

Regression guard for the gateway restart loop reported in #35098: files such
as gateway.lock / state.db / auth.json live directly under $HERMES_HOME (not in
a subdir), so the targeted subdir chown misses them. When created or rewritten
by `docker exec <container> hermes …` (root unless `-u` is passed) they land
root-owned and the unprivileged hermes runtime then hits PermissionError on next
startup.

The fix uses an explicit allowlist rather than a blanket `find -user root`
sweep, preserving the targeted-ownership contract from #19788 / PR #19795: a
bind-mounted $HERMES_HOME may contain host-owned files Hermes does not manage,
and those must never be chowned.

The s6-overlay rework moved bootstrap from docker/entrypoint.sh (now a shim) to
docker/stage2-hook.sh, installed as /etc/cont-init.d/01-hermes-setup. This test
targets that location.
"""
from __future__ import annotations

import os
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


def _toplevel_chown_loop(text: str) -> str:
    """Extract the `for f in … chown hermes:hermes "$HERMES_HOME/$f" … done`
    block that repairs top-level state-file ownership."""
    m = re.search(
        r"(for f in \\\n(?:.*\\\n)*?.*; do\n(?:.*\n)*?done)",
        text,
    )
    assert m, "stage2-hook.sh must contain the top-level-file chown for-loop (#35098)"
    block = m.group(1)
    assert 'chown hermes:hermes "$HERMES_HOME/$f"' in block, (
        "the top-level-file loop must chown each allowlisted file to hermes"
    )
    return block


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def test_toplevel_chown_loop_present(stage2_text: str) -> None:
    block = _toplevel_chown_loop(stage2_text)
    # The reported-broken files must be covered.
    for required in ("auth.json", "state.db", "gateway.lock", "gateway_state.json"):
        assert required in block, (
            f"top-level chown allowlist must include {required!r} (#35098)"
        )


def test_no_blanket_find_user_root_sweep(stage2_text: str) -> None:
    """The fix must NOT reintroduce a blanket `find … -user root` chown of
    $HERMES_HOME contents — that would clobber host-owned files in a bind mount
    (#19788 / PR #19795)."""
    assert not re.search(r"find\s+\"?\$\{?HERMES_HOME\}?\"?[^\n]*-user\s+root", stage2_text), (
        "stage2-hook.sh must not blanket-chown root-owned files under "
        "$HERMES_HOME via `find -user root`; use the targeted allowlist instead "
        "so host-owned bind-mounted files are preserved (#19788, #19795)."
    )


def _run_loop(text: str, present_files: list[str]) -> list[str]:
    """Run the extracted chown loop in a sandbox $HERMES_HOME, with `chown`
    stubbed to record which paths it was asked to touch. Returns the basenames
    the loop attempted to chown."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _toplevel_chown_loop(text)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        home = dpath / "home"
        home.mkdir()
        for f in present_files:
            (home / f).touch()
        # A non-allowlisted, "host-owned" file that must never be chowned.
        (home / "host_secret.json").touch()

        # Stub chown to record the basename of its last argument (the path),
        # so we observe exactly which files the allowlist loop selected
        # without needing real root privileges.
        script = (
            "set -e\n"
            f'HERMES_HOME="{home}"\n'
            f"{_path_guard_functions(text)}\n"
            f'chown() {{ for a in "$@"; do :; done; echo "${{a##*/}}" >> "{dpath}/chown.log"; }}\n'
            + block
        )
        script_path = dpath / "harness.sh"
        script_path.write_text(script)

        proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

        log = dpath / "chown.log"
        if not log.exists():
            return []
        return [ln for ln in log.read_text().splitlines() if ln]


def test_loop_chowns_present_allowlisted_files(stage2_text: str) -> None:
    touched = _run_loop(stage2_text, ["auth.json", "state.db", "gateway.lock"])
    assert "auth.json" in touched
    assert "state.db" in touched
    assert "gateway.lock" in touched


def test_loop_skips_nonallowlisted_host_file(stage2_text: str) -> None:
    """A file NOT on the allowlist (e.g. a host-owned file in a bind mount) must
    never be chowned, even if present."""
    touched = _run_loop(stage2_text, ["auth.json"])
    assert "host_secret.json" not in touched, (
        "the allowlist loop must not touch non-allowlisted files (#19788)"
    )


def test_loop_skips_symlinked_allowlisted_file(stage2_text: str, tmp_path: Path) -> None:
    """Even allowlisted state files must not be chowned through symlinks."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _toplevel_chown_loop(stage2_text)

    home = tmp_path / "home"
    home.mkdir()
    outside_auth = tmp_path / "outside-auth.json"
    outside_auth.touch()
    (home / "auth.json").symlink_to(outside_auth)

    log = tmp_path / "chown.log"
    script = (
        "set -e\n"
        f'HERMES_HOME="{home}"\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f'chown() {{ for a in "$@"; do :; done; echo "${{a##*/}}" >> "{log}"; }}\n'
        + block
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not log.exists(), "symlinked auth.json must not be passed to chown"
    assert "refusing chown through symlinked path" in proc.stdout


def test_loop_skips_allowlisted_file_under_symlinked_home(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    """A symlinked $HERMES_HOME must not let file chown reach its target."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _toplevel_chown_loop(stage2_text)

    real_home = tmp_path / "real-home"
    real_home.mkdir()
    (real_home / "auth.json").touch()
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")

    log = tmp_path / "chown.log"
    script = (
        "set -e\n"
        f'HERMES_HOME="{linked_home}"\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f'chown() {{ for a in "$@"; do :; done; echo "${{a##*/}}" >> "{log}"; }}\n'
        + block
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not log.exists(), "must not chown files through symlinked HERMES_HOME"
    assert "refusing chown through symlinked path" in proc.stdout


def test_loop_skips_absent_files(stage2_text: str) -> None:
    """Allowlisted files that don't exist are skipped (no spurious chown)."""
    touched = _run_loop(stage2_text, ["auth.json"])
    # state.db wasn't created, so it must not appear.
    assert "state.db" not in touched
