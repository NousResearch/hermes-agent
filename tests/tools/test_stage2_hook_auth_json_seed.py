"""Regression tests for the auth.json first-boot seed (#126950).

``docker/stage2-hook.sh`` seeds ``$HERMES_HOME/auth.json`` from
``HERMES_AUTH_JSON_BOOTSTRAP`` with a root-context redirect. Under the
container's ambient umask (022) that redirect creates the credential file
mode 0644 (world-readable) and only then tightens it to 0600 — and the
bare ``chmod 600`` runs unguarded under ``set -eu``, so a failing chmod
aborts the hook leaving the refresh token world-readable permanently.

Two tests, two layers:

- ``test_auth_json_seed_creates_owner_only_file`` pins the fix
  structurally: the seed block must establish ``umask 077`` before the
  first redirect that materialises ``auth.json`` (a ``>`` truncate
  preserves a pre-existing file's mode, so pre-creating 0600 closes the
  0644 window), and the re-tightening ``chmod`` must be guarded so it
  warns instead of aborting the hook.
- ``test_auth_json_seed_survives_failing_chmod`` runs the block under
  ``set -eu`` with a failing ``chmod``: the hook must exit 0 with a
  warning and the seeded content intact.

Why not assert ``stat -c %a == 600`` unconditionally: this repo's Windows
dev box runs MSYS with path conversion disabled, and its ``stat``
fabricates the mode from the *observer's* umask (same file reports
644/600/640 as the observer's umask changes) — verified, so a mode
assertion there can neither fail nor pass meaningfully. Where the
platform reports real modes (Linux CI/production), the behavioral test
additionally asserts 0600; see ``_modes_are_real``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = Path(
    os.environ.get(
        "HERMES_TEST_STAGE2_HOOK", REPO_ROOT / "docker" / "stage2-hook.sh"
    )
)

SEED = '{"providers":{"nous":{"refresh_token":"secret-token"}}}'


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _seed_block(text: str) -> str:
    start = text.index("# auth.json: bootstrap from env on first boot only.")
    end = text.index("# auth.json: re-seed a TERMINALLY-DEAD", start)
    return text[start:end]


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def _sh_home(home: Path) -> str:
    """MSYS sh honors umask/chmod only via the /tmp mount (this box
    disables path conversion and the /c mount ignores modes), so map the
    Windows temp dir onto /tmp; fall back to a drive-letter mapping."""
    posix = home.as_posix()
    if os.name == "nt":
        # tempfile/pytest use the Windows temp dir (GetTempPath, often in
        # 8.3 short form), not the MSYS $TEMP; that dir is also mounted
        # as /tmp, the only mount here that honors umask/chmod. resolve()
        # expands both sides to long form so relative_to can compare.
        try:
            rel = Path(home).resolve().relative_to(
                Path(tempfile.gettempdir()).resolve()
            )
            return "/tmp/" + rel.as_posix()
        except ValueError:
            pass
        m = re.match(r"^([A-Za-z]):(.*)$", posix)
        if m:
            return f"/{m.group(1).lower()}{m.group(2)}"
    return posix


def _run_seed(
    stage2_text: str,
    home: Path,
    break_chmod: bool = False,
) -> subprocess.CompletedProcess[str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    shadow = "chmod() { return 1; }\n" if break_chmod else ""
    script = (
        # Production runs the hook under `set -eu` — the sandbox must
        # match, or the tests cannot see unguarded-command defects that
        # would abort a real container boot.
        "set -eu\n"
        # Container PID 1 ambient umask.
        "umask 022\n"
        f"{shadow}"
        f'HERMES_AUTH_JSON_BOOTSTRAP=\'{SEED}\'\n'
        f'HERMES_HOME="{_sh_home(home)}"\n'
        # In tests we run unprivileged; as_hermes is a passthrough then.
        'as_hermes() { "$@"; }\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_seed_block(stage2_text)}\n"
        'stat -c %a "$HERMES_HOME/auth.json"\n'
    )
    return subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _modes_are_real(tmp_path: Path) -> bool:
    """True when the platform reports real file modes (not the case on
    this repo's MSYS dev box, where stat fabricates modes from the
    observer's umask)."""
    if shutil.which("sh") is None:
        return False
    probe = tmp_path / "mode-probe"
    probe.mkdir(exist_ok=True)
    home = _sh_home(probe)
    result = subprocess.run(
        [
            "sh",
            "-c",
            f'umask 077; : > "{home}/f"; umask 022; stat -c %a "{home}/f"',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip() == "600"


def test_auth_json_seed_creates_owner_only_file(stage2_text: str) -> None:
    """The seed must be 0600 from the first instant, not chmod-ed later."""
    block = _seed_block(stage2_text)
    seed_at = block.index(
        'printf \'%s\' "$HERMES_AUTH_JSON_BOOTSTRAP" > "$HERMES_HOME/auth.json"'
    )
    umask_at = block.find("umask 077")
    assert 0 <= umask_at < seed_at, (
        "auth.json must be pre-created under umask 077 before the seeding "
        "redirect (a '>' truncate preserves the file's mode, so 0600 from "
        "pre-create closes the ambient-umask 0644 window)"
    )
    assert re.search(
        r'chmod 600 "\$HERMES_HOME/auth\.json"[^\n]*(?:\\\n[^\n]*)*\|\|',
        block,
    ), "the re-tightening chmod must be guarded (warn, not abort under set -eu)"


def test_auth_json_seed_survives_failing_chmod(
    stage2_text: str, tmp_path: Path
) -> None:
    """A failing chmod must warn, not abort the hook with 0644 left behind."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_seed(stage2_text, home, break_chmod=True)
    assert result.returncode == 0, (
        f"seed must not abort the hook when chmod fails: {result.stderr}"
    )
    auth_path = home / "auth.json"
    assert auth_path.is_file(), "seed must create auth.json"
    assert auth_path.read_text() == SEED, "seed must write the bootstrap content"
    assert "Warning" in result.stdout, "failing chmod must warn, not die silent"
    if _modes_are_real(tmp_path):
        # Defense in depth made observable: with chmod broken, the file
        # must still be 0600 from the umask-077 pre-create alone.
        assert result.stdout.strip().splitlines()[-1] == "600", (
            f"auth.json must be owner-only, got mode {result.stdout.strip()}"
        )
