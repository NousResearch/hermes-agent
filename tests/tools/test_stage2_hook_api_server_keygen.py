"""Regression tests for the stage2 API_SERVER_KEY bootstrap (OOF-285).

The gateway's loopback api_server refuses to start without a strong
API_SERVER_KEY, and hosted cron fires are forwarded through it — so the
stage2 keygen must succeed even when no ``.env`` exists yet. Historically it
was gated on ``[ -f "$HERMES_HOME/.env" ]`` while the first-boot seed that
was supposed to create ``.env`` silently no-oped (``.env.example`` was
excluded from the image by ``.dockerignore``), leaving 40%+ of the hosted
fleet with no api_server and every scheduled cron fire silently lost.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# ``docker/stage2-hook.sh`` boots a Linux container.  Its permission
# contract is only observable on a POSIX filesystem: under Git Bash on
# NTFS a ``chmod 600`` reports back as ``0o666`` and a ``chmod 444`` does
# not stop the owner writing, so these two cases would assert nothing.
POSIX_FILE_MODES = pytest.mark.skipif(
    os.name == "nt",
    reason="container .env permission contract needs POSIX file modes",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

KEY_LINE_RE = re.compile(r"^API_SERVER_KEY=[0-9a-f]{64}$", re.MULTILINE)


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _keygen_block(text: str) -> str:
    start = text.index("# --- Ensure a gateway api_server key exists")
    end = text.index("# .env holds API keys and secrets", start)
    return text[start:end]


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def _run_keygen(
    stage2_text: str,
    home: Path,
    env_key: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    if env_key is None:
        env_setup = "unset API_SERVER_KEY\n"
    else:
        env_setup = f'API_SERVER_KEY="{env_key}"\n'
    script = (
        # Production runs the hook under `set -eu` (docker/stage2-hook.sh
        # shebang block) — the sandbox must match, or the tests cannot see
        # unguarded-command defects that would abort a real container boot.
        "set -eu\n"
        f"{env_setup}"
        f'HERMES_HOME="{home}"\n'
        # In tests we run unprivileged; as_hermes is a passthrough then.
        'as_hermes() { "$@"; }\n'
        f"{_path_guard_functions(stage2_text)}\n"
        f"{_keygen_block(stage2_text)}\n"
    )
    return subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_keygen_creates_env_when_missing(stage2_text: str, tmp_path: Path) -> None:
    """No .env at all (failed/absent first-boot seed) must still yield a key."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_keygen(stage2_text, home)
    assert result.returncode == 0, result.stderr
    env_path = home / ".env"
    assert env_path.is_file(), "keygen must create .env when it is missing"
    assert KEY_LINE_RE.search(env_path.read_text()), "generated key missing/malformed"


@POSIX_FILE_MODES
def test_keygen_creates_env_owner_only(stage2_text: str, tmp_path: Path) -> None:
    """The .env the keygen creates carries the API key — it must be 0600."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_keygen(stage2_text, home)
    assert result.returncode == 0, result.stderr
    env_path = home / ".env"
    mode = env_path.stat().st_mode & 0o777
    assert mode == 0o600, f".env must be owner-only, got {oct(mode)}"


def test_keygen_appends_to_existing_env_without_key(
    stage2_text: str, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("OTHER=1\nAPI_SERVER_KEY=\n")
    result = _run_keygen(stage2_text, home)
    assert result.returncode == 0, result.stderr
    content = (home / ".env").read_text()
    assert "OTHER=1" in content
    assert KEY_LINE_RE.search(content)
    # Exactly one real key assignment. (The stale empty `API_SERVER_KEY=` line
    # is dropped by GNU sed in the production container; BSD sed on macOS dev
    # hosts silently skips the -i invocation, so don't assert its removal.)
    assert len(re.findall(r"^API_SERVER_KEY=..+$", content, re.MULTILINE)) == 1


def test_keygen_never_overwrites_operator_key(
    stage2_text: str, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("API_SERVER_KEY=operator-provided-key-123\n")
    result = _run_keygen(stage2_text, home)
    assert result.returncode == 0, result.stderr
    content = (home / ".env").read_text()
    assert content == "API_SERVER_KEY=operator-provided-key-123\n"


def test_keygen_refuses_symlinked_env(stage2_text: str, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside.env"
    outside.write_text("HIJACK=1\n")
    (home / ".env").symlink_to(outside)
    result = _run_keygen(stage2_text, home)
    assert result.returncode == 0, result.stderr
    assert "refusing append" in (result.stdout + result.stderr)
    assert outside.read_text() == "HIJACK=1\n", "must not write through symlink"


def test_keygen_skips_when_container_env_provides_key(
    stage2_text: str, tmp_path: Path
) -> None:
    """`docker run -e API_SERVER_KEY=...` must win: no generated key.

    Hermes loads $HERMES_HOME/.env with override=True, so a key generated
    into .env would silently shadow the operator's env-provided credential
    and 401 every client still using it.
    """
    home = tmp_path / "home"
    home.mkdir()
    result = _run_keygen(stage2_text, home, env_key="operator-env-key-0123456789")
    assert result.returncode == 0, result.stderr
    assert "skipping generation" in (result.stdout + result.stderr)
    env_path = home / ".env"
    if env_path.exists():
        assert "API_SERVER_KEY=" not in env_path.read_text()


def test_keygen_env_key_with_existing_env_file_key_warns_not_clobbers(
    stage2_text: str, tmp_path: Path
) -> None:
    """Both container env AND .env carry keys: warn, touch nothing."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("API_SERVER_KEY=file-key-abcdef0123456789\n")
    result = _run_keygen(stage2_text, home, env_key="env-key-9876543210fedcba")
    assert result.returncode == 0, result.stderr
    assert "the .env value wins" in (result.stdout + result.stderr)
    content = (home / ".env").read_text()
    assert content == "API_SERVER_KEY=file-key-abcdef0123456789\n"


def test_keygen_env_key_drops_stale_empty_assignment(
    stage2_text: str, tmp_path: Path
) -> None:
    """Container env key + stale empty `API_SERVER_KEY=` line in .env.

    The empty assignment must be removed: .env is loaded with override=True,
    so python-dotenv would set API_SERVER_KEY to the empty string, clobbering
    the operator's container-provided key and failing the api_server startup
    guard — silently losing every cron fire.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("OTHER=1\nAPI_SERVER_KEY=\n")
    result = _run_keygen(stage2_text, home, env_key="operator-env-key-0123456789")
    assert result.returncode == 0, result.stderr
    assert "skipping generation" in (result.stdout + result.stderr)
    content = (home / ".env").read_text()
    assert "OTHER=1" in content
    # No generated (non-empty) key may appear — the operator's env key wins.
    assert not re.search(r"^API_SERVER_KEY=..+$", content, re.MULTILINE)
    # The stale empty line must be gone so override=True cannot clobber the
    # container key. (GNU sed only: BSD sed on macOS dev hosts silently skips
    # the -i invocation, same caveat as the append test above.)
    if _sed_is_gnu():
        assert "API_SERVER_KEY=" not in content


def _sed_is_gnu() -> bool:
    try:
        probe = subprocess.run(
            ["sed", "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and "GNU sed" in probe.stdout


@POSIX_FILE_MODES
def test_keygen_readonly_env_degrades_to_warning_not_boot_abort(
    stage2_text: str, tmp_path: Path
) -> None:
    """A keyless .env on a read-only volume must not abort stage2.

    The hook runs under `set -eu`; an unguarded failing append would kill the
    whole cont-init phase and the container boot. It must instead warn that
    the api_server will be unavailable and exit 0.
    """
    if os.geteuid() == 0:
        pytest.skip("running as root — file write perms are not enforced")
    home = tmp_path / "home"
    home.mkdir()
    env_path = home / ".env"
    env_path.write_text("OTHER=1\n")
    env_path.chmod(0o444)
    try:
        result = _run_keygen(stage2_text, home)
        assert result.returncode == 0, (
            f"stage2 keygen must not abort the boot on a read-only .env:\n"
            f"{result.stderr}"
        )
        assert "could not write API_SERVER_KEY" in (result.stdout + result.stderr)
        assert env_path.read_text() == "OTHER=1\n"
    finally:
        env_path.chmod(0o644)


def test_keygen_warns_on_weak_container_env_key(
    stage2_text: str, tmp_path: Path
) -> None:
    """A container key shorter than 16 chars now means api_server DOWN, not
    401s — the hook must say so in the boot log where the operator looks."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_keygen(stage2_text, home, env_key="short-key")
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "shorter than 16 characters" in out
    assert "skipping generation" in out


def test_keygen_weak_env_key_warning_suppressed_when_env_file_key_wins(
    stage2_text: str, tmp_path: Path
) -> None:
    """Weak container key + strong .env key: the .env key wins at runtime
    (override=True), the server DOES start — the 'will refuse to start'
    warning would be false and must not fire. The both-keys warning must."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("API_SERVER_KEY=strong-file-key-abcdef0123456789\n")
    result = _run_keygen(stage2_text, home, env_key="short-key")
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "shorter than 16 characters" not in out
    assert "the .env value wins" in out


def test_dockerignore_keeps_env_example_template() -> None:
    """The first-boot seed copies /opt/hermes/.env.example -> $HERMES_HOME/.env.

    ``.env.*`` in .dockerignore matches the template, so an explicit
    ``!.env.example`` re-include must appear AFTER it (last match wins), and
    no later rule may exclude it again (OOF-285).
    """
    if not DOCKERIGNORE.exists():
        pytest.skip(".dockerignore not present in this checkout")
    lines = [ln.strip() for ln in DOCKERIGNORE.read_text().splitlines()]
    rules = [ln for ln in lines if ln and not ln.startswith("#")]
    verdict = "excluded"  # default: not matched -> included
    for rule in rules:
        negate = rule.startswith("!")
        pattern = rule.lstrip("!")
        if pattern in (".env.example",) or _dockerignore_match(pattern):
            verdict = "included" if negate else "excluded"
    assert verdict == "included", (
        ".env.example must survive .dockerignore — docker/stage2-hook.sh "
        "seeds $HERMES_HOME/.env from it on first boot"
    )


def _dockerignore_match(pattern: str) -> bool:
    """Minimal dockerignore glob match of ``pattern`` against '.env.example'."""
    import fnmatch

    if pattern.endswith("/"):
        return False
    return fnmatch.fnmatch(".env.example", pattern)
