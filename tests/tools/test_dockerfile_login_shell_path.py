"""Contract + behavior tests for the login-shell PATH drop-in (#56634).

Debian's /etc/profile hardcodes PATH for login shells, discarding the image's
``ENV PATH`` *before* it sources /etc/profile.d/*.sh. init_session snapshots the
environment through a login shell (``bash -l -c ... export -p``), so without a
profile.d drop-in the snapshot loses the venv and a bare ``python3`` resolves to
the system interpreter (which lacks Hermes' deps, including Pillow). These tests
pin the drop-in into the image and prove -- with a real login-style shell, no
Docker build required -- that it survives the /etc/profile PATH reset.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DROPIN = REPO_ROOT / "docker" / "profile.d" / "99-hermes-venv-path.sh"

# Debian's /etc/profile PATH for a non-root shell, set before profile.d is
# sourced. Reproduced verbatim so the test exercises the exact reset the drop-in
# has to survive.
DEBIAN_NONROOT_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"


def _dockerfile_env_path_prepend() -> str:
    """The venv/bin prefix the Dockerfile's ``ENV PATH`` puts ahead of $PATH."""
    text = DOCKERFILE.read_text()
    match = re.search(r'ENV PATH="([^"]*):\$\{PATH\}"', text)
    assert match, "Dockerfile must set ENV PATH with a ${PATH} suffix"
    return match.group(1)


def _dropin_path_prepend() -> str:
    """The venv/bin prefix the drop-in prepends onto $PATH."""
    text = DROPIN.read_text()
    match = re.search(r'PATH="([^"]*):\$\{PATH\}"', text)
    assert match, "drop-in must prepend the venv dirs onto ${PATH}"
    return match.group(1)


def test_dockerfile_installs_login_shell_path_dropin() -> None:
    text = DOCKERFILE.read_text()
    assert DROPIN.exists(), "the login-shell PATH drop-in file must exist"
    # The drop-in must be copied into the directory /etc/profile is sourced from.
    assert re.search(
        r"COPY[^\n]*docker/profile\.d/99-hermes-venv-path\.sh\s+"
        r"/etc/profile\.d/99-hermes-venv-path\.sh",
        text,
    ), "Dockerfile must COPY the drop-in into /etc/profile.d/"


def test_dropin_matches_env_path_prepend() -> None:
    """Drift guard: the drop-in and ``ENV PATH`` must prepend the same dirs, so a
    future edit to one cannot silently diverge from the other."""
    assert _dropin_path_prepend() == _dockerfile_env_path_prepend()
    # And it must actually include the venv the whole bug is about.
    assert "/opt/hermes/.venv/bin" in _dropin_path_prepend()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_dropin_restores_venv_after_debian_profile_reset() -> None:
    """Reproduce /etc/profile's reset, then source the real drop-in as
    /etc/profile.d sourcing would, and confirm the venv dirs lead PATH."""
    script = (
        f'PATH="{DEBIAN_NONROOT_PATH}"\n'
        f'. "{DROPIN}"\n'
        'printf "%s" "$PATH"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    entries = result.stdout.split(":")
    # The privilege-drop shim dir leads, the venv is present, and both sit ahead
    # of the system interpreter dir that the reset left in front.
    assert entries[0] == "/opt/hermes/bin"
    assert "/opt/hermes/.venv/bin" in entries
    assert entries.index("/opt/hermes/.venv/bin") < entries.index("/usr/bin")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_dropin_is_idempotent_when_resourced() -> None:
    """Re-sourcing the drop-in (the profile chain can run more than once in a
    session) must not duplicate the venv entries or unboundedly grow PATH."""
    script = (
        f'PATH="{DEBIAN_NONROOT_PATH}"\n'
        f'. "{DROPIN}"\n'
        f'. "{DROPIN}"\n'
        'printf "%s" "$PATH"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(":").count("/opt/hermes/.venv/bin") == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_dropin_repairs_venv_present_but_not_leading() -> None:
    """If the venv is on PATH but NOT in front (so a bare ``python3`` still
    resolves to the system interpreter), the drop-in must still prepend the full
    prefix — the guard keys on the prefix leading PATH, not on mere presence."""
    script = (
        'PATH="/usr/bin:/opt/hermes/.venv/bin"\n'
        f'. "{DROPIN}"\n'
        'printf "%s" "$PATH"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    entries = result.stdout.split(":")
    assert entries[0] == "/opt/hermes/bin"
    assert entries.index("/opt/hermes/.venv/bin") < entries.index("/usr/bin")
