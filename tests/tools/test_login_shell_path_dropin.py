"""Behavior tests for the login-shell PATH drop-in (#56634).

Debian's /etc/profile hardcodes PATH for login shells, discarding the image's
``ENV PATH`` *before* it sources /etc/profile.d/*.sh. init_session snapshots the
environment through a login shell (``bash -l -c ... export -p``), so without a
profile.d drop-in the snapshot loses the venv and a bare ``python3`` resolves to
the system interpreter (which lacks Hermes' deps, including Pillow). These tests
reproduce that PATH reset in bash and source the real drop-in the way
/etc/profile.d sourcing would -- no Docker build required. The built-image
login-shell behavior is covered by tests/docker/test_login_shell_venv_path.py.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DROPIN = REPO_ROOT / "docker" / "profile.d" / "99-hermes-venv-path.sh"

# Debian's /etc/profile PATH for a non-root shell, set before profile.d is
# sourced. Reproduced verbatim so the test exercises the exact reset the drop-in
# has to survive.
DEBIAN_NONROOT_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"

# The exact prefix the drop-in prepends: the privilege-drop shim dir, the venv,
# and the data user's local bin -- mirrors the Dockerfile's ``ENV PATH``.
VENV_PREFIX = "/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_dropin_restores_venv_after_debian_profile_reset() -> None:
    """Reproduce /etc/profile's reset, then source the real drop-in as
    /etc/profile.d sourcing would: the full prefix must lead PATH, with the
    reset PATH intact behind it."""
    script = (
        f'PATH="{DEBIAN_NONROOT_PATH}"\n'
        f'. "{DROPIN}"\n'
        'printf "%s" "$PATH"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{VENV_PREFIX}:{DEBIAN_NONROOT_PATH}"


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
    assert result.stdout == f"{VENV_PREFIX}:{DEBIAN_NONROOT_PATH}"


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
    assert result.stdout == f"{VENV_PREFIX}:/usr/bin:/opt/hermes/.venv/bin"
