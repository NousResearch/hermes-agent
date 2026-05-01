"""Contract tests for Docker entrypoint volume bootstrapping."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"


@pytest.fixture(scope="module")
def entrypoint_text() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("Docker entrypoint script not present in this checkout")
    return ENTRYPOINT.read_text()


def test_entrypoint_exposes_hermes_on_volume_local_bin(entrypoint_text: str):
    """Fresh Docker volumes must make `hermes` available to later exec shells.

    The image PATH includes ``$HERMES_HOME/.local/bin`` so that commands run via
    ``docker compose exec ... bash`` can find user-local launchers.  A fresh
    mounted volume starts empty, so the entrypoint must create that directory
    and install a ``hermes`` launcher there; otherwise `which hermes` returns
    nothing in later Docker terminal sessions even though the app started.
    """
    assert '"$HERMES_HOME/.local/bin"' in entrypoint_text
    assert '"$HERMES_HOME/.local/bin/hermes"' in entrypoint_text
    assert '"${INSTALL_DIR}/.venv/bin/hermes"' in entrypoint_text
