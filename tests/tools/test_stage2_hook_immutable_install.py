"""Contract tests for the Docker stage2 immutable install-tree policy.

Hosted/container Hermes keeps user-writable state under HERMES_HOME
(/opt/data). The installed source, venv, TUI bundle, and node_modules under
/opt/hermes must remain root-owned/non-writable by the runtime hermes user so
an agent session cannot self-modify the installation and brick the gateway.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def test_stage2_does_not_chown_install_tree_to_hermes(stage2_text: str) -> None:
    assert "Fixing ownership of build trees under $INSTALL_DIR" not in stage2_text
    assert 'chown -R hermes:hermes \\\n        "$INSTALL_DIR/.venv"' not in stage2_text

    assert "venv_owner=$(stat -c %u \"$INSTALL_DIR/.venv\"" not in stage2_text
    assert "chown of build trees failed" not in stage2_text
    for install_tree in (
        '"$INSTALL_DIR/.venv" \\',
        '"$INSTALL_DIR/ui-tui" \\',
        '"$INSTALL_DIR/gateway" \\',
        '"$INSTALL_DIR/node_modules" \\',
    ):
        assert install_tree not in stage2_text, (
            f"stage2 must not chown {install_tree} back to hermes; "
            "the Dockerfile keeps /opt/hermes immutable and writable state "
            "belongs under HERMES_HOME"
        )


def test_stage2_documents_immutable_install_contract(stage2_text: str) -> None:
    assert "Immutable install tree" in stage2_text
    assert "PYTHONDONTWRITEBYTECODE" in stage2_text
    assert "HERMES_DISABLE_LAZY_INSTALLS=1" in stage2_text
    assert "/opt/hermes" in stage2_text
