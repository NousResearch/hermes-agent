"""Contract test: stage2-hook repairs ownership of the gateway install tree.

When HERMES_UID is remapped at container boot, ``usermod -u`` only rewrites
files under the hermes user's home directory ($HERMES_HOME == /opt/data).
Runtime-writable trees under ``/opt/hermes`` must be explicitly chowned to the
new UID before services drop privileges. ``/opt/hermes/gateway`` is one such
tree: Python writes ``__pycache__`` beneath the package on first import, which
fails with EACCES if the tree still belongs to the build-time UID (10000) after
a remap (#27221).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _install_dir_chown_block(text: str) -> str:
    match = re.search(
        r"(chown -R hermes:hermes \\\n"
        r"(?:\s+\"\$INSTALL_DIR/[^\"]+\" \\\n)+"
        r"\s+2>/dev/null \|\| \\\n"
        r"\s+echo \"\[stage2\] Warning: chown of build trees failed.*?\")",
        text,
        flags=re.DOTALL,
    )
    assert match, "stage2-hook.sh must repair ownership of runtime-writable install trees"
    return match.group(1)


def test_uid_remap_chowns_runtime_writable_gateway_tree(stage2_text: str) -> None:
    block = _install_dir_chown_block(stage2_text)
    assert '"$INSTALL_DIR/gateway"' in block, (
        "the build-tree ownership repair must chown $INSTALL_DIR/gateway so the "
        "gateway runtime can write Python cache artifacts after a UID remap (#27221)"
    )


def test_install_dir_chown_keeps_existing_runtime_writable_trees(stage2_text: str) -> None:
    block = _install_dir_chown_block(stage2_text)
    for required in (
        '"$INSTALL_DIR/.venv"',
        '"$INSTALL_DIR/ui-tui"',
        '"$INSTALL_DIR/node_modules"',
    ):
        assert required in block
