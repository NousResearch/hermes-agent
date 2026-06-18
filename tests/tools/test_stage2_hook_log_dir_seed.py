"""Contract test: the s6-overlay stage2 hook seeds $HERMES_HOME/logs/gateways
as the hermes user.

Regression guard for #45258: the per-profile gateway log service
(`gateway-<profile>/log/run`) creates `logs/gateways/` via `mkdir -p` but only
chowns the leaf `logs/gateways/<profile>`. If the first log service to boot
runs in root context, the `gateways/` parent is created root-owned and stays
that way; every profile registered later runs its log service as the dropped
hermes user and s6-log crash-loops on `mkdir: Permission denied`.

Seeding `logs/gateways` in stage2 (cont-init runs before any service starts)
guarantees the parent already exists hermes-owned by the time the first
log/run executes its `mkdir -p`.
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


def _seed_mkdir_block(text: str) -> str:
    """Extract the `as_hermes mkdir -p \\ ...` seed block."""
    m = re.search(r"as_hermes mkdir -p \\\n(?:[^\n]*\\\n)*[^\n]*\n", text)
    assert m, "stage2-hook.sh must contain the as_hermes mkdir -p seed block"
    return m.group(0)


def test_logs_gateways_is_seeded(stage2_text: str) -> None:
    block = _seed_mkdir_block(stage2_text)
    assert '"$HERMES_HOME/logs/gateways"' in block, (
        "logs/gateways must be seeded hermes-owned in stage2 so profiles "
        "added after first boot can create their log dirs (#45258)"
    )
    # The parent must also be seeded so mkdir -p inside the block never
    # creates logs/ implicitly with surprising ownership.
    assert '"$HERMES_HOME/logs"' in block


def test_logs_subtree_is_healed_when_chown_needed(stage2_text: str) -> None:
    """The needs_chown repair loop must cover the logs subtree recursively —
    that is what makes the seed entry above sufficient (no separate
    logs/gateways loop entry needed)."""
    m = re.search(r"for sub in ([^;]*); do", stage2_text)
    assert m, "stage2-hook.sh must contain the needs_chown subdir repair loop"
    assert "logs" in m.group(1).split(), (
        "the needs_chown loop must recursively chown logs/ — it covers "
        "logs/gateways, so the seed list does not need a loop twin"
    )
