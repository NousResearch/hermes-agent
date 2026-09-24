"""``checkpoint_footprint_notice``: tells a user their /rollback store is on and large.

Checkpoints were on by default Mar–May 2026 and that ``enabled: true`` persisted into user
configs after the default flipped back, so a GB-scale store can sit under a feature the user
never invokes. The notice fires only when checkpoints are enabled AND the store is at/over
``max_total_size_mb``; it names the opt-out.
"""

import os

import yaml

from hermes_constants import get_hermes_home
from tools.checkpoint_manager import CheckpointManager, checkpoint_footprint_notice


def _write_config(enabled: bool, cap_mb: int) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"checkpoints": {"enabled": enabled, "max_total_size_mb": cap_mb}}), encoding="utf-8")


def test_notice_only_when_enabled_and_over_cap(tmp_path, monkeypatch):
    base = get_hermes_home() / "checkpoints"
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", base)
    work = tmp_path / "proj"
    work.mkdir()
    (work / "blob.bin").write_bytes(os.urandom(2 * 1024 * 1024))  # incompressible → store > 1 MB
    _write_config(enabled=True, cap_mb=1)
    assert CheckpointManager(enabled=True, max_total_size_mb=1).ensure_checkpoint(str(work), "seed")

    notice = checkpoint_footprint_notice()
    assert notice

    _write_config(enabled=True, cap_mb=500)  # under the cap: no nag for a healthy store
    assert checkpoint_footprint_notice() is None

    _write_config(enabled=False, cap_mb=1)  # off: the store's size is irrelevant
    assert checkpoint_footprint_notice() is None


