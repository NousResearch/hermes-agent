"""A checkpoint project must never be a directory whose snapshot cannot succeed.

Two related failure modes repeat on every turn and turned into a standing health alert:

* the temp root itself becomes a project — anything written under ``/tmp`` without a project
  marker of its own resolves to ``/tmp`` (see ``get_working_dir_for_path``), so the store
  starts ingesting browser caches and root-owned system residue;
* a workdir that *contains* root-owned temp residue aborts ``git add -A`` with
  ``fatal: adding files failed`` / exit 128, because the agent user cannot read those entries.  The
  snapshot never succeeds, so the error repeats until the store is manually cleared.

Invariants asserted here: temp roots are never snapshot targets, their children still are,
and a workdir holding unreadable residue checkpoints cleanly.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.checkpoint_manager import CheckpointManager


@pytest.fixture()
def checkpoint_base(tmp_path):
    """Isolated checkpoint base — never writes to ~/.hermes/."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def mgr(checkpoint_base, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=True, max_snapshots=50)


class TestTempRootsAreNotSnapshotTargets:
    def test_temp_root_itself_is_skipped_but_children_stay_eligible(self, mgr, tmp_path):
        temp_root = str(Path(tempfile.gettempdir()).resolve())
        with patch.object(CheckpointManager, "_take", autospec=True, return_value=True) as take:
            assert mgr.ensure_checkpoint(temp_root, "temp root") is False
            assert take.call_count == 0, "temp root must be rejected before any snapshot work"

        # pytest tmp_path projects live *below* the temp root and must keep checkpointing.
        project = tmp_path / "project_below_temp_root"
        project.mkdir()
        (project / "main.py").write_text("print('hello')\n")
        with patch.object(CheckpointManager, "_take", autospec=True, return_value=True) as take:
            assert mgr.ensure_checkpoint(str(project), "below temp root") is True
            assert take.call_count == 1


class TestUnreadableTempResidue:
    @pytest.mark.platforms("linux")
    def test_workdir_containing_unreadable_residue_still_checkpoints(self, mgr, tmp_path):
        """``snap-private-tmp/``-style residue is unreadable to the agent user; the workdir
        around it must still snapshot instead of failing every turn."""
        project = tmp_path / "project_with_residue"
        project.mkdir()
        (project / "main.py").write_text("print('hello')\n")
        residue = project / "snap-private-tmp"
        residue.mkdir()
        secret = residue / "snapshot"
        secret.write_bytes(b"unreadable\n")
        secret.chmod(0o000)
        try:
            assert mgr.ensure_checkpoint(str(project), "residue present") is True
        finally:
            secret.chmod(0o600)
