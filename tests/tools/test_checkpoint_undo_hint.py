"""The /undo pointer's store-level contract: read-only over the store, hash refs every /rollback reads, and a
kept turn inside the checkpoint's own second is never rewound."""
from __future__ import annotations

import time

import pytest

import tools.checkpoint_manager as cm
from tools.checkpoint_undo_hint import rollback_rewinds_newest_turn, undo_rollback_hint

try:  # tolerant so each test, not collection, is what goes red on a tree without the shared resolver
    from tools.checkpoint_refs import resolve_checkpoint_ref
except ImportError:
    resolve_checkpoint_ref = None


def test_the_hint_never_stages_the_store_into_itself(tmp_path, monkeypatch):
    """The project IS the directory holding the store (HERMES_HOME as cwd): repeated hints add nothing to it."""
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    (tmp_path / "a.txt").write_text("original\n")
    mgr = cm.CheckpointManager(enabled=True, max_total_size_mb=0)
    since = time.time()
    assert mgr.ensure_checkpoint(str(tmp_path), "before edit")
    (tmp_path / "a.txt").write_text("edited\n")
    sizes, hints = [cm._dir_size_bytes(cm._store_path())], []
    for _ in range(3):
        hints.append(undo_rollback_hint(mgr, str(tmp_path), since, task_id=None))
        sizes.append(cm._dir_size_bytes(cm._store_path()))
    assert all(h is not None and h.files == ["a.txt"] for h in hints), hints
    assert len(set(sizes)) == 1, sizes


def test_a_kept_turn_in_the_checkpoints_own_second_is_not_rewound():
    """Git keeps whole seconds: a kept turn at T+0.2 and a checkpoint made at T+0.5 both read as second T."""
    kept = [{"role": "user", "timestamp": 1700000000.1}, {"role": "assistant", "timestamp": 1700000000.2}]
    assert rollback_rewinds_newest_turn(kept, [0], (1700000000.0, False)) is False
    assert rollback_rewinds_newest_turn(kept, [0], (1700000000.5, True)) is False
    assert rollback_rewinds_newest_turn(kept, [0], (1700000000.15, True)) is True  # that turn made it


@pytest.mark.parametrize("ref, expected", [("2", "bbbb"), ("1234", "1234"), ("0042", "0042"), ("3f9c2a1b", "3f9c2a1b")])
def test_a_rollback_argument_is_a_list_number_only_while_it_is_one(ref, expected):
    listed = [{"hash": "aaaa"}, {"hash": "bbbb"}]
    assert resolve_checkpoint_ref(listed, ref) == expected


def test_a_short_out_of_range_number_stays_an_error():
    with pytest.raises(ValueError, match="Invalid checkpoint number"):
        resolve_checkpoint_ref([{"hash": "aaaa"}], "9")
