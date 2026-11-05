"""``/undo`` puts back the files the undone turns wrote (tools/checkpoint_turns.py).

Real CheckpointManager + real git store under a temp HERMES_HOME; the turn boundary is the same
``begin_turn(user_ts, session)`` call the agent loop makes, the writes go through the same
``ensure_checkpoint`` (before the write) and ``record_agent_write`` (after it) the file tools use.
"""
from pathlib import Path

import pytest

from tools import checkpoint_manager as cm
from tools.checkpoint_turns import revert_turns


@pytest.fixture()
def project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", home / "checkpoints")
    work = tmp_path / "proj"
    work.mkdir()
    (work / "keep.txt").write_text("original\n")
    return work


def _agent_write(mgr, work: Path, name: str, text: str) -> Path:
    mgr.ensure_checkpoint(str(work), "before write_file")
    path = work / name
    path.write_text(text)
    mgr.record_agent_write(str(path))
    return path


def _turn(mgr, ts: float, session: str = "s1") -> None:
    mgr.new_turn()
    mgr.begin_turn(ts, session)


def test_undo_reverts_edits_and_creations_of_the_undone_turns_only(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "keep.txt", "turn1\n")
    _turn(mgr, 200.0)
    _agent_write(mgr, project, "keep.txt", "turn2\n")
    created = _agent_write(mgr, project, "new.txt", "made in turn 2\n")
    _turn(mgr, 300.0)
    _agent_write(mgr, project, "third.txt", "turn3\n")

    # /undo 2 from the newest turn: back to before turn 2 (user row ts 200).
    result = revert_turns(mgr, 200.0, {"s1"})

    assert (project / "keep.txt").read_text() == "turn1\n"  # turn 1's write survives
    assert not created.exists() and not (project / "third.txt").exists()
    assert sorted(Path(p).name for p in result["reverted"]) == ["keep.txt", "new.txt", "third.txt"]
    # Undoing turn 1 afterwards still reverts its write: the ledger handed turn 1 back.
    assert [Path(p).name for p in revert_turns(mgr, 100.0, {"s1"})["reverted"]] == ["keep.txt"]
    assert (project / "keep.txt").read_text() == "original\n"


def test_undo_keeps_hand_edits_other_conversations_and_turns_without_edits(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0, "other-conversation")
    _agent_write(mgr, project, "theirs.txt", "other session\n")
    _turn(mgr, 200.0)
    hand = _agent_write(mgr, project, "keep.txt", "agent\n")
    hand.write_text("user typed this\n")  # a hand edit after Hermes' write
    _turn(mgr, 300.0)  # a turn that wrote nothing

    assert revert_turns(mgr, 300.0, {"s1"})["reverted"] == []  # nothing to revert, nothing touched
    result = revert_turns(mgr, 200.0, {"s1"})

    assert hand.read_text() == "user typed this\n"
    assert result["skipped_user_edits"] == [str(hand)]
    assert (project / "theirs.txt").read_text() == "other session\n"
    assert revert_turns(cm.CheckpointManager(enabled=False), 100.0, {"other-conversation"})["reverted"] == []
