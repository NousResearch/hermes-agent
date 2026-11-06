"""A whole-tree /rollback after /undo never rewinds a kept turn that falls inside the checkpoint's own git second."""
from tools.checkpoint_undo_hint import rollback_rewinds_newest_turn


def test_a_kept_turn_in_the_checkpoints_own_second_is_not_rewound():
    """Git keeps whole seconds: a kept turn at T+0.2 and a checkpoint made at T+0.5 both read as second T."""
    kept = [{"role": "user", "timestamp": 1700000000.1}, {"role": "assistant", "timestamp": 1700000000.2}]
    assert rollback_rewinds_newest_turn(kept, [0], 1700000000.0) is False
    assert rollback_rewinds_newest_turn(kept, [0], 1699999999.0) is True  # the turn followed the checkpoint
