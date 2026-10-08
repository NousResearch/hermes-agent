"""A stopped child is stored and does not start the next parent turn."""

from tools.delegate_resume_policy import (
    completion_should_autoresume, hold_instead_of_turn, release_delegation_hold,
    session_blocks_synthetic_turn,
)


def test_interrupted_child_does_not_autoresume_and_a_real_prompt_clears_the_hold():
    stopped = {"type": "async_delegation", "status": "error", "results": [{"status": "interrupted"}]}
    assert completion_should_autoresume(stopped) is False
    assert hold_instead_of_turn(stopped, {}) is True
    single = {"type": "async_delegation", "status": "interrupted"}
    assert completion_should_autoresume(single, {}) is False
    mixed = {"type": "async_delegation", "status": "completed", "results": [
        {"status": "completed"}, {"status": "interrupted"}]}
    assert completion_should_autoresume(mixed) is True
    done = {"type": "async_delegation", "status": "completed", "results": [{"status": "completed"}]}
    assert completion_should_autoresume(done) is True
    paused = {"_turn_cancel_requested": True}
    assert completion_should_autoresume(done, paused) is False
    session = {"_delegation_hold": True, "_delegation_held_events": [done]}
    assert completion_should_autoresume(done, session) is False
    released = release_delegation_hold(session)
    assert released == [done]
    assert "_delegation_hold" not in session
    assert "_delegation_held_events" not in session
    assert released[0]["_released_by_user_turn"] is True
    assert completion_should_autoresume(done, session) is True


def test_interrupted_result_releases_only_after_real_user_turn():
    event = {"type": "async_delegation", "status": "interrupted"}
    session = {"_delegation_hold": True, "_delegation_held_events": [event]}
    assert hold_instead_of_turn(event, session) is True
    released = release_delegation_hold(session)
    assert released == [event]
    assert hold_instead_of_turn(event, session) is False
    session["_turn_cancel_requested"] = True
    assert hold_instead_of_turn(event, session) is True


def test_process_completion_does_not_autoresume_a_stopped_parent():
    event = {"type": "completion", "session_id": "proc_510149f07f72"}
    assert completion_should_autoresume(event, {}) is True
    stopped = {"_turn_cancel_requested": True}
    assert session_blocks_synthetic_turn(stopped) is True
    assert completion_should_autoresume(event, stopped) is False
    assert hold_instead_of_turn(event, stopped) is False
    assert session_blocks_synthetic_turn({"_closing": True}) is True
    assert session_blocks_synthetic_turn({"_delegation_hold": True}) is True
    assert session_blocks_synthetic_turn({}) is False
