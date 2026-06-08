from __future__ import annotations

from wisdom.capture import capture_text
from wisdom.interpret import interpret_capture


def test_deterministic_interpretation_created_explicitly(wisdom_db, wisdom_config):
    outcome = capture_text("Business idea: clients buy calm.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    interpretation = interpret_capture(wisdom_db, capture.id)
    assert interpretation is not None
    assert interpretation.method == "deterministic"
    assert interpretation.counterpoint
    assert interpretation.confidence <= 0.6
    assert wisdom_db.get_capture(capture.id).original_text == "Business idea: clients buy calm."


def test_interpret_missing_capture_returns_none(wisdom_db):
    assert interpret_capture(wisdom_db, 999) is None


def test_interpretation_is_idempotent(wisdom_db, wisdom_config):
    outcome = capture_text("Note this: one clear note.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    first = interpret_capture(wisdom_db, capture.id)
    second = interpret_capture(wisdom_db, capture.id)
    assert first is not None and second is not None
    assert first.id == second.id
