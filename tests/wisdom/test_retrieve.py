from __future__ import annotations

from wisdom.apply import create_application_proposals
from wisdom.capture import capture_text
from wisdom.interpret import interpret_capture
from wisdom.retrieve import get_original, inbox, search


def test_inbox_and_original_retrieval(wisdom_db, wisdom_config):
    outcome = capture_text("Health note: poor sleep changes decision quality.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    assert inbox(wisdom_db, limit=5)[0].id == capture.id
    assert get_original(wisdom_db, capture.id) == "Health note: poor sleep changes decision quality."


def test_search_finds_original_interpretation_and_application_text(wisdom_db, wisdom_config):
    outcome = capture_text("Life thought: courage is a habit.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    interpret_capture(wisdom_db, capture.id)
    create_application_proposals(wisdom_db, capture.id)
    assert search(wisdom_db, "courage", limit=5)[0].id == capture.id
    assert search(wisdom_db, "lightweight interpretation", limit=5)[0].id == capture.id
    assert search(wisdom_db, "writing seed", limit=5)[0].id == capture.id
