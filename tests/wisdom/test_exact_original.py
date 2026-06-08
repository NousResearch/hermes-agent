from __future__ import annotations

from wisdom.capture import capture_text
from wisdom.commands import WisdomCommandContext, handle_wisdom_command


def test_original_text_round_trips_exactly(wisdom_db, wisdom_config):
    original = "  Remember this: clients don't buy alpha.\nThey buy calm.  "
    outcome = capture_text(original, config=wisdom_config, db=wisdom_db)
    assert outcome.status == "captured"
    capture = outcome.capture
    assert capture is not None
    assert wisdom_db.get_capture(capture.id).original_text == original
    assert wisdom_db.get_capture(capture.id).cleaned_text != original


def test_wisdom_original_command_returns_exact_original(wisdom_db, wisdom_config):
    original = "Clients need windshields, not rear-view mirrors."
    response = handle_wisdom_command(
        f"capture {original}",
        context=WisdomCommandContext(channel="test"),
        config=wisdom_config,
        db=wisdom_db,
    )
    assert response.startswith("Captured #")
    assert handle_wisdom_command("original 1", config=wisdom_config, db=wisdom_db) == original


def test_secret_like_capture_is_blocked_not_stored(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Remember this: api_key = sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        config=wisdom_config,
        db=wisdom_db,
    )
    assert outcome.status == "blocked_secret"
    assert wisdom_db.counts()["captures"] == 0
