from __future__ import annotations

from wisdom.capture import capture_text


def test_capture_text_creates_active_capture(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Investing thought: position sizing matters more than thesis confidence.",
        config=wisdom_config,
        db=wisdom_db,
    )
    assert outcome.status == "captured"
    assert outcome.capture is not None
    assert outcome.capture.category == "investing"
    assert outcome.capture.source_type == "thought"
    assert outcome.capture.status == "active"


def test_off_setting_disables_capture(wisdom_db, wisdom_config):
    wisdom_db.set_setting("enabled", "false")
    outcome = capture_text("Remember this: no write", config=wisdom_config, db=wisdom_db)
    assert outcome.status == "disabled"
    assert wisdom_db.counts()["captures"] == 0


def test_secret_capture_blocked(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Remember this: Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        config=wisdom_config,
        db=wisdom_db,
    )
    assert outcome.status == "blocked_secret"
    assert wisdom_db.counts()["raw_events"] == 0


def test_secret_context_note_blocked(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Remember this: harmless note",
        context_note="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        config=wisdom_config,
        db=wisdom_db,
    )
    assert outcome.status == "blocked_secret"
    assert wisdom_db.counts()["raw_events"] == 0
