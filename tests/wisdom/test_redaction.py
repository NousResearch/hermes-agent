from __future__ import annotations

import json

from wisdom.capture import capture_text
from wisdom.redaction import detect_secret_like_text, ensure_salt, redact_for_log, stable_hash


def test_detects_secret_like_text_without_over_redacting_normal_thoughts(wisdom_home):
    assert detect_secret_like_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    assert detect_secret_like_text("password = verysecretvalue")
    assert not detect_secret_like_text("clients buy peace of mind, not alpha")


def test_redact_for_log_masks_secret():
    redacted = redact_for_log("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "[REDACTED]" in redacted


def test_stable_hash_uses_local_salt(wisdom_home):
    salt = ensure_salt()
    first = stable_hash("chat-123", salt=salt, prefix="sess_")
    second = stable_hash("chat-123", salt=salt, prefix="sess_")
    assert first == second
    assert first != "chat-123"
    salt_path = wisdom_home / "wisdom" / "salt"
    assert salt_path.exists()


def test_raw_platform_ids_are_not_stored_in_metadata(wisdom_db, wisdom_config):
    outcome = capture_text(
        "Remember this: metadata should not store raw ids.",
        config=wisdom_config,
        db=wisdom_db,
        session_key="raw-chat-id",
        message_ref="raw-message-id",
        metadata={"chat_id": "raw-chat-id", "user_id": "raw-user-id", "safe": "ok"},
    )
    assert outcome.status == "captured"
    row = wisdom_db.conn.execute("SELECT * FROM raw_events").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata == {"safe": "ok", "trigger": "remember this:"}
    assert row["session_key_hash"] != "raw-chat-id"
    assert row["message_ref_hash"] != "raw-message-id"
