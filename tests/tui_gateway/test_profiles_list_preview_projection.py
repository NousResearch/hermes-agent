"""Tests: profiles.list roster previews project rows like the transcript adapters.

Why: ``_latest_message_preview`` used to truncate the newest RAW active user/assistant row, so a
mid-turn steer previewed the ``[OUT-OF-BAND USER MESSAGE — ...]`` marker instead of the user's
words, hidden / untyped ``[System:`` / model-only rows replaced the excerpt with their internal
bodies, and structured multimodal content leaked its storage encoding (#136219). The roster now
reads a bounded newest page through ``get_messages`` (storage owns ordering and decoding) and
reuses the same read-side shaping as ``session.resume`` and the REST history projection.

Contract under test:
- A steer row previews the user's own words (the issue's repro script).
- hidden / untyped ``[System:`` / model-only rows are skipped, and several consecutive invisible
  rows do not prevent selecting the preceding visible conversational message.
- Structured (multimodal) content previews its text parts, never the stored encoding.
- Plain text and the 80-char truncation are unchanged; a page with no visible row yields "".
"""

from __future__ import annotations

from agent.prompt_builder import steer_user_row
from tui_gateway.methods_profiles import _latest_message_preview


def _db(tmp_path):
    from hermes_state import SessionDB

    return SessionDB(db_path=tmp_path / "state.db")


def test_steer_row_previews_the_users_words(tmp_path):
    db = _db(tmp_path)
    db.create_session("demo", "cli")
    db.append_messages_batch(
        "demo", [steer_user_row("Please focus on the delivery plan.")]
    )
    assert _latest_message_preview(db, "demo") == "Please focus on the delivery plan."


def test_hidden_system_and_model_only_rows_fall_through_to_visible_text(tmp_path):
    db = _db(tmp_path)
    db.create_session("demo", "cli")
    db.append_message("demo", "user", "What changed in the release?")
    db.append_message("demo", "user", "[System: model switched to glm-5.3]")
    db.append_message(
        "demo", "user", "internal scaffolding body", display_kind="hidden"
    )
    db.append_message(
        "demo", "user", "merged model-only row", display_metadata={"model_only": True}
    )
    assert _latest_message_preview(db, "demo") == "What changed in the release?"


def test_structured_multimodal_content_previews_text_not_encoding(tmp_path):
    db = _db(tmp_path)
    db.create_session("demo", "cli")
    db.append_message(
        "demo",
        "user",
        [
            {"type": "text", "text": "What is on this label?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8gd29ybGQ="},
            },
        ],
    )
    preview = _latest_message_preview(db, "demo")
    assert "What is on this label?" in preview
    assert "base64" not in preview and "aGVsbG8" not in preview


def test_plain_text_preview_and_truncation_are_unchanged(tmp_path):
    db = _db(tmp_path)
    db.create_session("demo", "cli")
    db.append_message("demo", "user", "short question")
    db.append_message("demo", "assistant", "a definitive answer")
    assert _latest_message_preview(db, "demo") == "a definitive answer"
    db.append_message("demo", "user", " ".join(["word"] * 30))
    preview = _latest_message_preview(db, "demo")
    assert len(preview) == 83 and preview.endswith("...")


def test_no_visible_row_yields_empty_preview(tmp_path):
    db = _db(tmp_path)
    db.create_session("demo", "cli")
    db.append_message("demo", "user", "gone", display_kind="hidden")
    db.append_message("demo", "user", "[System: archived notice]")
    assert _latest_message_preview(db, "demo") == ""
