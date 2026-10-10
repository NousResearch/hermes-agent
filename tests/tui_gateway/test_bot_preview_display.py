"""Bot previews select user-visible text before flattening and truncating."""

from agent.prompt_builder import steer_user_row
from agent.first_task_prompt import MARKER
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from hermes_state import SessionDB
from tui_gateway.methods_profiles import _latest_message_preview


def test_preview_unwraps_steering_and_skips_consecutive_hidden_rows(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        db.append_messages_batch("bot", [steer_user_row("Please focus on the delivery plan.")])
        assert _latest_message_preview(db, "bot") == "Please focus on the delivery plan."
        db.append_messages_batch("bot", [
            {"role": "user", "content": f"internal heartbeat {i}", "display_kind": "hidden"}
            for i in range(70)
        ])
        assert _latest_message_preview(db, "bot") == "Please focus on the delivery plan."
    finally:
        db.close()


def test_preview_decodes_multimodal_and_keeps_untyped_marker_text(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        db.append_message("bot", "user", [{"type": "text", "text": "Look here"},
                                          {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}}])
        assert _latest_message_preview(db, "bot") == "Look here [image]"
        db.append_message("bot", "user", "[System: a literal marker I am asking you about]")
        assert _latest_message_preview(db, "bot") == "[System: a literal marker I am asking you about]"
    finally:
        db.close()


def test_preview_skips_notices_inactive_empty_and_model_only_without_a_row_cap(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        db.append_message("bot", "assistant", "The last real reply.")
        db.append_messages_batch("bot", [
            {"role": "user", "content": f"notice {i}", "display_kind": "process_complete",
             "display_metadata": {"display_text": "Completed"}} for i in range(70)
        ])
        db.append_message("bot", "user", "private model row", display_metadata={"model_only": True})
        db.append_message("bot", "assistant", "   ")
        inactive = db.append_message("bot", "user", "inactive request")
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (inactive,)))
        assert _latest_message_preview(db, "bot") == "The last real reply."
        db.append_messages_batch("bot", [steer_user_row("[System: please explain this]")])
        assert _latest_message_preview(db, "bot") == "[System: please explain this]"
        db.append_messages_batch("bot", [steer_user_row("word " * 30)])
        expected = " ".join(("word " * 30).split())
        assert _latest_message_preview(db, "bot") == expected[:80] + "..."
    finally:
        db.close()


def test_empty_and_all_hidden_conversation_has_no_preview(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        assert _latest_message_preview(db, "bot") == ""
        db.append_message("bot", "user", "snapshot", display_kind="hidden")
        db.append_message("bot", "user", "   ")
        assert _latest_message_preview(db, "bot") == ""
        db.append_messages_batch("bot", [{"role": "user", "content": f"{SUMMARY_PREFIX}old\n{_SUMMARY_END_MARKER}",
                                          COMPRESSED_SUMMARY_METADATA_KEY: True}])
        assert _latest_message_preview(db, "bot") == ""
    finally:
        db.close()


def test_empty_projected_steer_and_first_task_only_tail_leave_previous_preview(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        db.append_message("bot", "assistant", "The last real reply.")
        db.append_messages_batch("bot", [steer_user_row("   ")])
        assert _latest_message_preview(db, "bot") == "The last real reply."
        db.append_message("bot", "user", MARKER + "model-only first-task body")
        assert _latest_message_preview(db, "bot") == "The last real reply."
    finally:
        db.close()
