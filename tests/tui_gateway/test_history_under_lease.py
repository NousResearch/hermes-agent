"""Desktop prepares the exact model projection only after native admission."""
from types import SimpleNamespace

import pytest

from tests.tui_gateway.test_out_of_band_history_adoption import _seed
from tui_gateway.server import _history_after_admission


@pytest.mark.parametrize("change", ["append_after_staging", "edit", "rotate", "unchanged"])
def test_admitted_desktop_projection(tmp_path, change):
    db, session = _seed(tmp_path)
    state = SimpleNamespace(agent=SimpleNamespace(_session_db=db), history=None, history_version=None)
    target = "s1"
    if change == "append_after_staging":
        own = db.append_message("s1", "user", "next")
        session["_submit_user_row"] = {"role": "user", "content": "next", "_row_id": own}
        db.append_message("s1", "user", "external")
        db.append_message("s1", "assistant", "external answer")
    elif change == "edit":
        db.set_user_message_content("s1", session["history"][0]["_row_id"], "corrected")
    elif change == "rotate":
        db.end_session("s1", "compression")
        db.create_session("tip", source="desktop", parent_session_id="s1")
        db.append_message("tip", "user", "compressed")
        target = "tip"
    state.agent._pending_cli_user_message = session.pop("_submit_user_row", None)
    try:
        result = _history_after_admission(session, state, target)
        text = [m["content"] for m in result]
        if change == "append_after_staging":
            assert text == ["My codeword is MANGO.", "OK", "external", "external answer"]
        elif change == "edit":
            assert text[0] == "corrected"
        elif change == "rotate":
            assert text == ["compressed"]
        else:
            assert result[0] is session["history"][0]
        assert state.history == result
        assert state.history_version == session["history_version"]
    finally:
        db.close()


def test_compacted_staged_input_is_not_replayed(tmp_path):
    db, session = _seed(tmp_path)
    own = db.append_message("s1", "user", "next", timestamp=123.5)
    session["_submit_user_row"] = {"role": "user", "content": "next", "timestamp": 123.5, "_row_id": own}
    db.archive_and_compact("s1", [
        {"role": "assistant", "content": "summary", "_compressed_summary": True},
        {"role": "user", "content": "next", "timestamp": 123.5},
    ], tail_count=1)
    state = SimpleNamespace(agent=SimpleNamespace(
        _session_db=db, _pending_cli_user_message=session.pop("_submit_user_row")))
    try:
        result = _history_after_admission(session, state, "s1")
        assert all(m["content"] != "next" for m in result)
    finally:
        db.close()


def test_desktop_keeps_live_image_payload(tmp_path):
    db, session = _seed(tmp_path)
    db.append_message("s1", "user", "inspect image")
    db.append_message("s1", "assistant", None, tool_calls=[{
        "id": "call", "type": "function", "function": {"name": "view", "arguments": "{}"}}])
    image_id = db.append_message("s1", "tool", "[screenshot]", tool_call_id="call")
    db.append_message("s1", "assistant", "image inspected")
    session["history"] = db.get_messages_as_conversation("s1", include_row_ids=True)
    image = next(m for m in session["history"] if m["_row_id"] == image_id)
    image["content"] = [{"type": "image_url", "image_url": {"url": "fixture"}}]
    state = SimpleNamespace(agent=SimpleNamespace(_session_db=db))
    try:
        result = _history_after_admission(session, state, "s1")
        assert any(m is image for m in result)
    finally:
        db.close()
