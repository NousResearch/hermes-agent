"""REST pages, resumed history and roster previews share typed display semantics."""

from copy import deepcopy

import pytest

from agent.first_task_prompt import MARKER

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY, HISTORICAL_TASK_HEADING, SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
)
from agent.prompt_builder import steer_user_row
from hermes_cli.web_routers.sessions import _project_for_display
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.methods_profiles import _latest_message_preview


def test_persisted_producer_rows_agree_across_transports_and_keep_identity(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        rows = [
            {"role": "user", "content": "question", "message_uid": "first"},
            {**steer_user_row("[System: please explain this]"), "message_uid": "steer"},
            {"role": "user", "content": "model-only heartbeat", "display_kind": "hidden"},
            {"role": "user", "content": "internal completion with full results", "display_kind": "process_complete",
             "display_metadata": {"display_text": "Background task finished."}},
        ]
        db.append_messages_batch("bot", rows)
        stored = db.get_messages("bot")
        before = deepcopy(stored)
        rest = _project_for_display(stored)
        gateway = server._history_to_messages(db.get_messages_as_conversation("bot", repair_alternation=False, include_row_ids=True))
        assert [m["id"] for m in rest] == [m["id"] for m in stored]
        assert [m["content"] for m in rest] == [m["content"] for m in stored]
        assert rest[1]["display_content"] == "[System: please explain this]"
        assert rest[2]["display_kind"] == "hidden"
        assert rest[3]["display_metadata"]["display_text"] == "Background task finished."
        assert [m["text"] for m in gateway] == ["question", "[System: please explain this]", "internal completion with full results"]
        assert gateway[1]["row_id"] == rest[1]["id"]
        assert stored == before
        assert _latest_message_preview(db, "bot") == "[System: please explain this]"
    finally:
        db.close()


def test_direct_completion_dispatch_persists_the_same_kind_as_batches(monkeypatch, tmp_path):
    event = {"type": "completion", "session_id": "p1", "exit_code": 0, "command": "printf hello"}
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
    submitted = []
    monkeypatch.setattr(server, "_notif_submit", lambda *a, **kw: submitted.append((a, kw)))
    server._notif_dispatch_event("sid", {}, event, "model-facing completion")
    args, kwargs = submitted[0]
    assert args[3] == "model-facing completion"
    assert kwargs["display_kind"] == "process_complete"
    assert kwargs["display_metadata"]["display_text"]
    from types import SimpleNamespace
    from agent.session_persistence import _db_flush_row

    durable = _db_flush_row(SimpleNamespace(), {"role": "user", "content": args[3], **kwargs}, False)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("bot", "cli")
        db.append_messages_batch("bot", [durable])
        stored = db.get_messages("bot")[0]
        assert stored["display_kind"] == "process_complete"
        assert stored["display_metadata"]["display_text"] == kwargs["display_metadata"]["display_text"]
        assert server._history_to_messages([stored])[0]["text"] == args[3]
        assert _latest_message_preview(db, "bot") == ""
    finally:
        db.close()


@pytest.mark.parametrize("text", ["[System: explain this]", "literal" + MARKER + "quoted marker", "[Triggering message id: `42` — use as `message_id` for reply/react/pin via the discord tools.]\nactual words"])
def test_typed_steering_is_authoritative_and_keeps_quoted_producer_markers(text):
    row = steer_user_row(text)
    assert server._history_to_messages([row])[0]["text"] == text
    assert _project_for_display([row])[0]["display_content"] == text


def test_history_to_messages_drops_pure_compaction_scaffolding():

    summary = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {"role": "user", "content": summary, COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "assistant", "content": "real answer"},
        ]
    ) == [{"role": "assistant", "text": "real answer"}]

def test_history_to_messages_preserves_live_ask_without_compaction_scaffolding():

    carrier = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}\n\n"
        "test the browser controller"
    )

    assert server._history_to_messages(
        [
            {
                "role": "user",
                "content": carrier,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                "tool_calls": [{"id": "stale"}],
                "reasoning": "internal compaction reasoning",
            }
        ]
    ) == [{"role": "user", "text": "test the browser controller"}]

def test_history_to_messages_unwraps_merged_assistant_carrier():

    carrier = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
        "real completed answer\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {
                "role": "assistant",
                "content": carrier,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                "tool_calls": [{"id": "stale"}],
                "reasoning_details": [{"summary": "internal"}],
            }
        ]
    ) == [{"role": "assistant", "text": "real completed answer"}]
