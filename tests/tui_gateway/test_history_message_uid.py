"""The resume projection carries each row's ``message_uid`` beside its ``row_id``.

Desktop pairs a live part with its stored copy by the uid a compaction keeps (agent/live_row_identity.py),
so a session opened from ``session.resume`` rows needs it as much as one read from the REST page."""
from tests.fixtures.storage_read import conversation, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle

from persistence.conversation.messages import ConversationRequest
import tui_gateway.server as srv


def test_resume_rows_carry_the_stored_uid(tmp_path):
    db = storage_handle(tmp_path)
    seed_session(db, "s", "test")
    seed_messages(db, "s", {"role": "user", "content": "look"}, {"role": "assistant", "content": "looked"})

    stored = transcript_rows(db, "s")
    projected = srv._history_to_messages(  # server.py-bound
        conversation(db, "s", ConversationRequest(include_row_ids=True)))

    assert [(m["row_id"], m["message_uid"]) for m in projected] == [(r["id"], r["message_uid"]) for r in stored]
    assert all(isinstance(m["message_uid"], str) and m["message_uid"] for m in projected)
