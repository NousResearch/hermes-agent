"""A branch copies provider metadata from the lineage without its display fence."""

import threading

from hermes_state import SessionDB
from tui_gateway import server


def test_branch_lineage_retains_raw_provider_reasoning(tmp_path):
    raw = "<memory-context>PRIVATE_BRANCH_REASONING</memory-context>"
    details = [{"type": "reasoning.text", "text": raw, "signature": raw}]
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="tui")
        db.append_message("parent", "user", content="question")
        db.append_message("parent", "assistant", content="answer", reasoning_details=details,
                          api_content=raw + "answer")
        model, display = db.get_resume_conversations("parent")
        assert model[-1]["reasoning_details"] == details
        assert "PRIVATE_BRANCH_REASONING" not in repr(display[-1]["reasoning_details"])
        # UI projection drops raw replay sidecars; the internal transcript keeps them.
        assert "api_content" not in repr(server._history_to_messages(display))
        branch = server._branch_source_history(
            db, {"history_lock": threading.RLock(), "history": []}, "parent")
        assert branch[-1]["reasoning_details"] == details
        assert branch[-1]["api_content"] == raw + "answer"
