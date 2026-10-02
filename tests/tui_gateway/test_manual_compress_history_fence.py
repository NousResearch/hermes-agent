"""Manual compression must not publish a snapshot invalidated before lease admission."""
import threading
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("compress_args,stale", [
    ("", False), ("", True), ("", "validation_error"), ("", "empty_summary"),
    ("here 2", True), ("here 2", "validation_error"),
])
def test_manual_compress_rejects_history_rewritten_before_admission(
    tmp_path, monkeypatch, stale, in_place, compress_args,
):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.context_compressor import SUMMARY_PREFIX
    from tui_gateway import server

    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "manual-fence-parent"
    db.create_session(parent, source="cli")
    for i in range(12):
        db.append_message(parent, "user", f"question {i} " + "x" * 80)
        db.append_message(parent, "assistant", f"answer {i} " + "y" * 80)
    before = db.get_messages_as_conversation(parent)
    agent = AIAgent(api_key="fixture-only", base_url="http://127.0.0.1:9/v1",
                    model="test/model", platform="cli", quiet_mode=True,
                    session_db=db, session_id=parent, skip_context_files=True,
                    skip_memory=True, enabled_toolsets=[])
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": SUMMARY_PREFIX + " earlier turns"}, before[-1].copy()]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = in_place
    agent._compression_feasibility_checked = True
    monkeypatch.setattr(agent, "_build_system_prompt", lambda *args, **kwargs: "fixture prompt")
    session = {"agent": agent, "history_lock": threading.Lock(), "history": list(before),
               "history_version": 1, "running": False, "session_key": parent, "cwd": str(tmp_path)}
    if stale is True:
        # A real guarded edit wins after the host snapshot but before the durable lease.
        replacement = [{"role": "user", "content": "edited question"},
                       {"role": "assistant", "content": "edited answer"}]
        acquire = db.try_acquire_compression_lock

        def edit_then_acquire(*args, **kwargs):
            with session["history_lock"]:
                db.replace_messages(parent, replacement, reject_active_turn_lease=True)
                session["history"] = replacement
                session["history_version"] += 1
            return acquire(*args, **kwargs)

        monkeypatch.setattr(db, "try_acquire_compression_lock", edit_then_acquire)
    try:
        if stale == "validation_error":
            from agent.conversation_compression_manual import compress_now, parse_compress_args

            def fail_validation():
                assert db.get_compression_lock_holder(parent) is not None
                raise RuntimeError("validation failed")

            with pytest.raises(RuntimeError, match="validation failed"):
                compress_now(agent, before, parse_compress_args(compress_args), snapshot_is_current=fail_validation)
            assert db.get_compression_lock_holder(parent) is None
            assert agent.session_id == parent
            assert db.get_messages_as_conversation(parent) == before
            return
        if stale == "empty_summary":
            compressor.compress.return_value = []
        removed, _ = server._compress_session_history(
            session, focus_topic=compress_args, before_messages=before, history_version=1,
        )
        assert db.get_compression_lock_holder(parent) is None
        if stale is True:
            assert removed == 0
            assert agent.session_id == parent, "stale compression rotated the live agent"
            assert db.find_live_compression_child(parent) is None
            assert db.get_session(parent)["ended_at"] is None
            assert [m["content"] for m in db.get_messages_as_conversation(parent)] == [
                "edited question", "edited answer"]
            assert session["history"] == replacement
        elif stale == "empty_summary":
            assert removed == 0
            assert agent.session_id == parent
            assert session["history"] == before
        else:
            assert removed > 0
            assert (agent.session_id == parent) is in_place
            assert (db.find_live_compression_child(parent) is None) is in_place
    finally:
        agent.close()
        db.close()
