"""Manual compression must not publish a snapshot invalidated before lease admission."""
import logging
import threading
from unittest.mock import MagicMock

import pytest

from persistence.conversation.messages import NewMessage, ReplaceMode
from persistence.sqlite.conversation._store.compression import find_live_compression_child
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation, session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _agent_with_history(tmp_path, monkeypatch, in_place=False):
    from run_agent import AIAgent
    from agent.context_compressor import SUMMARY_PREFIX

    db = storage_handle(tmp_path)
    parent = "manual-fence-parent"
    seed_session(db, parent, "cli")
    for i in range(12):
        seed_messages(db, parent, {"role": "user", "content": f"question {i} " + "x" * 80},
                      {"role": "assistant", "content": f"answer {i} " + "y" * 80})
    before = conversation(db, parent)
    agent = AIAgent(api_key="fixture-only", base_url="http://127.0.0.1:9/v1",
                    model="test/model", platform="cli", quiet_mode=True,
                    persistence=db, session_id=parent, skip_context_files=True,
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
    return db, agent, parent, before


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("compress_args,stale", [("", False), ("", True), ("here 2", True)])
def test_manual_compress_rejects_history_rewritten_before_admission(
    tmp_path, monkeypatch, caplog, stale, in_place, compress_args,
):
    from tui_gateway import server

    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    db, agent, parent, before = _agent_with_history(tmp_path, monkeypatch, in_place)
    session = {"agent": agent, "history_lock": threading.Lock(), "history": list(before),
               "history_version": 1, "running": False, "session_key": parent, "cwd": str(tmp_path)}
    if stale:
        # A real guarded edit wins after the host snapshot but before the durable lease.
        replacement = [{"role": "user", "content": "edited question"},
                       {"role": "assistant", "content": "edited answer"}]
        acquire = db.coordination.try_acquire_compression_lock

        def edit_then_acquire(*args, **kwargs):
            with session["history_lock"]:
                db.messages.replace_transcript(parent, [NewMessage.from_live(m) for m in replacement],
                                               ReplaceMode.DESTRUCTIVE, reject_active_turn_lease=True)
                session["history"] = replacement
                session["history_version"] += 1
            return acquire(*args, **kwargs)

        monkeypatch.setattr(db.coordination, "try_acquire_compression_lock", edit_then_acquire)
    try:
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            removed, _ = server._compress_session_history(
                session, focus_topic=compress_args, before_messages=before, history_version=1,
            )
        assert db.coordination.get_compression_lock_holder(session_id=parent) is None
        if stale:
            assert removed == 0
            assert agent.session_id == parent, "stale compression rotated the live agent"
            assert find_live_compression_child(raw_holder(db), parent) is None
            assert session_row(db, parent)["ended_at"] is None
            assert [m["content"] for m in conversation(db, parent)] == [
                "edited question", "edited answer"]
            assert session["history"] == replacement
            assert '"failure_class":"snapshot_stale"' in caplog.text
        else:
            assert removed > 0
            assert (agent.session_id == parent) is in_place
            assert (find_live_compression_child(raw_holder(db), parent) is None) is in_place
    finally:
        agent.close()


def test_manual_compress_releases_lease_when_snapshot_check_raises(tmp_path, monkeypatch):
    from agent.conversation_compression_manual import compress_now, parse_compress_args

    db, agent, parent, before = _agent_with_history(tmp_path, monkeypatch)

    def fail_validation():
        assert db.coordination.get_compression_lock_holder(session_id=parent) is not None
        raise RuntimeError("validation failed")

    try:
        with pytest.raises(RuntimeError, match="validation failed"):
            compress_now(agent, before, parse_compress_args(""), snapshot_is_current=fail_validation)
        assert db.coordination.get_compression_lock_holder(session_id=parent) is None
        assert agent.session_id == parent
        assert conversation(db, parent) == before
    finally:
        agent.close()
