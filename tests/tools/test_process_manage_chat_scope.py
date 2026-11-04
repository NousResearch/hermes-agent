"""process_manage: a live process handle is not authority across the chats one gateway serves.

``list`` already shows a turn only its own chat's processes and retained receipts require the
owning session; the live poll/log/wait/write/kill path must honour the same boundary.
"""

import json
import time

from gateway.session_context import clear_session_vars, set_session_vars
from tools.process_registry import _handle_process, process_registry


def _turn(chat, key, session_id):
    return set_session_vars(platform="telegram", chat_id=chat, chat_type="dm", user_id=chat,
                            session_key=key, session_id=session_id)


def _act(action, handle, task_id):
    return json.loads(_handle_process({"action": action, "session_id": handle}, task_id=task_id))


def test_other_chat_cannot_read_or_kill_a_live_process_by_handle():
    owner_key = "agent:main:telegram:dm:111"
    tok = _turn("111", owner_key, "owner-session")
    try:
        proc = process_registry.spawn_local("echo OWNER-OUTPUT; sleep 30", task_id="owner-task",
                                            session_key=owner_key)
    finally:
        clear_session_vars(tok)
    try:
        deadline = time.monotonic() + 10
        while "OWNER-OUTPUT" not in proc.output_buffer and time.monotonic() < deadline:
            time.sleep(0.05)

        tok = _turn("222", "agent:main:telegram:dm:222", "other-session")
        try:
            for action, handle in (("poll", proc.id), ("log", proc.id[5:11]), ("kill", proc.id)):
                result = _act(action, handle, "other-task")
                assert result["status"] == "not_found", (action, result)
                assert "OWNER-OUTPUT" not in json.dumps(result)
        finally:
            clear_session_vars(tok)
        assert not proc.exited

        # The owning chat keeps its process across /new (same key, new session id and task).
        tok = _turn("111", owner_key, "owner-session-after-new")
        try:
            assert "OWNER-OUTPUT" in _act("log", proc.id, "owner-task-2")["output"]
            assert _act("kill", proc.id, "owner-task-2")["status"] in {"killed", "already_exited"}
        finally:
            clear_session_vars(tok)
    finally:
        process_registry.kill_process(proc.id)
