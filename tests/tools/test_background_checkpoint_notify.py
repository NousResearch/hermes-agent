"""A notify=true background job must survive a host restart with its completion contract.

The spawn writes ``processes.json`` before the terminal tool sets ``notify_on_complete``
and the chat routing; recovery in a new host reads only that file.
"""

import json

from gateway.session_context import clear_session_vars, set_session_vars
from tools.process_registry import ProcessRegistry, _checkpoint_path, process_registry
from tools.terminal_tool import terminal_tool


def test_restart_recovery_rearms_the_completion_watcher():
    tokens = set_session_vars(platform="telegram", chat_id="42", user_id="7",
                              session_key="agent:main:telegram:dm:42")
    try:
        out = json.loads(terminal_tool("sleep 30", background=True, notify_on_complete=True, task_id="t1"))
    finally:
        clear_session_vars(tokens)
    sid = out["session_id"]
    try:
        entry = next(e for e in json.loads(_checkpoint_path().read_text()) if e["session_id"] == sid)
        assert entry["notify_on_complete"] is True
        assert (entry["watcher_platform"], entry["watcher_chat_id"]) == ("telegram", "42")

        restarted = ProcessRegistry()  # a new host reading the same home
        assert restarted.recover_from_checkpoint() == 1
        assert [(w["session_id"], w["platform"], w["chat_id"], w["notify_on_complete"])
                for w in restarted.pending_watchers] == [(sid, "telegram", "42", True)]
    finally:
        process_registry.kill_process(sid)
