"""The per-turn cleanup keeps a session's version-checked write baselines.

The gateway runs every turn of a session under ``task_id=session_id``, and the end-of-turn
cleanup used to drop that task's whole read tracker. The next turn's ``write_file`` then
refused a file the agent itself wrote one turn earlier ("has not seen its full current
content"). Baselines are checked against the file's current version on use, so carrying them
over still refuses a file that changed in between. A task id the next turn does not reuse keeps
nothing, and session close still clears everything.
"""

import json
from types import SimpleNamespace

from agent.chat_completion_helpers import cleanup_task_resources
from tools.file_tools import _read_tracker, clear_file_ops_cache
from tools.registry import registry
from tools.terminal_tool_lifecycle import cleanup_vm

_REFUSED = "has not seen its full current content"


def _write(path, task_id, content):
    result = registry.dispatch("write_file", {"path": str(path), "content": content}, task_id=task_id)
    return json.loads(result)


def _end_turn(task_id, session_id):
    cleanup_task_resources(SimpleNamespace(verbose_logging=False, session_id=session_id), task_id)


def test_own_file_from_an_earlier_turn_can_be_overwritten(tmp_path):
    path, task = tmp_path / "chart.py", "20261017_000000_turns"
    try:
        assert "error" not in _write(path, task, "first\n")
        _end_turn(task, session_id=task)
        written = _write(path, task, "second\n")
        assert "error" not in written, written
        assert path.read_text(encoding="utf-8-sig") == "second\n"
    finally:
        clear_file_ops_cache(task)


def test_a_file_changed_between_turns_is_still_refused(tmp_path):
    path, task = tmp_path / "notes.txt", "20261017_000000_external"
    try:
        assert "error" not in _write(path, task, "agent version\n")
        _end_turn(task, session_id=task)
        path.write_text("edited by someone else, longer than before\n", encoding="utf-8")
        refused = _write(path, task, "agent overwrite\n")
        assert _REFUSED in refused.get("error", ""), refused
        assert path.read_text(encoding="utf-8-sig") == "edited by someone else, longer than before\n"
    finally:
        clear_file_ops_cache(task)


def test_a_task_id_the_next_turn_does_not_reuse_keeps_nothing(tmp_path):
    task = "4f0c2b9e-per-turn-uuid"
    try:
        assert "error" not in _write(tmp_path / "once.txt", task, "x\n")
        _end_turn(task, session_id="the-session")
        assert task not in _read_tracker
    finally:
        clear_file_ops_cache(task)


def test_session_close_still_clears_the_baselines(tmp_path):
    path, task = tmp_path / "closed.txt", "20261017_000000_closed"
    try:
        assert "error" not in _write(path, task, "first\n")
        _end_turn(task, session_id=task)
        cleanup_vm(task)
        assert task not in _read_tracker
        assert _REFUSED in _write(path, task, "second\n").get("error", "")
    finally:
        clear_file_ops_cache(task)
