"""Terminal file views of a file this task already read in full through read_file get a re-read hint."""

import json
import shlex

from tools.terminal_reread_hint import REREAD_HINT, viewed_path
from tools.terminal_tool_result import finalize_foreground_result


def _finalize(command, tmp_path, task_id):
    raw = {"output": "x", "returncode": 0}
    return json.loads(finalize_foreground_result(
        command=command, result=raw, env=None, env_type="local", effective_task_id=task_id, task_id=task_id,
        session_id=None, session_key=task_id, workdir=None, command_cwd=str(tmp_path), approval_note=None,
    ))


def test_reread_hint_for_unchanged_file_and_not_after_change(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "mod.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")
    task = "reread-hint-task"
    assert _finalize("cat mod.py", tmp_path, task).get("reread") is None  # never read: no hint
    assert "1|a = 1" in json.loads(read_file_tool(str(target), task_id=task))["content"]

    for command in ("cat mod.py", "sed -n 1,2p mod.py", "head -n 5 mod.py", "tail mod.py",
                    "less " + shlex.quote(str(target))):
        assert _finalize(command, tmp_path, task).get("reread") == REREAD_HINT, command
    assert _finalize("cat mod.py | grep a", tmp_path, task).get("reread") is None
    assert _finalize("cat mod.py", tmp_path, "another-task").get("reread") is None

    target.write_text("a = 1\nb = 3\n", encoding="utf-8")
    assert _finalize("cat mod.py", tmp_path, task).get("reread") is None


def test_viewed_path_only_accepts_plain_single_file_views():
    assert viewed_path("sed -n '10,40p' a.py") == "a.py"
    assert viewed_path("head -20 a.py") == "a.py"
    for command in ("cat a.py b.py", "sed -i s/a/b/ a.py", "cat a.py > b", "grep x a.py", "cat $(ls)", "cat"):
        assert viewed_path(command) is None, command
