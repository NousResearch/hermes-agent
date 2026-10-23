"""An agent's whole-file knowledge survives its own turn boundary.

Gateway turns run under a fresh task_id per message and per-turn cleanup drops the previous
task's read tracker, so ``write_file`` refused to overwrite a file the same agent wrote one
message earlier ("exists but this task has not seen its full current content"). The next turn
now inherits the previous turn's byte-snapshot baselines; a file changed in between, or a task
that is not the agent's own previous turn, is still refused.
"""

import json
from types import SimpleNamespace

from agent.turn_context import _bind_turn_identity
from tools.file_tools import clear_file_ops_cache
from tools.registry import registry

_REFUSED = "has not seen its full current content"


def _write(path, task_id, content):
    return json.loads(registry.dispatch("write_file", {"path": str(path), "content": content}, task_id=task_id))


def _next_turn(agent):
    task_id, _turn = _bind_turn_identity(agent, None, None, None, None, None)
    return task_id


def _agent():
    return SimpleNamespace(session_id="sess-turns")


def test_next_turn_may_overwrite_the_file_its_previous_turn_wrote(tmp_path):
    agent, path = _agent(), tmp_path / "scratch.py"
    first = _next_turn(agent)
    assert "error" not in _write(path, first, "v1\n")
    clear_file_ops_cache(first)  # per-turn cleanup of a non-persistent env

    second = _next_turn(agent)
    assert second != first
    result = _write(path, second, "v2\n")
    assert "error" not in result, result
    assert path.read_text() == "v2\n"
    clear_file_ops_cache(second)


def test_file_changed_between_turns_is_still_refused(tmp_path):
    agent, path = _agent(), tmp_path / "scratch.py"
    first = _next_turn(agent)
    assert "error" not in _write(path, first, "v1\n")
    clear_file_ops_cache(first)
    path.write_text("edited by someone else\n")

    second = _next_turn(agent)
    assert _REFUSED in _write(path, second, "v2\n").get("error", "")
    assert path.read_text() == "edited by someone else\n"
    clear_file_ops_cache(second)


def test_a_different_agent_does_not_inherit(tmp_path):
    path = tmp_path / "scratch.py"
    writer = _agent()
    first = _next_turn(writer)
    assert "error" not in _write(path, first, "v1\n")
    clear_file_ops_cache(first)

    other = _next_turn(_agent())
    assert _REFUSED in _write(path, other, "v2\n").get("error", "")
    clear_file_ops_cache(other)
