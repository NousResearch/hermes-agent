"""write_file's staleness refusal names why a read was not recorded as complete.

A whole-file read can be stamped partial for reasons that have nothing to do with
pagination (the repeat guard, redacted content). The refusal used to blame
offset/limit pagination in every case and send the model back to read more pages,
which cannot fix any of them.
"""

import json

from tools.file_tools import clear_file_ops_cache
from tools.registry import registry


def _call(name, path, task_id, **arguments):
    result = registry.dispatch(name, {"path": str(path), **arguments}, task_id=task_id)
    assert isinstance(result, str)
    return json.loads(result)


def _refusal_after(path, task_id):
    refused = _call("write_file", path, task_id, content="replacement\n")
    assert refused.get("stale_write_blocked"), refused
    return refused["error"]


def test_refusal_after_the_repeat_guard_does_not_blame_pagination(tmp_path):
    path = tmp_path / "loop.txt"
    path.write_text("".join(f"line {i}\n" for i in range(50)), encoding="utf-8")
    task = "stale-reason-repeat"
    try:
        # Read, rewrite, read: the fourth identical read trips the repeat guard.
        for _ in range(3):
            _call("read_file", path, task)
            assert "error" not in _call("write_file", path, task, content="x\n")
        _call("read_file", path, task)

        error = _refusal_after(path, task)
        assert "offset/limit pagination" not in error
        assert "repeated" in error and "patch" in error
        # A blocked repeat read cannot be retried to heal this baseline.
        assert "Reload the current contents with read_file" not in error
    finally:
        clear_file_ops_cache(task)


def test_refusal_after_a_redacted_whole_read_does_not_blame_pagination(tmp_path):
    path = tmp_path / "secrets.txt"
    path.write_text("API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF\nother=1\n", encoding="utf-8")
    task = "stale-reason-redacted"
    try:
        assert "error" not in _call("read_file", path, task)

        error = _refusal_after(path, task)
        assert "offset/limit pagination" not in error
        assert "redacted" in error
    finally:
        clear_file_ops_cache(task)
