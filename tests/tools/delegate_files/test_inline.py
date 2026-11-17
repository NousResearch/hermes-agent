"""delegate_task ``files``: named ranges reach the child's first turn, capped, inside its workspace only, and
count as already read in the child's read_file dedup state."""

import json

import pytest

from tools.delegate_files import inline as dtf
from tools.delegate_tool_progress import child_task_message


def _workspace(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "small.py").write_text("".join(f"line {n}\n" for n in range(1, 21)), encoding="utf-8")
    (ws / "big.py").write_text("".join(f"row {n}\n" for n in range(1, 1001)), encoding="utf-8")
    return ws


@pytest.fixture(autouse=True)
def _no_terminal_cwd(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)


class _Parent:
    def __init__(self, ws):
        self.terminal_cwd = str(ws)


def _brief(tmp_path, entries, context="ctx"):
    ws = tmp_path / "repo" if (tmp_path / "repo").is_dir() else _workspace(tmp_path)
    files, err = dtf.resolve_task_files([{"goal": "g", "files": entries}], None, _Parent(ws))
    return ws, files, err, (child_task_message("do it", dtf.with_files(context, files[0])) if not err else None)


def test_files_land_in_first_turn_after_context_with_cap_marker(tmp_path):
    _ws, _files, err, message = _brief(tmp_path, ["small.py:3-5", "big.py"])
    assert err is None
    context_at, files_at, task_at = message.index("CONTEXT:\nctx"), message.index(dtf.FILES_HEADING), message.index("TASK:\ndo it")
    assert context_at < files_at < task_at
    assert "small.py:3-5\n```\n3|line 3\n4|line 4\n5|line 5\n```" in message
    assert "line 6" not in message
    assert f"{dtf.MAX_FILE_LINES}|row {dtf.MAX_FILE_LINES}\n" in message
    assert f"row {dtf.MAX_FILE_LINES + 1}\n" not in message
    assert f"[truncated: {1000 - dtf.MAX_FILE_LINES} more lines, read_file to see the rest]" in message


def test_files_total_cap_and_outside_workspace_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 1000)
    _ws, files, err, message = _brief(tmp_path, ["big.py"])
    assert err is None and len(files[0].block.encode()) <= 1000 and "[truncated: " in message

    outside = tmp_path / "secret.txt"
    outside.write_text("nope\n", encoding="utf-8")
    for entry in (str(outside), "../secret.txt"):
        _ws, files, err, _msg = _brief(tmp_path, [entry])
        assert files == [] and "outside the child's workspace" in err


def test_inlined_range_is_marked_read_for_the_child(tmp_path):
    from tools.file_tools import read_file_tool

    ws, files, err, _msg = _brief(tmp_path, ["small.py:3-5", "small.py"])
    assert err is None
    dtf.mark_files_read("child-files-task", files[0])
    stub = json.loads(read_file_tool(str(ws / "small.py"), offset=3, limit=3, task_id="child-files-task"))
    assert stub.get("dedup") is True and "content" not in stub
    whole = json.loads(read_file_tool(str(ws / "small.py"), task_id="child-files-task"))
    assert whole.get("dedup") is True
    other = json.loads(read_file_tool(str(ws / "small.py"), offset=6, limit=2, task_id="child-files-task"))
    assert "6|line 6" in other["content"]


def test_files_refused_when_the_child_works_on_another_host(tmp_path, monkeypatch):
    """ssh/docker/modal: the child's tools see the backend, so this host's same-named copy is never inlined."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    _ws, files, err, _msg = _brief(tmp_path, ["small.py"])
    assert files == [] and "local terminal backend" in err


def test_delegate_task_passes_files_through_the_default_backend(tmp_path, monkeypatch):
    """Real delegate_task and real task routing: the stock backend accepts ``files`` and the child is built with the
    FILES block in its context; a refused entry fails the call before any child is built."""
    import types

    import hermes_yaml as yaml
    from tools import delegate_tool as dt

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"delegation": {
        "model": "pinned-model", "base_url": "http://127.0.0.1:9/v1", "api_key": "k"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    ws = _workspace(tmp_path)
    monkeypatch.setenv("TERMINAL_CWD", str(ws))
    built = []
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", lambda **kw: built.append(kw) or types.SimpleNamespace(
        tool_progress_callback=None, model=kw.get("model")))
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    parent = types.SimpleNamespace(session_id="root", model="m", provider="openrouter", tool_progress_callback=None,
                                   _delegate_spinner=None, request_overrides={}, _safe_print=lambda line: None)

    out = json.loads(dt.delegate_task(tasks=[{"goal": "edit the small module", "context": "ctx",
                                              "files": ["small.py:1-2"]}], parent_agent=parent))
    assert out["results"][0]["status"] == "completed", out
    assert built[0]["context"].startswith("ctx\n\n" + dtf.FILES_HEADING) and "1|line 1\n2|line 2\n" in built[0]["context"]

    refused = json.loads(dt.delegate_task(tasks=[{"goal": "edit the small module", "files": ["../x.py"]}],
                                          parent_agent=parent))
    assert "outside the child's workspace" in refused["error"] and len(built) == 1


def test_files_follow_read_file_lines_cap_every_byte_and_tie_marks_to_the_shown_snapshot(tmp_path, monkeypatch):
    """Ranges count newline bytes only (a form feed is not a line break), labels and markers count against the
    total cap, a file that changes before the child starts is not marked read, and worktree isolation refuses."""
    from tools.file_tools import read_file_tool

    ws = _workspace(tmp_path)
    (ws / "ff.py").write_text("a\fb\nsecond\nthird\n", encoding="utf-8")
    _ws, files, err, message = _brief(tmp_path, ["ff.py:2-2"])
    assert err is None and "2|second\n" in message and "third" not in message

    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 2000)
    _ws, files, err, _msg = _brief(tmp_path, ["small.py"] * 40)
    assert len(files[0].block.encode()) <= 2000 and "over the 2000-byte cap: small.py" in files[0].block

    _ws, files, err, _msg = _brief(tmp_path, ["ff.py"])
    (ws / "ff.py").write_text("changed\n", encoding="utf-8")
    dtf.mark_files_read("snapshot-task", files[0])
    assert "1|changed" in json.loads(read_file_tool(str(ws / "ff.py"), task_id="snapshot-task"))["content"]

    monkeypatch.setattr("tools.delegate_tool_config._get_worktree_isolation", lambda: True)
    _ws, files, err, _msg = _brief(tmp_path, ["small.py"])
    assert files == [] and "worktree_isolation" in err


def test_files_refuse_range_starting_beyond_eof(tmp_path):
    """A stale parent range must fail before spawn rather than inlining an empty, 'complete' snippet."""
    _ws, files, err, message = _brief(tmp_path, ["small.py:21-25"])
    assert files == []
    assert "'files' entry 'small.py:21-25' starts beyond the end of the file" in err
    assert message is None
