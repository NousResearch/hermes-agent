"""delegate_task ``files``: named ranges reach the child's first turn, capped, inside its workspace only, and
count as already read in the child's read_file dedup state."""

import json

from tools import delegate_tool_files as dtf
from tools.delegate_tool_progress import child_task_message


def _workspace(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "small.py").write_text("".join(f"line {n}\n" for n in range(1, 21)), encoding="utf-8")
    (ws / "big.py").write_text("".join(f"row {n}\n" for n in range(1, 1001)), encoding="utf-8")
    return ws


def _brief(tmp_path, entries, context="ctx"):
    ws = tmp_path / "repo" if (tmp_path / "repo").is_dir() else _workspace(tmp_path)
    specs, err = dtf.coerce_task_files([{"goal": "g", "files": entries}], None)
    assert err is None
    files, err = dtf.resolve_task_files(specs, str(ws))
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
    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 200)
    _ws, files, err, message = _brief(tmp_path, ["big.py"])
    assert err is None and len(files[0].block.encode()) < 400 and "[truncated: " in message

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
