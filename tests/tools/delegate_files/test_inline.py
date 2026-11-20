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


def test_skipped_names_stay_inside_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 2000)
    ws = _workspace(tmp_path)
    names = [f"{'d' * 60}_{i}.py" for i in range(8)]
    for name in names:
        (ws / name).write_text("x = 1\n", encoding="utf-8")
    _ws, files, err, _msg = _brief(tmp_path, ["big.py", *names])
    assert err is None and "not inlined" in files[0].block and len(files[0].block.encode()) <= 2000


def test_long_backtick_fences_stay_inside_the_cap(tmp_path, monkeypatch):
    """A body holding a long backtick run widens both fences; the entry still fits the total cap."""
    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 2000)
    ws = _workspace(tmp_path)
    (ws / "ticks.md").write_text("".join(f"{'`' * 300} {n}\n" for n in range(1, 50)), encoding="utf-8")
    _ws, files, err, _msg = _brief(tmp_path, ["ticks.md"])
    assert err is None and len(files[0].block.encode()) <= 2000
    assert f"1|{'`' * 300} 1\n" in files[0].block and "not inlined" not in files[0].block


def test_fence_refit_drops_a_row_per_pass(tmp_path, monkeypatch):
    """A fence-widening row followed by a row too long to fit leaves slack; each re-render still drops a row."""
    monkeypatch.setattr(dtf, "MAX_TOTAL_BYTES", 8000)
    ws = _workspace(tmp_path)
    reads = []
    real = dtf._read_lines
    monkeypatch.setattr(dtf, "_read_lines", lambda *a: reads.append(1) or real(*a))
    for k in range(2500, 2800):  # some k lands the one-row render a few bytes over the cap
        (ws / "t.md").write_text("`" * k + "\n" + "x" * 20000 + "\n", encoding="utf-8")
        reads.clear()
        _ws, files, err, _msg = _brief(tmp_path, ["t.md"])
        assert err is None and len(files[0].block.encode()) <= 8000
        assert len(reads) <= 3, (k, len(reads))


def test_a_huge_unrequested_line_is_never_held_in_memory(tmp_path):
    """A short line 1 then a 32 MiB newline-free tail (a minified file): reading 1-1 stays small in memory."""
    import tracemalloc

    ws = _workspace(tmp_path)
    with open(ws / "min.js", "wb") as out:
        out.write(b"ok\n")
        out.writelines(b"x" * (1 << 20) for _ in range(32))
    tracemalloc.start()
    try:
        _ws, files, err, _msg = _brief(tmp_path, ["min.js:1-1"])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert err is None and "1|ok\n" in files[0].block and len(files[0].block) < 1000
    assert peak < 8 * (1 << 20), peak


def test_a_label_longer_than_the_remaining_budget_is_skipped_not_rendered(tmp_path):
    """After a first file fills most of the cap, a later entry whose label alone overflows is listed, not shown."""
    ws = _workspace(tmp_path)
    (ws / "wide.txt").write_text("".join("x" * 142 + "\n" for _ in range(400)), encoding="utf-8")
    deep = ws
    for i in range(7):
        deep = deep / (f"{i}" + "d" * 250)
    deep.mkdir(parents=True)
    (deep / "ok.txt").write_text("ok\n", encoding="utf-8")
    long_path = str((deep / "ok.txt").relative_to(ws))
    _ws, files, err, _msg = _brief(tmp_path, ["wide.txt", long_path])
    assert err is None and len(files[0].block.encode()) <= dtf.MAX_TOTAL_BYTES
    assert "not inlined" in files[0].block and "1|ok" not in files[0].block


def test_chunked_reads_keep_the_unterminated_last_line_and_a_bom_crlf_row_that_fits(tmp_path):
    ws = _workspace(tmp_path)
    # A last line with no newline, spanning a chunk boundary, is still a row.
    (ws / "tail.txt").write_bytes(b"a\n" + b"b" * (dtf._CHUNK + 5))
    rows, wanted, _sha = dtf._read_lines(ws / "tail.txt", 2, 2, 1 << 20)
    assert wanted == 1 and rows == [f"2|{'b' * (dtf._CHUNK + 5)}\n"]
    # Line 1 is 8 bytes on disk (BOM + CRLF) but its row "1|abc\n" is 6: it fits a 6-byte budget.
    (ws / "bom.txt").write_bytes(b"\xef\xbb\xbfabc\r\n")
    assert dtf._read_lines(ws / "bom.txt", 1, 1, 6)[0] == ["1|abc\n"]


def test_cut_files_stay_unmarked_and_denied_or_binary_files_fail_the_call(tmp_path):
    from tools.file_tools import read_file_tool

    ws, files, err, _msg = _brief(tmp_path, ["big.py"])
    dtf.mark_files_read("child-cut-task", files[0])
    assert "401|row 401" in json.loads(read_file_tool(str(ws / "big.py"), task_id="child-cut-task"))["content"]
    (ws / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    (ws / "blob.py").write_bytes(b"a\0b\n")
    for entry, words in ((".env", ".env"), ("blob.py", "binary file")):
        _ws, files, err, _msg = _brief(tmp_path, [entry])
        assert files == [] and words in err, err
