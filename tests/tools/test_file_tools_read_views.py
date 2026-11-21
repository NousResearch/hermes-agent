"""read_file / search_files cover the shell one-liners models otherwise send to the terminal:
``tail -n N`` (negative offset), ``ls -la`` (a directory path), and, inside execute_code, results a
cell can use without parsing (``text`` without the ``N|`` gutter, match dicts, no char budget)."""

import json

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
from tools.file_operations_common import DIR_LISTING_MAX_ENTRIES
from tools.file_tools_read_tracking import reset_file_dedup


def _model(task, name, **args):
    return json.loads(handle_function_call(name, args, task_id=task))


def _cell(task, name, **args):
    return json.loads(_default_dispatch(task)(name, args))


def test_negative_offset_reads_the_last_n_lines_with_their_real_numbers(tmp_path):
    reset_file_dedup("tail")
    log = tmp_path / "run.log"
    log.write_text("".join(f"line {i}\n" for i in range(1, 501)), encoding="utf-8")

    result = _model("tail", "read_file", path=str(log), offset=-3)

    assert result["content"] == "498|line 498\n499|line 499\n500|line 500"
    assert result["total_lines"] == 500
    # The offset places the window; limit still bounds the page.
    assert _model("tail", "read_file", path=str(log), offset=-12, limit=2)["content"] == "489|line 489\n490|line 490"
    # Asking for more lines than the file has is the whole file, not an error.
    assert _model("tail", "read_file", path=str(log), offset=-900)["content"].startswith("1|line 1\n")


def test_a_tail_follows_a_log_that_rotates_between_the_count_and_the_read(tmp_path, monkeypatch):
    reset_file_dedup("rotate")
    log = tmp_path / "app.log"
    log.write_text("one\ntwo\nthree\n", encoding="utf-8")
    from tools.file_operations import ShellFileOperations
    real_read = ShellFileOperations.read_file
    calls = []

    def rotating_read(self, path, offset=1, limit=2000):
        calls.append(offset)
        if len(calls) == 2:  # right after the count probe, the log is rotated to one line
            log.write_text("new\n", encoding="utf-8")
        return real_read(self, path, offset, limit)

    monkeypatch.setattr(ShellFileOperations, "read_file", rotating_read)
    assert _model("rotate", "read_file", path=str(log), offset=-1)["content"] == "1|new"

    # A log that changes again during the one retry is reported, never served as a stale tail.
    reset_file_dedup("grow")
    log.write_text("a\nb\nc\n", encoding="utf-8")
    calls.clear()

    def growing_read(self, path, offset=1, limit=2000):
        calls.append(offset)
        if len(calls) in (2, 3):  # one line appended after the count probe and again after the read
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"grow{len(calls)}\n")
        return real_read(self, path, offset, limit)

    monkeypatch.setattr(ShellFileOperations, "read_file", growing_read)
    grown = _model("grow", "read_file", path=str(log), offset=-1)
    assert "File changed during tail read" in grown["error"] and len(calls) == 3


def test_a_directory_path_returns_a_listing_with_types_and_sizes(tmp_path):
    reset_file_dedup("ls")
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "data.json").write_text("{}\n", encoding="utf-8")

    shown = _model("ls", "read_file", path=str(root))
    assert shown["is_directory"] and shown["total_entries"] == 2
    rows = shown["content"].splitlines()[1:]
    assert rows[0].endswith("sub/") and rows[1].split()[0] == "3" and rows[1].endswith("data.json")
    assert "entries" not in shown  # the model reads the table; only a cell gets the dicts

    entries = _cell("ls", "read_file", path=str(root))["entries"]
    sub_size = (root / "sub").lstat().st_size
    assert [(e["name"], e["type"], e["size"]) for e in entries] == [("sub", "dir", sub_size), ("data.json", "file", 3)]
    assert rows[0].split()[0] == str(sub_size)  # a dir shows its size like ls -la, never a "-"

    big = tmp_path / "big"
    big.mkdir()
    for i in range(DIR_LISTING_MAX_ENTRIES + 7):
        (big / f"f{i:04d}").touch()
    capped = _cell("ls", "read_file", path=str(big))
    assert capped["truncated"] and capped["total_entries"] == DIR_LISTING_MAX_ENTRIES + 7
    assert len(capped["entries"]) == len(capped["content"].splitlines()) - 1 == DIR_LISTING_MAX_ENTRIES


def test_cell_reads_and_searches_come_back_ready_to_use(tmp_path, monkeypatch):
    reset_file_dedup("cell-shape")
    src = tmp_path / "mod.py"
    src.write_text("".join(f"value_{i} = {i}\n" for i in range(1500)), encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: 1_000)

    cell_read = _cell("cell-shape", "read_file", path=str(src))
    assert cell_read["text"] == src.read_text(encoding="utf-8").removesuffix("\n")
    assert not cell_read.get("truncated")  # code consumes it; the model-facing char budget does not apply
    assert _cell("cell-shape", "read_file", path=str(tmp_path / "empty.txt"))["text"] == ""

    model_read = _model("cell-shape", "read_file", path=str(src))
    assert "text" not in model_read and model_read["truncated_by"] == "bytes"

    many = _cell("cell-shape", "search_files", pattern=r"value_\d+ = 7\d?$", path=str(tmp_path))
    assert {"path": str(src), "line": 8, "content": "value_7 = 7"} in many["matches"]
    assert len(many["matches"]) == 11 and "matches_text" not in many
    assert _cell("cell-shape", "search_files", pattern="no_such_symbol", path=str(tmp_path))["matches"] == []
    assert _cell("cell-shape", "search_files", pattern="*.absent", target="files", path=str(tmp_path))["files"] == []
