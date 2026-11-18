"""read_file / search_files cover the shell one-liners models otherwise send to the terminal:
``tail -n N`` (negative offset), ``ls -la`` (a directory path), and, inside execute_code, results a
cell can use without parsing (``text`` without the ``N|`` gutter, match dicts, no char budget)."""

import json

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
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
    # Asking for more lines than the file has is the whole file, not an error.
    assert _model("tail", "read_file", path=str(log), offset=-900)["content"].startswith("1|line 1\n")


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
    assert [(e["name"], e["type"], e["size"]) for e in entries] == [("sub", "dir", None), ("data.json", "file", 3)]


def test_cell_reads_and_searches_come_back_ready_to_use(tmp_path, monkeypatch):
    reset_file_dedup("cell-shape")
    src = tmp_path / "mod.py"
    src.write_text("".join(f"value_{i} = {i}\n" for i in range(1500)), encoding="utf-8")
    monkeypatch.setattr("tools.file_tools._get_max_read_chars", lambda: 1_000)

    cell_read = _cell("cell-shape", "read_file", path=str(src))
    assert cell_read["text"] == src.read_text(encoding="utf-8").removesuffix("\n")
    assert not cell_read.get("truncated")  # code consumes it; the model-facing char budget does not apply

    model_read = _model("cell-shape", "read_file", path=str(src))
    assert "text" not in model_read and model_read["truncated_by"] == "bytes"

    found = _cell("cell-shape", "search_files", pattern=r"value_\d+ = 7$", path=str(tmp_path))
    assert {"path": str(src), "line": 8, "content": "value_7 = 7"} in found["matches"]
    assert "matches_text" not in found
