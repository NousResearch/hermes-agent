"""A repeat of a truncated read_file must say where the rest starts.

The dedup stub and its BLOCK escalation used to answer an identical re-read with
"the earlier result is still current" and no ``next_offset``. When that earlier
result was truncated, the model was told it already had a file it had seen only
the first page of, with nothing naming the line to continue from.
"""
import json

import pytest

from tools import file_tools
from tools.file_tools import read_file_tool
from tools.file_tools_read_tracking import _read_tracker


@pytest.fixture(autouse=True)
def _fresh_tracker():
    _read_tracker.clear()
    yield
    _read_tracker.clear()


def _write_lines(path, n):
    path.write_text("".join(f"line {i}\n" for i in range(1, n + 1)), encoding="utf-8")
    return str(path)


def test_repeat_of_line_limited_read_names_next_offset_through_the_block(tmp_path):
    target = _write_lines(tmp_path / "long.txt", 300)

    first, stub, block = (json.loads(read_file_tool(target, limit=100, task_id="t"))
                          for _ in range(3))

    assert first["truncated"] is True and "content" in first
    assert stub["dedup"] is True and "content" not in stub
    assert stub["next_offset"] == 101 and stub["total_lines"] == 300
    assert "offset=101" in stub["message"]
    assert block["error"].startswith("BLOCKED") and block["next_offset"] == 101
    assert "offset=101" in block["error"]

    # The hint leads somewhere: the continuation is a real read, not another stub.
    rest = json.loads(read_file_tool(target, offset=stub["next_offset"], limit=100, task_id="t"))
    assert rest["content"].startswith("101|line 101")


def test_repeat_of_char_budget_read_names_the_budget_next_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "_get_max_read_chars", lambda: 400)
    target = _write_lines(tmp_path / "wide.txt", 200)

    first = json.loads(read_file_tool(target, task_id="t"))
    stub = json.loads(read_file_tool(target, task_id="t"))

    assert first["truncated_by"] == "bytes"
    assert stub["dedup"] is True and stub["next_offset"] == first["next_offset"]
    assert f"offset={first['next_offset']}" in stub["message"]


def test_repeat_of_a_complete_read_stays_a_plain_stub(tmp_path):
    target = _write_lines(tmp_path / "short.txt", 20)

    read_file_tool(target, task_id="t")
    stub = json.loads(read_file_tool(target, task_id="t"))

    assert stub["dedup"] is True
    assert "next_offset" not in stub and "truncated" not in stub
    assert "offset=" not in stub["message"]


def test_continuation_survives_cap_eviction_while_its_read_is_still_deduped(tmp_path, monkeypatch):
    # A write leaves its range's continuation behind with no dedup entry; once the
    # continuation map overflows, FIFO eviction must not drop a hint whose read is
    # still deduped (it would stub with "still current" and no next_offset again).
    from tools import file_tools_read_tracking
    monkeypatch.setattr(file_tools_read_tracking, "_DEDUP_CAP", 2)
    a, b, c = (_write_lines(tmp_path / f"{n}.txt", 300) for n in "abc")

    read_file_tool(a, limit=100, task_id="t")
    read_file_tool(b, limit=100, task_id="t")
    file_tools_read_tracking._update_read_timestamp(b, "t")  # b written: dedup entry gone
    file_tools_read_tracking.reset_file_dedup("t")           # compaction boundary
    read_file_tool(a, limit=100, task_id="t")                # recovery read of a
    read_file_tool(c, limit=100, task_id="t")                # overflows the maps
    stub = json.loads(read_file_tool(a, limit=100, task_id="t"))

    assert stub["dedup"] is True
    assert stub["next_offset"] == 101 and "offset=101" in stub["message"]
