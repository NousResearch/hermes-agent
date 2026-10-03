"""Over-budget write_file edits report exact +/- alongside the capped preview.

The stored ``inline_diff`` is a display preview capped at
``_MAX_INLINE_DIFF_LINES`` rendered lines. File rows must not count that
preview, or every large edit shows the cap remainder instead of its real
additions.
"""

import json

from agent.display import capture_local_edit_snapshot
from tui_gateway import server as progress


def test_prepare_attaches_exact_counts_for_over_budget_edit(monkeypatch, tmp_path):
    target = tmp_path / "page.html"
    target.write_text("".join(f"old {i}\n" for i in range(10)), encoding="utf-8")
    snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})
    target.write_text(
        "".join(f"old {i}\n" for i in range(10)) + "".join(f"new {i}\n" for i in range(200)),
        encoding="utf-8",
    )

    sid, call_id = "diff-counts", "call_0"
    monkeypatch.setitem(progress._sessions, sid, {"edit_snapshots": {call_id: snapshot}})

    out = progress._prepare_tool_result_metadata(
        sid,
        call_id,
        "write_file",
        {"path": str(target)},
        json.dumps({"success": True, "bytes_written": 1234}),
    )

    metadata = out["tool_result_metadata"]
    assert metadata["lines_added"] == 200
    assert metadata["lines_removed"] == 0
    # The preview stays capped; the counts carry the truth.
    assert len(metadata["inline_diff"].splitlines()) <= 82


def test_prepare_omits_counts_for_noop_edit(monkeypatch, tmp_path):
    target = tmp_path / "same.txt"
    target.write_text("unchanged\n", encoding="utf-8")
    snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

    sid, call_id = "diff-counts-noop", "call_0"
    monkeypatch.setitem(progress._sessions, sid, {"edit_snapshots": {call_id: snapshot}})

    out = progress._prepare_tool_result_metadata(
        sid,
        call_id,
        "write_file",
        {"path": str(target)},
        json.dumps({"success": True, "bytes_written": 10}),
    )

    assert out == {}
