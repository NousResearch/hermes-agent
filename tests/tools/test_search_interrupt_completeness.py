"""Interrupted content-search output is partial, even below the page limit."""

import pytest

from tools.file_operations_common import ExecuteResult
from tools.file_operations_search import _FILE_ROW_MARK, _parse_search_output


@pytest.mark.parametrize("mode,row", [
    ("content", "./plain.py:1:hit\n"),
    ("files_only", _FILE_ROW_MARK + "./plain.py\n"),
    ("count", "./plain.py:1\n"),
])
@pytest.mark.parametrize("has_row", [False, True])
@pytest.mark.parametrize("exit_code,marker,reason", [
    (130, "\n[Command interrupted]", "search_interrupted"),
    (124, "\n[Command timed out after 60s]", "search_timeout"),
    (0, "", None),
])
def test_termination_preserves_rows_and_discloses_incompleteness(
    mode, row, has_row, exit_code, marker, reason,
):
    result = _parse_search_output(
        ExecuteResult(stdout=(row if has_row else "") + marker, exit_code=exit_code),
        mode, limit=50, offset=0, context=0,
    )
    output = result.to_dict()
    assert result.error is None
    assert result.total_count == int(has_row)
    if has_row:
        if mode == "content":
            assert [(m.path, m.line_number, m.content) for m in result.matches] == [
                ("./plain.py", 1, "hit"),
            ]
        elif mode == "files_only":
            assert result.files == ["./plain.py"]
        else:
            assert result.counts == {"./plain.py": 1}
    assert result.limit_reason == reason
    assert result.truncated is bool(reason)
    assert output.get("total_count_is_lower_bound", False) is bool(reason)
