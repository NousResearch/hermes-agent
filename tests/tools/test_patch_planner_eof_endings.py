"""Line endings at an unterminated last line survive insertions and reordered hunks."""
import pytest

from tests.tools.test_patch_planner_indent_carryover import _apply


@pytest.mark.parametrize("body, expected", [
    (" last\n+new\n", "a\nlast\nnew\n"),
    ("-last\n+new\n+more\n", "a\nnew\nmore\n"),
    (" a\n last\n+new\n", "a\nlast\nnew\n"),
])
def test_insert_after_unterminated_last_line_does_not_merge(body, expected):
    result, store = _apply("a\nlast", body)
    assert result.success, result.error
    assert store.files["f"] == expected


@pytest.mark.parametrize("body", ["-old\n+new\n", "+new\n-old\n"])
def test_one_for_one_replacement_keeps_missing_final_newline(body):
    result, store = _apply("keep\nold", body)
    assert result.success, result.error
    assert store.files["f"] == "keep\nnew"
