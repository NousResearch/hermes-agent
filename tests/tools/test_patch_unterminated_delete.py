"""Deleting all content of an unterminated file is a valid update."""
from tests.tools.test_patch_planner_indent_carryover import _apply


def test_delete_only_unterminated_line():
    result, store = _apply("old", "-old\n")
    assert result.success, result.error
    assert store.files["f"] == ""
