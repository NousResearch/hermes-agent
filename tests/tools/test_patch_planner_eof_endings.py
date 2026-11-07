"""Line endings at an unterminated last line survive insertions and reordered hunks."""
import pytest

from tests.tools.test_patch_planner_indent_carryover import _apply


@pytest.mark.parametrize("body, expected", [
    (" last\n+new\n", "a\nlast\nnew\n"),
    ("-last\n+new\n+more\n", "a\nnew\nmore"),
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


def test_multi_line_replacement_of_unterminated_tail_keeps_missing_final_newline():
    result, store = _apply("keep\nold\nlast", "-old\n-last\n+a\n+b\n")
    assert result.success, result.error
    assert store.files["f"] == "keep\na\nb"


def test_anchor_cursor_follows_lines_removed_above_it():
    src = "def a():\n    d1\n    d2\n# anchor\nx = old\n# other\nx = old\n"
    star = "*" * 3
    from tests.tools.test_patch_planner_indent_carryover import Store
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    body = ("@@\n # anchor\n@@ def a(): @@\n-    d1\n-    d2\n@@\n-x = old\n+x = new\n")
    ops, err = parse_v4a_patch(f"{star} Begin Patch\n{star} Update File: f\n{body}{star} End Patch\n")
    assert err is None
    store = Store(src)
    result = apply_v4a_operations(ops, store)
    assert not result.success
    assert store.files["f"] == src and not store.writes
