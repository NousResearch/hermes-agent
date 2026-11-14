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


@pytest.mark.parametrize("source, expected", [
    ("foo\nbar\n", "baz\nbar\n"),
    ("foo\n", "baz\n"),
    ("foo\r\nbar\r\n", "baz\r\nbar\r\n"),
])
def test_trailing_blank_context_line_adds_no_blank_line(source, expected):
    result, store = _apply(source, "-foo\n+baz\n \n")
    assert result.success, result.error
    assert store.files["f"] == expected


def test_a_raising_backend_read_is_a_fixed_code_not_its_text():
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    class Boom:
        def read_file_raw(self, path):
            raise RuntimeError("SYNTHETIC_SECRET_9XYZ")

    star = "*" * 3
    ops, err = parse_v4a_patch(f"{star} Begin Patch\n{star} Update File: f\n@@\n-a\n+b\n{star} End Patch\n")
    assert err is None
    result = apply_v4a_operations(ops, Boom())
    assert not result.success and "SYNTHETIC_SECRET_9XYZ" not in repr(result)
    assert "READ_ERROR" in result.error
