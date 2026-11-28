"""Raw-substring edges respect identifiers; a Move keeps the file's sha baseline."""
from types import SimpleNamespace

import pytest

from tests.tools.test_patch_planner_indent_carryover import _apply
from tools.patch_parser import _resolve_operations, parse_v4a_patch


@pytest.mark.parametrize("source, body", [
    ("self.old_value = 1\n", "-value = 1\n+value = 2\n"),
    ("my_name = 1\n", "-name = 1\n+name = 2\n"),
    ("name = 1\n", "-e \n+E \n"),
])
def test_fragment_of_another_identifier_is_refused(source, body):
    result, store = _apply(source, body)
    assert not result.success
    assert store.files["f"] == source and store.writes == []


def test_mid_line_search_ending_in_space_still_edits_at_a_token_start():
    result, store = _apply("prefix = 1\n", "-prefix \n+head \n")
    assert result.success, result.error
    assert store.files["f"] == "head = 1\n"


def test_update_after_move_keeps_the_source_sha_baseline():
    class Store:
        def read_file_raw(self, path):
            if path != "f":
                return SimpleNamespace(error="missing", content=None, not_found=True)
            return SimpleNamespace(error=None, content="old\n", not_found=False, _content_sha256="SHA-F")

    star = "*" * 3
    text = (f"{star} Begin Patch\n{star} Move File: f -> g\n"
            f"{star} Update File: g\n@@\n-old\n+new\n{star} End Patch\n")
    ops, err = parse_v4a_patch(text)
    assert err is None
    resolved = _resolve_operations(ops, Store())
    assert not resolved.errors, resolved.errors
    updates = [c for c in resolved.commits if c.op.file_path == "g"]
    assert updates and updates[0].read_sha == "SHA-F"


@pytest.mark.parametrize("source, body", [
    ("timeout = 1.5\n", "-timeout = 1\n+timeout = 2\n"),
    ("x = 1.0\n", "-x = 1\n+x = 2\n"),
])
def test_decimal_continuation_is_refused(source, body):
    result, store = _apply(source, body)
    assert not result.success
    assert store.files["f"] == source and store.writes == []


@pytest.mark.parametrize("source, body, expected", [
    ("items.append(x)\n", "-.append(x)\n+.extend(x)\n", "items.extend(x)\n"),
    ("foo(bar)\n", "-(bar)\n+(baz)\n", "foo(baz)\n"),
    ("v = 1 + 2.5\n", "-1 +\n+3 +\n", "v = 3 + 2.5\n"),
])
def test_punctuation_edged_needles_may_sit_inside_a_line(source, body, expected):
    result, store = _apply(source, body)
    assert result.success, result.error
    assert store.files["f"] == expected


def test_crlf_mid_line_edit_keeps_crlf():
    result, store = _apply("a = 1\r\nvalue = old\r\nz = 2\r\n", "-= old\n+= new\n")
    assert result.success, result.error
    assert store.files["f"] == "a = 1\r\nvalue = new\r\nz = 2\r\n"


def test_nfkc_and_unicode_space_equivalences_match_whole_lines():
    result, store = _apply("x\u00a0=\u30001\n", "-x = 1\n+x = 2\n")
    assert result.success, result.error
    assert store.files["f"] == "x = 2\n"
