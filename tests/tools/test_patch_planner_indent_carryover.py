"""Normalized whole-line matches carry the matched source indent onto added lines.

Exact matches stay byte-verbatim; inconsistent or already-indented searches write verbatim.
"""
from types import SimpleNamespace

from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


class Store:
    def __init__(self, text):
        self.files = {"f": text}
        self.writes = []

    def read_file_raw(self, path):
        return SimpleNamespace(error=None, content=self.files[path], not_found=False)

    def write_file(self, path, content, pre_content=None):
        self.writes.append(path)
        self.files[path] = content
        return SimpleNamespace(error=None)


def _apply(text, body):
    star = "*" * 3
    ops, err = parse_v4a_patch(f"{star} Begin Patch\n{star} Update File: f\n@@\n{body}{star} End Patch\n")
    assert err is None
    store = Store(text)
    return apply_v4a_operations(ops, store), store


def test_normalized_match_carries_the_consistent_source_indent():
    result, store = _apply("def t(x):\n    y = 1\n    return y\n", "-y = 1\n+y = 3\n+z = 4\n return y\n")
    assert result.success, result.error
    assert store.files["f"] == "def t(x):\n    y = 3\n    z = 4\n    return y\n"


def test_inconsistent_source_indent_is_written_verbatim():
    result, store = _apply("def t(x):\n    y = 1\n  return y\n", "-y = 1\n+y = 3\n return y\n")
    assert result.success, result.error
    assert store.files["f"] == "def t(x):\ny = 3\n  return y\n"


def test_exact_match_stays_verbatim():
    result, store = _apply("y = 1\n    return y\n", "-y = 1\n+  y = 3\n")
    assert result.success, result.error
    assert store.files["f"] == "  y = 3\n    return y\n"


def test_indented_search_line_gets_no_carry_over():
    result, store = _apply("    y  =  1\n", "-  y = 1\n+y = 3\n")
    assert result.success, result.error
    assert store.files["f"] == "y = 3\n"
