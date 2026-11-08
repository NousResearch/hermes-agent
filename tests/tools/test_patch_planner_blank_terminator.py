"""An empty patch line is a whole physical blank line, never a line terminator.

Regression for PR #122566 audit finding: ``-`` (empty removed line) matched the
``\\n`` that ends the previous line, so ``foo\\nbar\\n`` + ``-`` / ``-bar`` wrote
``foo`` joined to the replacement instead of refusing.
"""
import itertools

import pytest

from tests.tools.test_patch_planner_indent_carryover import Store
from tools.file_operations_common import ReadResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

STAR = "*" * 3


def _run(files, body_by_file):
    text = f"{STAR} Begin Patch\n"
    for name, body in body_by_file:
        text += f"{STAR} Update File: {name}\n@@\n{body}"
    text += f"{STAR} End Patch\n"
    ops, err = parse_v4a_patch(text)
    assert err is None, err
    store = Store(files)
    return apply_v4a_operations(ops, store), store


@pytest.mark.parametrize("source, body", [
    ("foo\nbar\n", "-\n-bar\n+NEW\n"),
    ("foo\nbar\n", "-foo\n-\n"),
    ("foo\r\nbar\r\n", "-\n-bar\n-\n"),
])
def test_empty_line_never_matches_a_terminator(source, body):
    result, store = _run(source, [("f", body)])
    assert not result.success
    assert store.files["f"] == source and not store.writes


class _TwoFiles(Store):
    def __init__(self, files):
        super().__init__("")
        self.files = dict(files)

    def read_file_raw(self, path):
        if path not in self.files:
            return ReadResult(content="", total_lines=0, file_size=0, error="missing", not_found=True)
        text = self.files[path]
        return ReadResult(content=text, total_lines=text.count("\n") + 1, file_size=len(text.encode()))


def test_false_blank_in_one_file_refuses_every_file():
    files = {"a.py": "foo\nbar\n", "b.py": "old\n"}
    text = (f"{STAR} Begin Patch\n{STAR} Update File: b.py\n@@\n-old\n+NEW\n"
            f"{STAR} Update File: a.py\n@@\n-foo\n-\n{STAR} End Patch\n")
    ops, err = parse_v4a_patch(text)
    assert err is None
    store = _TwoFiles(files)
    result = apply_v4a_operations(ops, store)
    assert not result.success
    assert store.files == files and not store.writes


@pytest.mark.parametrize("source, body", [
    ("bar\n    baz\n", "-bar\n-    \n+NEW\n"),
    ("foo   \nbar\n", "-   \n-bar\n+NEW\n"),
    ("foo\r\n  bar\r\n", "-foo\n-  \n"),
    ("   baz\n", "-   \n"),
    ("foo   \nbar\n", "-   \n"),
])
def test_whitespace_only_line_never_matches_a_terminator_or_indent(source, body):
    result, store = _run(source, [("f", body)])
    assert not result.success
    assert store.files["f"] == source and not store.writes


def test_real_whitespace_only_line_is_still_deleted():
    result, store = _run("foo\n   \nbar\n", [("f", "-foo\n-   \n")])
    assert result.success, result.error
    assert store.files["f"] == "bar\n"


@pytest.mark.parametrize("source, body, expected", [
    ("foo\n\nbar\n", "-\n", "foo\nbar\n"),
    ("foo\n\nbar\n", "-\n-bar\n+NEW\n", "foo\nNEW\n"),
    ("foo\n\nbar\n", "-foo\n-\n", "bar\n"),
    ("foo\r\n\r\nbar\r\n", "-foo\n-\n", "bar\r\n"),
])
def test_real_blank_line_still_deleted(source, body, expected):
    result, store = _run(source, [("f", body)])
    assert result.success, result.error
    assert store.files["f"] == expected


def _lines(source: str):
    """Physical lines of a terminated file (the closed oracle domain)."""
    eol = "\r\n" if "\r\n" in source else "\n"
    return source.split(eol)[:-1], eol


def _expected(source: str, search: list[str], repl: list[str]):
    """Exact whole-line oracle: refuse unless exactly one physical window matches."""
    lines, eol = _lines(source)
    n = len(search)
    hits = [i for i in range(len(lines) - n + 1) if lines[i:i + n] == search]
    if len(hits) != 1:
        return None
    i = hits[0]
    return "".join(x + eol for x in lines[:i] + repl + lines[i + n:])


def _cases():
    """Files and removal patches over {blank, a, b}, LF and CRLF."""
    alpha = ["", "a", "b"]
    for flen in range(1, 5):
        for content in itertools.product(alpha, repeat=flen):
            for eol in ("\n", "\r\n"):
                source = "".join(x + eol for x in content)
                for slen in range(1, 4):
                    for search in itertools.product(alpha, repeat=slen):
                        for repl in ([], ["X"]):
                            yield source, list(search), repl


def _check(source, search, repl):
    body = "".join(f"-{s}\n" for s in search) + "".join(f"+{r}\n" for r in repl)
    result, store = _run(source, [("f", body)])
    want = _expected(source, search, repl)
    if want is None:
        assert not result.success, (source, body)
        assert store.files["f"] == source and not store.writes, (source, body)
        return False
    assert result.success, (source, body, result.error)
    assert store.files["f"] == want, (source, body, store.files["f"])
    return True


def test_removed_empty_lines_agree_with_exact_line_oracle():
    """Bounded exhaustive agreement with the exact whole-line oracle."""
    outcomes = [_check(*case) for case in _cases()]
    assert len(outcomes) > 5000 and any(outcomes) and not all(outcomes)
