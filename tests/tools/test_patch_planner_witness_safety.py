"""Round-2 witness safety for the production V4A planner.

Nine standalone cases. Expected bytes are hand-computed, not taken from
the unhinted oracle or from planner internals: a refusal writes nothing
and leaves every original byte; a unique edit lands only on the matched
bytes, including CR line endings. Hinted cases stay outside the oracle.
"""

import pytest

from tools.file_operations_common import ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

NBSP = "\u00a0"
IDEOGRAPHIC_SPACE = "\u3000"
_FIXED = "fixed in t_6e12e276"


def _known(shape):
    return f"known V4A defect {shape}, {_FIXED}"


class MemoryFiles:
    """In-memory stand-in for the planner's file ops. No disk writes."""

    def __init__(self, files):
        self.files = dict(files)
        self.writes = []

    def read_file_raw(self, path):
        if path not in self.files:
            return ReadResult(
                content="",
                total_lines=0,
                file_size=0,
                error="file not found",
                not_found=True,
            )
        text = self.files[path]
        return ReadResult(
            content=text,
            total_lines=text.count("\n") + 1,
            file_size=len(text.encode()),
        )

    def write_file(self, path, content, pre_content=None):
        self.writes.append((path, content))
        self.files[path] = content
        return WriteResult(bytes_written=len(content.encode()), verified=True)


def _apply(files, text):
    ops, error = parse_v4a_patch(text)
    assert error is None, error
    store = MemoryFiles(files)
    return apply_v4a_operations(ops, store), store


def _update(body):
    return "*** Begin Patch\n" + body + "\n*** End Patch\n"


def _assert_refused(result, store, source):
    assert result.success is False
    assert store.writes == []
    assert store.files == source


def _assert_landed(result, store, path, expected):
    assert result.success is True, result.error
    assert store.files[path] == expected
    assert store.writes == [(path, expected)]


@pytest.mark.xfail(
    strict=True,
    reason=_known("ascii-normalized-partial-decoy"),
)
def test_ascii_normalized_partial_decoy_refuses():
    """``value  =  10`` is not a match for ``value = 1``."""
    source = {"f.py": "# CANARY_HINT_TOKEN\nvalue  =  10\n"}
    body = (
        "*** Update File: f.py\n"
        "@@ CANARY_HINT_TOKEN @@\n"
        "-value = 1\n"
        "+CHANGED"
    )
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("nbsp-normalized-substring"),
)
def test_nbsp_normalized_substring_refuses():
    """A NBSP line is not a match for an ASCII ``value = 1`` search."""
    raw = f"# CANARY_HINT_TOKEN\nvalue{NBSP}={NBSP}10\n"
    source = {"f.py": raw}
    body = (
        "*** Update File: f.py\n"
        "@@ CANARY_HINT_TOKEN @@\n"
        "-value = 1\n"
        "+CHANGED"
    )
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("ideographic-space-normalized-substring"),
)
def test_ideographic_space_normalized_substring_refuses():
    """An ideographic-space line is not an ASCII ``value = 1`` match."""
    raw = f"# CANARY_HINT_TOKEN\nvalue{IDEOGRAPHIC_SPACE}={IDEOGRAPHIC_SPACE}10\n"
    source = {"f.py": raw}
    body = (
        "*** Update File: f.py\n"
        "@@ CANARY_HINT_TOKEN @@\n"
        "-value = 1\n"
        "+CHANGED"
    )
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("partial-consumption-unicode-substring"),
)
def test_partial_consumption_then_unicode_substring_refuses():
    """A later unmatched hunk must discard an earlier partial edit."""
    source = {"f.py": f'ZZZname = "edge-1"\nvalue{NBSP}={NBSP}10\n'}
    body = (
        "*** Update File: f.py\n"
        "@@\n"
        "-ZZZn\n"
        "+QQ\n"
        "@@\n"
        "-value = 1\n"
        "+CHANGED"
    )
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("false-noop-swallows-refusal"),
)
def test_false_noop_swallows_normalized_substring_refusal():
    """A replacement that already exists must not hide a bad search."""
    source = {"f.py": "keep = hostish_d\nvalue  =  10\n"}
    body = "*** Update File: f.py\n@@\n-value = 1\n+keep = hostish_d"
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("partial-write-false-noop"),
)
def test_partial_write_via_false_noop_refuses():
    """A swallowed later hunk must not keep an earlier edit."""
    source = {"f.py": "value=1\nkeep = hostish_d\n"}
    body = (
        "*** Update File: f.py\n"
        "@@\n"
        "-value=1\n"
        "+value=9\n"
        "@@\n"
        "-value = 1\n"
        "+keep = hostish_d"
    )
    result, store = _apply(source, _update(body))
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("cross-file-false-noop"),
)
def test_cross_file_false_noop_is_atomic():
    """A swallowed hunk in one file must not allow another file to be written."""
    source = {"a.py": "value=1\nkeep = hostish_d\n", "b.py": "old\n"}
    text = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "@@\n"
        "-value=1\n"
        "+value=9\n"
        "@@\n"
        "-value = 1\n"
        "+keep = hostish_d\n"
        "*** Update File: b.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )
    result, store = _apply(source, text)
    _assert_refused(result, store, source)


@pytest.mark.xfail(
    strict=True,
    reason=_known("crlf-non-final-changed-line"),
)
def test_crlf_preserved_on_non_final_changed_line():
    """A non-final replacement keeps the file's CR line endings."""
    source = {"f.py": "a = 1\r\nb = 2\r\nc = 3\r\n"}
    body = "*** Update File: f.py\n@@\n a = 1\n-b = 2\n+B\n c = 3"
    expected = "a = 1\r\nB\r\nc = 3\r\n"
    result, store = _apply(source, _update(body))
    _assert_landed(result, store, "f.py", expected)


@pytest.mark.xfail(
    strict=True,
    reason=_known("crlf-multiline-replacement"),
)
def test_crlf_preserved_inside_one_multiline_replacement():
    """Every line of a multi-line replacement keeps CR."""
    source = {"f.py": "foo\r\nbar\r\n"}
    body = "*** Update File: f.py\n@@\n-foo\n-bar\n+FOO\n+BAR"
    expected = "FOO\r\nBAR\r\n"
    result, store = _apply(source, _update(body))
    _assert_landed(result, store, "f.py", expected)
