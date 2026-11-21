"""Round-2 witness safety for the production V4A planner.

Nine standalone cases. Expected bytes are hand-computed, not taken from
the unhinted oracle or from planner internals: a refusal writes nothing
and leaves every original byte; a unique edit lands only on the matched
bytes, including CR line endings. Hinted cases stay outside the oracle.

The known baseline write fails as ``_KnownPlannerDefect``. It is not an
xfail. A parse failure, an exception, or any other write is a real error.
"""

import pytest

from tools.file_operations_common import PatchResult, ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

NBSP = "\u00a0"
IDEOGRAPHIC_SPACE = "\u3000"
class _KnownPlannerDefect(AssertionError):
    """The pinned baseline shape, and nothing else."""


_WITNESSES = (
    (
        "test_ascii_normalized_partial_decoy_refuses",
        "ascii-normalized-partial-decoy",
    ),
    (
        "test_nbsp_normalized_substring_refuses",
        "nbsp-normalized-substring",
    ),
    (
        "test_ideographic_space_normalized_substring_refuses",
        "ideographic-space-normalized-substring",
    ),
    (
        "test_partial_consumption_then_unicode_substring_refuses",
        "partial-consumption-unicode-substring",
    ),
    (
        "test_false_noop_swallows_normalized_substring_refusal",
        "false-noop-swallows-refusal",
    ),
    (
        "test_partial_write_via_false_noop_refuses",
        "partial-write-false-noop",
    ),
    (
        "test_cross_file_false_noop_is_atomic",
        "cross-file-false-noop",
    ),
    (
        "test_crlf_preserved_on_non_final_changed_line",
        "crlf-non-final-changed-line",
    ),
    (
        "test_crlf_preserved_inside_one_multiline_replacement",
        "crlf-multiline-replacement",
    ),
)


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
    if error or not ops:
        pytest.fail("patch did not parse")
    store = MemoryFiles(files)
    return apply_v4a_operations(ops, store), store


def _update(body):
    return "*** Begin Patch\n" + body + "\n*** End Patch\n"


def _witnessed_write(result, store, files, writes, modified):
    return (
        result.success is True
        and result.error is None
        and result.no_change is False
        and result.files_modified == modified
        and not result.files_created
        and not result.files_deleted
        and store.files == files
        and store.writes == writes
    )


def _disagree(shape, result, store):
    pytest.fail(f"unrelated disagreement for {shape}")


def _refuse_or_known(result, store, source, files, writes, modified, shape):
    if result.success is False and not store.writes and store.files == source:
        return
    if _witnessed_write(result, store, files, writes, modified):
        raise _KnownPlannerDefect(shape)
    _disagree(shape, result, store)


def _land_or_known(result, store, path, expected, witnessed, modified, shape):
    if (
        result.success is True
        and result.error is None
        and store.files.get(path) == expected
        and store.writes == [(path, expected)]
    ):
        return
    if _witnessed_write(
        result, store, {path: witnessed}, [(path, witnessed)], modified
    ):
        raise _KnownPlannerDefect(shape)
    _disagree(shape, result, store)


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
    witnessed = "# CANARY_HINT_TOKEN\nCHANGED0\n"
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "ascii-normalized-partial-decoy",
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
    witnessed = "# CANARY_HINT_TOKEN\nCHANGED0\n"
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "nbsp-normalized-substring",
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
    witnessed = "# CANARY_HINT_TOKEN\nCHANGED0\n"
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "ideographic-space-normalized-substring",
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
    witnessed = 'QQame = "edge-1"\nCHANGED0\n'
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "partial-consumption-unicode-substring",
    )


def test_false_noop_swallows_normalized_substring_refusal():
    """A replacement that already exists must not hide a bad search."""
    source = {"f.py": "keep = hostish_d\nvalue  =  10\n"}
    body = "*** Update File: f.py\n@@\n-value = 1\n+keep = hostish_d"
    result, store = _apply(source, _update(body))
    witnessed = "keep = hostish_d\nkeep = hostish_d0\n"
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "false-noop-swallows-refusal",
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
    witnessed = "value=9\nkeep = hostish_d\n"
    _refuse_or_known(
        result,
        store,
        source,
        {"f.py": witnessed},
        [("f.py", witnessed)],
        ["f.py"],
        "partial-write-false-noop",
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
    _refuse_or_known(
        result,
        store,
        source,
        {
            "a.py": "value=9\nkeep = hostish_d\n",
            "b.py": "new\n",
        },
        [
            ("a.py", "value=9\nkeep = hostish_d\n"),
            ("b.py", "new\n"),
        ],
        ["a.py", "b.py"],
        "cross-file-false-noop",
    )


def test_crlf_preserved_on_non_final_changed_line():
    """A non-final replacement keeps the file's CR line endings."""
    source = {"f.py": "a = 1\r\nb = 2\r\nc = 3\r\n"}
    body = "*** Update File: f.py\n@@\n a = 1\n-b = 2\n+B\n c = 3"
    expected = "a = 1\r\nB\r\nc = 3\r\n"
    result, store = _apply(source, _update(body))
    _land_or_known(
        result,
        store,
        "f.py",
        expected,
        "a = 1\nB\nc = 3\n",
        ["f.py"],
        "crlf-non-final-changed-line",
    )


def test_crlf_preserved_inside_one_multiline_replacement():
    """Every line of a multi-line replacement keeps CR."""
    source = {"f.py": "foo\r\nbar\r\n"}
    body = "*** Update File: f.py\n@@\n-foo\n-bar\n+FOO\n+BAR"
    expected = "FOO\r\nBAR\r\n"
    result, store = _apply(source, _update(body))
    _land_or_known(
        result,
        store,
        "f.py",
        expected,
        "FOO\nBAR\n",
        ["f.py"],
        "crlf-multiline-replacement",
    )


def test_witnesses_have_no_xfail_markers():
    """The nine cases are real failures, not expected xfails."""
    assert len(_WITNESSES) == 9
    for name, _shape in _WITNESSES:
        marks = getattr(globals()[name], "pytestmark", [])
        if not isinstance(marks, (list, tuple)):
            marks = [marks]
        xfails = [mark for mark in marks if getattr(mark, "name", None) == "xfail"]
        assert not xfails, name
    assert not issubclass(pytest.fail.Exception, _KnownPlannerDefect)
    assert not issubclass(RuntimeError, _KnownPlannerDefect)


def test_failing_parse_is_not_a_known_planner_defect(monkeypatch):
    """A parse failure inside a witness must fail, not raise the known defect."""

    def bad(_text):
        return [], "syntax error"

    monkeypatch.setattr(f"{__name__}.parse_v4a_patch", bad)
    with pytest.raises(pytest.fail.Exception, match="patch did not parse"):
        test_ascii_normalized_partial_decoy_refuses()


def test_unrelated_wrong_write_is_not_swallowed(monkeypatch):
    """A different wrong write is not the witnessed baseline defect."""

    def other_write(_ops, file_ops):
        file_ops.write_file("f.py", "NOT-WITNESSED\n")
        return PatchResult(success=True, files_modified=["f.py"])

    monkeypatch.setattr(f"{__name__}.apply_v4a_operations", other_write)
    with pytest.raises(pytest.fail.Exception, match="unrelated disagreement"):
        test_nbsp_normalized_substring_refuses()


def test_unrelated_crlf_write_is_not_swallowed(monkeypatch):
    """Mixed endings are not the witnessed all-LF context-window defect."""

    def mixed(_ops, file_ops):
        file_ops.write_file("f.py", "a = 1\r\nB\nc = 3\r\n")
        return PatchResult(success=True, files_modified=["f.py"])

    monkeypatch.setattr(f"{__name__}.apply_v4a_operations", mixed)
    with pytest.raises(pytest.fail.Exception, match="unrelated disagreement"):
        test_crlf_preserved_on_non_final_changed_line()


def test_planner_exception_is_not_swallowed_as_known_defect(monkeypatch):
    """An unexpected planner exception must not become the known defect."""

    def boom(_ops, _store):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(f"{__name__}.apply_v4a_operations", boom)
    with pytest.raises(RuntimeError, match="planner exploded"):
        test_partial_write_via_false_noop_refuses()
