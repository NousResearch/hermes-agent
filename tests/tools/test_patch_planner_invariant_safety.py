"""Named invariant safety checks for the production V4A planner.

Unhinted cases use real ``parse_v4a_patch`` / ``apply_v4a_operations`` against
an in-memory store. Hinted cases are handwritten safety checks only: a refusal
writes nothing, an unnamed file is never written, a named-file success is only
the unique expected bytes, and diagnostics do not quote source or hint text.
They do not choose a hint site or assume a hint window matched.

Each test is one invariant or one defect class. The known baseline shape
fails as ``_KnownPlannerDefect``; it is not an xfail. A different failure
stays a real error. Expected bytes are hand-computed.
"""

import pytest

from tools.file_operations_common import ReadResult, WriteResult
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

NBSP = "\u00a0"
IDEOGRAPHIC_SPACE = "\u3000"
class _KnownPlannerDefect(AssertionError):
    """The pinned baseline shape, and nothing else."""


class MemoryFiles:
    """In-memory stand-in. No disk writes."""

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


def _update(path, body):
    return f"*** Begin Patch\n*** Update File: {path}\n{body}\n*** End Patch\n"


def _apply(files, text):
    ops, error = parse_v4a_patch(text)
    if error or not ops:
        pytest.fail("patch did not parse")
    store = MemoryFiles(files)
    return apply_v4a_operations(ops, store), store


def _reported(result):
    return result.files_modified + result.files_created + result.files_deleted


def _clean_refusal(result, store, source):
    return (
        result.success is False
        and result.no_change is False
        and not store.writes
        and store.files == source
        and not _reported(result)
    )


def _landed(result, store, path, expected):
    return (
        result.success is True
        and result.error is None
        and store.files == {path: expected}
        and store.writes == [(path, expected)]
        and result.files_modified == [path]
    )


def _require_refusal(result, store, source, shape):
    if not _clean_refusal(result, store, source):
        pytest.fail(f"{shape} was not a clean refusal")


def _spec_open_unique_or_refusal(result, store, source, path, expected):
    """Hint windows are open: refuse unchanged, or write only ``expected``.

    A partial or wrong write to the named file is neither outcome. Other files
    stay byte-for-byte identical in both.
    """
    if (
        result.success is False
        and not store.writes
        and store.files == source
        and not _reported(result)
    ):
        return
    others = {name: text for name, text in source.items() if name != path}
    if (
        result.success is True
        and result.error is None
        and result.no_change is False
        and store.writes == [(path, expected)]
        and store.files.get(path) == expected
        and {name: store.files.get(name) for name in others} == others
        and set(_reported(result)) == {path}
    ):
        return
    pytest.fail("named file was partially or wrongly written")


def _diagnostics(result):
    return "\n".join(
        part for part in (result.error, result.note, result.diff) if part
    )


def _assert_absent(result, pairs):
    diagnostic = _diagnostics(result)
    for kind, token in pairs:
        if token and token in diagnostic:
            raise AssertionError(f"diagnostic contains {kind}")


def _prefix_false_success(source_text, shape):
    """A non-matching ``value = 1`` search must refuse, not write."""
    source = {"f.py": source_text}
    result, store = _apply(source, _update("f.py", "@@\n-value = 1\n+CHANGED"))
    if _clean_refusal(result, store, source):
        return
    if (
        result.success is True
        and store.files.get("f.py") == "CHANGED0\n"
        and store.writes == [("f.py", "CHANGED0\n")]
    ):
        raise _KnownPlannerDefect("prefix match left a suffix")
    pytest.fail(f"unrelated disagreement for {shape}")


# --- refusal atomicity: a refusal writes nothing ---------------------------


def test_refusal_atomicity_not_found():
    """A missing unhinted search leaves the only file untouched."""
    source = {"f.py": "a\n"}
    result, store = _apply(source, _update("f.py", "@@\n-old\n+new"))
    _require_refusal(result, store, source, "not-found")


def test_refusal_atomicity_ambiguous_exact():
    """Two identical unhinted lines refuse instead of taking the first."""
    source = {"f.py": "old\nold\n"}
    result, store = _apply(source, _update("f.py", "@@\n-old\n+new"))
    _require_refusal(result, store, source, "ambiguous-exact")


def test_refusal_atomicity_later_hunk_miss():
    """A later miss discards an earlier hunk that would have matched."""
    source = {"f.py": "a = old\nb = old\n"}
    body = "@@\n-a = old\n+a = new\n@@\n-missing = old\n+missing = new"
    result, store = _apply(source, _update("f.py", body))
    _require_refusal(result, store, source, "later-hunk-miss")


def test_refusal_atomicity_cross_file_miss():
    """A miss in a later file does not write the earlier file."""
    source = {"a.py": "old\n", "b.py": "other\n"}
    text = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old\n+new\n"
        "*** Update File: b.py\n@@\n-absent\n+replacement\n"
        "*** End Patch\n"
    )
    result, store = _apply(source, text)
    _require_refusal(result, store, source, "cross-file-miss")


def test_refusal_atomicity_partial_write_false_noop():
    """A swallowed later hunk must not keep an earlier edit."""
    source = {"f.py": "value=1\nkeep = hostish_d\n"}
    body = "@@\n-value=1\n+value=9\n@@\n-missing = old\n+keep = hostish_d"
    result, store = _apply(source, _update("f.py", body))
    if _clean_refusal(result, store, source):
        return
    kept = "value=9\nkeep = hostish_d\n"
    if result.success is True and store.files == {"f.py": kept} and store.writes == [("f.py", kept)]:
        raise _KnownPlannerDefect("earlier hunk kept after a swallowed refusal")
    pytest.fail("unrelated disagreement for partial-write-false-noop")


def test_extra_write_is_not_classified_as_known_false_noop(monkeypatch):
    from tools.file_operations_common import PatchResult

    def extra_write(_ops, store):
        store.write_file("f.py", "value=9\nkeep = hostish_d\n")
        store.write_file("other.py", "unexpected\n")
        return PatchResult(success=True, files_modified=["f.py", "other.py"])

    monkeypatch.setattr(f"{__name__}.apply_v4a_operations", extra_write)
    with pytest.raises(pytest.fail.Exception, match="unrelated disagreement"):
        test_refusal_atomicity_partial_write_false_noop()


def test_refusal_atomicity_cross_file_false_noop():
    """A swallowed hunk in one file must not allow another file to be written."""
    source = {"a.py": "value=1\nkeep = hostish_d\n", "b.py": "old\n"}
    text = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "@@\n-value=1\n+value=9\n"
        "@@\n-missing = old\n+keep = hostish_d\n"
        "*** Update File: b.py\n"
        "@@\n-old\n+new\n"
        "*** End Patch\n"
    )
    result, store = _apply(source, text)
    if _clean_refusal(result, store, source):
        return
    if (
        result.success is True
        and store.files.get("a.py") == "value=9\nkeep = hostish_d\n"
        and store.files.get("b.py") == "new\n"
        and store.files == {"a.py": "value=9\nkeep = hostish_d\n", "b.py": "new\n"}
        and store.writes == [("a.py", "value=9\nkeep = hostish_d\n"), ("b.py", "new\n")]
    ):
        raise _KnownPlannerDefect("swallowed hunk still wrote both files")
    pytest.fail("unrelated disagreement for cross-file-false-noop")


# --- no false success: a refusal is not success or a no-op write -----------


def test_no_false_success_when_replacement_is_absent():
    """A missing search whose replacement is also absent is a refusal."""
    source = {"f.py": "keep = hostish_d\n"}
    result, store = _apply(
        source, _update("f.py", "@@\n-missing = old\n+other = new")
    )
    _require_refusal(result, store, source, "replacement-absent")


def test_unique_replacement_applies_exact_bytes():
    """A unique whole-line edit is a real write, not a no-op success."""
    source = {"f.py": "a\nold\ntail\n"}
    expected = "a\nnew\ntail\n"
    result, store = _apply(source, _update("f.py", "@@\n-old\n+new"))
    assert _landed(result, store, "f.py", expected)
    assert store.writes == [("f.py", expected)]
    assert result.no_change is False


def test_no_false_success_substring_prefix_write():
    """``value = 10`` is not a successful match for ``value = 1``."""
    _prefix_false_success("value = 10\n", "substring-prefix-write")


def test_no_false_success_normalized_space_prefix_write():
    """A spaced ``value = 10`` line is not a match for ``value = 1``."""
    _prefix_false_success("value  =  10\n", "normalized-space-prefix-write")


def test_no_false_success_nbsp_prefix_write():
    """NBSP in a longer token is not an ASCII ``value = 1`` match."""
    raw = f"value{NBSP}={NBSP}10\n"
    _prefix_false_success(raw, "nbsp-normalized-prefix-write")


def test_no_false_success_ideographic_space_prefix_write():
    """An ideographic space is not an ASCII ``value = 1`` match."""
    raw = f"value{IDEOGRAPHIC_SPACE}={IDEOGRAPHIC_SPACE}10\n"
    _prefix_false_success(raw, "ideographic-space-prefix-write")


def test_no_false_success_false_noop_swallows_refusal():
    """A replacement that already exists must not hide a missing search."""
    source = {"f.py": "keep = hostish_d\n"}
    body = "@@\n-missing = old\n+keep = hostish_d"
    result, store = _apply(source, _update("f.py", body))
    if _clean_refusal(result, store, source):
        return
    if (
        result.success is True
        and result.no_change is False
        and store.files == source
        and store.writes == [("f.py", source["f.py"])]
    ):
        raise _KnownPlannerDefect("missing search reported as success")
    pytest.fail("unrelated disagreement for false-noop-swallows-refusal")


# --- EOL preserved per file -------------------------------------------------


def test_eol_preserved_on_whole_line_crlf_replacement():
    """A unique whole-line replacement keeps the file's CR endings."""
    source = {"f.py": "a = 1\r\nb = 2\r\nc = 3\r\n"}
    expected = "a = 1\r\nB\r\nc = 3\r\n"
    result, store = _apply(source, _update("f.py", "@@\n-b = 2\n+B"))
    assert _landed(result, store, "f.py", expected)


def test_eol_preserved_per_file():
    """Each file keeps its own ending when both edits are unique."""
    source = {"a.py": "old\n", "b.py": "old\r\n"}
    text = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old\n+new\n"
        "*** Update File: b.py\n@@\n-old\n+new\n"
        "*** End Patch\n"
    )
    result, store = _apply(source, text)
    assert result.success is True, "unique two-file edit did not apply"
    assert store.files["a.py"] == "new\n"
    assert store.files["b.py"] == "new\r\n"
    assert store.writes == [("a.py", "new\n"), ("b.py", "new\r\n")]


def test_eol_preserved_when_hunk_has_context():
    """Context lines in a CRLF hunk must not strip CR from that window."""
    source = {"f.py": "a = 1\r\nb = 2\r\nc = 3\r\n"}
    expected = "a = 1\r\nB\r\nc = 3\r\n"
    observed = "a = 1\nB\nc = 3\n"
    body = "@@\n a = 1\n-b = 2\n+B\n c = 3"
    result, store = _apply(source, _update("f.py", body))
    if _landed(result, store, "f.py", expected):
        return
    if result.success is True and store.files == {"f.py": observed} and store.writes == [("f.py", observed)]:
        raise _KnownPlannerDefect("context hunk stripped CR inside its window")
    pytest.fail("unrelated disagreement for crlf-context-window")


def test_eol_preserved_inside_multiline_replacement():
    """Every line of a multi-line replacement keeps CR."""
    source = {"f.py": "foo\r\nbar\r\n"}
    expected = "FOO\r\nBAR\r\n"
    result, store = _apply(source, _update("f.py", "@@\n-foo\n-bar\n+FOO\n+BAR"))
    if _landed(result, store, "f.py", expected):
        return
    if result.success is True and store.files == {"f.py": "FOO\nBAR\n"} and store.writes == [("f.py", "FOO\nBAR\n")]:
        raise _KnownPlannerDefect("multiline replacement dropped CR")
    pytest.fail("unrelated disagreement for crlf-multiline-replacement")


def test_eol_preserved_on_addition_only():
    """An EOF insertion in a CRLF file uses that file's ending."""
    source = {"f.py": "first\r\n"}
    expected = "first\r\nlast\r\n"
    result, store = _apply(source, _update("f.py", "@@\n+last"))
    if _landed(result, store, "f.py", expected):
        return
    if result.success is True and store.files == {"f.py": "first\r\nlast\n"} and store.writes == [("f.py", "first\r\nlast\n")]:
        raise _KnownPlannerDefect("added line used LF in a CRLF file")
    pytest.fail("unrelated disagreement for crlf-addition-only")


# --- diagnostics quote neither source nor hint text ------------------------


def test_errors_omit_source_on_not_found():
    """A not-found error quotes neither the nearby line nor the search."""
    secret = "OPAQUE_SECRET_7X4Y"
    search = "ABSENT_SEARCH_7X4Y"
    source = {"f.py": f"password = {secret}\n"}
    result, store = _apply(source, _update("f.py", f"@@\n-{search}\n+changed"))
    _require_refusal(result, store, source, "not-found-excerpt")
    assert result.error
    _assert_absent(
        result, (("source excerpt", secret), ("search excerpt", search))
    )


def test_errors_omit_source_on_ambiguous():
    """An ambiguity error does not quote the repeated source line."""
    secret = "OPAQUE_LINE_7X4Y"
    source = {"f.py": f"{secret}\n{secret}\n"}
    result, store = _apply(source, _update("f.py", f"@@\n-{secret}\n+changed"))
    _require_refusal(result, store, source, "ambiguous-excerpt")
    assert result.error
    _assert_absent(result, (("source excerpt", secret),))


def test_errors_omit_source_on_similar_section():
    """A similar-section suffix names lines, not the nearby source."""
    secret = "OPAQUE_SECRET_7X4Y"
    source = {"f.py": f"password = {secret}\n"}
    result, store = _apply(source, _update("f.py", "@@\n-tail\n+new"))
    _require_refusal(result, store, source, "similar-section-excerpt")
    assert result.error
    _assert_absent(result, (("source excerpt", secret), ("search excerpt", "tail")))


def test_errors_omit_source_on_redacted_matches():
    """A multi-match diagnostic redacts every listed source line."""
    secret = "OPAQUE_SRC_7X4Y"
    line = f"call(foo,  bar) # {secret}"
    source = {"f.py": f"{line}\n{line}\n"}
    result, store = _apply(source, _update("f.py", "@@\n-call(foo, bar)\n+changed"))
    _require_refusal(result, store, source, "redacted-matches")
    assert result.error
    _assert_absent(
        result,
        (
            ("source excerpt", secret),
            ("source line", line),
            ("search excerpt", "call(foo, bar)"),
        ),
    )


def test_errors_omit_hint_and_source_on_hinted_miss():
    """A handwritten hinted miss quotes neither the hint nor nearby source."""
    hint = "OPAQUE_HINT_7X4Y"
    secret = "OPAQUE_SECRET_7X4Y"
    source = {"f.py": f"password = {secret}\nvalue = 1\n"}
    body = f"@@ {hint} @@\n-password = different\n+changed"
    result, store = _apply(source, _update("f.py", body))
    _require_refusal(result, store, source, "hinted-miss")
    assert result.error
    _assert_absent(
        result,
        (
            ("hint excerpt", hint),
            ("source excerpt", secret),
            ("search excerpt", "different"),
        ),
    )


def test_errors_omit_hint_on_hinted_ambiguous():
    """A handwritten hinted ambiguity does not quote the hint or the line."""
    hint = "OPAQUE_HINT_7X4Y"
    secret = "OPAQUE_SRC_7X4Y"
    source = {"f.py": f"{hint}\nold\nold\nsecret = {secret}\n"}
    result, store = _apply(source, _update("f.py", f"@@ {hint} @@\n-old\n+new"))
    _require_refusal(result, store, source, "hinted-ambiguous")
    assert result.error
    _assert_absent(
        result, (("hint excerpt", hint), ("source excerpt", secret))
    )


def test_hinted_update_does_not_write_unnamed_file():
    """A hint that occurs only in an unnamed file does not write that file.

    The named file is the unique line ``old``. Hint-window bounds are open, so
    a refusal may leave every file unchanged, and a success may write only
    ``new`` with that file's ending. Any other named-file bytes fail.
    """
    hint = "OPAQUE_HINT_7X4Y"
    secret = "OPAQUE_OTHER_7X4Y"
    source = {
        "a.py": "old\n",
        "b.py": f"{hint}\nold\nsecret = {secret}\n",
    }
    result, store = _apply(source, _update("a.py", f"@@ {hint} @@\n-old\n+new"))
    _spec_open_unique_or_refusal(result, store, source, "a.py", "new\n")
    _assert_absent(result, (("hint excerpt", hint), ("source excerpt", secret)))


# --- named defect classes outside the four safety invariants ---------------


def test_defect_blank_delete_keeps_line():
    """Deleting a non-blank line removes that pair, including its ending."""
    source = {"f.py": "a\nold\nb\n"}
    expected = "a\nb\n"
    result, store = _apply(source, _update("f.py", "@@\n-old\n"))
    if _landed(result, store, "f.py", expected) and store.writes == [
        ("f.py", expected)
    ]:
        return
    if (
        result.success is True
        and store.files.get("f.py") == "a\n\nb\n"
        and store.writes == [("f.py", "a\n\nb\n")]
    ):
        raise _KnownPlannerDefect("deletion left a blank terminated line")
    pytest.fail("unrelated disagreement for blank-delete-keeps-line")


def test_defect_empty_search_refuses_unique_blank():
    """A unique blank line is a deletion, not an empty-search refusal."""
    source = {"f.py": "a\n\nb\n"}
    expected = "a\nb\n"
    result, store = _apply(source, _update("f.py", "@@\n-\n"))
    if _landed(result, store, "f.py", expected):
        return
    if (
        _clean_refusal(result, store, source)
        and result.error
        and "empty search text" in result.error
    ):
        raise _KnownPlannerDefect("unique blank deletion refused as empty search")
    pytest.fail("unrelated disagreement for empty-search")


def test_defect_exact_tier_lost_to_substring():
    """A unique exact line wins over a longer substring decoy."""
    source = {"f.py": "value = 10\nvalue = 1\n"}
    expected = "value = 10\nCHANGED\n"
    result, store = _apply(source, _update("f.py", "@@\n-value = 1\n+CHANGED"))
    if _landed(result, store, "f.py", expected) and store.writes == [
        ("f.py", expected)
    ]:
        return
    if (
        _clean_refusal(result, store, source)
        and result.error
        and "ambiguous" in result.error
    ):
        raise _KnownPlannerDefect("exact line refused beside a substring decoy")
    pytest.fail("unrelated disagreement for exact-tier-lost-to-substring")
