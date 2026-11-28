"""The whole-file write guard vs. a file read in windows.

A ``read_file`` page is capped (default 2000 lines, ~100K chars), so a large file can only
be seen in windows. Reading every window must authorize the overwrite, and one window must
not. Two defects this pins:

* the last page of a fully paged read was stamped "partial" from its own ``offset``, so
  ``file_state`` refused the overwrite the read had just authorized, and its advice ("read
  the remaining pages") led to the same stamp and the same refusal;
* read coverage was recorded as ``offset + limit - 1`` rather than where the page stopped,
  claiming lines never served -- a later page could then "complete" the file and bless an
  overwrite of unseen content.

These are unit tests over the tracking/guarding layer, which is where both defects lived.
They do not build a shell: the tool-level behavior is exercised by
``tests/manual/write_guard_paged_read_check.py``, which needs a working terminal backend
(the pytest home-IO guard blocks this install's real one).
"""

import sys

import pytest

from tools.file_tools_read_tracking import (
    _baseline_key,
    _file_version,
    _has_full_write_baseline,
    _mark_full_write_baseline,
    _note_read_coverage,
    _read_tracker,
)
from tools.file_operations_common import ReadResult


def _task(name):
    return _read_tracker.setdefault(name, {"full_write_baselines": {}, "blind_patches": {}})


def test_read_result_emits_next_offset_only_when_set():
    plain = ReadResult(content="a", total_lines=1, file_size=2).to_dict()
    assert "next_offset" not in plain

    truncated = ReadResult(content="a", total_lines=10, file_size=99,
                           truncated=True, next_offset=501).to_dict()
    assert truncated["next_offset"] == 501


def test_coverage_merges_windows_into_a_complete_read():
    data = _task("cov-merge")
    version = ("v",)

    complete, _ = _note_read_coverage(data, "C:/x/f.md", version, 1, 500, 3000, False)
    assert complete is False
    complete, _ = _note_read_coverage(data, "C:/x/f.md", version, 501, 1000, 3000, False)
    assert complete is False
    complete, _ = _note_read_coverage(data, "C:/x/f.md", version, 1001, 3000, 3000, False)
    assert complete is True


def test_partial_coverage_never_reports_complete():
    data = _task("cov-gap")
    _note_read_coverage(data, "C:/x/g.md", ("v",), 1, 500, 3000, False)
    complete, _ = _note_read_coverage(data, "C:/x/g.md", ("v",), 2501, 3000, 3000, False)
    assert complete is False, "a gap of unseen lines must not count as a whole-file read"


def test_redacted_page_is_reported_redacted():
    data = _task("cov-redacted")
    complete, redacted = _note_read_coverage(data, "C:/x/h.md", ("v",), 1, 10, 10, True)
    assert complete is True and redacted is True


def test_new_version_restarts_coverage():
    data = _task("cov-version")
    _note_read_coverage(data, "C:/x/i.md", ("v1",), 1, 500, 3000, False)
    complete, _ = _note_read_coverage(data, "C:/x/i.md", ("v2",), 2501, 3000, 3000, False)
    assert complete is False, "pages of a file that changed do not describe the new bytes"


def test_baseline_round_trip_authorizes_later_write(tmp_path):
    path = tmp_path / "round.md"
    path.write_text("one\ntwo\n", encoding="utf-8")
    version = _file_version(str(path))
    assert version is not None

    assert _has_full_write_baseline(str(path), "t-baseline") is False
    _mark_full_write_baseline(str(path), "t-baseline", version[-1].hex())
    assert _has_full_write_baseline(str(path), "t-baseline") is True


def test_baseline_is_dropped_when_the_file_changes(tmp_path):
    path = tmp_path / "changed.md"
    path.write_text("one\n", encoding="utf-8")
    _mark_full_write_baseline(str(path), "t-drop", _file_version(str(path))[-1].hex())

    path.write_text("one\ntwo\n", encoding="utf-8")
    assert _has_full_write_baseline(str(path), "t-drop") is False


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"),
                    reason="case-insensitive filesystem needed")
def test_baseline_key_ignores_path_case():
    assert _baseline_key("C:/Users/x/Notes.md") == _baseline_key("c:/users/X/notes.MD")


def test_baseline_key_is_exact_on_case_sensitive_filesystems():
    if sys.platform in ("win32", "darwin"):
        pytest.skip("case-insensitive platform")
    assert _baseline_key("/tmp/Notes.md") != _baseline_key("/tmp/notes.md")
