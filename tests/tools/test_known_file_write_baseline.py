"""Whole-file knowledge survives what does not change it.

A narrower reread of unchanged bytes keeps the write_file baseline, and so does the task's
own patch of a file it knew in full: ``write_file -> read_file -> patch -> write_file`` used
to refuse the last write although the same task made the only change. The baseline survives
the patch exactly when the task held one for the pre-patch bytes and the patch's write is what
is on disk. A blind patch, a sibling or external writer, or an incomplete paged read stays
refused.
"""

import json
import os

import pytest

from tools import file_operations
from tools.file_tools import clear_file_ops_cache
from tools.file_tools_read_tracking import reset_file_dedup
from tools.registry import registry

_BLIND = "Your patch changed this file"


def _call(name, path, task_id, **arguments):
    result = registry.dispatch(name, {"path": str(path), **arguments}, task_id=task_id)
    assert isinstance(result, str)
    return json.loads(result)


def test_partial_reread_keeps_an_unchanged_full_read_or_write(tmp_path):
    original = "first\nsecond\nthird\n"
    for source in ("read", "write", "pages"):
        task = f"known-{source}"
        path = tmp_path / f"{source}.txt"
        try:
            if source == "write":
                assert "error" not in _call("write_file", path, task, content=original)
            else:
                path.write_text(original, encoding="utf-8")
                if source == "pages":
                    for offset in (1, 2, 3):
                        assert "error" not in _call("read_file", path, task, offset=offset, limit=1)
                else:
                    assert "error" not in _call("read_file", path, task)
            assert "error" not in _call("read_file", path, task, offset=2, limit=2)
            written = _call("write_file", path, task, content="replacement\n")
            assert "error" not in written, (source, written)
            assert path.read_text(encoding="utf-8") == "replacement\n"
        finally:
            clear_file_ops_cache(task)


def test_partial_read_cannot_refresh_a_changed_full_baseline(tmp_path):
    original = "first\nsecond\nthird\n"
    for source in ("write", "pages"):
        path = tmp_path / f"changed-{source}.txt"
        task = f"snapshot-{source}"
        try:
            if source == "write":
                assert "error" not in _call("write_file", path, task, content=original)
            else:
                path.write_text(original, encoding="utf-8")
                assert "error" not in _call("read_file", path, task, offset=1, limit=1)
            stamp = path.stat()
            path.write_text("other\nsecond\nthird\n", encoding="utf-8")
            # mtime alone cannot identify bytes: editors/copy tools can preserve it.
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            assert "error" not in _call("read_file", path, task, offset=2, limit=2)
            refused = _call("write_file", path, task, content="replacement\n")
            assert refused.get("stale_write_blocked"), (source, refused)
            assert path.read_text(encoding="utf-8") == "other\nsecond\nthird\n"
            # Reading all of the new version is recovery, not a bypass.
            assert "error" not in _call("read_file", path, task)
            assert "error" not in _call("write_file", path, task, content="merged\n")
            assert path.read_text(encoding="utf-8") == "merged\n"
        finally:
            clear_file_ops_cache(task)

    # A page that hides part of a line never supplies whole-file knowledge.
    from tools.tool_output_limits import get_max_line_length

    path = tmp_path / "clamped.txt"
    path.write_text("x" * (get_max_line_length() + 10) + "\nlast\n", encoding="utf-8")
    try:
        assert "error" not in _call("read_file", path, "clamped")
        refused = _call("write_file", path, "clamped", content="replacement\n")
        assert refused.get("stale_write_blocked"), refused
        assert path.read_text(encoding="utf-8").startswith("x" * (get_max_line_length() + 10))
    finally:
        clear_file_ops_cache("clamped")


def _dispatch(name, task_id, **arguments):
    result = registry.dispatch(name, arguments, task_id=task_id)
    assert isinstance(result, str)
    return json.loads(result)


def _ok(result):
    assert "error" not in result, result
    return result


def _write(path, task_id, content):
    return _dispatch("write_file", task_id, path=str(path), content=content)


def _read(path, task_id, **paging):
    return _ok(_dispatch("read_file", task_id, path=str(path), **paging))


def _replace(path, task_id, old, new):
    return _ok(_dispatch("patch", task_id, path=str(path), old_string=old, new_string=new))


def _v4a(task_id, *updates):
    body = "".join(f"*** Update File: {p}\n@@\n-{old}\n+{new}\n" for p, old, new in updates)
    return _ok(_dispatch("patch", task_id, mode="patch", patch=f"*** Begin Patch\n{body}*** End Patch"))


def _blocked(result):
    assert result.get("stale_write_blocked") is True, result
    return result["error"]


@pytest.fixture
def task(request):
    task_id = f"own-patch-{request.node.name}"
    yield task_id
    clear_file_ops_cache(task_id)


def test_own_replace_patch_after_full_read_keeps_the_baseline(tmp_path, task):
    path = tmp_path / "probe.txt"
    _ok(_write(path, task, "first=old\nsecond=old\n"))
    _read(path, task)
    _replace(path, task, "first=old", "first=new")

    written = _write(path, task, "first=new\nsecond=new\n")

    assert "error" not in written and not written.get("stale_write_blocked"), written
    assert path.read_text() == "first=new\nsecond=new\n"


def test_own_patch_after_own_write_keeps_the_baseline(tmp_path, task):
    path = tmp_path / "written.txt"
    _ok(_write(path, task, "a=1\nb=1\n"))
    _replace(path, task, "a=1", "a=2")
    _replace(path, task, "b=1", "b=2")

    _ok(_write(path, task, "a=3\nb=3\n"))
    assert path.read_text() == "a=3\nb=3\n"


def test_own_v4a_patch_keeps_the_baseline_for_every_file(tmp_path, task):
    one, two = tmp_path / "one.txt", tmp_path / "two.txt"
    for p in (one, two):
        p.write_text("alpha\nbeta\n")
        _read(p, task)
    _v4a(task, (one, "alpha", "ALPHA"))
    _ok(_write(one, task, "single\n"))
    assert one.read_text() == "single\n"

    _read(one, task)
    _v4a(task, (one, "single", "SINGLE"), (two, "alpha", "ALPHA"))
    for p in (one, two):
        _ok(_write(p, task, f"replaced {p.name}\n"))
        assert p.read_text() == f"replaced {p.name}\n"

    # Two operations on one file chain: the second reads what the first wrote.
    _v4a(task, (one, f"replaced {one.name}", "first"), (one, "first", "second"))
    assert one.read_text() == "second\n"
    _ok(_write(one, task, "chained\n"))

    # A file the patch created holds only bytes the task wrote, like write_file creating it.
    added = tmp_path / "added.txt"
    _ok(_dispatch("patch", task, mode="patch",
                  patch=f"*** Begin Patch\n*** Add File: {added}\n+created\n*** End Patch"))
    _ok(_write(added, task, "replaced\n"))
    assert added.read_text() == "replaced\n"


def test_own_patch_keeps_crlf_and_bom_files_writable(tmp_path, task):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    _read(path, task)
    _replace(path, task, "one", "ONE")
    assert path.read_bytes() == b"\xef\xbb\xbfONE\r\ntwo\r\n"
    _ok(_write(path, task, "three\n"))


def test_blind_patch_never_gains_a_baseline(tmp_path, task):
    path = tmp_path / "blind.txt"
    path.write_text("one\ntwo\nthree\n")
    _replace(path, task, "one", "ONE")

    error = _blocked(_write(path, task, "x\n"))
    assert _BLIND in error
    assert path.read_text() == "ONE\ntwo\nthree\n"

    # A partial view is not full knowledge either.
    path.write_text("one\ntwo\nthree\n")
    _read(path, task, offset=1, limit=1)
    _replace(path, task, "two", "TWO")
    assert _BLIND in _blocked(_write(path, task, "x\n"))
    assert path.read_text() == "one\nTWO\nthree\n"

    # Recovery: read every line, then the overwrite lands.
    _read(path, task)
    _ok(_write(path, task, "merged\n"))
    assert path.read_text() == "merged\n"


def test_sibling_patch_voids_the_other_task_and_gives_the_patcher_nothing(tmp_path, task):
    path = tmp_path / "shared.txt"
    path.write_text("left\nright\n")
    sibling = f"{task}-sibling"
    try:
        _read(path, task)
        _replace(path, sibling, "right", "RIGHT")

        mine = _blocked(_write(path, task, "mine\n"))
        assert _BLIND not in mine and sibling in mine
        assert _BLIND in _blocked(_write(path, sibling, "theirs\n"))
        assert path.read_text() == "left\nRIGHT\n"

        _read(path, task)
        _ok(_write(path, task, "mine\n"))
        _read(path, sibling)
        _ok(_write(path, sibling, "theirs\n"))
        assert path.read_text() == "theirs\n"
    finally:
        clear_file_ops_cache(sibling)


@pytest.mark.parametrize("preserve_mtime", [False, True], ids=["mtime-moves", "mtime-restored"])
def test_external_edit_after_own_patch_is_refused(tmp_path, task, preserve_mtime):
    path = tmp_path / "external.txt"
    path.write_text("one\ntwo\n")
    _read(path, task)
    _replace(path, task, "one", "ONE")
    stamp = path.stat()
    path.write_text("ONE\nEXTERNAL\n")
    if preserve_mtime:
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    else:
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))

    error = _blocked(_write(path, task, "clobber\n"))
    assert _BLIND not in error
    assert path.read_text() == "ONE\nEXTERNAL\n"

    _read(path, task)
    _ok(_write(path, task, "merged\n"))
    assert path.read_text() == "merged\n"


def test_write_landing_after_the_patch_write_gives_no_baseline(tmp_path, task, monkeypatch):
    path = tmp_path / "raced-after.txt"
    path.write_text("one\ntwo\n")
    original = file_operations.ShellFileOperations.patch_replace

    def patch_then_external_write(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        stamp = os.stat(path)
        path.write_text("ONE\nRACED\n")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return result

    _read(path, task)
    monkeypatch.setattr(file_operations.ShellFileOperations, "patch_replace", patch_then_external_write)
    _replace(path, task, "one", "ONE")
    monkeypatch.undo()

    _blocked(_write(path, task, "clobber\n"))
    assert path.read_text() == "ONE\nRACED\n"


def test_write_landing_before_the_patch_read_gives_no_baseline(tmp_path, task, monkeypatch):
    """The patch applied its delta to bytes the task never saw in full."""
    path = tmp_path / "raced-before.txt"
    path.write_text("one\ntwo\n")
    original = file_operations.ShellFileOperations.patch_replace

    def external_write_then_patch(self, *args, **kwargs):
        path.write_text("one\nUNSEEN\n")
        return original(self, *args, **kwargs)

    _read(path, task)
    monkeypatch.setattr(file_operations.ShellFileOperations, "patch_replace", external_write_then_patch)
    _replace(path, task, "one", "ONE")
    monkeypatch.undo()
    assert path.read_text() == "ONE\nUNSEEN\n"

    _blocked(_write(path, task, "clobber\n"))
    assert path.read_text() == "ONE\nUNSEEN\n"


def test_paged_read_carries_only_when_every_page_was_read(tmp_path, task):
    content = "".join(f"line {i}\n" for i in range(1, 2501))
    complete, missing = tmp_path / "complete.txt", tmp_path / "missing.txt"
    for p in (complete, missing):
        p.write_text(content)
        assert _read(p, task).get("truncated")
    _read(complete, task, offset=2001)

    for p in (complete, missing):
        _replace(p, task, "line 1\n", "LINE 1\n")
    _ok(_write(complete, task, "merged\n"))
    assert complete.read_text() == "merged\n"
    assert _BLIND in _blocked(_write(missing, task, "merged\n"))
    assert missing.read_text().startswith("LINE 1\nline 2\n")


@pytest.mark.parametrize("reread", [False, True], ids=["no-reread", "reread"])
def test_own_patch_after_compaction_keeps_the_baseline(tmp_path, task, reread):
    path = tmp_path / "compacted.txt"
    path.write_text("one\ntwo\n")
    _read(path, task)
    reset_file_dedup(task)
    if reread:
        _read(path, task)
    _replace(path, task, "one", "ONE")
    _ok(_write(path, task, "merged\n"))
    assert path.read_text() == "merged\n"


def test_third_patch_failure_hint_requires_a_full_reread_before_write_file(tmp_path, task):
    path = tmp_path / "anchor.txt"
    path.write_text("alpha\n")
    for _ in range(3):
        failed = _dispatch("patch", task, path=str(path), old_string="missing", new_string="x")
    hint = failed["_hint"]
    assert "failure #3" in hint
    assert "re-read the file in full, then use write_file to replace it" in hint
    assert "use write_file to replace the entire file" not in hint
