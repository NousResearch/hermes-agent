"""A narrower read must not discard unchanged whole-file knowledge."""

import json
import os

from tools.file_tools import clear_file_ops_cache
from tools.registry import registry


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
