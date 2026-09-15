"""Real-file regressions for archive rollback and no-op batches."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import memory_tool as mt


def test_empty_archive_batch_does_not_touch_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path / "memories")
    assert mt.archive_entries("memory", []) == ([], None)
    assert not (tmp_path / "memories").exists()


def test_failed_append_cannot_erase_another_process_record(tmp_path, monkeypatch):
    """Hold the failing writer open while a second real process tries to append."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path / "memories")
    path = mt.get_memory_archive_path()
    path.parent.mkdir()
    original = '{"id": "existing"}\n'
    concurrent = '{"id": "concurrent"}\n'
    path.write_text(original, encoding="utf-8")
    real_open = open
    children: list[subprocess.Popen[str]] = []

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, data):
            # Persist a real partial write, not just an exception before any I/O.
            self.handle.write(data[:5])
            self.handle.flush()
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "from tools import memory_tool as mt; "
                 "print('ready', flush=True); "
                 f"mt._archive_append_lines([{concurrent!r}])"],
                cwd=Path(__file__).resolve().parents[2],
                env=os.environ.copy(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            children.append(child)
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "ready"
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # Expected: the other process waits for our archive lock.
            raise OSError("disk full after partial write")

    def failing_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if Path(file) == path and mode in ("a", "ab"):
            return FailingWriter(handle)
        return handle

    monkeypatch.setattr(mt, "open", failing_open, raising=False)
    try:
        with pytest.raises(OSError, match="disk full"):
            mt._archive_append_lines(['{"id": "failed"}\n'])
        assert children
        child = children[0]
        _, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stderr
        assert path.read_text(encoding="utf-8") == original + concurrent
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
