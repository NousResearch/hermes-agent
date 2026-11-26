"""Large transcript reference scans stay complete, and failed scans keep archives."""

import os
import time

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.errors import TransientStoreError
from persistence_assembly import open_persistence, profile_scope
from tools.tool_result_storage import _sweep_archive_dir, referenced_archive_refs


@pytest.fixture
def busy_home(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession("s1", "cli"))
        store.messages.append_batch("s1", [
            NewMessage("tool", f"cache/pruned/{i}.json" if i % 2 else "display",
                       api_content=None if i % 2 else f"cache/pruned/{i}.json", tool_call_id=f"c{i}")
            for i in range(20_001)
        ])
    return tmp_path


def _old_archives(home):
    archive_dir = home / "cache" / "pruned"
    archive_dir.mkdir(parents=True)
    for name in ("0.json", "20000.json", "orphan.json"):
        path = archive_dir / name
        path.write_text("original")
        os.utime(path, (1, 1))
    return archive_dir


def test_sweep_removes_only_unreferenced_files_past_the_old_row_cap(busy_home):
    """Past 20,000 referencing rows the set is still read whole, so an orphan is swept and referenced files stay."""
    assert referenced_archive_refs(busy_home) == {f"cache/pruned/{i}.json" for i in range(20_001)}
    archive_dir = _old_archives(busy_home)
    assert _sweep_archive_dir(archive_dir, time.time() - 3600) == 1
    assert {p.name for p in archive_dir.iterdir()} == {"0.json", "20000.json"}


@pytest.mark.parametrize("error", [TransientStoreError, OSError])
def test_mid_iteration_error_keeps_every_archive(busy_home, monkeypatch, error):
    """A read that fails part-way answers "unknown": the sweep fails closed and keeps every file."""
    from persistence.sqlite.conversation.messages import SQLiteMessageRepository

    def failing_rows(self):
        yield ("cache/pruned/0.json", None)
        raise error("synthetic read failure")

    monkeypatch.setattr(SQLiteMessageRepository, "spillover_reference_rows", failing_rows)
    archive_dir = _old_archives(busy_home)
    assert referenced_archive_refs(busy_home) is None
    assert _sweep_archive_dir(archive_dir, time.time() - 3600) == 0
    assert {p.name for p in archive_dir.iterdir()} == {"0.json", "20000.json", "orphan.json"}
