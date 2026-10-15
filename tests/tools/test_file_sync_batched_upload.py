"""A sync that fails part-way keeps what the remote already acknowledged.

The whole upload used to be one transfer committed only on success, so a remote that cannot
write every file within the transfer timeout (thousands of skill files on a slow filesystem)
restarted from zero on every cycle and never converged.
"""

from tools.environments import file_sync
from tools.environments.file_sync import FileSyncManager


def _host_files(tmp_path, count, size=8):
    files = []
    for index in range(count):
        path = tmp_path / f"skill{index}" / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * size)
        files.append((str(path), f"/root/.hermes/skills/skill{index}/SKILL.md"))
    return files


def test_a_failed_batch_keeps_the_batches_before_it(tmp_path, monkeypatch):
    monkeypatch.setattr(file_sync, "_UPLOAD_BATCH_FILES", 2)
    files = _host_files(tmp_path, 5)
    sent, failures = [], [RuntimeError("SSH bulk upload timed out")]

    def bulk_upload(batch):
        if len(sent) == 4 and failures:
            raise failures.pop()
        sent.extend(remote for _, remote in batch)

    manager = FileSyncManager(get_files_fn=lambda: files, upload_fn=lambda *args: None,
                              delete_fn=lambda paths: None, bulk_upload_fn=bulk_upload)
    manager.sync(force=True)
    assert sent == [remote for _, remote in files[:4]]

    manager.sync(force=True)
    assert sent[4:] == [files[4][1]]


def test_batches_are_bounded_by_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(file_sync, "_UPLOAD_BATCH_BYTES", 20)
    batches = list(file_sync._upload_batches(_host_files(tmp_path, 5, size=8)))
    assert [len(batch) for batch in batches] == [2, 2, 1]
