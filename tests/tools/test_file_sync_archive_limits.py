import io
import logging
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.environments.file_sync import FileSyncManager


def _download_with(files: dict[str, bytes]):
    def download(dest: Path) -> None:
        with tarfile.open(dest, "w") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    return download


def _manager(host_file: Path, files: dict[str, bytes]) -> FileSyncManager:
    return FileSyncManager(
        get_files_fn=lambda: [(str(host_file), "/root/.hermes/skill.md")],
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=_download_with(files),
    )


def _run_sync_back(manager: FileSyncManager, archive_path: Path) -> None:
    temporary_file = nullcontext(SimpleNamespace(name=str(archive_path)))
    with patch(
        "tools.environments.file_sync.tempfile.NamedTemporaryFile",
        return_value=temporary_file,
    ):
        manager._sync_back_impl()


def test_sync_back_refuses_excessive_member_count(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"remote_version",
        "root/.hermes/extra.md": b"extra",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_MEMBERS", 1):
            _run_sync_back(manager, tmp_path / "members.tar")

    assert host_file.read_bytes() == b"original"
    assert any("member" in record.message.lower() for record in caplog.records)


def test_sync_back_refuses_excessive_expanded_size(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"remote_version",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_EXPANDED_BYTES", 1):
            _run_sync_back(manager, tmp_path / "expanded.tar")

    assert host_file.read_bytes() == b"original"
    assert any("expanded" in record.message.lower() for record in caplog.records)
