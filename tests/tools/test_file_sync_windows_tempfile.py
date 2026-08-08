import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

from tools.environments.file_sync import FileSyncManager


def test_sync_back_download_receives_reopenable_path(tmp_path):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    archive_paths: list[Path] = []

    def download(dest: Path) -> None:
        archive_paths.append(dest)
        assert not dest.exists()
        with tarfile.open(dest, "w") as archive:
            data = b"remote_version"
            member = tarfile.TarInfo("root/.hermes/skill.md")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host_file), "/root/.hermes/skill.md")],
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=download,
    )
    manager._pushed_hashes["/root/.hermes/skill.md"] = "0" * 64

    manager._sync_back_impl()

    assert host_file.read_bytes() == b"remote_version"
    assert archive_paths and not archive_paths[0].exists()


def test_sync_back_maps_files_when_relpath_uses_windows_separators(
    tmp_path, monkeypatch
):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")

    def download(dest: Path) -> None:
        with tarfile.open(dest, "w") as archive:
            data = b"remote_version"
            member = tarfile.TarInfo("root/.hermes/skill.md")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    monkeypatch.setattr(
        "tools.environments.file_sync.os.path.relpath",
        lambda *_args: r"root\.hermes\skill.md",
    )

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host_file), "/root/.hermes/skill.md")],
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=download,
    )
    manager._pushed_hashes["/root/.hermes/skill.md"] = "0" * 64

    manager._sync_back_impl()

    assert host_file.read_bytes() == b"remote_version"
