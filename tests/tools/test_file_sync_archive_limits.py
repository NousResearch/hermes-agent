import io
import logging
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.file_sync import FileSyncManager, _extract_sync_back_archive


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


def _download_members(members: list[tuple[tarfile.TarInfo, bytes | None]]):
    def download(dest: Path) -> None:
        with tarfile.open(dest, "w") as archive:
            for member, data in members:
                archive.addfile(member, io.BytesIO(data) if data is not None else None)

    return download


@contextmanager
def _temporary_archive(path: Path):
    try:
        yield SimpleNamespace(name=str(path))
    finally:
        path.unlink(missing_ok=True)


def _run_sync_back(manager: FileSyncManager, archive_path: Path) -> list[Path]:
    staging_paths: list[Path] = []
    real_temporary_directory = tempfile.TemporaryDirectory

    def recording_temporary_directory(*args, **kwargs):
        directory = real_temporary_directory(*args, **kwargs)
        staging_paths.append(Path(directory.name))
        return directory

    with patch(
        "tools.environments.file_sync.tempfile.NamedTemporaryFile",
        return_value=_temporary_archive(archive_path),
    ), patch(
        "tools.environments.file_sync.tempfile.TemporaryDirectory",
        side_effect=recording_temporary_directory,
    ):
        manager._sync_back_impl()
    assert not archive_path.exists()
    assert all(not path.exists() for path in staging_paths)
    return staging_paths


def _assert_rejected(
    manager: FileSyncManager,
    host_file: Path,
    archive_path: Path,
    *,
    expect_staging: bool = True,
) -> None:
    staging_paths = _run_sync_back(manager, archive_path)
    assert bool(staging_paths) is expect_staging
    assert host_file.read_bytes() == b"original"


def test_sync_back_refuses_excessive_member_count(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"remote_version",
        "root/.hermes/extra.md": b"extra",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_MEMBERS", 1):
            _assert_rejected(manager, host_file, tmp_path / "members.tar")

    assert any("member" in record.message.lower() for record in caplog.records)


def test_sync_back_refuses_excessive_expanded_size(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"ab",
        "root/.hermes/extra.md": b"cd",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_EXPANDED_BYTES", 3):
            _assert_rejected(manager, host_file, tmp_path / "expanded.tar")

    assert any("expanded" in record.message.lower() for record in caplog.records)


def test_sync_back_refuses_excessive_per_member_size(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"remote_version",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_MEMBER_BYTES", 1):
            _assert_rejected(manager, host_file, tmp_path / "member-size.tar")

    assert any("member" in record.message.lower() for record in caplog.records)
    assert any("cap" in record.message.lower() for record in caplog.records)


def test_sync_back_refuses_oversized_downloaded_archive(tmp_path, caplog):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {
        "root/.hermes/skill.md": b"remote_version",
    })

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        with patch("tools.environments.file_sync._SYNC_BACK_MAX_BYTES", 1):
            _assert_rejected(
                manager,
                host_file,
                tmp_path / "download.tar",
                expect_staging=False,
            )

    assert any("remote tar is" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.md",
        "..\\escape.md",
        "/absolute/escape.md",
        "C:\\absolute\\escape.md",
    ],
)
def test_sync_back_refuses_traversal_and_absolute_paths(
    tmp_path, caplog, member_name
):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    manager = _manager(host_file, {member_name: b"remote_version"})

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        _assert_rejected(manager, host_file, tmp_path / "unsafe-path.tar")

    assert not (tmp_path / "escape.md").exists()
    assert any("unsafe" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE],
)
def test_sync_back_refuses_link_members(tmp_path, caplog, member_type):
    host_file = tmp_path / "host_skill.md"
    host_file.write_bytes(b"original")
    member = tarfile.TarInfo("root/.hermes/alias.md")
    member.type = member_type
    member.linkname = "root/.hermes/skill.md"
    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host_file), "/root/.hermes/skill.md")],
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=_download_members([(member, None)]),
    )

    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        _assert_rejected(manager, host_file, tmp_path / "unsafe-member.tar")

    assert any("link" in record.message.lower() for record in caplog.records)


def test_safe_archive_extracts_regular_files_and_directories(tmp_path):
    archive_path = tmp_path / "valid.tar"
    directory = tarfile.TarInfo("root/.hermes")
    directory.type = tarfile.DIRTYPE
    file_member = tarfile.TarInfo("root/.hermes/skill.md")
    payload = b"remote_version"
    file_member.size = len(payload)
    _download_members([(directory, None), (file_member, payload)])(archive_path)

    destination = tmp_path / "out"
    destination.mkdir()
    with tarfile.open(archive_path) as archive:
        _extract_sync_back_archive(archive, str(destination))

    assert (destination / "root" / ".hermes" / "skill.md").read_bytes() == payload
