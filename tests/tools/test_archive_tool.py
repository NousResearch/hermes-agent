import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

from tools.archive_tool import extract_archive_tool


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _make_zip_with_symlink(path: Path, link_name: str, target: str) -> None:
    info = zipfile.ZipInfo(link_name)
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, target)


def _make_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_archive_tool_extracts_zip(tmp_path):
    archive = tmp_path / "sample.zip"
    _make_zip(archive, {"nested/hello.txt": b"hi"})

    result = json.loads(extract_archive_tool(str(archive)))

    assert result["success"] is True
    output_dir = Path(result["output_dir"])
    assert (output_dir / "nested" / "hello.txt").read_text() == "hi"
    assert "nested/hello.txt" in result["extracted_files"]


def test_extract_archive_tool_extracts_tar_gz(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    _make_tar_gz(archive, {"hello.txt": b"hi"})

    result = json.loads(extract_archive_tool(str(archive)))

    assert result["success"] is True
    output_dir = Path(result["output_dir"])
    assert (output_dir / "hello.txt").read_text() == "hi"


def test_extract_archive_tool_blocks_zip_slip(tmp_path):
    archive = tmp_path / "escape.zip"
    _make_zip(archive, {"../../escape.txt": b"pwnd"})

    result = json.loads(extract_archive_tool(str(archive)))

    assert result["success"] is False
    assert "unsafe archive member path" in result["error"].lower()
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_tool_rejects_zip_symlink(tmp_path):
    archive = tmp_path / "symlink.zip"
    _make_zip_with_symlink(archive, "link", "target.txt")

    result = json.loads(extract_archive_tool(str(archive)))

    assert result["success"] is False
    assert "unsupported archive member type" in result["error"].lower()


def test_extract_archive_tool_rejects_symlinked_destination(tmp_path):
    archive = tmp_path / "sample.zip"
    _make_zip(archive, {"nested/hello.txt": b"hi"})
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlink_dir = tmp_path / "linked-out"
    os.symlink(real_dir, symlink_dir)

    result = json.loads(extract_archive_tool(str(archive), output_dir=str(symlink_dir)))

    assert result["success"] is False
    assert "symlinked destination" in result["error"].lower()


def test_extract_archive_tool_rejects_unsupported_extension(tmp_path):
    archive = tmp_path / "sample.bin"
    archive.write_bytes(b"not an archive")

    result = json.loads(extract_archive_tool(str(archive)))

    assert result["success"] is False
    assert "unsupported archive type" in result["error"].lower()
