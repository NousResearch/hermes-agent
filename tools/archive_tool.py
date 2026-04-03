#!/usr/bin/env python3
"""Archive extraction tool for ZIP and tar archives."""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import mkdtemp
from typing import Iterable

from tools.registry import registry

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)

_MAX_EXTRACTED_FILE_LIST = 200
_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 250 * 1024 * 1024
_MAX_ARCHIVE_ENTRIES = 10_000


def _normalize_archive_member_parts(member_name: str) -> list[str]:
    """Return safe path parts for an archive member."""
    normalized_name = (member_name or "").replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(member_name or "")

    if (
        not normalized_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"Unsafe archive member path: {member_name}")

    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return parts


def _resolve_output_dir(archive_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        raw_target = Path(output_dir).expanduser()
        target = (Path.cwd() / raw_target) if not raw_target.is_absolute() else raw_target
        if target.exists() and target.is_symlink():
            raise ValueError(f"Refusing to extract into symlinked destination: {target}")
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    base_name = archive_path.name
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip", ".tar"):
        if base_name.lower().endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    safe_stem = base_name or "archive"
    return Path(mkdtemp(prefix=f"extract_{safe_stem}_", dir=str(archive_path.parent)))


def _record_path(collected: list[str], relative_path: Path) -> None:
    if len(collected) < _MAX_EXTRACTED_FILE_LIST:
        collected.append(relative_path.as_posix())


def _validate_archive_size(archive_path: Path) -> None:
    archive_size = archive_path.stat().st_size
    if archive_size > _MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Archive exceeds size limit ({_MAX_ARCHIVE_BYTES // (1024 * 1024)}MB): {archive_path}"
        )


def _ensure_no_symlink_components(destination: Path, target: Path) -> None:
    current = destination
    for part in target.relative_to(destination).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"Refusing to extract through symlinked path: {current}")


def _is_zip_symlink(member: zipfile.ZipInfo) -> bool:
    mode = (member.external_attr >> 16) & 0xFFFF
    return (mode & 0o170000) == 0o120000


def _extract_zip(archive_path: Path, destination: Path) -> list[str]:
    extracted_files: list[str] = []
    extracted_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"Archive has too many entries ({len(infos)} > {_MAX_ARCHIVE_ENTRIES})")
        for member in infos:
            if _is_zip_symlink(member):
                raise ValueError(f"Unsupported archive member type: {member.filename}")
            parts = _normalize_archive_member_parts(member.filename)
            target = destination.joinpath(*parts)
            _ensure_no_symlink_components(destination, target.parent if not member.is_dir() else target)

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            extracted_bytes += member.file_size
            if extracted_bytes > _MAX_EXTRACTED_BYTES:
                raise ValueError(
                    f"Archive extracted content exceeds size limit ({_MAX_EXTRACTED_BYTES // (1024 * 1024)}MB)"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            _record_path(extracted_files, target.relative_to(destination))
    return extracted_files


def _iter_tar_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"Unsupported archive member type: {member.name}")
        yield member


def _extract_tar(archive_path: Path, destination: Path) -> list[str]:
    extracted_files: list[str] = []
    extracted_bytes = 0
    with tarfile.open(archive_path, "r:*") as tf:
        members = list(_iter_tar_members(tf))
        if len(members) > _MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"Archive has too many entries ({len(members)} > {_MAX_ARCHIVE_ENTRIES})")
        for member in members:
            parts = _normalize_archive_member_parts(member.name)
            target = destination.joinpath(*parts)
            _ensure_no_symlink_components(destination, target.parent if not member.isdir() else target)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member type: {member.name}")

            extracted_bytes += max(0, member.size)
            if extracted_bytes > _MAX_EXTRACTED_BYTES:
                raise ValueError(
                    f"Archive extracted content exceeds size limit ({_MAX_EXTRACTED_BYTES // (1024 * 1024)}MB)"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)
            _record_path(extracted_files, target.relative_to(destination))

            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                pass
    return extracted_files


def _detect_archive_type(archive_path: Path) -> str | None:
    lower_name = archive_path.name.lower()
    for suffix in _SUPPORTED_ARCHIVE_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    return None


def extract_archive_tool(archive_path: str, output_dir: str | None = None, task_id: str | None = None) -> str:
    del task_id  # reserved for parity with other tool handlers

    try:
        path = Path(archive_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()

        if not path.exists():
            return json.dumps({"success": False, "error": f"Archive not found: {path}"})
        if not path.is_file():
            return json.dumps({"success": False, "error": f"Not a file: {path}"})

        _validate_archive_size(path)

        archive_type = _detect_archive_type(path)
        if archive_type is None:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Unsupported archive type. Supported: "
                        + ", ".join(_SUPPORTED_ARCHIVE_SUFFIXES)
                    ),
                }
            )

        destination = _resolve_output_dir(path, output_dir)
        if archive_type == ".zip":
            extracted_files = _extract_zip(path, destination)
        else:
            extracted_files = _extract_tar(path, destination)

        return json.dumps(
            {
                "success": True,
                "archive_path": str(path),
                "archive_type": archive_type,
                "output_dir": str(destination),
                "extracted_count": len(extracted_files),
                "extracted_files": extracted_files,
                "truncated": len(extracted_files) >= _MAX_EXTRACTED_FILE_LIST,
            }
        )
    except (ValueError, tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        logger.warning("Archive extraction failed for %s: %s", archive_path, exc)
        return json.dumps({"success": False, "error": str(exc)})


def check_archive_tool_requirements() -> bool:
    return True


EXTRACT_ARCHIVE_SCHEMA = {
    "name": "extract_archive",
    "description": (
        "Safely extract a ZIP or tar archive that already exists on disk. "
        "Use this for uploaded project bundles or compressed datasets when you "
        "need Hermes to unpack them on the host machine instead of relying on shell unzip commands."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "archive_path": {
                "type": "string",
                "description": "Path to a .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, or .txz archive.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional destination directory. Omit to extract into a new sibling directory next to the archive.",
            },
        },
        "required": ["archive_path"],
    },
}


registry.register(
    name="extract_archive",
    toolset="file",
    schema=EXTRACT_ARCHIVE_SCHEMA,
    handler=lambda args, **kw: extract_archive_tool(
        archive_path=args.get("archive_path", ""),
        output_dir=args.get("output_dir"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_archive_tool_requirements,
    emoji="🗜️",
)
