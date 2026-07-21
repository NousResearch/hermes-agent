"""Audit records for skill-package validation.

Hermes deliberately reuses its existing terminal/sandbox tools to execute a
skill's tests.  This module records the resulting command, exit status, and
package digest; it does not introduce a second execution engine or silently run
code while loading a skill.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.skill_sidecar_io import atomic_write_sidecar, read_sidecar, sidecar_lock

VALIDATION_FILE = ".validation.json"
VALIDATION_LOCK_FILE = ".validation.lock"
LIFECYCLE_LOCK_FILE = ".lifecycle.lock"
VALIDATION_SCHEMA = "hermes.skill-validation.v1"
MAX_VALIDATION_OUTPUT_CHARS = 16_384
MAX_VALIDATION_RECORD_BYTES = 65_536
MAX_VALIDATED_PACKAGE_FILES = 2_048
MAX_VALIDATED_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_VALIDATION_SIGNATURE_ENTRIES = 16_384
_EXCLUDED_FILES = {
    ".memory.md",
    ".memory.lock",
    VALIDATION_FILE,
    VALIDATION_LOCK_FILE,
    LIFECYCLE_LOCK_FILE,
}
_GENERATED_PACKAGE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
}
_GENERATED_PACKAGE_SUFFIXES = {".pyc", ".pyo"}


def _is_generated_package_artifact(relative_path: Path) -> bool:
    return (
        any(part in _GENERATED_PACKAGE_DIRS for part in relative_path.parts)
        or relative_path.suffix in _GENERATED_PACKAGE_SUFFIXES
        or relative_path.name == ".coverage"
        or relative_path.name.startswith(".coverage.")
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validation_path(skill_dir: Path) -> Path:
    return skill_dir / VALIDATION_FILE


def skill_content_digest(skill_dir: Path) -> str:
    """Hash stable package content while excluding mutable lifecycle sidecars."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(skill_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative_path = path.relative_to(skill_dir)
        if _is_generated_package_artifact(relative_path):
            continue
        if path.is_symlink():
            raise ValueError(
                f"validated skill packages cannot contain symlinks: {relative_path}"
            )
        if not path.is_file():
            continue
        if len(relative_path.parts) == 1 and path.name in _EXCLUDED_FILES:
            continue
        file_count += 1
        if file_count > MAX_VALIDATED_PACKAGE_FILES:
            raise ValueError("validated skill package has too many files")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ValueError(
                f"cannot safely read package file: {relative_path}"
            ) from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(
                    f"package entry is not a regular file: {relative_path}"
                )
            total_bytes += opened.st_size
            if total_bytes > MAX_VALIDATED_PACKAGE_BYTES:
                raise ValueError("validated skill package is too large")
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(
                        f"package file changed while hashing: {relative_path}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise ValueError(f"package file changed while hashing: {relative_path}")
            data = b"".join(chunks)
        finally:
            os.close(fd)
        relative = relative_path.as_posix().encode("utf-8")
        mode = stat.S_IMODE(opened.st_mode)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    atomic_write_sidecar(path.parent, path.name, data)


def _invalidate_skill_validation_locked(skill_dir: Path) -> None:
    """Mark an existing validation record stale after package content changes."""
    path = validation_path(skill_dir)
    tests_dir = skill_dir / "tests"
    tests_collected = (
        len(list(tests_dir.rglob("test_*.py"))) if tests_dir.is_dir() else 0
    )
    if not path.exists() and tests_collected == 0:
        return
    # A package still under lifecycle construction must never be un-hidden by an
    # intermediate mutation (e.g. adding scripts before tests). Preserve the
    # ``draft`` state so it stays out of discovery until the lifecycle registers
    # it explicitly.
    existing = read_skill_validation(skill_dir)
    if isinstance(existing, dict) and existing.get("status") == "draft":
        _write_draft_validation_locked(skill_dir, tests_collected)
        return
    status = "pending" if tests_collected else "static"
    record = {
        "schema": VALIDATION_SCHEMA,
        "status": status,
        "reason": (
            "skill package changed"
            if tests_collected
            else "text-only skill changed; static checks retained"
        ),
        "tests_collected": tests_collected,
        "content_digest": skill_content_digest(skill_dir),
        "updated_at": _now_iso(),
    }
    if tests_collected:
        record["validation_token"] = secrets.token_urlsafe(24)
    _atomic_json_write(path, record)


def _write_draft_validation_locked(
    skill_dir: Path, tests_collected: Optional[int] = None
) -> None:
    """Write a ``draft`` record that fails the discovery gate closed."""
    if tests_collected is None:
        tests_dir = skill_dir / "tests"
        tests_collected = (
            len(list(tests_dir.rglob("test_*.py"))) if tests_dir.is_dir() else 0
        )
    record = {
        "schema": VALIDATION_SCHEMA,
        "status": "draft",
        "reason": "skill package under lifecycle construction",
        "tests_collected": tests_collected,
        "content_digest": skill_content_digest(skill_dir),
        "updated_at": _now_iso(),
    }
    _atomic_json_write(validation_path(skill_dir), record)


def record_draft_validation(skill_dir: Path) -> None:
    """Atomically stamp a package as an undiscoverable lifecycle draft."""
    with sidecar_lock(skill_dir, VALIDATION_LOCK_FILE):
        _write_draft_validation_locked(skill_dir)


def invalidate_skill_validation(skill_dir: Path) -> None:
    """Atomically invalidate validation evidence after package mutation."""
    with sidecar_lock(skill_dir, VALIDATION_LOCK_FILE):
        _invalidate_skill_validation_locked(skill_dir)


def _record_skill_validation_locked(
    skill_dir: Path,
    validation: Optional[Dict[str, Any]] = None,
    *,
    approval_id: str | None = None,
) -> Dict[str, Any]:
    """Record static validation or evidence from a test command run separately."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return {"success": False, "error": "SKILL.md is missing"}
    if approval_id:
        if not approval_id.isalnum() or len(approval_id) > 64:
            return {"success": False, "error": "invalid validation approval id"}
        existing_record = read_skill_validation(skill_dir)
        if existing_record and existing_record.get("approval_id") == approval_id:
            existing_status = existing_record.get("status")
            result = {
                "success": existing_status == "passed",
                "validation_status": existing_status,
                **existing_record,
            }
            if existing_status == "failed":
                result["refinement_required"] = True
                result["error"] = (
                    "Skill tests failed; refine the package and validate again."
                )
            return result

    tests = (
        sorted((skill_dir / "tests").rglob("test_*.py"))
        if (skill_dir / "tests").is_dir()
        else []
    )
    digest = skill_content_digest(skill_dir)

    if not tests:
        record = {
            "schema": VALIDATION_SCHEMA,
            "status": "static",
            "tests_collected": 0,
            "content_digest": digest,
            "validated_at": _now_iso(),
        }
        if approval_id:
            record["approval_id"] = approval_id
        _atomic_json_write(validation_path(skill_dir), record)
        return {"success": True, "validation_status": "static", **record}

    if not isinstance(validation, dict):
        token = secrets.token_urlsafe(24)
        pending = {
            "schema": VALIDATION_SCHEMA,
            "status": "pending",
            "reason": "test evidence required",
            "tests_collected": len(tests),
            "validation_token": token,
            "content_digest": digest,
            "updated_at": _now_iso(),
        }
        if approval_id:
            pending["approval_id"] = approval_id
        _atomic_json_write(validation_path(skill_dir), pending)
        return {
            "success": False,
            "error": (
                "This skill contains tests. Run them through the active terminal or "
                "sandbox, then retry validate with the issued content_digest and "
                "validation_token plus command, exit_code, and output evidence."
            ),
            "validation_status": "pending",
            **pending,
        }

    command = validation.get("command")
    exit_code = validation.get("exit_code")
    output = validation.get("output", "")
    tested_digest = validation.get("content_digest")
    tested_token = validation.get("validation_token")
    if not isinstance(tested_digest, str) or tested_digest != digest:
        return {
            "success": False,
            "error": (
                "validation.content_digest must match the package digest returned "
                "before the test run; package content changed or evidence is stale"
            ),
            "validation_status": "pending",
            "content_digest": digest,
        }
    issued = read_skill_validation(skill_dir)
    if (
        not isinstance(tested_token, str)
        or not issued
        or issued.get("status") != "pending"
        or issued.get("content_digest") != digest
        or issued.get("validation_token") != tested_token
    ):
        return {
            "success": False,
            "error": (
                "validation.validation_token must match the pending token issued "
                "before the test run; request a new validation challenge"
            ),
            "validation_status": "pending",
            "content_digest": digest,
        }
    if not isinstance(command, str) or not command.strip():
        return {
            "success": False,
            "error": "validation.command must be a non-empty string",
        }
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return {"success": False, "error": "validation.exit_code must be an integer"}
    if not isinstance(output, str):
        return {"success": False, "error": "validation.output must be a string"}

    from agent.redact import redact_sensitive_for_persistence

    command = redact_sensitive_for_persistence(command)
    output = redact_sensitive_for_persistence(output)

    status = "passed" if exit_code == 0 else "failed"
    record = {
        "schema": VALIDATION_SCHEMA,
        "status": status,
        "tests_collected": len(tests),
        "command": command.strip(),
        "exit_code": exit_code,
        "output": output[-MAX_VALIDATION_OUTPUT_CHARS:],
        "content_digest": digest,
        "validated_at": _now_iso(),
    }
    if approval_id:
        record["approval_id"] = approval_id
    _atomic_json_write(validation_path(skill_dir), record)

    result = {
        "success": exit_code == 0,
        "validation_status": status,
        "tests_collected": len(tests),
        "content_digest": digest,
        "test_output": record["output"],
    }
    if exit_code != 0:
        result["refinement_required"] = True
        result["error"] = "Skill tests failed; refine the package and validate again."
    return result


def record_skill_validation(
    skill_dir: Path,
    validation: Optional[Dict[str, Any]] = None,
    *,
    approval_id: str | None = None,
) -> Dict[str, Any]:
    """Consume validation challenges atomically across threads and processes."""
    with sidecar_lock(skill_dir, VALIDATION_LOCK_FILE):
        return _record_skill_validation_locked(
            skill_dir, validation, approval_id=approval_id
        )


def validation_sidecar_signature(skill_roots) -> tuple:
    """Return a bounded cross-process cache key for opted-in package state."""
    from agent.skill_utils import EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS

    entries = []
    scanned_dirs = 0
    scanned_entries = 0
    max_signature_entries = MAX_VALIDATION_SIGNATURE_ENTRIES

    def uncacheable(reason: str) -> tuple:
        # A fresh nonce disables cache hits when a bounded scan cannot prove that
        # every opted-in package is represented in the signature.
        entries.append((reason, os.urandom(8).hex()))
        return tuple(sorted(entries, key=str))

    for root in skill_roots:
        root_path = Path(root)
        entries.append(("root", str(root_path)))
        if not root_path.exists():
            continue
        scan_errors = []
        for current_root, dirnames, filenames in os.walk(
            root_path, followlinks=False, onerror=scan_errors.append
        ):
            scanned_dirs += 1
            if scanned_dirs > max_signature_entries:
                return uncacheable("validation_signature_directory_limit")
            dirnames[:] = sorted(
                dirname for dirname in dirnames if dirname not in EXCLUDED_SKILL_DIRS
            )
            filenames.sort()
            if "SKILL.md" not in filenames:
                continue
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in SKILL_SUPPORT_DIRS
            ]
            skill_dir = Path(current_root)
            sidecar = skill_dir / VALIDATION_FILE
            try:
                stat_result = os.lstat(sidecar)
            except FileNotFoundError:
                continue
            except OSError as exc:
                entries.append((str(sidecar), "error", exc.errno))
                continue
            entries.append((
                str(sidecar),
                stat_result.st_mtime_ns,
                stat_result.st_ctime_ns,
                stat_result.st_size,
                stat_result.st_ino,
                stat_result.st_mode,
            ))
            try:
                skill_dir_stat = os.lstat(skill_dir)
                entries.append((
                    str(skill_dir),
                    skill_dir_stat.st_mtime_ns,
                    skill_dir_stat.st_ctime_ns,
                    skill_dir_stat.st_ino,
                    skill_dir_stat.st_mode,
                ))
            except OSError as exc:
                entries.append((str(skill_dir), "error", exc.errno))

            package_entries = 0
            package_scan_errors = []
            for package_root, package_dirs, package_files in os.walk(
                skill_dir, followlinks=False, onerror=package_scan_errors.append
            ):
                package_dirs[:] = sorted(
                    dirname
                    for dirname in package_dirs
                    if dirname not in _GENERATED_PACKAGE_DIRS
                )
                package_files.sort()
                package_root_path = Path(package_root)
                symlink_dirs = []
                for dirname in list(package_dirs):
                    directory = package_root_path / dirname
                    if directory.is_symlink():
                        symlink_dirs.append(directory)
                        package_dirs.remove(dirname)
                package_paths = [
                    *symlink_dirs,
                    *(package_root_path / name for name in package_files),
                ]
                for package_path in package_paths:
                    relative = package_path.relative_to(skill_dir)
                    if _is_generated_package_artifact(relative):
                        continue
                    if len(relative.parts) == 1 and relative.name in _EXCLUDED_FILES:
                        continue
                    package_entries += 1
                    scanned_entries += 1
                    if package_entries > MAX_VALIDATED_PACKAGE_FILES:
                        entries.append((str(skill_dir), "package_limit_exceeded"))
                        break
                    if scanned_entries > max_signature_entries:
                        return uncacheable("validation_signature_entry_limit")
                    try:
                        package_stat = os.lstat(package_path)
                    except OSError as exc:
                        entries.append((str(package_path), "error", exc.errno))
                        continue
                    entries.append((
                        str(package_path),
                        package_stat.st_mtime_ns,
                        package_stat.st_ctime_ns,
                        package_stat.st_size,
                        package_stat.st_ino,
                        package_stat.st_mode,
                    ))
                if package_entries > MAX_VALIDATED_PACKAGE_FILES:
                    break
            for exc in package_scan_errors:
                entries.append((str(skill_dir), "package_scan_error", exc.errno))
        for exc in scan_errors:
            entries.append((str(root_path), "skill_scan_error", exc.errno))
    return tuple(sorted(entries, key=str))


def validation_allows_discovery(skill_dir: Path) -> bool:
    """Preserve legacy skills, but gate packages that opted into validation."""
    record = read_skill_validation(skill_dir)
    if record is None:
        tests_dir = skill_dir / "tests"
        has_tests = tests_dir.is_dir() and any(tests_dir.rglob("test_*.py"))
        if has_tests:
            # A tested package with no validation record has not been through
            # the lifecycle gate. Fail closed unconditionally: never discover a
            # code-backed skill whose tests were not recorded as passing,
            # regardless of platform sidecar support.
            return False
        return True
    return record.get("status") in {"passed", "static"}


def read_skill_validation(skill_dir: Path) -> Optional[Dict[str, Any]]:
    """Return a validation record, marking it stale in-memory if content changed."""
    path = validation_path(skill_dir)
    from tools.skill_sidecar_io import secure_sidecar_io_available

    if not secure_sidecar_io_available():
        if not os.path.lexists(path):
            return None
        return {"status": "invalid", "reason": "secure sidecar I/O is unavailable"}
    if path.is_symlink():
        return {
            "status": "invalid",
            "reason": "validation record cannot be a symlink",
        }
    try:
        data, _ = read_sidecar(
            skill_dir,
            VALIDATION_FILE,
            max_bytes=MAX_VALIDATION_RECORD_BYTES,
        )
    except ValueError:
        return {"status": "invalid", "reason": "validation record is too large"}
    except OSError:
        return {"status": "invalid", "reason": "validation record is unreadable"}
    if data is None:
        return None
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return {"status": "invalid", "reason": "validation record is unreadable"}
    if not isinstance(record, dict):
        return {"status": "invalid", "reason": "validation record is not an object"}
    if record.get("schema") != VALIDATION_SCHEMA:
        return {"status": "invalid", "reason": "unsupported validation schema"}
    status = record.get("status")
    if status not in {"draft", "pending", "passed", "failed", "static"}:
        return {"status": "invalid", "reason": "unknown validation status"}
    tests_collected = record.get("tests_collected")
    if (
        not isinstance(tests_collected, int)
        or isinstance(tests_collected, bool)
        or tests_collected < 0
    ):
        return {"status": "invalid", "reason": "invalid tests_collected value"}

    actual_tests = (
        len(list((skill_dir / "tests").rglob("test_*.py")))
        if (skill_dir / "tests").is_dir()
        else 0
    )
    if tests_collected != actual_tests:
        return {"status": "invalid", "reason": "tests_collected does not match package"}
    record_approval_id = record.get("approval_id")
    if record_approval_id is not None and (
        not isinstance(record_approval_id, str)
        or not record_approval_id.isalnum()
        or len(record_approval_id) > 64
    ):
        return {"status": "invalid", "reason": "invalid validation approval id"}

    common = {"schema", "status", "tests_collected", "content_digest", "approval_id"}
    allowed_by_status = {
        "draft": common | {"reason", "updated_at"},
        "pending": common | {"validation_token", "reason", "updated_at"},
        "static": common | {"reason", "validated_at", "updated_at"},
        "passed": common | {"command", "exit_code", "output", "validated_at"},
        "failed": common | {"command", "exit_code", "output", "validated_at"},
    }
    if set(record) - allowed_by_status[status]:
        return {"status": "invalid", "reason": "validation record has unknown fields"}

    for timestamp_key in ("validated_at", "updated_at"):
        if timestamp_key in record and (
            not isinstance(record[timestamp_key], str)
            or len(record[timestamp_key]) > 128
        ):
            return {"status": "invalid", "reason": "invalid validation timestamp"}

    if status == "draft":
        if not isinstance(record.get("reason"), str) or not isinstance(
            record.get("updated_at"), str
        ):
            return {"status": "invalid", "reason": "invalid draft validation record"}
    elif status == "pending":
        token = record.get("validation_token")
        if (
            tests_collected <= 0
            or not isinstance(token, str)
            or not 24 <= len(token) <= 128
            or not isinstance(record.get("reason"), str)
            or not isinstance(record.get("updated_at"), str)
        ):
            return {"status": "invalid", "reason": "invalid pending validation record"}
    elif status == "static":
        if tests_collected != 0 or not (
            isinstance(record.get("validated_at"), str)
            or isinstance(record.get("updated_at"), str)
        ):
            return {"status": "invalid", "reason": "invalid static validation record"}
    else:
        exit_code = record.get("exit_code")
        if (
            tests_collected <= 0
            or not isinstance(record.get("command"), str)
            or not record["command"].strip()
            or len(record["command"]) > 4096
            or not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not isinstance(record.get("output"), str)
            or len(record["output"]) > MAX_VALIDATION_OUTPUT_CHARS
            or not isinstance(record.get("validated_at"), str)
        ):
            return {"status": "invalid", "reason": "invalid test evidence fields"}
        if (status == "passed" and exit_code != 0) or (
            status == "failed" and exit_code == 0
        ):
            return {"status": "invalid", "reason": "status contradicts exit code"}

    content_digest = record.get("content_digest")
    if (
        not isinstance(content_digest, str)
        or len(content_digest) != 64
        or any(character not in "0123456789abcdef" for character in content_digest)
    ):
        return {"status": "invalid", "reason": "invalid content digest"}
    try:
        current_digest = skill_content_digest(skill_dir)
    except ValueError as exc:
        return {"status": "invalid", "reason": str(exc)}
    if record.get("content_digest") != current_digest:
        if status == "draft":
            record = {
                "schema": VALIDATION_SCHEMA,
                "status": "draft",
                "tests_collected": tests_collected,
                "reason": "skill package under lifecycle construction",
                "content_digest": current_digest,
            }
        elif tests_collected == 0:
            record = {
                "schema": VALIDATION_SCHEMA,
                "status": "static",
                "tests_collected": 0,
                "reason": "text-only skill changed; frontmatter is rechecked on load",
                "content_digest": current_digest,
            }
        else:
            record = {
                "schema": VALIDATION_SCHEMA,
                "status": "pending",
                "tests_collected": tests_collected,
                "reason": "skill package changed; request a new validation challenge",
                "content_digest": current_digest,
            }
    allowed_fields = {
        "schema",
        "status",
        "tests_collected",
        "approval_id",
        "validation_token",
        "content_digest",
        "reason",
        "command",
        "exit_code",
        "output",
        "validated_at",
        "updated_at",
    }
    return {
        key: value
        for key, value in record.items()
        if key in allowed_fields and value is not None
    }
