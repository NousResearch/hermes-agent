"""Owned lifecycle and inode admission for local shell snapshots.

Snapshot files may contain environment-carried secrets.  Cleanup therefore uses
an exact per-session owner marker and never deletes arbitrary name matches.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import time
import uuid
from typing import Callable, cast

RUN = "RUN"
DEFER = "DEFER_NEW_SNAPSHOT"
FAIL_CLOSED = "FAIL_CLOSED"

_SESSION_RE = re.compile(r"^[0-9a-f]{12}$")
_MARKER_RE = re.compile(r"^hermes-session-([0-9a-f]{12})\.owner\.json$")


@dataclass(frozen=True)
class SnapshotLifecycleSettings:
    ttl_seconds: float = 86_400.0
    min_free_inode_ratio: float = 0.15
    critical_free_inode_ratio: float = 0.10
    min_free_inodes: int = 10_000
    critical_free_inodes: int = 1_000


@dataclass(frozen=True)
class InodeAdmission:
    outcome: str
    reason: str
    free_inode_ratio: float | None
    free_inodes: int | None = None


@dataclass(frozen=True)
class InodeHeadroom:
    free_inode_ratio: float | None
    free_inodes: int | None


@dataclass(frozen=True)
class OwnedArtifacts:
    session_id: str
    snapshot_path: str
    cwd_path: str
    marker_path: str
    pid: int
    uid: int | None
    hostname: str
    created_at: float


def settings_from_environment() -> SnapshotLifecycleSettings:
    """Load internal values bridged from ``terminal:`` config settings."""
    def _float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return float("nan")

    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            return -1

    return SnapshotLifecycleSettings(
        ttl_seconds=_float("TERMINAL_SNAPSHOT_TTL_SECONDS", 86_400.0),
        min_free_inode_ratio=_float("TERMINAL_SNAPSHOT_MIN_FREE_INODE_RATIO", 0.15),
        critical_free_inode_ratio=_float(
            "TERMINAL_SNAPSHOT_CRITICAL_FREE_INODE_RATIO", 0.10
        ),
        min_free_inodes=_int("TERMINAL_SNAPSHOT_MIN_FREE_INODES", 10_000),
        critical_free_inodes=_int("TERMINAL_SNAPSHOT_CRITICAL_FREE_INODES", 1_000),
    )


def decide_inode_admission(
    free_inode_ratio: float | None,
    settings: SnapshotLifecycleSettings,
    *,
    free_inodes: int | None = None,
) -> InodeAdmission:
    """Pure decision core using relative pressure plus absolute headroom.

    Quota/container views can expose a very small ``f_favail / f_files`` ratio
    while still providing millions of usable inodes.  Ratio-only defaults would
    disable the terminal on such hosts.  Absolute low headroom remains decisive;
    a low ratio is pressure only when absolute headroom is also low or unknown.
    """
    critical = settings.critical_free_inode_ratio
    minimum = settings.min_free_inode_ratio
    if (
        not all(math.isfinite(value) for value in (critical, minimum, settings.ttl_seconds))
        or not (0 <= critical < minimum <= 1)
        or settings.ttl_seconds < 0
        or isinstance(settings.min_free_inodes, bool)
        or isinstance(settings.critical_free_inodes, bool)
        or not isinstance(settings.min_free_inodes, int)
        or not isinstance(settings.critical_free_inodes, int)
        or not (0 <= settings.critical_free_inodes < settings.min_free_inodes)
    ):
        return InodeAdmission(FAIL_CLOSED, "INVALID_THRESHOLDS", free_inode_ratio, free_inodes)
    ratio_valid = (
        free_inode_ratio is not None
        and math.isfinite(free_inode_ratio)
        and 0 <= free_inode_ratio <= 1
    )
    count_valid = (
        free_inodes is not None
        and not isinstance(free_inodes, bool)
        and isinstance(free_inodes, int)
        and free_inodes >= 0
    )
    ratio_value = float(cast(float, free_inode_ratio)) if ratio_valid else None
    count_value = int(cast(int, free_inodes)) if count_valid else None
    if not ratio_valid and not count_valid:
        return InodeAdmission(
            FAIL_CLOSED, "INODE_MEASUREMENT_UNAVAILABLE", free_inode_ratio, free_inodes
        )
    if count_value is not None and count_value < settings.critical_free_inodes:
        return InodeAdmission(FAIL_CLOSED, "INODES_CRITICAL", free_inode_ratio, free_inodes)
    if (
        ratio_value is not None
        and ratio_value < critical
        and (count_value is None or count_value < settings.min_free_inodes)
    ):
        return InodeAdmission(FAIL_CLOSED, "INODES_CRITICAL", free_inode_ratio, free_inodes)
    if count_value is not None and count_value <= settings.min_free_inodes:
        return InodeAdmission(DEFER, "INODE_PRESSURE", free_inode_ratio, free_inodes)
    if ratio_value is not None and ratio_value <= minimum and count_value is None:
        return InodeAdmission(DEFER, "INODE_PRESSURE", free_inode_ratio, free_inodes)
    return InodeAdmission(RUN, "INODES_AVAILABLE", free_inode_ratio, free_inodes)


def measure_inode_headroom(path: str | Path) -> InodeHeadroom:
    """Measure POSIX inode ratio and absolute unprivileged headroom once."""
    if os.name == "nt":
        return InodeHeadroom(1.0, 2**31 - 1)
    try:
        values = os.statvfs(os.fspath(path))
    except (AttributeError, OSError):
        return InodeHeadroom(None, None)
    if values.f_files <= 0:
        return InodeHeadroom(None, int(values.f_favail))
    return InodeHeadroom(values.f_favail / values.f_files, int(values.f_favail))


def measure_free_inode_ratio(path: str | Path) -> float | None:
    """Compatibility wrapper for callers that need only the ratio."""
    return measure_inode_headroom(path).free_inode_ratio


def _paths(root: Path, session_id: str) -> tuple[Path, Path, Path]:
    if not _SESSION_RE.fullmatch(session_id):
        raise ValueError("snapshot session id must be exactly 12 lowercase hex characters")
    return (
        root / f"hermes-snap-{session_id}.sh",
        root / f"hermes-cwd-{session_id}.txt",
        root / f"hermes-session-{session_id}.owner.json",
    )


def _write_private_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except (OSError, NotImplementedError):
            pass
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def prepare_owned_artifacts(
    temp_root: str | Path,
    session_id: str,
    *,
    now: float | None = None,
    pid: int | None = None,
    uid: int | None = None,
    hostname: str | None = None,
) -> OwnedArtifacts:
    try:
        root = Path(temp_root).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"snapshot temp root is not a real directory: {temp_root}") from exc
    if not root.is_dir():
        raise RuntimeError(f"snapshot temp root is not a real directory: {root}")
    snapshot, cwd, marker = _paths(root, session_id)
    created_at = time.time() if now is None else float(now)
    process_id = os.getpid() if pid is None else int(pid)
    if uid is None and hasattr(os, "getuid"):
        uid = os.getuid()
    host = socket.gethostname() if hostname is None else str(hostname)
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "pid": process_id,
        "uid": uid,
        "hostname": host,
        "created_at": created_at,
    }
    _write_private_json(marker, payload)
    return OwnedArtifacts(
        session_id=session_id,
        snapshot_path=str(snapshot),
        cwd_path=str(cwd),
        marker_path=str(marker),
        pid=process_id,
        uid=uid,
        hostname=host,
        created_at=created_at,
    )


def _load_valid_marker(
    marker: Path,
    *,
    expected_session_id: str,
    uid: int | None,
    hostname: str,
) -> dict | None:
    try:
        info = marker.lstat()
        if not stat.S_ISREG(info.st_mode) or marker.is_symlink():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != 1:
        return None
    if payload.get("session_id") != expected_session_id:
        return None
    if payload.get("hostname") != hostname:
        return None
    if payload.get("uid") != uid:
        return None
    if not isinstance(payload.get("pid"), int):
        return None
    if not isinstance(payload.get("created_at"), (int, float)):
        return None
    return payload


def _unlink_regular(path: Path, removed: list[str]) -> None:
    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) and not path.is_symlink():
            path.unlink()
            removed.append(str(path))
    except OSError:
        pass


def cleanup_owned_artifacts(owned: OwnedArtifacts) -> list[str]:
    """Delete only artifacts authenticated by the exact session owner marker."""
    marker = Path(owned.marker_path)
    payload = _load_valid_marker(
        marker,
        expected_session_id=owned.session_id,
        uid=owned.uid,
        hostname=owned.hostname,
    )
    if payload is None:
        return []

    removed: list[str] = []
    snapshot = Path(owned.snapshot_path)
    cwd = Path(owned.cwd_path)
    _unlink_regular(snapshot, removed)
    _unlink_regular(cwd, removed)
    for temporary in snapshot.parent.glob(f"{snapshot.name}.tmp.*"):
        _unlink_regular(temporary, removed)
    _unlink_regular(marker, removed)
    return removed


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_stale_owned_artifacts(
    temp_root: str | Path,
    settings: SnapshotLifecycleSettings,
    *,
    now: float | None = None,
    uid: int | None = None,
    hostname: str | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> list[str]:
    """Reap TTL-expired, dead, self-owned sessions; refuse every ambiguous entry."""
    try:
        root = Path(temp_root).resolve(strict=True)
    except OSError:
        return []
    if not root.is_dir() or settings.ttl_seconds < 0:
        return []
    current = time.time() if now is None else float(now)
    if uid is None and hasattr(os, "getuid"):
        uid = os.getuid()
    host = socket.gethostname() if hostname is None else str(hostname)
    reaped: list[str] = []

    for marker in root.glob("hermes-session-*.owner.json"):
        match = _MARKER_RE.fullmatch(marker.name)
        if not match:
            continue
        session_id = match.group(1)
        payload = _load_valid_marker(
            marker,
            expected_session_id=session_id,
            uid=uid,
            hostname=host,
        )
        if payload is None:
            continue
        if current - float(payload["created_at"]) < settings.ttl_seconds:
            continue
        if pid_alive(int(payload["pid"])):
            continue
        snapshot, cwd, _ = _paths(root, session_id)
        owned = OwnedArtifacts(
            session_id=session_id,
            snapshot_path=str(snapshot),
            cwd_path=str(cwd),
            marker_path=str(marker),
            pid=int(payload["pid"]),
            uid=uid,
            hostname=host,
            created_at=float(payload["created_at"]),
        )
        cleanup_owned_artifacts(owned)
        if not marker.exists():
            reaped.append(session_id)
    return reaped
