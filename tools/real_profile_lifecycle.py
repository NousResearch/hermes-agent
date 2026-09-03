"""Persistent lifecycle tracking for real-profile Chromium processes.

The real-profile browser path launches the user's signed browser binary
directly on a Hermes-owned profile copy.  Unlike an ``agent-browser`` daemon,
that child is not discoverable from a socket directory after the owning Hermes
process crashes.  This module records enough identity to reap only the exact
process we launched, while allowing several live Hermes processes to share the
same copy-browser safely.

Records are deliberately narrow: only the managed profile path and the exact
browser binary are accepted.  Every destructive operation revalidates the
PID, process start time, executable, and ``--user-data-dir`` argument before
delegating to the shared process-tree terminator.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RECORD_VERSION = 1
_BROWSER_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RECORD_DIRNAME = "real-profile"


def _browser_name_is_safe(browser: str) -> bool:
    return isinstance(browser, str) and bool(_BROWSER_NAME.fullmatch(browser))


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()).expanduser()


def _managed_profile_dir(browser: str) -> Path:
    return (_hermes_home() / "browser-profile" / browser).resolve(strict=False)


def _canonical_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _state_dir(*, create: bool = True) -> Optional[Path]:
    try:
        path = _hermes_home() / "cache" / "browser-use" / _RECORD_DIRNAME
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("Could not resolve real-profile lifecycle state: %s", exc)
        return None
    if not create:
        return path if path.is_dir() else None
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        return path
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not create real-profile lifecycle state: %s", exc)
        return None


def _record_path(browser: str, *, create: bool = False) -> Optional[Path]:
    if not _browser_name_is_safe(browser):
        return None
    state_dir = _state_dir(create=create)
    return state_dir / f"{browser}.json" if state_dir else None


def _safe_start_time(pid: Optional[int]) -> Optional[int]:
    if not pid:
        return None
    try:
        from tools.process_registry import ProcessRegistry

        return ProcessRegistry._safe_host_start_time(int(pid))
    except Exception:
        return None


def _owner_entry(pid: int) -> dict[str, Optional[int]]:
    return {"pid": int(pid), "start_time": _safe_start_time(int(pid))}


def _normalize_owner_entries(record: dict[str, Any]) -> list[dict[str, Optional[int]]]:
    """Read current and legacy owner fields into a de-duplicated list."""
    raw = record.get("owners")
    candidates: list[Any] = raw if isinstance(raw, list) else []
    if not candidates and record.get("owner_pid") is not None:
        candidates = [
            {
                "pid": record.get("owner_pid"),
                "start_time": record.get("owner_start_time"),
            }
        ]

    owners: list[dict[str, Optional[int]]] = []
    seen: set[tuple[int, Optional[int]]] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("pid"))
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        raw_start = item.get("start_time")
        try:
            start_time = int(raw_start) if raw_start is not None else None
        except (TypeError, ValueError):
            start_time = None
        key = (pid, start_time)
        if key in seen:
            continue
        seen.add(key)
        owners.append({"pid": pid, "start_time": start_time})
    return owners


def _with_owner_fields(
    record: dict[str, Any], owners: list[dict[str, Optional[int]]]
) -> dict[str, Any]:
    """Return a record with the multi-owner list and legacy primary fields."""
    updated = dict(record)
    updated["owners"] = owners
    if owners:
        primary = owners[-1]
        updated["owner_pid"] = primary["pid"]
        updated["owner_start_time"] = primary["start_time"]
    else:
        updated.pop("owner_pid", None)
        updated.pop("owner_start_time", None)
    return updated


def _write_record(browser: str, record: dict[str, Any]) -> bool:
    path = _record_path(browser, create=True)
    if path is None:
        return False
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{browser}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            os.chmod(handle.name, 0o600)
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not persist real-profile lifecycle state for %s: %s", browser, exc
        )
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _read_record(browser: str) -> Optional[dict[str, Any]]:
    path = _record_path(browser, create=False)
    if path is None:
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _unlink_record(browser: str) -> None:
    path = _record_path(browser, create=False)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(
            "Could not remove real-profile lifecycle state for %s: %s", browser, exc
        )


def _record_target(
    record: dict[str, Any],
) -> Optional[tuple[str, Path, Path, int, Optional[int]]]:
    browser_value = record.get("browser")
    if not isinstance(browser_value, str) or not _browser_name_is_safe(browser_value):
        return None
    browser = browser_value
    profile_dir = _canonical_path(record.get("profile_dir"))
    binary = _canonical_path(record.get("binary"))
    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool) or not isinstance(raw_pid, (int, str)):
        return None
    try:
        expected_pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if expected_pid <= 0 or profile_dir is None or binary is None:
        return None
    try:
        managed_profile = _managed_profile_dir(browser)
    except (OSError, RuntimeError, ValueError):
        return None
    if profile_dir != managed_profile:
        return None
    raw_start = record.get("target_start_time", record.get("start_time"))
    try:
        target_start = int(raw_start) if raw_start is not None else None
    except (TypeError, ValueError):
        return None
    return browser, profile_dir, binary, expected_pid, target_start


def _process_is_alive(proc: Any) -> bool:
    try:
        import psutil

        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _process_user_data_dir(cmdline: list[str]) -> Optional[Path]:
    for index, arg in enumerate(cmdline):
        if not isinstance(arg, str):
            continue
        if arg.startswith("--user-data-dir="):
            return _canonical_path(arg.split("=", 1)[1])
        if arg == "--user-data-dir" and index + 1 < len(cmdline):
            return _canonical_path(cmdline[index + 1])
    return None


def _process_binary(proc: Any, cmdline: list[str]) -> Optional[Path]:
    try:
        executable = proc.exe()
    except Exception:
        executable = cmdline[0] if cmdline else None
    return _canonical_path(executable)


def _target_matches(
    pid: int,
    profile_dir: Path,
    binary: Path,
    expected_start: Optional[int],
) -> bool:
    """Verify a recorded PID is the exact direct browser incarnation."""
    if expected_start is None:
        # A target without an identity fingerprint cannot be safely killed
        # after a restart.  A current Popen handle may still be cleaned up by
        # its owner, but the cross-process reaper must leave this record alone.
        return False
    try:
        import psutil

        proc = psutil.Process(int(pid))
        if not _process_is_alive(proc):
            return False
        if _safe_start_time(int(pid)) != expected_start:
            return False
        cmdline = proc.cmdline()
        # Renderer/GPU children can inherit the profile argument.  The main
        # process is the only exact executable/profile match without a
        # Chromium process-type switch.
        if any(arg.startswith("--type=") for arg in cmdline):
            return False
        if _process_binary(proc, cmdline) != binary:
            return False
        return _process_user_data_dir(cmdline) == profile_dir
    except Exception:
        return False


def _matching_browser_processes(
    profile_dir: Path, binary: Path
) -> list[tuple[int, int]]:
    """Find uniquely identifiable main Chromium processes for a profile copy."""
    matches: list[tuple[int, int]] = []
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                pid = int(proc.info["pid"])
                cmdline = [
                    arg for arg in (proc.info.get("cmdline") or [])
                    if isinstance(arg, str)
                ]
                if not cmdline or any(arg.startswith("--type=") for arg in cmdline):
                    continue
                if _process_binary(proc, cmdline) != binary:
                    continue
                if _process_user_data_dir(cmdline) != profile_dir:
                    continue
                start_time = _safe_start_time(pid)
                if start_time is not None and _process_is_alive(proc):
                    matches.append((pid, start_time))
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
                continue
    except Exception:
        return []
    return matches


def _owner_is_alive(owner: dict[str, Optional[int]]) -> bool:
    pid = owner.get("pid")
    if not pid:
        return False
    try:
        from gateway.status import _pid_exists

        if not _pid_exists(int(pid)):
            return False
    except Exception:
        return False
    expected_start = owner.get("start_time")
    if expected_start is None:
        # Legacy records have no owner fingerprint.  Treat a live PID as live
        # to avoid killing an unrelated process; this may leak until the
        # record is manually removed, which is safer than a false kill.
        return True
    return _safe_start_time(int(pid)) == expected_start


def _live_owners(record: dict[str, Any]) -> list[dict[str, Optional[int]]]:
    return [
        owner for owner in _normalize_owner_entries(record) if _owner_is_alive(owner)
    ]


def _base_record(
    browser: str,
    profile_dir: Path,
    binary: Path,
    pid: int,
    target_start: Optional[int],
) -> dict[str, Any]:
    now = time.time()
    return {
        "version": _RECORD_VERSION,
        "browser": browser,
        "pid": int(pid),
        "profile_dir": str(profile_dir),
        "binary": str(binary),
        "target_start_time": target_start,
        # Keep the old field name for operators/scripts that already inspect
        # real-profile records, while the new field is more explicit.
        "start_time": target_start,
        "started_at": now,
    }


def register_real_profile_chrome(
    browser: str,
    profile_dir: str,
    binary: str,
    pid: int,
) -> Optional[dict[str, Any]]:
    """Persist ownership for a just-launched real-profile browser.

    Returns the normalized target record on success.  ``None`` means the
    profile/binary was not a managed target or state could not be persisted;
    callers must terminate the just-launched process in that case.
    """
    if not _browser_name_is_safe(browser):
        return None
    profile = _canonical_path(profile_dir)
    executable = _canonical_path(binary)
    if (
        profile is None
        or executable is None
        or profile != _managed_profile_dir(browser)
    ):
        return None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None

    target_start = _safe_start_time(pid)
    record = _base_record(browser, profile, executable, pid, target_start)
    current = _read_record(browser)
    current_target_matches = False
    if current:
        parsed = _record_target(current)
        if parsed and parsed[1:] == (profile, executable, pid, target_start):
            current_target_matches = True
        elif parsed:
            old_browser, old_profile, old_binary, old_pid, old_start = parsed
            if _target_matches(old_pid, old_profile, old_binary, old_start):
                logger.warning(
                    "Refusing to replace live real-profile %s browser pid %d with pid %d",
                    old_browser,
                    old_pid,
                    pid,
                )
                return None

    owners = _normalize_owner_entries(current or {}) if current_target_matches else []
    owners = [owner for owner in owners if _owner_is_alive(owner)]
    owners.append(_owner_entry(os.getpid()))
    record = _with_owner_fields(record, owners)
    if not _write_record(browser, record):
        return None
    return record


def claim_real_profile_chrome(
    browser: str,
    profile_dir: str,
    binary: str,
) -> Optional[dict[str, Any]]:
    """Claim a browser already serving the managed profile copy.

    This is used when a new Hermes process reuses an existing shared
    agent-browser session.  If no record exists (for example, a browser was
    launched by a Hermes build before persistent tracking), the exact process
    is discovered from its executable/profile arguments and recorded.  The
    function fails closed when the process identity is ambiguous.
    """
    if not _browser_name_is_safe(browser):
        return None
    profile = _canonical_path(profile_dir)
    executable = _canonical_path(binary)
    if (
        profile is None
        or executable is None
        or profile != _managed_profile_dir(browser)
    ):
        return None

    current = _read_record(browser)
    target: Optional[tuple[int, int]] = None
    if current:
        parsed = _record_target(current)
        if parsed is None:
            current = None
        else:
            _, current_profile, current_binary, pid, start_time = parsed
            if (
                current_profile == profile
                and current_binary == executable
                and start_time is not None
                and _target_matches(pid, profile, executable, start_time)
            ):
                target = (pid, start_time)
            elif not _target_matches(pid, current_profile, current_binary, start_time):
                # The record is stale; replace it below after discovering the
                # live process.  Never kill or overwrite a live mismatched one.
                current = None
            else:
                return None

    if target is None:
        matches = _matching_browser_processes(profile, executable)
        if len(matches) != 1:
            return None
        target = matches[0]
        current = None

    pid, target_start = target
    record = _base_record(browser, profile, executable, pid, target_start)
    if current:
        # Preserve the original launch timestamp and existing owners when the
        # record already describes this exact target.
        record["started_at"] = current.get("started_at", record["started_at"])
        owners = [
            owner
            for owner in _normalize_owner_entries(current)
            if _owner_is_alive(owner)
        ]
    else:
        owners = []
    current_owner = _owner_entry(os.getpid())
    owners = [owner for owner in owners if owner != current_owner]
    owners.append(current_owner)
    record = _with_owner_fields(record, owners)
    if not _write_record(browser, record):
        return None
    return record


def _remove_current_owner(record: dict[str, Any]) -> list[dict[str, Optional[int]]]:
    current_pid = os.getpid()
    current_start = _safe_start_time(current_pid)
    owners = _normalize_owner_entries(record)
    remaining = []
    for owner in owners:
        if owner.get("pid") != current_pid:
            remaining.append(owner)
            continue
        if current_start is None or owner.get("start_time") in (None, current_start):
            continue
        remaining.append(owner)
    return [owner for owner in remaining if _owner_is_alive(owner)]


def retire_real_profile_chrome(
    browser: str,
    profile_dir: str,
    binary: str,
    pid: Optional[int],
) -> Optional[bool]:
    """Release this Hermes owner and stop the target when no owner remains.

    Returns ``True`` when the target was stopped or was already gone,
    ``False`` when another live Hermes owner still uses it, and ``None`` when
    no trustworthy record describes the supplied process.
    """
    if not _browser_name_is_safe(browser):
        return None
    record = _read_record(browser)
    if not record:
        return None
    parsed = _record_target(record)
    profile = _canonical_path(profile_dir)
    executable = _canonical_path(binary)
    if not parsed or profile is None or executable is None:
        return None
    _, recorded_profile, recorded_binary, recorded_pid, recorded_start = parsed
    if recorded_profile != profile or recorded_binary != executable:
        return None
    if pid is not None and int(pid) != recorded_pid:
        return None

    owners = _remove_current_owner(record)
    if owners:
        updated = _with_owner_fields(record, owners)
        return False if _write_record(browser, updated) else False

    if recorded_start is None:
        # The current owner can still use its Popen handle as a last-resort
        # cleanup path, but a cross-process caller must not unlink this record
        # or signal a PID without an incarnation fingerprint.
        return False

    if _target_matches(recorded_pid, recorded_profile, recorded_binary, recorded_start):
        try:
            from tools.process_registry import ProcessRegistry

            ProcessRegistry._terminate_host_pid(
                recorded_pid, expected_start=recorded_start
            )
        except Exception as exc:
            logger.debug(
                "Could not stop real-profile Chrome pid %d: %s", recorded_pid, exc
            )
            return False
    _unlink_record(browser)
    return True


def has_live_real_profile_owner(browser: str) -> bool:
    """Return whether a tracked real-profile browser still has a live owner."""
    record = _read_record(browser)
    return bool(record and _live_owners(record))


def reap_orphaned_real_profile_chrome() -> int:
    """Reap direct real-profile Chrome processes whose owners are gone."""
    state_dir = _state_dir(create=False)
    if state_dir is None:
        return 0
    reaped = 0
    for record_path in sorted(state_dir.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            # A partially-written/malformed record is not evidence that a
            # process is ours.  Leave it for inspection; never guess a PID.
            continue
        if not isinstance(record, dict):
            continue
        parsed = _record_target(record)
        if parsed is None:
            continue
        browser, profile_dir, binary, pid, target_start = parsed
        if _live_owners(record):
            continue
        if target_start is None:
            logger.warning(
                "Refusing to reap real-profile Chrome pid %d (%s): "
                "no process start-time fingerprint",
                pid,
                browser,
            )
            continue
        if not _target_matches(pid, profile_dir, binary, target_start):
            # The target already exited or the PID was recycled.  Removing a
            # stale record is safe because no process was signalled.
            try:
                record_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        try:
            from tools.process_registry import ProcessRegistry

            ProcessRegistry._terminate_host_pid(pid, expected_start=target_start)
        except Exception as exc:
            logger.debug(
                "Could not reap orphaned real-profile Chrome pid %d: %s", pid, exc
            )
            continue

        # Do not discard a record if the exact process is still alive after a
        # failed/denied tree-kill; the next sweep can retry safely.
        if not _target_matches(pid, profile_dir, binary, target_start):
            try:
                record_path.unlink(missing_ok=True)
            except OSError:
                pass
            reaped += 1
            logger.info("Reaped orphaned real-profile Chrome pid %d (%s)", pid, browser)
    return reaped
