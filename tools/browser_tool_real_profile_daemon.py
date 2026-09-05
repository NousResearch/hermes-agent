"""Durable ownership and cleanup for the shared real-profile attach daemon."""

import json
import os
import shutil
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Optional

from tools.browser_tool_origin import origin as _bt

_OWNER_SUFFIX = ".owners.json"
_LOCK_SUFFIX = ".owners.lock"


def _socket_dir() -> str:
    return os.path.join(_bt._socket_safe_tmpdir(), f"agent-browser-{_bt._REAL_PROFILE_SESSION}")


def _owner_path(socket_dir: str, session_name: str) -> Path:
    return Path(socket_dir) / f"{session_name}{_OWNER_SUFFIX}"


def _lock_path(socket_dir: str, session_name: str) -> Path:
    del session_name
    return Path(f"{socket_dir}{_LOCK_SUFFIX}")


def _process_start_time(pid: int) -> Optional[int]:
    from gateway.status import get_process_start_time

    return get_process_start_time(pid)


@contextmanager
def _locked(socket_dir: str, session_name: str):
    """Serialize owner read-modify-write and cleanup across Hermes processes."""
    path = _lock_path(socket_dir, session_name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\n")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            with suppress(OSError):
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def legacy_state_path() -> Optional[Path]:
    """Return pre-fix real-profile state outside the managed socket directory, if any."""
    env = _bt._build_browser_env()
    explicit = str(env.get("AGENT_BROWSER_SOCKET_DIR") or "").strip()
    xdg_runtime = str(env.get("XDG_RUNTIME_DIR") or "").strip()
    home = str(env.get("HOME") or "").strip()
    if explicit:
        legacy_dir = Path(explicit)
    elif xdg_runtime:
        legacy_dir = Path(xdg_runtime) / "agent-browser"
    elif home:
        legacy_dir = Path(home) / ".agent-browser"
    else:
        import tempfile

        legacy_dir = Path(tempfile.gettempdir()) / "agent-browser"
    if os.path.normcase(os.path.realpath(legacy_dir)) == os.path.normcase(os.path.realpath(_socket_dir())):
        return None
    session_name = _bt._REAL_PROFILE_SESSION
    for suffix in ("pid", "sock", "port", "engine", "stream", "version"):
        candidate = legacy_dir / f"{session_name}.{suffix}"
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return None


def _owner_identity() -> Optional[dict[str, int]]:
    owner_start = _process_start_time(os.getpid())
    if owner_start is None:
        return None
    return {"pid": os.getpid(), "start_time": owner_start}


def _read_owners(socket_dir: str, session_name: str) -> Optional[list[dict[str, int]]]:
    path = _owner_path(socket_dir, session_name)
    try:
        if path.is_symlink():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("session_name") != session_name or not isinstance(payload.get("owners"), list):
            return None
        owners = []
        for value in payload["owners"]:
            pid, started = int(value["pid"]), int(value["start_time"])
            if pid <= 0 or started <= 0:
                return None
            owners.append({"pid": pid, "start_time": started})
        return owners
    except FileNotFoundError:
        return []
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _write_legacy_owner_pid(socket_dir: str, session_name: str, owner_pid: int) -> bool:
    """Keep the existing reaper marker on a live owner for mixed-version safety."""
    try:
        from utils import atomic_write_text

        atomic_write_text(
            Path(socket_dir) / f"{session_name}.owner_pid",
            str(owner_pid),
            create_mode=0o600,
        )
        return True
    except OSError:
        return False


def _write_owners(socket_dir: str, session_name: str, owners: list[dict[str, int]]) -> bool:
    try:
        from utils import atomic_json_write

        atomic_json_write(
            _owner_path(socket_dir, session_name),
            {"session_name": session_name, "owners": owners},
            mode=0o600,
        )
        return True
    except OSError as exc:
        _bt.logger.warning("Cannot publish real-profile daemon owners for %s: %s", session_name, exc)
        return False


def _owner_alive(owner: dict[str, int]) -> Optional[bool]:
    """True for the recorded incarnation, False when dead/reused, None if ambiguous."""
    from gateway.status import _pid_exists

    pid = owner["pid"]
    if not _pid_exists(pid):
        return False
    current_start = _process_start_time(pid)
    if current_start is None:
        return None
    return current_start == owner["start_time"]


def _classify_owners(owners: list[dict[str, int]]) -> tuple[list[dict[str, int]], bool]:
    live = []
    ambiguous = False
    for owner in owners:
        state = _owner_alive(owner)
        if state is True:
            live.append(owner)
        elif state is None:
            ambiguous = True
    return live, ambiguous


def register(socket_dir: str, session_name: str) -> bool:
    """Add this process incarnation as an owner; fail closed on ambiguous state."""
    current = _owner_identity()
    if current is None:
        _bt.logger.warning("Cannot register real-profile daemon owner: process start time unavailable")
        return False
    try:
        with _locked(socket_dir, session_name):
            owners = _read_owners(socket_dir, session_name)
            if owners is None:
                _bt.logger.warning("Cannot register real-profile daemon owner: owner metadata is ambiguous")
                return False
            if not owners and (Path(socket_dir) / f"{session_name}.pid").exists():
                _bt.logger.warning("Cannot adopt real-profile daemon without durable owner metadata")
                return False
            live, ambiguous = _classify_owners(owners)
            if ambiguous:
                _bt.logger.warning("Cannot register real-profile daemon owner: owner identity is unreadable")
                return False
            if current not in live:
                live.append(current)
            return _write_owners(socket_dir, session_name, live)
    except OSError as exc:
        _bt.logger.warning("Cannot lock real-profile daemon ownership for %s: %s", session_name, exc)
        return False


@contextmanager
def exclusive_current_owner(socket_dir: str, session_name: str):
    """Yield whether this process is the sole provably live owner, while holding the owner lock."""
    current = _owner_identity()
    if current is None:
        yield False
        return
    try:
        with _locked(socket_dir, session_name):
            owners = _read_owners(socket_dir, session_name)
            if owners is None or current not in owners:
                yield False
                return
            live, ambiguous = _classify_owners(owners)
            if not ambiguous and live != owners:
                _write_owners(socket_dir, session_name, live)
            yield not ambiguous and live == [current]
    except OSError:
        yield False


def _daemon_pid(socket_dir: str, session_name: str) -> tuple[Optional[int], bool]:
    """Return ``(pid, trustworthy)``; a missing file is trustworthy absence, junk is ambiguous."""
    path = Path(socket_dir) / f"{session_name}.pid"
    try:
        if path.is_symlink():
            return None, False
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None, True
    except OSError:
        return None, False
    try:
        pid = int(raw)
    except ValueError:
        return None, False
    return (pid, True) if pid > 0 else (None, False)


def _stable_file(path: Path) -> Optional[tuple[str, os.stat_result]]:
    try:
        before = path.stat()
        text = path.read_text(encoding="utf-8")
        after = path.stat()
    except OSError:
        return None
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    ):
        return None
    return text, after


def _file_still_matches(path: Path, expected_text: str, expected_stat: os.stat_result) -> bool:
    current = _stable_file(path)
    if current is None:
        return False
    text, stat = current
    return text == expected_text and (
        stat.st_dev,
        stat.st_ino,
        stat.st_mtime_ns,
        stat.st_size,
    ) == (
        expected_stat.st_dev,
        expected_stat.st_ino,
        expected_stat.st_mtime_ns,
        expected_stat.st_size,
    )


def _claim_and_remove(
    socket_dir: str,
    evidence_name: str,
    expected_text: str,
    expected_stat: os.stat_result,
) -> bool:
    """Rename the inspected directory and delete only that exact state generation."""
    claim = f"{socket_dir}.reap-{os.getpid()}-{time.time_ns()}"
    try:
        os.rename(socket_dir, claim)
    except OSError:
        return False
    if not _file_still_matches(Path(claim) / evidence_name, expected_text, expected_stat):
        return False
    shutil.rmtree(claim, ignore_errors=True)
    return not os.path.exists(claim)


def _terminate_exact_daemon(socket_dir: str, session_name: str, daemon_pid: int) -> bool:
    """Terminate only the verified daemon incarnation bound to this socket directory."""
    from gateway.status import _pid_exists
    from tools.browser_tool_lifecycle import _verify_reapable_browser_daemon
    from tools.process_registry import ProcessRegistry

    daemon_start = _process_start_time(daemon_pid)
    if daemon_start is None:
        _bt.logger.warning(
            "Refusing to reap real-profile daemon PID %d: no process-start fingerprint", daemon_pid
        )
        return False
    if not _verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
        return False
    if _process_start_time(daemon_pid) != daemon_start:
        _bt.logger.warning(
            "Refusing to reap real-profile daemon PID %d: process incarnation changed", daemon_pid
        )
        return False
    try:
        ProcessRegistry._terminate_host_pid(daemon_pid, expected_start=daemon_start)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return not _pid_exists(daemon_pid)


def _remove_dead_state(socket_dir: str, session_name: str, *, allow_pidless: bool) -> bool:
    """Remove state only when no live daemon can still reference it."""
    from gateway.status import _pid_exists

    daemon_pid, pid_trustworthy = _daemon_pid(socket_dir, session_name)
    if not pid_trustworthy:
        _bt.logger.warning("Preserving real-profile daemon state %s: daemon PID is ambiguous", socket_dir)
        return False
    if daemon_pid is None and not allow_pidless:
        return False
    evidence_name = f"{session_name}.pid" if daemon_pid is not None else f"{session_name}{_OWNER_SUFFIX}"
    evidence = _stable_file(Path(socket_dir) / evidence_name)
    if evidence is None:
        return False
    expected_text, expected_stat = evidence
    if daemon_pid is not None and _pid_exists(daemon_pid):
        if not _terminate_exact_daemon(socket_dir, session_name, daemon_pid):
            return False
    if not _file_still_matches(Path(socket_dir) / evidence_name, expected_text, expected_stat):
        return False
    return _claim_and_remove(socket_dir, evidence_name, expected_text, expected_stat)


def cleanup_owned() -> bool:
    """Release this owner and stop the daemon only when no other live owner remains."""
    session_name, socket_dir = _bt._REAL_PROFILE_SESSION, _socket_dir()
    if not os.path.isdir(socket_dir):
        return False
    current = _owner_identity()
    if current is None:
        _bt.logger.warning("Refusing real-profile daemon cleanup: current owner identity is unavailable")
        return False
    try:
        with _locked(socket_dir, session_name):
            owners = _read_owners(socket_dir, session_name)
            if owners is None or current not in owners:
                _bt.logger.warning("Refusing real-profile daemon cleanup: ownership is ambiguous")
                return False
            remaining, ambiguous = _classify_owners([owner for owner in owners if owner != current])
            if ambiguous:
                _bt.logger.warning("Preserving real-profile daemon: another owner identity is unreadable")
                return False
            if remaining:
                if _write_owners(socket_dir, session_name, remaining):
                    _write_legacy_owner_pid(socket_dir, session_name, remaining[0]["pid"])
                return False
            return _remove_dead_state(socket_dir, session_name, allow_pidless=False)
    except OSError as exc:
        _bt.logger.warning("Cannot lock real-profile daemon cleanup: %s", exc)
        return False


def reap_orphans() -> int:
    """Stop the shared daemon only after every recorded Hermes owner is provably gone."""
    session_name, socket_dir = _bt._REAL_PROFILE_SESSION, _socket_dir()
    if not os.path.isdir(socket_dir):
        return 0
    try:
        with _locked(socket_dir, session_name):
            owners = _read_owners(socket_dir, session_name)
            if owners is None or not owners:
                _bt.logger.warning("Preserving real-profile daemon state %s: owner metadata is ambiguous", socket_dir)
                return 0
            live, ambiguous = _classify_owners(owners)
            if ambiguous:
                return 0
            if live:
                if live != owners and _write_owners(socket_dir, session_name, live):
                    _write_legacy_owner_pid(socket_dir, session_name, live[0]["pid"])
                return 0
            return int(_remove_dead_state(socket_dir, session_name, allow_pidless=False))
    except OSError as exc:
        _bt.logger.warning("Cannot lock real-profile daemon orphan cleanup: %s", exc)
        return 0
