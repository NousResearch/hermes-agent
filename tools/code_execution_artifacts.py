"""Private, owner-scoped retention for ``execute_code`` overflow artifacts.

Artifacts are written where the selected execution backend can read them: local
runs use a contained directory under the active ``HERMES_HOME``; remote runs
write directly to the backend's own temp directory.  Nothing is added to the
cache mount list.  Records are bounded by bytes, count, and age, and session
cleanup removes exactly the current profile/owner's records.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import posixpath
import shlex
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_ARTIFACT_BYTES = 5_000_000
MAX_ARTIFACTS_PER_OWNER = 4
MAX_ARTIFACTS_PER_PROCESS = 64
ARTIFACT_MAX_AGE_SECONDS = 24 * 60 * 60
_REAPER_INTERVAL_SECONDS = 5 * 60


@dataclass(frozen=True)
class _Artifact:
    scope: tuple[str, str]
    path: str
    root: str
    created_at: float
    env: Any | None


_lock = threading.Lock()
_records: dict[tuple[str, str], list[_Artifact]] = {}
# Local profile roots seen by this process -> monotonic time of their last crash-leftover scan.
_local_root_scans: dict[str, float] = {}
_reaper_started = False


def _scope(owner: str) -> tuple[str, str]:
    from hermes_constants import hermes_home_key

    return hermes_home_key(), str(owner or "default")


def _scope_digest(scope: tuple[str, str]) -> str:
    return hashlib.sha256((scope[0] + "\0" + scope[1]).encode("utf-8")).hexdigest()[:16]


def _bounded_redacted_text(text: str) -> str:
    """Apply the inline output pass to at most ``MAX_ARTIFACT_BYTES`` bytes."""
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    data = str(text or "").encode("utf-8", errors="replace")
    if len(data) > MAX_ARTIFACT_BYTES:
        marker = b"\n\n[... retained execute_code artifact capped at 5,000,000 bytes ...]"
        data = data[: MAX_ARTIFACT_BYTES - len(marker)] + marker
    decoded = data.decode("utf-8", errors="replace")
    return redact_sensitive_text(strip_ansi(decoded), code_file=True)


def _local_root() -> Path:
    from hermes_constants import get_hermes_home
    from tools.path_security import validate_within_dir
    from tools.spill_safety import ensure_spill_dir

    home = get_hermes_home()
    root = home / "cache" / "exec"
    if error := validate_within_dir(root, home):
        raise OSError(f"execute_code artifact directory escapes HERMES_HOME: {error}")
    ensure_spill_dir(root, private=True)
    _prune_stale_local_root(root)
    _ensure_reaper()
    return root


def _prune_stale_local_root(root: Path, *, force: bool = False) -> None:
    """Reclaim crash-leftover ``owner-*`` dirs older than the retention window.

    Unregistered dirs (left by a crashed process) are invisible to ``_records``, so the
    root itself is rescanned: on first use, at most once per reaper interval from the
    write path, and on every reaper pass (``force``). A leftover that was still young at
    the first scan is therefore removed once it ages past the window, without a restart.
    """
    root_key = str(root)
    now = time.monotonic()
    with _lock:
        last = _local_root_scans.get(root_key)
        if not force and last is not None and now - last < _REAPER_INTERVAL_SECONDS:
            return
        _local_root_scans[root_key] = now
    cutoff = time.time() - ARTIFACT_MAX_AGE_SECONDS
    try:
        candidates = list(root.glob("owner-*"))
    except OSError:
        return
    for candidate in candidates:
        try:
            st = candidate.lstat()
            if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_mtime >= cutoff:
                continue
            shutil.rmtree(candidate)
        except OSError:
            logger.debug("Could not prune stale execute_code artifact dir %s", candidate)


def _remote_root(env: Any, scope: tuple[str, str]) -> str:
    from tools.code_execution_rpc import _execute_checked, _private_dirs_cmd
    from tools.code_execution_tool import _env_temp_dir

    base = _env_temp_dir(env)
    root = posixpath.join(base, f"hermes_exec_artifacts_{_scope_digest(scope)}")
    _execute_checked(env, _private_dirs_cmd(root), "execute_code artifact directory setup", timeout=15)
    return root


def _safe_kind(kind: str) -> str:
    return "diagnostic" if kind == "diagnostic" else "output"


def retain_artifact(owner: str, text: str, *, env: Any | None = None,
                    kind: str = "output") -> Optional[str]:
    """Retain text privately for *owner* and return its backend-visible path."""
    scope = _scope(owner)
    filename = f"{_safe_kind(kind)}-{time.time_ns()}-{uuid.uuid4().hex[:8]}.txt"
    content = _bounded_redacted_text(text)
    try:
        if env is None:
            from tools.spill_safety import ensure_spill_dir, write_text_exclusive

            root_path = _local_root() / f"owner-{_scope_digest(scope)}"
            ensure_spill_dir(root_path, private=True)
            path = root_path / filename
            write_text_exclusive(path, content, private=True, errors="replace")
            root, visible = str(root_path), str(path)
        else:
            from tools.code_execution_rpc import _remote_write

            root = _remote_root(env, scope)
            visible = posixpath.join(root, filename)
            _remote_write(env, visible, content, check=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - retention is best-effort
        logger.debug("Failed to retain execute_code %s artifact: %s", kind, exc, exc_info=True)
        return None

    _register(_Artifact(scope=scope, path=visible, root=root,
                        created_at=time.time(), env=env))
    return visible


def _register(record: _Artifact) -> None:
    with _lock:
        _records.setdefault(record.scope, []).append(record)
        doomed = _pop_excess_locked(time.time())
    for old in doomed:
        _delete(old)
    _ensure_reaper()


def _ensure_reaper() -> None:
    global _reaper_started
    with _lock:
        if _reaper_started:
            return
        _reaper_started = True
    threading.Thread(target=_reaper_loop, name="hermes-exec-artifact-reaper",
                     daemon=True).start()


def _pop_excess_locked(now: float) -> list[_Artifact]:
    doomed: list[_Artifact] = []
    cutoff = now - ARTIFACT_MAX_AGE_SECONDS
    for scope in list(_records):
        keep = []
        for record in _records[scope]:
            (doomed if record.created_at < cutoff else keep).append(record)
        if len(keep) > MAX_ARTIFACTS_PER_OWNER:
            keep.sort(key=lambda item: item.created_at)
            doomed.extend(keep[:-MAX_ARTIFACTS_PER_OWNER])
            keep = keep[-MAX_ARTIFACTS_PER_OWNER:]
        if keep:
            _records[scope] = keep
        else:
            _records.pop(scope, None)
    remaining = sorted(
        (record for records in _records.values() for record in records),
        key=lambda item: item.created_at,
    )
    excess = max(0, len(remaining) - MAX_ARTIFACTS_PER_PROCESS)
    for record in remaining[:excess]:
        records = _records.get(record.scope, [])
        if record in records:
            records.remove(record)
            doomed.append(record)
            if not records:
                _records.pop(record.scope, None)
    return doomed


def _delete(record: _Artifact) -> None:
    if record.env is None:
        try:
            Path(record.path).unlink(missing_ok=True)
            Path(record.root).rmdir()
        except OSError:
            pass
        return
    try:
        record.env.execute(f"rm -f {shlex.quote(record.path)}", cwd="/", timeout=15)
    except Exception:
        logger.debug("Could not remove remote execute_code artifact %s", record.path)


def cleanup_expired_artifacts() -> int:
    """Delete registered artifacts past their bounded retention window.

    Also rescans every local profile root this process has used, so unregistered
    crash leftovers are held to the same age bound as registered records.
    """
    with _lock:
        doomed = _pop_excess_locked(time.time())
        local_roots = list(_local_root_scans)
    for record in doomed:
        _delete(record)
    for root in local_roots:
        _prune_stale_local_root(Path(root), force=True)
    return len(doomed)


def _reaper_loop() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL_SECONDS)
        try:
            cleanup_expired_artifacts()
        except Exception:
            logger.debug("execute_code artifact reaper failed", exc_info=True)


def cleanup_artifacts_for_owner(owner: str) -> None:
    """Delete the active profile's artifacts for one session owner."""
    scope = _scope(owner)
    with _lock:
        doomed = _records.pop(scope, [])
    _delete_grouped(doomed)


def cleanup_artifacts_where(owner_matches: Callable[[str], bool]) -> None:
    """Delete active-profile artifacts whose raw owner satisfies *owner_matches*."""
    from hermes_constants import hermes_home_key

    home = hermes_home_key()
    with _lock:
        scopes = [scope for scope in _records
                  if scope[0] == home and owner_matches(scope[1])]
        doomed = [record for scope in scopes for record in _records.pop(scope, [])]
    _delete_grouped(doomed)


def _delete_grouped(records: list[_Artifact]) -> None:
    remote_roots: dict[tuple[int, str], _Artifact] = {}
    for record in records:
        if record.env is None:
            _delete(record)
        else:
            remote_roots[(id(record.env), record.root)] = record
    for record in remote_roots.values():
        env = record.env
        if env is None:  # narrowed above; defensive for type checkers
            continue
        try:
            env.execute(f"rm -rf {shlex.quote(record.root)}", cwd="/", timeout=15)
        except Exception:
            logger.debug("Could not remove remote execute_code artifact dir %s", record.root)


def retain_local_runner_artifact(owner: str, raw_path: str, *, allowed_root: str,
                                 kind: str) -> Optional[str]:
    """Re-home a kernel-runner raw spill after a contained, no-symlink read."""
    path = Path(raw_path)
    try:
        root = Path(allowed_root).resolve(strict=True)
        if path.parent.resolve(strict=True) != root or path.is_symlink():
            return None
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, os.O_RDONLY | nofollow)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            chunks: list[bytes] = []
            remaining = MAX_ARTIFACT_BYTES + 1
            while remaining > 0:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(fd)
        text = b"".join(chunks).decode("utf-8-sig", errors="replace")
    except OSError:
        return None
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    return retain_artifact(owner, text, kind=kind)


def retain_remote_runner_artifact(owner: str, raw_path: str, *, allowed_root: str,
                                  env: Any, kind: str) -> Optional[str]:
    """Re-home a remote runner spill without allowing it to name an arbitrary backend path."""
    root = posixpath.normpath(allowed_root)
    path = posixpath.normpath(raw_path)
    if not root.startswith("/") or not path.startswith(root.rstrip("/") + "/"):
        return None
    try:
        result = env.execute(f"cat {shlex.quote(path)}", cwd="/", timeout=60)
        if result.get("returncode", 1) != 0:
            return None
        text = str(result.get("output", "") or "")
    except Exception:
        return None
    finally:
        try:
            env.execute(f"rm -f {shlex.quote(path)}", cwd="/", timeout=15)
        except Exception:
            pass
    return retain_artifact(owner, text, env=env, kind=kind)


def _cleanup_all() -> None:
    with _lock:
        doomed = [record for records in _records.values() for record in records]
        _records.clear()
    _delete_grouped(doomed)


atexit.register(_cleanup_all)
