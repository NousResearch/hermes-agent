"""Private, versioned host policy independent of profile/config snapshots.

An absent authority preserves unmanaged behavior. Once a ledger exists, removing
its policy is an error, not an implicit opt-out. This is cooperative OS-account
policy, not a sandbox against other programs running as that same account.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
import threading
import time
from pathlib import Path

from hermes_platform.host.identity import account_home

_LOCK = threading.Lock()
_OPEN_LOCKS: set[int] = set()
_MAX_BYTES = 2_000_000
LOCK_TIMEOUT_SECONDS = 5.0


class HostPolicyError(ValueError):
    """Installed host policy is unusable; reject new governed work."""


def policy_root() -> Path:
    return account_home() / ".local" / "state" / "hermes" / "host-policy"


def _private_stat(info: os.stat_result, *, directory: bool = False) -> None:
    valid_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    mode = 0o700 if directory else 0o600
    if not valid_type or (not directory and info.st_nlink != 1):
        raise HostPolicyError("Host policy requires owner-only regular files and a private directory")
    wrong_owner = hasattr(os, "getuid") and info.st_uid != os.getuid()
    if wrong_owner or stat.S_IMODE(info.st_mode) != mode:
        raise HostPolicyError("Host policy requires owner-only regular files and a private directory")


def check_root(root: Path) -> None:
    if os.name == "nt":
        raise HostPolicyError("Installed host policy requires POSIX; Windows ACL support is not qualified")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise HostPolicyError("Host policy authority must not contain symlink components")
    _private_stat(root.stat(), directory=True)


def _unique_object(pairs: list) -> dict:
    data = {}
    for key, value in pairs:
        if key in data:
            raise HostPolicyError("Duplicate host policy state field")
        data[key] = value
    return data


def read_private_json(path: Path) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        _private_stat(path.lstat())
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            _private_stat(info)
            if info.st_size > _MAX_BYTES or (info.st_dev, info.st_ino) != (
                path.lstat().st_dev, path.lstat().st_ino
            ):
                raise HostPolicyError("Host policy file is oversized or changed during access")
            content = stream.read(_MAX_BYTES + 1)
            if len(content) > _MAX_BYTES:
                raise HostPolicyError("Host policy file exceeds the size limit")
            data = json.loads(content, object_pairs_hook=_unique_object)
    except (OSError, ValueError, RecursionError) as exc:
        raise HostPolicyError("Cannot read valid private host policy state") from exc
    if not isinstance(data, dict):
        raise HostPolicyError("Host policy state must be a JSON object")
    return data


def validate_policy(data: dict) -> dict:
    if set(data) - {"version", "revision", "admission", "approvals"}:
        raise HostPolicyError("Unsupported host policy fields")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise HostPolicyError("Unsupported host policy version")
    if not isinstance(data.get("revision"), str) or not data["revision"].strip():
        raise HostPolicyError("Host policy revision must be a nonempty string")
    admission = data.get("admission")
    if not isinstance(admission, dict) or set(admission) != {"mode", "max_workers"}:
        raise HostPolicyError("Host policy admission requires mode and max_workers")
    if not isinstance(admission["mode"], str) or admission["mode"] not in {"disabled", "observe", "enforce"}:
        raise HostPolicyError("Invalid host admission mode")
    if type(admission["max_workers"]) is not int or admission["max_workers"] < 1:
        raise HostPolicyError("Host admission max_workers must be a positive integer")
    from tools.approval_host_policy import validate_approvals_policy
    return {**data, "approvals": validate_approvals_policy(data.get("approvals", {}))}


def load_host_policy() -> dict | None:
    try:
        root = policy_root()
        try:
            root.lstat()
        except FileNotFoundError:
            return None
        check_root(root)
        path = root / "policy.json"
        try:
            path.lstat()
        except FileNotFoundError:
            try:
                (root / "ledger.json").lstat()
            except FileNotFoundError:
                return None
            else:
                raise HostPolicyError("Installed host admission ledger has no policy; restore the reviewed policy")
        return validate_policy(read_private_json(path))
    except (OSError, KeyError) as exc:
        raise HostPolicyError("Cannot access the host policy authority") from exc


def atomic_private_json(path: Path, data: dict) -> None:
    """Caller holds the persistent authority lock; never replace that lock file."""
    if path.is_symlink():
        raise HostPolicyError("Refusing a symlinked host state file")
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        # The rename, not just the contents, must survive an account/host crash.
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _after_fork() -> None:
    global _LOCK
    for fd in _OPEN_LOCKS:
        os.close(fd)
    _OPEN_LOCKS.clear()
    _LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


@contextlib.contextmanager
def policy_lock(*, create: bool = True):
    """Short bounded OS + thread transaction; no callbacks or worker waits inside."""
    from tools.browser_launcher_holders import _lock_nt, _lock_posix
    root = policy_root()
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    if not _LOCK.acquire(timeout=LOCK_TIMEOUT_SECONDS):
        raise HostPolicyError("Host admission transaction stayed locked")
    fd = None
    held = False
    try:
        check_root(root)
        path = root / ".admission.lock"
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.set_inheritable(fd, False)
        _OPEN_LOCKS.add(fd)
        _private_stat(os.fstat(fd))
        lock = _lock_nt if os.name == "nt" else lambda handle, _root, blocking: _lock_posix(handle, blocking)
        while not held:
            held = lock(fd, root, False)
            if not held:
                if time.monotonic() >= deadline:
                    raise HostPolicyError("Host admission authority stayed locked")
                time.sleep(0.01)
        opened, current = os.fstat(fd), path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise HostPolicyError("Host admission lock identity changed; repair the authority")
        yield
    except OSError as exc:
        raise HostPolicyError("Cannot lock the host admission authority") from exc
    finally:
        if fd is not None:
            if held and os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            _OPEN_LOCKS.discard(fd)
            os.close(fd)
        _LOCK.release()
