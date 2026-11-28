"""How a folder lock's acquisition is classified: a lock another holder has (contention) is waited for
(or, out of lock order, refused after the bounded wait), never skipped; a lock that cannot be used here
(any other OS error) is skipped and the write keeps the profile's own lock, as on main. Platform error
codes are injected so the msvcrt (Windows) behavior is exercised on any OS. Fixtures and helpers come
from test_skill_lock_owning_root.py."""

import errno
import io
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from tests.tools.test_skill_lock_owning_root import _folder, _lock, _locks_module, _make_fleet


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    return _make_fleet(tmp_path, monkeypatch)


class _FakeLocking:
    """A stand-in for the platform lock call that reports *code* for every attempt on *path* until
    released (or forever), and grants every other attempt."""

    def __init__(self, path, code, release_after=None):
        self.path, self.code, self.granted = str(path), code, []
        self.release_at = None if release_after is None else time.monotonic() + release_after

    def held(self):
        return self.release_at is None or time.monotonic() < self.release_at

    def attempt(self, name, unlock):
        if unlock:
            return
        if name == self.path and self.held():
            raise OSError(self.code, "injected")
        self.granted.append(name)


class _HeldRaw(io.RawIOBase):
    """The raw file under a BUFFERED open of the lock file: its writes fail with EACCES while the fake
    holder keeps the lock, as on Windows for a byte range another process locked. Buffering defers a
    write to flush or close, so this catches an initializing write that fails late (unit 6 round 3)."""

    def __init__(self, path, fake):
        self.fd, self._fake = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND), fake

    def readable(self):
        return True

    def writable(self):
        return True

    def seekable(self):
        return True

    def fileno(self):
        return self.fd

    def readinto(self, buf):
        data = os.read(self.fd, len(buf))
        buf[:len(data)] = data
        return len(data)

    def write(self, data):
        if self._fake.held():
            raise OSError(errno.EACCES, "injected: byte locked by another process")
        return os.write(self.fd, data)

    def seek(self, pos, whence=0):
        return os.lseek(self.fd, pos, whence)

    def close(self):
        if not self.closed:
            super().close()
            os.close(self.fd)


def _track(monkeypatch, fake, block_writes=False, fail_close=False):
    """Name every descriptor opened (raw os.open or builtin open) so the fake lock call knows which
    file it is locking. With *block_writes*, writing the held lock file fails with EACCES while the
    fake holds it, raw or buffered; with *fail_close*, closing it fails with EACCES too."""
    names, mine = {}, set()
    real_open, real_os_open, real_write, real_close = open, os.open, os.write, os.close

    def os_open(path, *a, **kw):
        fd = real_os_open(path, *a, **kw)
        names[fd] = str(path)
        if str(path) == fake.path:
            mine.add(fd)
        return fd

    def os_write(fd, data):
        if block_writes and fd in mine and fake.held():
            raise OSError(errno.EACCES, "injected: byte locked by another process")
        return real_write(fd, data)

    def os_close(fd):
        mine.discard(fd)
        real_close(fd)
        if fail_close and names.get(fd) == fake.path and fake.held():
            raise OSError(errno.EACCES, "injected: close failed")

    def tracking_open(path, mode="r", *a, **kw):
        if block_writes and str(path) == fake.path:
            fd = io.TextIOWrapper(io.BufferedRandom(_HeldRaw(path, fake)), encoding=kw.get("encoding"))
        else:
            fd = real_open(path, mode, *a, **kw)
        names[fd.fileno()] = str(path)
        return fd

    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(os, "write", os_write)
    monkeypatch.setattr(os, "close", os_close)
    monkeypatch.setattr("builtins.open", tracking_open)
    return names


def _inject_fcntl(monkeypatch, fake, **kw):
    """A self-contained fcntl (POSIX flock constants), so this also runs where fcntl is absent."""
    from tools import skill_usage as su
    names = _track(monkeypatch, fake, **kw)

    def flock(fd, op):
        fake.attempt(names.get(fd) if isinstance(fd, int) else fd.name, op == 8)

    monkeypatch.setattr(su, "msvcrt", None)
    monkeypatch.setattr(su, "fcntl", SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=flock))


def _inject_msvcrt(monkeypatch, fake, **kw):
    """msvcrt.locking: LK_LOCK gives up with EDEADLOCK-style errors after its own retries; LK_NBLCK
    reports EACCES; both are contention."""
    from tools import skill_usage as su
    names = _track(monkeypatch, fake, **kw)
    monkeypatch.setattr(su, "fcntl", None)
    monkeypatch.setattr(su, "msvcrt", SimpleNamespace(
        LK_LOCK=1, LK_NBLCK=2, LK_UNLCK=0,
        locking=lambda fileno, mode, n: fake.attempt(names.get(fileno), mode == 0)))


def _patch_shared(text, old="Step 1: Do the thing."):
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(action="patch", name="shared-skill", old_string=old, new_string=text))


@pytest.mark.parametrize("platform", ["fcntl", "msvcrt"])
@pytest.mark.parametrize("code", [errno.EACCES, errno.EAGAIN, errno.EDEADLK])
def test_a_held_folder_lock_is_waited_for_never_skipped(fleet, monkeypatch, platform, code):
    """Another holder has the shared skill's folder lock for 0.6 s: the write waits for it (round 3
    skipped it on Windows, where a held lock surfaces as a plain OSError, and wrote at once)."""
    _locks_module()
    lock = _folder(fleet, fleet["shared"] / "shared-skill")
    fake = _FakeLocking(lock, code, release_after=0.6)
    (_inject_fcntl if platform == "fcntl" else _inject_msvcrt)(monkeypatch, fake)
    started = time.monotonic()
    result = _patch_shared("Step 1: Waited.")
    assert result["success"] is True, result
    assert time.monotonic() - started >= 0.5, "the write did not wait for the held folder lock"
    assert str(lock) in fake.granted, "the folder lock was skipped instead of taken"


@pytest.mark.parametrize("platform", ["fcntl", "msvcrt"])
@pytest.mark.parametrize("code", [errno.ENOLCK, errno.EIO, errno.ETIMEDOUT])
def test_an_unusable_folder_lock_falls_back_to_the_own_lock(fleet, monkeypatch, platform, code):
    """Locking is unavailable for the folder lock (not held by anyone): the write proceeds at once
    under the profile's own lock, as on main, in order and out of order. ETIMEDOUT is an OSError
    Python raises as TimeoutError; it is not contention, so it must not be read as busy."""
    _locks_module()
    lock = _folder(fleet, fleet["shared"] / "shared-skill")
    fake = _FakeLocking(lock, code)
    (_inject_fcntl if platform == "fcntl" else _inject_msvcrt)(monkeypatch, fake)
    from tools.skill_manager_tool import _skill_mutation_lock
    box = {}

    def writes():
        box["top"] = _patch_shared("Step 1: Own lock only.")
        with _skill_mutation_lock("zz-local"):  # its own lock sorts after the folder lock: out of order
            box["nested"] = _patch_shared("Step 1: Nested, own lock only.", old="Step 1: Own lock only.")

    worker = threading.Thread(target=writes, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "an unusable lock was waited for as if another holder had it"
    assert box["top"]["success"] is True and box["nested"]["success"] is True, box
    assert str(lock) not in fake.granted and str(_lock(fleet["b"] / "skills", "shared-skill")) in fake.granted


@pytest.mark.parametrize("platform", ["fcntl", "msvcrt"])
@pytest.mark.parametrize("shape", ["single", "batch"])
def test_a_folder_lock_busy_out_of_order_refuses_a_nested_call_of_either_shape(fleet, monkeypatch, platform, shape):
    """A hook's nested skill_manage call (inside another skill's locks) needs the shared skill's
    folder lock out of lock order while another holder keeps it: after the bounded wait the call is
    refused as a result, single call or operations array (round 3 raised TimeoutError for the
    array), and nothing is written."""
    locks = _locks_module()
    monkeypatch.setattr(locks, "_OUT_OF_ORDER_WAIT", 0.3)
    lock = _folder(fleet, fleet["shared"] / "shared-skill")
    fake = _FakeLocking(lock, errno.EACCES)
    (_inject_fcntl if platform == "fcntl" else _inject_msvcrt)(monkeypatch, fake)
    from tools.skill_manager_tool import _skill_mutation_lock, skill_manage
    skill_md = fleet["shared"] / "shared-skill" / "SKILL.md"
    original = skill_md.read_text(encoding="utf-8")
    old, new = "Step 1: Do the thing.", "Step 1: Nested."
    box = {}

    def nested_write():
        with _skill_mutation_lock("zz-local"):
            try:
                if shape == "single":
                    box["result"] = json.loads(skill_manage(action="patch", name="shared-skill",
                                                            old_string=old, new_string=new))
                else:
                    box["result"] = json.loads(skill_manage(action="patch", name="", operations=[
                        {"action": "patch", "name": "shared-skill", "old_string": old, "new_string": new}]))
            except TimeoutError as exc:
                box["result"] = {"raised": repr(exc)}

    worker = threading.Thread(target=nested_write, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "the nested call never returned"
    assert box["result"].get("success") is False and "busy" in box["result"].get("error", ""), box["result"]
    assert skill_md.read_text(encoding="utf-8") == original


def test_a_first_use_racing_another_holders_initialisation_waits_never_skips(fleet, monkeypatch):
    """Windows, first use: the folder lock file is still empty while another process holds its first
    byte (it locked before its own initializing write landed). Writing that byte fails with EACCES;
    that is contention, so this write waits and then takes the lock: never skipped, never truncating."""
    _locks_module()
    lock = _folder(fleet, fleet["shared"] / "shared-skill")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")
    fake = _FakeLocking(lock, errno.EACCES, release_after=0.6)
    _inject_msvcrt(monkeypatch, fake, block_writes=True)
    started = time.monotonic()
    result = _patch_shared("Step 1: Waited for the first holder.")
    assert result["success"] is True, result
    assert time.monotonic() - started >= 0.5, "the write did not wait for the holder of the first byte"
    assert str(lock) in fake.granted, "the folder lock was skipped instead of taken"


@pytest.mark.parametrize("shape", ["single", "batch"])
@pytest.mark.parametrize("fail_close", [False, True])
def test_a_first_use_still_held_out_of_order_is_refused_whatever_cleanup_does(fleet, monkeypatch, shape,
                                                                              fail_close):
    """Windows, first use, out of lock order (a hook's nested call): the lock file is empty and another
    holder keeps its first byte past the bounded wait, so writing that byte keeps failing with EACCES.
    The call is refused before it writes, single call or operations array, even when closing the lock
    file fails too: the busy outcome is decided when the error code is read, and no later cleanup can
    turn it into "unusable, skip it" (unit 6 round 3 lost it to a buffered stream's close)."""
    locks = _locks_module()
    monkeypatch.setattr(locks, "_OUT_OF_ORDER_WAIT", 0.3)
    lock = _folder(fleet, fleet["shared"] / "shared-skill")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("", encoding="utf-8")
    fake = _FakeLocking(lock, errno.EACCES)
    _inject_msvcrt(monkeypatch, fake, block_writes=True, fail_close=fail_close)
    from tools.skill_manager_tool import _skill_mutation_lock, skill_manage
    skill_md = fleet["shared"] / "shared-skill" / "SKILL.md"
    original = skill_md.read_text(encoding="utf-8")
    old, new = "Step 1: Do the thing.", "Step 1: Nested."
    box = {}

    def nested_write():
        with _skill_mutation_lock("zz-local"):
            try:
                if shape == "single":
                    box["result"] = json.loads(skill_manage(action="patch", name="shared-skill",
                                                            old_string=old, new_string=new))
                else:
                    box["result"] = json.loads(skill_manage(action="patch", name="", operations=[
                        {"action": "patch", "name": "shared-skill", "old_string": old, "new_string": new}]))
            except OSError as exc:
                box["result"] = {"raised": repr(exc)}

    worker = threading.Thread(target=nested_write, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "the nested call never returned"
    assert box["result"].get("success") is False and "busy" in box["result"].get("error", ""), box["result"]
    assert skill_md.read_text(encoding="utf-8") == original
    assert str(lock) not in fake.granted
