"""The Windows holder lock (``tools/browser_launcher_holders.records_lock``) retries only lock contention, and only
for a bounded time: a lock leaked by a hung process fails the caller with a clear error instead of hanging it,
and any other ``msvcrt`` error surfaces at once."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import browser_launcher_holders as holders


@pytest.mark.parametrize("code", sorted({errno.EACCES, errno.EDEADLK, getattr(errno, "EDEADLOCK", errno.EDEADLK)}))
def test_contention_is_retried_until_the_deadline_then_fails_naming_the_dir(code, tmp_path):
    busy = OSError(code, "locked by another process")
    assert holders.nt_lock_failure(busy, tmp_path, expired=False) is None
    failure = holders.nt_lock_failure(busy, tmp_path, expired=True)
    assert isinstance(failure, TimeoutError) and str(tmp_path) in str(failure)


@pytest.mark.parametrize("code", [errno.EBADF, errno.EINVAL, errno.EIO])
def test_any_other_lock_error_is_raised_at_once(code, tmp_path):
    broken = OSError(code, "not a contention")
    assert holders.nt_lock_failure(broken, tmp_path, expired=False) is broken


@pytest.mark.platforms("windows")
def test_a_lock_held_by_a_hung_process_times_out(tmp_path, monkeypatch):
    holder = subprocess.Popen(
        [sys.executable, "-c", "import msvcrt, os, sys\n"
         "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
         "msvcrt.locking(fd, msvcrt.LK_LOCK, 1)\nprint('held', flush=True)\nsys.stdin.readline()\n",
         str(Path(tmp_path, holders._LOCK_NAME))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(holders, "_NT_LOCK_TIMEOUT_S", 0.0)
        with pytest.raises(TimeoutError), holders.records_lock(tmp_path):
            pass
    finally:
        holder.communicate("\n", timeout=30)
    assert os.path.exists(tmp_path / holders._LOCK_NAME)
