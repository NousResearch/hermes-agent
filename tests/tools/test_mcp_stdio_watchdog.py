import unittest.mock as mock

import psutil

from tools import mcp_stdio_watchdog


def test_is_orphaned_true_when_ppid_changes():
    # The original parent went away and the watchdog was reparented.
    assert mcp_stdio_watchdog._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_starttime_mismatch():
    # Same ppid but a different starttime means the PID was reused by an
    # unrelated process. On Linux the /proc starttime won't be 0.0, so this
    # exercises the mismatch branch directly.
    me = psutil.Process()
    assert mcp_stdio_watchdog._is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    # Snapshot the fingerprint with the same helper the implementation compares
    # against, so units match (ticks on Linux, epoch via psutil otherwise).
    me = psutil.Process()
    starttime = mcp_stdio_watchdog._read_proc_starttime(me.pid)
    if starttime is not None:
        snapshot = float(starttime)
    else:
        snapshot = me.create_time()
    assert mcp_stdio_watchdog._is_orphaned(me.pid, snapshot, getppid=lambda: me.pid) is False


def test_read_proc_starttime_handles_comm_with_spaces_and_parens():
    # /proc/<pid>/stat field 2 (comm) can contain spaces and parens. A naive
    # split() over the whole record corrupts every field index after it; the
    # helper must split on the last ')'.
    me = psutil.Process()
    fake_pid = me.pid
    real_stat = open(f'/proc/{fake_pid}/stat').read()
    real_tail = real_stat.rsplit(')', 1)[1]
    fake_stat = f'{fake_pid} (my proc (with) parens){real_tail}'

    m = mock.mock_open(read_data=fake_stat)
    with mock.patch('builtins.open', m):
        result = mcp_stdio_watchdog._read_proc_starttime(fake_pid)

    expected = int(real_tail.split()[19])
    assert result == expected


def test_is_orphaned_fallback_preserves_pid_reuse_guard(monkeypatch):
    # On non-Linux (or when /proc is unavailable) the fallback must compare the
    # psutil create_time() fingerprint rather than degrading to bare pid_exists().
    me = psutil.Process()
    real_create_time = me.create_time()

    monkeypatch.setattr(mcp_stdio_watchdog, '_read_proc_starttime', lambda pid: None)

    assert (
        mcp_stdio_watchdog._is_orphaned(
            me.pid, real_create_time + 9999, getppid=lambda: me.pid
        )
        is True
    )
    assert (
        mcp_stdio_watchdog._is_orphaned(
            me.pid, real_create_time, getppid=lambda: me.pid
        )
        is False
    )
