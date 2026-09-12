"""A completed group must not turn successful native search into an OS error."""
import signal
from types import SimpleNamespace

import pytest

from tools.environments import local

pytestmark = pytest.mark.macos_only


@pytest.mark.parametrize(("returncode", "group_state", "allowed"), [
    (0, "gone", True),
    (None, "gone", False),
    (0, "live", False),
    (0, "denied", False),
])
def test_permission_error_requires_reaped_leader_and_absent_group(
    monkeypatch, returncode, group_state, allowed,
):
    import psutil
    from unittest.mock import Mock

    proc = SimpleNamespace(pid=12345, poll=lambda: returncode)
    descendants = [object()]
    monkeypatch.setattr(local.os, "getpgid", lambda pid: proc.pid)
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        children=lambda recursive: descendants))
    sweep = Mock()
    monkeypatch.setattr(local, "_sweep_escaped_descendants", sweep)

    def killpg(pgid, sig):
        assert pgid == proc.pid
        if sig == signal.SIGTERM or group_state == "denied":
            raise PermissionError("signal denied")
        assert sig == 0
        if group_state == "gone":
            raise ProcessLookupError("group gone")

    monkeypatch.setattr(local.os, "killpg", killpg)
    if allowed:
        local._kill_process_group_posix(proc)
        sweep.assert_called_once_with(descendants, proc.pid)
    else:
        with pytest.raises(PermissionError, match="signal denied"):
            local._kill_process_group_posix(proc)
        sweep.assert_not_called()
