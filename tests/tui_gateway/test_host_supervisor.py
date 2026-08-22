from unittest.mock import call, patch

import pytest

from tui_gateway import host_supervisor as hs


@pytest.mark.parametrize("pid_exists", [True, False])
def test_pid_alive_delegates_without_signaling(pid_exists):
    with patch("gateway.status._pid_exists", return_value=pid_exists) as exists, \
         patch.object(hs.os, "kill") as kill:
        assert hs._pid_alive(1234) is pid_exists

    exists.assert_called_once_with(1234)
    kill.assert_not_called()


def test_pid_alive_nonpositive_short_circuits_without_delegating():
    with patch("gateway.status._pid_exists", return_value=True) as exists, \
         patch.object(hs.os, "kill") as kill:
        assert hs._pid_alive(0) is False
        assert hs._pid_alive(-1) is False
        exists.assert_not_called()
        kill.assert_not_called()

        # Positive control keeps this test red against the legacy implementation
        # instead of only pinning behavior that already existed.
        assert hs._pid_alive(1234) is True

    exists.assert_called_once_with(1234)
    kill.assert_not_called()


def test_pid_alive_assumes_alive_when_canonical_helper_raises():
    with patch(
        "gateway.status._pid_exists",
        side_effect=RuntimeError("process status unavailable"),
    ) as exists, patch.object(hs.os, "kill") as kill:
        assert hs._pid_alive(1234) is True

    exists.assert_called_once_with(1234)
    kill.assert_not_called()


def test_terminate_pid_falls_back_to_sigterm_without_sigkill(monkeypatch):
    monkeypatch.delattr(hs.signal, "SIGKILL", raising=False)
    supervisor = hs.HostSupervisor(
        autostart=False,
        expected_build_sha="test",
        expected_hermes_home="test",
    )

    with patch.object(hs.os, "kill") as kill:
        supervisor._terminate_pid(1234, timeout=0)

    assert kill.call_args_list == [
        call(1234, hs.signal.SIGTERM),
        call(1234, hs.signal.SIGTERM),
    ]
