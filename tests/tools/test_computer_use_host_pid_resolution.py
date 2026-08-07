"""Regression tests for verified window-to-process PID resolution.

``_resolve_host_pid`` must never return a PID derived from process-name
matching alone: the first process whose cmdline merely *contains*
``app_name`` can belong to an unrelated window (maintainer review on
#73007).  The only acceptable source is the window's own ``_NET_WM_PID``
read through the X server, cross-checked against ``/proc``.
"""

from __future__ import annotations

from unittest.mock import patch

from tools.computer_use.cua_backend import (
    _proc_cmdline_matches,
    _resolve_host_pid,
    _x11_window_pid,
)


# ---------------------------------------------------------------------------
# _x11_window_pid: reads _NET_WM_PID through the X server
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout


def test_x11_window_pid_parses_xprop_output():
    with patch("tools.computer_use.cua_backend.shutil.which", return_value="/usr/bin/xprop"), patch(
        "tools.computer_use.cua_backend.subprocess.run",
        return_value=_FakeResult(0, "  _NET_WM_PID(CARDINAL) = 4321\n"),
    ) as run:
        assert _x11_window_pid(77) == 4321

    assert run.call_args.args[0] == ["xprop", "-id", "77", "_NET_WM_PID"]


def test_x11_window_pid_falls_back_to_xdotool():
    def fake_run(cmd, *a, **k):
        if cmd[0] == "xprop":
            return _FakeResult(1, "")
        return _FakeResult(0, "4321\n")

    with patch("tools.computer_use.cua_backend.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), patch(
        "tools.computer_use.cua_backend.subprocess.run", side_effect=fake_run
    ):
        assert _x11_window_pid(77) == 4321


def test_x11_window_pid_none_when_no_reader_available():
    with patch("tools.computer_use.cua_backend.shutil.which", return_value=None):
        assert _x11_window_pid(77) is None


def test_x11_window_pid_none_when_window_has_no_net_wm_pid():
    with patch("tools.computer_use.cua_backend.shutil.which", return_value="/usr/bin/xprop"), patch(
        "tools.computer_use.cua_backend.subprocess.run",
        return_value=_FakeResult(0, "  _NET_WM_PID(CARDINAL) = 0\n"),
    ):
        assert _x11_window_pid(77) is None


# ---------------------------------------------------------------------------
# _proc_cmdline_matches: /proc cross-check
# ---------------------------------------------------------------------------

def test_proc_cmdline_matches_true_on_substring():
    with patch("builtins.open") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"/usr/lib/zen/bin/zen\x00--type=browser\x00"
        assert _proc_cmdline_matches(1234, "zen") is True


def test_proc_cmdline_matches_false_when_missing():
    with patch("builtins.open") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"/usr/bin/other-app\x00"
        assert _proc_cmdline_matches(1234, "zen") is False


def test_proc_cmdline_matches_false_on_oserror():
    with patch("builtins.open", side_effect=OSError("no such process")):
        assert _proc_cmdline_matches(999999, "zen") is False


# ---------------------------------------------------------------------------
# _resolve_host_pid: verified mapping only — never a cmdline guess
# ---------------------------------------------------------------------------

def test_resolve_returns_pid_when_net_wm_pid_is_crosschecked():
    with patch("tools.computer_use.cua_backend._x11_window_pid", return_value=4321), patch(
        "tools.computer_use.cua_backend._proc_cmdline_matches", return_value=True
    ):
        assert _resolve_host_pid(77, "zen") == 4321


def test_resolve_returns_none_when_net_wm_pid_does_not_match_cmdline():
    # The core regression: window 77 belongs to process A, but the first
    # /proc cmdline hit for "zen" is process B.  Previously we returned B.
    with patch("tools.computer_use.cua_backend._x11_window_pid", return_value=4321), patch(
        "tools.computer_use.cua_backend._proc_cmdline_matches", return_value=False
    ):
        assert _resolve_host_pid(77, "zen") is None


def test_resolve_returns_none_when_no_net_wm_pid():
    with patch("tools.computer_use.cua_backend._x11_window_pid", return_value=None):
        assert _resolve_host_pid(77, "zen") is None


def test_resolve_guards_bad_inputs():
    assert _resolve_host_pid(0, "zen") is None
    assert _resolve_host_pid(-3, "zen") is None
    assert _resolve_host_pid(77, "") is None
