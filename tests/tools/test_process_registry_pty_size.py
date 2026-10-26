"""PTY background sessions adopt the real terminal's size (#7321).

Drives the real ``spawn_local(..., use_pty=True)`` with the PTY backend stubbed through
``sys.modules`` (both ``ptyprocess`` and ``winpty``), so the tests are host-independent.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture()
def registry():
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _no_systemd_scope(monkeypatch):
    """Keep a probe on a real systemd host out of the stubbed spawn."""
    import tools.process_registry as _pr

    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)


class TestPtyTerminalSize:
    """A PTY background session must adopt the real terminal's size.

    ``_spawn_local_pty`` spawned with a hardcoded ``dimensions=(30, 120)`` and never put
    COLUMNS/LINES in the child env, so every interactive tool run through
    ``terminal(background=true, use_pty=true)`` wrapped its output at 120 columns no matter
    how wide the user's terminal actually was.
    """

    @staticmethod
    def _spawn_pty(registry):
        """Spawn a PTY session against a stubbed PTY backend; return the spawn kwargs."""
        pty_proc = MagicMock()
        pty_proc.pid = 5555
        pty_module = MagicMock()
        pty_module.PtyProcess.spawn = MagicMock(return_value=pty_proc)
        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch.dict("sys.modules", {"ptyprocess": pty_module, "winpty": pty_module}), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("top", cwd="/tmp", use_pty=True)

        assert pty_module.PtyProcess.spawn.called, "PTY spawn should have been attempted"
        return pty_module.PtyProcess.spawn.call_args.kwargs

    def test_pty_adopts_the_real_terminal_size(self, registry, monkeypatch):
        """A 203x51 terminal gives a 203x51 PTY, and the child is told so."""
        monkeypatch.delenv("COLUMNS", raising=False)
        monkeypatch.delenv("LINES", raising=False)
        monkeypatch.setattr(os, "get_terminal_size", lambda *a: os.terminal_size((203, 51)))

        kwargs = self._spawn_pty(registry)

        assert kwargs["dimensions"] == (51, 203)  # ptyprocess takes (rows, cols)
        assert kwargs["env"]["COLUMNS"] == "203"
        assert kwargs["env"]["LINES"] == "51"

    def test_pty_falls_back_to_30x120_with_no_terminal(self, registry, monkeypatch):
        """Headless hosts (gateway, cron) keep the historical size, and env agrees with it."""
        monkeypatch.delenv("COLUMNS", raising=False)
        monkeypatch.delenv("LINES", raising=False)

        def _no_tty(*_args):
            raise OSError("not a terminal")

        monkeypatch.setattr(os, "get_terminal_size", _no_tty)

        kwargs = self._spawn_pty(registry)

        assert kwargs["dimensions"] == (30, 120)
        assert (kwargs["env"]["COLUMNS"], kwargs["env"]["LINES"]) == ("120", "30")

    def test_pty_defaults_pagers_to_cat_but_keeps_the_users(self, registry, monkeypatch):
        """A PTY is a real TTY, so git/man would page and hang on `q`: pagers default to cat."""
        monkeypatch.delenv("GIT_PAGER", raising=False)
        monkeypatch.setenv("PAGER", "less -R")

        env = self._spawn_pty(registry)["env"]

        assert (env["GIT_PAGER"], env["PAGER"]) == ("cat", "less -R")
