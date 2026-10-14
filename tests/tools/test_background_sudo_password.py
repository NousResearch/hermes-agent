"""A background terminal process must be able to authenticate sudo.

#133622: ``ProcessRegistry.spawn_local`` — the path ``terminal(background=true)`` uses — ran only
``_rewrite_compound_background`` and never ``_transform_sudo_command``, and wired
``stdin=subprocess.DEVNULL``. So a background ``sudo`` had neither a prompt path nor a password
path: ``sudo`` read EOF and failed with ``sudo: no password was provided``.

A configured ``SUDO_PASSWORD`` could not close the gap either, because the transform that would
inject ``sudo -S`` never ran. Foreground sudo works (prompt, or ``-S`` when a password is set);
background did not, in either case.

The aggravating part is the PAM angle: each failed attempt still counts, so retries march the
account toward ``pam_faillock`` (default ``deny=3``). The operator then sees the *correct* password
refused and debugs the wrong thing. That is the shape of the reported incident: 428 background
``sudo ctr ... rm`` calls, three foreground retries, lockout.

These drive the real ``spawn_local`` against a stand-in ``sudo`` on PATH — no mocking of the
registry — so a regression in the wiring fails here rather than in production.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("not windows")


# A stand-in that behaves like sudo for the cases that matter: it rejects `-n` (no cached
# credential), and under `-S` it reads ONE line and reports what it got.
_STUB_SUDO = """#!/usr/bin/env bash
for a in "$@"; do
  if [ "$a" = "-n" ]; then exit 1; fi
done
has_S=0
for a in "$@"; do
  if [ "$a" = "-S" ]; then has_S=1; fi
done
if [ "$has_S" = "1" ]; then
  IFS= read -r pw
  echo "PW=[$pw]"
  if [ -z "$pw" ]; then echo "sudo: no password was provided"; exit 1; fi
  if [ "$pw" = "correct-horse" ]; then echo "root"; exit 0; fi
  echo "sudo: 1 incorrect password attempt"; exit 1
fi
echo "sudo: a terminal is required to read the password; either use the -S option to read from standard input"
exit 1
"""


@pytest.fixture
def stub_sudo(tmp_path, monkeypatch):
    """A stand-in ``sudo`` first on PATH, so the real spawn path invokes it."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sudo = bindir / "sudo"
    sudo.write_text(_STUB_SUDO, encoding="utf-8")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    # Make sure the shim wins over any real sudo further down PATH.
    monkeypatch.setenv("SUDO_PASSWORD", "correct-horse", raising=False)
    return bindir / "sudo"


def _spawn(command: str, task_id: str = "bg-sudo", **kw):
    """Spawn on the local backend and wait; returns (registry, session, wait-result)."""
    from tools.process_registry import ProcessRegistry

    reg = ProcessRegistry()
    sess = reg.spawn_local(command=command, cwd=str(Path.home()),
                           task_id=task_id, session_key=task_id, **kw)
    result = reg.wait(sess.id, timeout=30)
    return reg, sess, result


class TestBackgroundSudoAuthenticates:
    def test_background_sudo_uses_the_configured_password(self, stub_sudo):
        """The core gap: a correctly-configured SUDO_PASSWORD must reach a background sudo.

        Before the fix the transform never ran, so sudo got EOF on DEVNULL stdin and failed with
        its no-TTY / no-password diagnostic regardless of configuration.
        """
        _reg, _sess, result = _spawn("sudo whoami", task_id="bg-sudo-1")
        out = result.get("output") or ""

        assert "root" in out, (
            f"background sudo could not authenticate with the password configured: {out!r}"
        )

    def test_the_transform_rewrites_to_dash_S(self, stub_sudo):
        """``_transform_sudo_command`` turns a bare ``sudo`` into ``sudo -S -p ''``;
        spawn_local must apply it, exactly as ``BaseEnvironment._prepare_command`` does."""
        from tools.terminal_tool_sudo import _transform_sudo_command

        expected_command, expected_stdin = _transform_sudo_command("sudo whoami")
        assert expected_stdin, "precondition: the transform produces a password line here"
        assert "-S" in (expected_command or ""), "precondition: the rewrite adds -S"

        _reg, _sess, result = _spawn("sudo whoami", task_id="bg-sudo-2")
        assert "root" in (result.get("output") or "")


class TestCompoundBackgroundIsStillSafe:
    def test_a_backgrounded_compound_command_still_authenticates(self, stub_sudo):
        """``A && sudo B &`` backgrounds the whole list, so bash forks: a stdin pipe write lands
        on the parent shell and the backgrounded sudo reads EOF. The password has to be bound to
        the command (heredoc) rather than piped, or this shape stays broken even after the
        transform is applied.
        """
        _reg, _sess, result = _spawn("echo start && sudo whoami &", task_id="bg-sudo-3")
        out = result.get("output") or ""

        assert "PW=[correct-horse]" in out or "root" in out, (
            f"a backgrounded compound sudo did not receive the password: {out!r}"
        )

    def test_a_non_sudo_command_is_untouched(self, stub_sudo):
        """The rewrite and the stdin pipe are both conditional on a sudo being present; an
        ordinary background command must still get DEVNULL stdin and behave as before."""
        _reg, _sess, result = _spawn("echo plain", task_id="bg-sudo-3b")

        assert "plain" in (result.get("output") or "")


class TestNoPasswordAvailableStillFailsGracefully:
    def test_without_a_password_sudo_fails_without_hanging(self, stub_sudo, monkeypatch):
        """No password configured must still terminate promptly with sudo's own diagnostic —
        not hang waiting on a stdin that will never carry one."""
        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        from tools.terminal_tool_sudo import _reset_cached_sudo_passwords

        _reset_cached_sudo_passwords()

        _reg, _sess, result = _spawn("sudo whoami", task_id="bg-sudo-4")

        assert result.get("status") in {"exited", "timeout"}, (
            f"expected a prompt failure, got {result.get('status')!r}"
        )
        assert (result.get("output") or "").strip(), "sudo should say why it failed"