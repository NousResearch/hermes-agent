"""Regression tests for issue #31015 — Termux:API app detection.

`/voice on` was reporting "Termux:API Android app is not installed"
even on devices where the app *is* installed and the
`termux-microphone-record` binary works fine. The cause was a single
brittle probe (`pm list packages com.termux.api`) that returns a
false negative on some Android versions / Termux configurations.

These tests pin the new probe ladder:

  1. `pm list packages` confirms the app → True (back-compat).
  2. `pm` missing or non-zero → fall back to
     `cmd package list packages` (modern Android API 28+).
  3. Both probes inconclusive but `termux-microphone-record` on PATH
     → trust the binary, return True.
  4. At least one probe ran cleanly and definitively did not list
     the package → return False (the genuine "CLI without app"
     case the existing warning was written for).
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_run_dispatcher(behaviors):
    """Build a fake `subprocess.run` that returns / raises per-command.

    `behaviors` maps the first arg of each invocation to either a
    SimpleNamespace (returncode, stdout, stderr) or an Exception class
    instance to raise.
    """
    def _fake_run(cmd, **kwargs):
        key = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
        action = behaviors.get(key)
        if action is None:
            raise AssertionError(f"Unexpected probe command: {cmd!r}")
        if isinstance(action, BaseException):
            raise action
        return action
    return _fake_run


def _force_termux(monkeypatch):
    monkeypatch.setattr("tools.voice_mode._is_termux_environment", lambda: True)


# ── _termux_api_app_installed: probe ladder ───────────────────────────────


class TestTermuxApiAppInstalledProbeLadder:
    """The probe ladder is the heart of the #31015 fix — keep its truth
    table pinned so future "simplifications" can't silently regress it."""

    def test_returns_false_outside_termux(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._is_termux_environment", lambda: False)
        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is False

    def test_pm_confirms_package_returns_true(self, monkeypatch):
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(
                returncode=0,
                stdout="package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_pm_clean_miss_then_cmd_confirms(self, monkeypatch):
        """`pm` ran cleanly with no match → fall through to `cmd package`,
        which finds the app. Some devices return empty `pm` output for
        the calling user even when the package is installed."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(returncode=0, stdout="", stderr=""),
            "cmd": SimpleNamespace(
                returncode=0,
                stdout="package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_pm_missing_then_cmd_confirms(self, monkeypatch):
        """`pm` not on PATH → FileNotFoundError → fall through to `cmd`."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": FileNotFoundError("pm: command not found"),
            "cmd": SimpleNamespace(
                returncode=0,
                stdout="package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_pm_timeout_then_cmd_confirms(self, monkeypatch):
        """A hung `pm` (5s timeout) must not block detection."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": subprocess.TimeoutExpired(cmd="pm", timeout=5),
            "cmd": SimpleNamespace(
                returncode=0,
                stdout="package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_pm_nonzero_exit_then_cmd_confirms(self, monkeypatch):
        """Non-zero exit (e.g. permission denied for the calling user)
        is treated as inconclusive, not as a definitive "no"."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(returncode=1, stdout="", stderr="permission denied"),
            "cmd": SimpleNamespace(
                returncode=0,
                stdout="package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_both_probes_inconclusive_but_binary_present_returns_true(self, monkeypatch):
        """Core #31015 case: both probes fail or are unavailable, but the
        `termux-microphone-record` binary is on PATH. Trust the binary —
        a false positive only surfaces a precise runtime error, while a
        false negative blocks /voice on entirely (the user-reported
        symptom)."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": FileNotFoundError("pm: command not found"),
            "cmd": FileNotFoundError("cmd: command not found"),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)
        monkeypatch.setattr(
            "tools.voice_mode.shutil.which",
            lambda name: "/data/data/com.termux/files/usr/bin/termux-microphone-record"
            if name == "termux-microphone-record" else None,
        )

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True

    def test_both_probes_inconclusive_and_no_binary_returns_false(self, monkeypatch):
        """Without the binary on PATH there's nothing to trust — fall
        through to False so the user gets the install hint."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": FileNotFoundError("pm: command not found"),
            "cmd": FileNotFoundError("cmd: command not found"),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)
        monkeypatch.setattr("tools.voice_mode.shutil.which", lambda name: None)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is False

    def test_both_probes_clean_no_match_returns_false(self, monkeypatch):
        """Clean (returncode=0) probes that don't list the package are
        authoritative — the app is genuinely missing.  Don't promote
        this to True via the binary fallback."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(returncode=0, stdout="", stderr=""),
            "cmd": SimpleNamespace(returncode=0, stdout="", stderr=""),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)
        monkeypatch.setattr(
            "tools.voice_mode.shutil.which",
            lambda name: "/data/data/com.termux/files/usr/bin/termux-microphone-record"
            if name == "termux-microphone-record" else None,
        )

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is False

    def test_match_is_case_insensitive(self, monkeypatch):
        """Defensive against ROMs that capitalise the prefix differently."""
        _force_termux(monkeypatch)
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(
                returncode=0,
                stdout="Package:com.termux.api\n",
                stderr="",
            ),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import _termux_api_app_installed
        assert _termux_api_app_installed() is True


# ── End-to-end through detect_audio_environment ───────────────────────────


class TestDetectAudioEnvironmentTermuxFallback:
    """The point of the fix is that #31015 users with the binary on PATH
    no longer see the misleading 'Termux:API Android app is not installed'
    warning when the package-manager probe is inconclusive."""

    def test_inconclusive_probes_with_binary_does_not_emit_app_warning(
        self, monkeypatch
    ):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)

        # No sounddevice — we go down the Termux:API branch.
        monkeypatch.setattr(
            "tools.voice_mode._import_audio",
            lambda: (_ for _ in ()).throw(ImportError("no audio libs")),
        )
        monkeypatch.setattr(
            "tools.voice_mode._termux_microphone_command",
            lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record",
        )
        # Both probes fail (the #31015 reproduction).
        run = _make_run_dispatcher({
            "pm": FileNotFoundError("pm: command not found"),
            "cmd": FileNotFoundError("cmd: command not found"),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)
        monkeypatch.setattr(
            "tools.voice_mode.shutil.which",
            lambda name: "/data/data/com.termux/files/usr/bin/termux-microphone-record"
            if name == "termux-microphone-record" else None,
        )

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is True, (
            f"Voice mode should be available when the binary is on PATH "
            f"and probes are inconclusive (issue #31015). Got: {result}"
        )
        assert not any(
            "Termux:API Android app is not installed" in w
            for w in result["warnings"]
        ), (
            "The misleading 'app is not installed' warning must not fire "
            "when probes are inconclusive but the binary works (issue "
            f"#31015). warnings={result['warnings']!r}"
        )
        assert any(
            "Termux:API microphone recording available" in n
            for n in result.get("notices", [])
        )

    def test_clean_probes_no_match_still_blocks(self, monkeypatch):
        """The genuine "CLI installed without the app" case still blocks
        with the existing warning — important so users don't lose the
        install hint when the package manager *can* tell us the truth."""
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)

        monkeypatch.setattr(
            "tools.voice_mode._import_audio",
            lambda: (_ for _ in ()).throw(ImportError("no audio libs")),
        )
        monkeypatch.setattr(
            "tools.voice_mode._termux_microphone_command",
            lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record",
        )
        run = _make_run_dispatcher({
            "pm": SimpleNamespace(returncode=0, stdout="", stderr=""),
            "cmd": SimpleNamespace(returncode=0, stdout="", stderr=""),
        })
        monkeypatch.setattr("tools.voice_mode.subprocess.run", run)

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is False
        assert any(
            "Termux:API Android app is not installed" in w
            for w in result["warnings"]
        )
