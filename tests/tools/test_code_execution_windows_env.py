"""Tests for execute_code env scrubbing on Windows.

On Windows the child process needs a small set of OS-essential env vars
(SYSTEMROOT, WINDIR, COMSPEC, ...) to run.  Without SYSTEMROOT in particular,
``socket.socket(AF_INET, SOCK_STREAM)`` fails inside the sandbox with
WinError 10106 (Winsock can't locate mswsock.dll) and no tool call over
loopback TCP can ever succeed.

These tests cover ``_scrub_child_env`` directly so they run on every OS
— the logic is conditional on a passed-in ``is_windows`` flag, not on
the host platform.  We also keep a live Winsock smoke test that only runs
on a real Windows host.
"""

import os
import socket
import subprocess
import sys
import textwrap
import unittest.mock as mock

import pytest

from tools.code_execution_tool import (
    _SAFE_ENV_PREFIXES,
    _SECRET_SUBSTRINGS,
    _WINDOWS_ESSENTIAL_ENV_VARS,
    _scrub_child_env,
)


def _no_passthrough(_name):
    return False


class TestWindowsEssentialAllowlist:
    """The allowlist itself — contents, shape, and invariants."""

    def test_contains_winsock_required_vars(self):
        # Without SYSTEMROOT the child cannot initialize Winsock.
        assert "SYSTEMROOT" in _WINDOWS_ESSENTIAL_ENV_VARS

    def test_contains_subprocess_required_vars(self):
        # Without COMSPEC, subprocess can't resolve the default shell.
        assert "COMSPEC" in _WINDOWS_ESSENTIAL_ENV_VARS

    def test_contains_user_profile_vars(self):
        # os.path.expanduser("~") on Windows uses USERPROFILE.
        assert "USERPROFILE" in _WINDOWS_ESSENTIAL_ENV_VARS
        assert "APPDATA" in _WINDOWS_ESSENTIAL_ENV_VARS
        assert "LOCALAPPDATA" in _WINDOWS_ESSENTIAL_ENV_VARS

    def test_contains_only_uppercase_names(self):
        # Windows env var names are case-insensitive but we canonicalize to
        # uppercase for the membership check (``k.upper() in _WINDOWS_...``).
        for name in _WINDOWS_ESSENTIAL_ENV_VARS:
            assert name == name.upper(), f"{name!r} should be uppercase"

    def test_no_overlap_with_secret_substrings(self):
        # Sanity: none of the essential OS vars should look like secrets.
        # If this ever fires, we'd have a precedence ordering bug (secrets
        # are blocked *before* the essentials check).
        for name in _WINDOWS_ESSENTIAL_ENV_VARS:
            assert not any(s in name for s in _SECRET_SUBSTRINGS), (
                f"{name!r} looks secret-like — would be blocked before the "
                "essentials allowlist can match"
            )


class TestScrubChildEnvWindows:
    """Verify _scrub_child_env passes Windows essentials through when
    is_windows=True and blocks them when is_windows=False (so POSIX hosts
    don't inherit pointless Windows vars)."""

    def _sample_windows_env(self):
        """A realistic subset of what os.environ looks like on Windows."""
        return {
            "SYSTEMROOT": r"C:\Windows",
            "SystemDrive": "C:",        # Windows preserves native case
            "WINDIR": r"C:\Windows",
            "ComSpec": r"C:\Windows\System32\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD;.PY",
            "USERPROFILE": r"C:\Users\alice",
            "APPDATA": r"C:\Users\alice\AppData\Roaming",
            "LOCALAPPDATA": r"C:\Users\alice\AppData\Local",
            "PATH": r"C:\Windows\System32;C:\Python311",
            "HOME": r"C:\Users\alice",
            "TEMP": r"C:\Users\alice\AppData\Local\Temp",
            # Should still be blocked:
            "OPENAI_API_KEY": "sk-secret",
            "GITHUB_TOKEN": "ghp_secret",
            "MY_PASSWORD": "hunter2",
            # Not matched by any rule — should be dropped on both OSes:
            "RANDOM_UNKNOWN_VAR": "value",
        }

    def test_windows_essentials_passed_through_when_is_windows_true(self):
        env = self._sample_windows_env()
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=_no_passthrough,
                                    is_windows=True)

        # Every essential var from the sample env should survive.
        assert scrubbed["SYSTEMROOT"] == r"C:\Windows"
        assert scrubbed["SystemDrive"] == "C:"  # case preserved
        assert scrubbed["WINDIR"] == r"C:\Windows"
        assert scrubbed["ComSpec"] == r"C:\Windows\System32\cmd.exe"
        assert scrubbed["PATHEXT"] == ".COM;.EXE;.BAT;.CMD;.PY"
        assert scrubbed["USERPROFILE"] == r"C:\Users\alice"
        assert scrubbed["APPDATA"].endswith("Roaming")
        assert scrubbed["LOCALAPPDATA"].endswith("Local")

        # Safe-prefix vars still pass (baseline behavior).
        assert "PATH" in scrubbed
        assert "HOME" in scrubbed
        assert "TEMP" in scrubbed

    def test_secrets_still_blocked_on_windows(self):
        """The Windows allowlist must NOT defeat the secret-substring block.

        This is the key security invariant: essentials are allowed by
        *exact name*, and the secret-substring block runs before the
        essentials check anyway, so a variable named e.g. ``API_KEY`` can
        never sneak through just because we added Windows support.
        """
        env = self._sample_windows_env()
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=_no_passthrough,
                                    is_windows=True)
        assert "OPENAI_API_KEY" not in scrubbed
        assert "GITHUB_TOKEN" not in scrubbed
        assert "MY_PASSWORD" not in scrubbed

    def test_unknown_vars_still_dropped_on_windows(self):
        env = self._sample_windows_env()
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=_no_passthrough,
                                    is_windows=True)
        assert "RANDOM_UNKNOWN_VAR" not in scrubbed

    def test_essentials_blocked_when_is_windows_false(self):
        """On POSIX hosts, Windows-specific vars should not pass — they
        have no meaning and could confuse child tooling."""
        env = self._sample_windows_env()
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=_no_passthrough,
                                    is_windows=False)
        # Safe prefixes still match (PATH, HOME, TEMP).
        assert "PATH" in scrubbed
        assert "HOME" in scrubbed
        assert "TEMP" in scrubbed
        # But Windows OS vars should be dropped.
        assert "SYSTEMROOT" not in scrubbed
        assert "WINDIR" not in scrubbed
        assert "ComSpec" not in scrubbed
        assert "APPDATA" not in scrubbed

    def test_case_insensitive_essential_match(self):
        """Windows env var names are case-insensitive at the OS level but
        Python preserves whatever case os.environ reported.  The scrubber
        must normalize to uppercase for the membership check."""
        env = {
            "SystemRoot": r"C:\Windows",       # mixed case
            "comspec": r"C:\Windows\System32\cmd.exe",  # lowercase
            "APPDATA": r"C:\Users\x\AppData\Roaming",   # uppercase
        }
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=_no_passthrough,
                                    is_windows=True)
        assert "SystemRoot" in scrubbed
        assert "comspec" in scrubbed
        assert "APPDATA" in scrubbed


class TestScrubChildEnvPassthroughInteraction:
    """The passthrough hook runs *before* the secret block, so a skill
    can legitimately forward a third-party API key.  The Windows
    essentials addition must not interfere with that."""

    def test_passthrough_wins_over_secret_block(self):
        env = {"TENOR_API_KEY": "x", "PATH": "/bin"}
        scrubbed = _scrub_child_env(env,
                                    is_passthrough=lambda k: k == "TENOR_API_KEY",
                                    is_windows=False)
        assert scrubbed.get("TENOR_API_KEY") == "x"
        assert scrubbed.get("PATH") == "/bin"

    def test_passthrough_still_works_on_windows(self):
        env = {
            "TENOR_API_KEY": "x",
            "SYSTEMROOT": r"C:\Windows",
            "OPENAI_API_KEY": "sk-secret",  # not passthrough
        }
        scrubbed = _scrub_child_env(
            env,
            is_passthrough=lambda k: k == "TENOR_API_KEY",
            is_windows=True,
        )
        assert scrubbed.get("TENOR_API_KEY") == "x"
        assert scrubbed.get("SYSTEMROOT") == r"C:\Windows"
        assert "OPENAI_API_KEY" not in scrubbed


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Winsock-specific regression — only meaningful on Windows",
)
class TestWindowsSocketSmokeTest:
    """Integration-ish smoke test: spawn a child Python with a scrubbed
    env and confirm it can create an AF_INET socket.  This is the
    regression that motivated the fix — without SYSTEMROOT the child
    hits WinError 10106 before any RPC is attempted."""

    def test_child_can_create_socket_with_scrubbed_env(self):
        scrubbed = _scrub_child_env(os.environ, is_passthrough=_no_passthrough)

        # Build a tiny child script that simply opens an AF_INET socket.
        script = textwrap.dedent("""
            import socket, sys
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.close()
                print("OK")
                sys.exit(0)
            except OSError as exc:
                print(f"FAIL: {exc}")
                sys.exit(1)
        """).strip()

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=scrubbed,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Child failed to create socket with scrubbed env:\n"
            f"  stdout={result.stdout!r}\n"
            f"  stderr={result.stderr!r}\n"
            f"  scrubbed keys={sorted(scrubbed.keys())}"
        )
        assert "OK" in result.stdout
