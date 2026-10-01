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

Background on the companion Windows bug: the sandbox writes
``hermes_tools.py`` and ``script.py`` into a temp dir, and those files
must be written as UTF-8 on every platform — the generated stub contains
em-dash/en-dash characters in docstrings, and the default ``open(path, "w")``
on Windows uses the system locale (cp1252 typically), corrupting those
bytes.  The child then fails to import with a SyntaxError:
``'utf-8' codec can't decode byte 0x97``.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from tools.code_execution_env import (
    _SECRET_SUBSTRINGS,
    _WINDOWS_ESSENTIAL_ENV_VARS,
    _scrub_child_env,
)
from tools import code_execution_env


def _no_passthrough(_name):
    return False


class TestWindowsEssentialAllowlist:
    """The allowlist itself — contents, shape, and invariants."""



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


# ``windows_only`` rather than ``skipif(sys.platform != "win32")``: the
# dedicated Windows CI job selects its files by grepping for the marker, so a
# bare skipif is invisible to it — the file is never imported there and these
# tests run on no host at all.
@pytest.mark.windows_only
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


# ---------------------------------------------------------------------------
# POSIX scrubbing contract
# ---------------------------------------------------------------------------

class TestPosixEquivalence:
    """POSIX-mode scrubbing: safe prefixes and the HERMES_* operational
    allowlist pass; secret-looking names (incl. DSN/WEBHOOK) and every other
    HERMES_* var are dropped (#27303); Windows mode only ever adds essentials."""

    _POSIX_SYNTHETIC_ENV = {
        # Safe-prefix matches
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/alice",
        "USER": "alice",
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "en_US.UTF-8",
        "TERM": "xterm-256color",
        "SHELL": "/bin/zsh",
        "LOGNAME": "alice",
        "TMPDIR": "/tmp",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "XDG_CONFIG_HOME": "/home/alice/.config",
        "PYTHONPATH": "/opt/lib",
        "VIRTUAL_ENV": "/home/alice/.venv",
        "CONDA_PREFIX": "/opt/conda",
        # HERMES_* handling (#27303): only the operational allowlist passes;
        # every other HERMES_* is dropped (the broad prefix was removed).
        "HERMES_HOME": "/home/alice/.hermes",        # allowlisted → kept
        "HERMES_PROFILE": "default",                 # allowlisted → kept
        "HERMES_INTERACTIVE": "1",                   # not allowlisted → dropped
        "HERMES_BASE_URL": "https://api.internal",   # not allowlisted → dropped
        "HERMES_KANBAN_DB": "postgres://u:p@h/db",   # not allowlisted → dropped
        # Secret-substring blocks
        "OPENAI_API_KEY": "sk-xxx",
        "GITHUB_TOKEN": "ghp_xxx",
        "AWS_SECRET_ACCESS_KEY": "yyy",
        "MY_PASSWORD": "hunter2",
        "SENTRY_DSN": "https://abc@sentry.io/1",     # DSN substring → blocked
        "SLACK_WEBHOOK": "https://hooks.slack/x",    # WEBHOOK substring → blocked
        # Uncategorized — must be dropped
        "RANDOM_UNKNOWN": "drop-me",
        "DISPLAY": ":0",
        "SSH_AUTH_SOCK": "/run/user/1000/ssh-agent",
        # Passthrough candidate (also matches secret block by default)
        "TENOR_API_KEY": "tenor-xxx",
    }

    _WINDOWS_SYNTHETIC_ENV = {
        # Windows-essential names (must be dropped on POSIX, passed on Win)
        "SYSTEMROOT": r"C:\Windows",
        "SystemDrive": "C:",
        "WINDIR": r"C:\Windows",
        "ComSpec": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT",
        "USERPROFILE": r"C:\Users\alice",
        "APPDATA": r"C:\Users\alice\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\alice\AppData\Local",
        # Safe-prefix matches (cross-platform)
        "PATH": r"C:\Python311;C:\Windows\System32",
        "HOME": r"C:\Users\alice",
        "TEMP": r"C:\Users\alice\AppData\Local\Temp",
        # Secret-looking (always blocked)
        "OPENAI_API_KEY": "sk-xxx",
        "GITHUB_TOKEN": "ghp_xxx",
    }



    def test_posix_scrub_keeps_safe_vars_and_drops_secrets(self):
        scrubbed = _scrub_child_env(self._POSIX_SYNTHETIC_ENV,
                                    is_passthrough=_no_passthrough,
                                    is_windows=False)
        for kept in ("PATH", "HOME", "LANG", "LC_CTYPE", "XDG_RUNTIME_DIR",
                     "VIRTUAL_ENV", "HERMES_HOME", "HERMES_PROFILE"):
            assert scrubbed.get(kept) == self._POSIX_SYNTHETIC_ENV[kept], kept
        for dropped in ("OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                        "MY_PASSWORD", "SENTRY_DSN", "SLACK_WEBHOOK", "TENOR_API_KEY",
                        "HERMES_INTERACTIVE", "HERMES_BASE_URL", "HERMES_KANBAN_DB",
                        "RANDOM_UNKNOWN", "SSH_AUTH_SOCK"):
            assert dropped not in scrubbed, dropped

    def test_windows_mode_is_strict_superset_of_posix_mode(self):
        """Correctness check on the NEW behavior: is_windows=True must
        keep everything POSIX mode keeps, and *may* add Windows
        essentials.  It must never drop a var that POSIX mode would keep
        — if it did, we'd have broken same-host reuse of the scrubber."""
        env = {**self._POSIX_SYNTHETIC_ENV, **self._WINDOWS_SYNTHETIC_ENV}
        posix_result = _scrub_child_env(env,
                                        is_passthrough=lambda _: False,
                                        is_windows=False)
        windows_result = _scrub_child_env(env,
                                          is_passthrough=lambda _: False,
                                          is_windows=True)
        missing = set(posix_result) - set(windows_result)
        assert not missing, (
            f"is_windows=True dropped vars that is_windows=False kept: {missing}"
        )
        # And any extras must come from the Windows essentials allowlist.
        extras = set(windows_result) - set(posix_result)
        for k in extras:
            assert k.upper() in _WINDOWS_ESSENTIAL_ENV_VARS, (
                f"Unexpected extra var in windows-mode output: {k} "
                f"(not in _WINDOWS_ESSENTIAL_ENV_VARS)"
            )


# ---------------------------------------------------------------------------
# UTF-8 stdio regression test
# ---------------------------------------------------------------------------
#
# The third Windows-specific sandbox bug: after the UTF-8 file-write fix
# let the child import hermes_tools, a user script that printed non-ASCII
# to stdout still crashed with:
#
#     UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
#                         in position N: character maps to <undefined>
#
# Python's sys.stdout on Windows is bound to the console code page
# (cp1252 on US-locale installs) when the process is attached to a pipe
# without PYTHONIOENCODING set.  LLM-generated scripts routinely print
# em-dashes, arrows, accented chars, emoji — all of which break.
#
# Fix: spawn the child with PYTHONIOENCODING=utf-8 and PYTHONUTF8=1.
# The latter also makes open()'s default encoding UTF-8 (PEP 540),
# belt-and-suspenders for user scripts that do their own file I/O.


class TestChildStdioIsUtf8:
    """Verify the sandbox child is spawned with UTF-8 stdio encoding,
    so LLM scripts can print non-ASCII without crashing on Windows."""

    def test_live_child_can_print_non_ascii(self):
        """Live regression: spawn a Python child with the same env
        treatment the sandbox uses (PYTHONIOENCODING=utf-8 + PYTHONUTF8=1)
        and verify it can print em-dashes, arrows, and emoji to stdout
        without crashing.  This is the exact scenario that broke in live
        usage.

        Runs on every OS — on POSIX the fix is belt-and-suspenders but
        still load-bearing for C.ASCII locale environments.
        """
        script = textwrap.dedent("""
            import sys
            # Mix of chars that cp1252 can't encode: arrow, emoji.
            print("em-dash \\u2014 arrow \\u2192 emoji \\U0001f680")
            sys.exit(0)
        """).strip()

        # The production child-env builder must set UTF-8 stdio itself.
        scrubbed = _configured_timezone_child_env()
        assert scrubbed.get("PYTHONIOENCODING") == "utf-8"

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=scrubbed,
            capture_output=True,
            timeout=15,
            # Don't decode at the subprocess boundary — we want to check
            # the raw bytes match UTF-8, same as what the sandbox does.
        )
        assert result.returncode == 0, (
            f"Child crashed printing non-ASCII:\n"
            f"  stdout (raw): {result.stdout!r}\n"
            f"  stderr (raw): {result.stderr!r}"
        )
        decoded = result.stdout.decode("utf-8")
        assert "\u2014" in decoded, f"em-dash missing from output: {decoded!r}"
        assert "\u2192" in decoded, f"arrow missing from output: {decoded!r}"
        assert "\U0001f680" in decoded, f"emoji missing from output: {decoded!r}"


def _configured_timezone_child_env():
    return code_execution_env._build_child_env(
        rpc_endpoint="socket",
        rpc_token="token",
        tmpdir="/tmp/hermes-code-execution-test",
        child_python=sys.executable,
    )




@pytest.mark.windows_only
def test_windows_live_child_offset_matches_os_zone_when_timezone_is_configured(monkeypatch):
    """The user-visible contract of #112233: with ``timezone:`` configured, a real Windows child
    must report the OS zone's UTC offset — an IANA name in ``TZ`` made the MSVC runtime derive
    ``time.timezone == 0`` (+01:00 instead of -07:00) while ``time.tzname`` still read correctly."""
    import datetime
    import json

    monkeypatch.setattr("hermes_time.get_timezone_name", lambda: "America/Los_Angeles")
    child_env = _configured_timezone_child_env()
    result = subprocess.run(
        [sys.executable, "-c",
         "import json, time, datetime; print(json.dumps([time.timezone, "
         "datetime.datetime.now().astimezone().utcoffset().total_seconds()]))"],
        env=child_env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    child_timezone, child_offset = json.loads(result.stdout.strip())
    # The test process itself has no TZ override, so its view IS the OS zone.
    assert child_offset == datetime.datetime.now().astimezone().utcoffset().total_seconds()
    assert child_timezone == time.timezone
