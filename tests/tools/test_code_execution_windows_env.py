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

Also covers the companion Windows bug: the sandbox writes
``hermes_tools.py`` and ``script.py`` into a temp dir, and those files
must be written as UTF-8 on every platform — the generated stub contains
em-dash/en-dash characters in docstrings, and the default ``open(path, "w")``
on Windows uses the system locale (cp1252 typically), corrupting those
bytes.  The child then fails to import with a SyntaxError:
``'utf-8' codec can't decode byte 0x97``.
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


# ---------------------------------------------------------------------------
# POSIX equivalence guard
# ---------------------------------------------------------------------------

def _legacy_posix_scrubber(source_env, is_passthrough):
    """Verbatim copy of the pre-Windows-fix inline scrubbing logic.

    This is the oracle used by TestPosixEquivalence to prove the refactor
    did not change POSIX behavior.  DO NOT edit this to "match" a future
    production change — if _scrub_child_env's POSIX behavior legitimately
    needs to evolve, delete this function and adjust the equivalence test
    on purpose, so the churn is visible in review.
    """
    _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                          "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                          "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA",
                          "HERMES_")
    _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                          "PASSWD", "AUTH")
    out = {}
    for k, v in source_env.items():
        if is_passthrough(k):
            out[k] = v
            continue
        if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
            continue
        if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
            out[k] = v
    return out


class TestPosixEquivalence:
    """Lock in the invariant that _scrub_child_env(env, is_windows=False)
    behaves *bit-for-bit identically* to the pre-refactor inline scrubber.

    If this ever fails, it means somebody changed POSIX env-scrubbing
    behavior — maybe on purpose, maybe not.  Either way it should land
    as a deliberate, reviewed change (update _legacy_posix_scrubber
    above in the same PR).

    Rationale: the Windows-essentials patch refactored the scrubber into
    a helper.  Linux/macOS must not regress.  This class gates that.
    """

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
        "HERMES_HOME": "/home/alice/.hermes",
        "HERMES_INTERACTIVE": "1",
        # Secret-substring blocks
        "OPENAI_API_KEY": "sk-xxx",
        "GITHUB_TOKEN": "ghp_xxx",
        "AWS_SECRET_ACCESS_KEY": "yyy",
        "MY_PASSWORD": "hunter2",
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

    @pytest.mark.parametrize("env_name,env", [
        ("posix_synthetic", _POSIX_SYNTHETIC_ENV),
        ("windows_synthetic_on_posix", _WINDOWS_SYNTHETIC_ENV),
    ])
    @pytest.mark.parametrize("pt_name,pt", [
        ("no_passthrough", lambda _: False),
        ("tenor_passthrough", lambda k: k == "TENOR_API_KEY"),
        ("all_passthrough", lambda _: True),
    ])
    def test_posix_behavior_unchanged(self, env_name, env, pt_name, pt):
        """For every combination of (env shape × passthrough rule), the
        new helper with is_windows=False must produce the exact same dict
        as the legacy inline scrubber.

        We parametrize over three passthrough rules to cover the full
        surface: no passthrough, single-var passthrough (the common
        skill-registered case), and everything-passes (edge case that
        could expose precedence bugs)."""
        expected = _legacy_posix_scrubber(env, pt)
        actual = _scrub_child_env(env, is_passthrough=pt, is_windows=False)
        assert actual == expected, (
            f"POSIX behavior regressed for env={env_name}, passthrough={pt_name}\n"
            f"  only in legacy: {sorted(set(expected) - set(actual))}\n"
            f"  only in new:    {sorted(set(actual) - set(expected))}\n"
            f"  value diffs:    {[k for k in expected if k in actual and expected[k] != actual[k]]}"
        )

    def test_posix_behavior_unchanged_on_real_os_environ(self):
        """Bonus check against the actual os.environ of the host running
        the test.  This covers vars we might not have thought to put in
        the synthetic fixtures."""
        expected = _legacy_posix_scrubber(os.environ, lambda _: False)
        actual = _scrub_child_env(os.environ,
                                  is_passthrough=lambda _: False,
                                  is_windows=False)
        assert actual == expected, (
            "POSIX-mode scrubber diverged from legacy behavior on real "
            f"os.environ (host platform={sys.platform})"
        )

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
# UTF-8 file-write regression test
# ---------------------------------------------------------------------------
#
# The sandbox writes two Python files into a temp dir — the generated
# ``hermes_tools.py`` stub, and the LLM's ``script.py``.  Both contain
# non-ASCII characters in practice: the stub has em-dashes in docstrings
# ("``tcp://host:port`` — the parent falls back..."), and user scripts
# routinely contain non-ASCII strings, comments, or Unicode identifiers.
#
# On Windows, ``open(path, "w")`` without encoding= uses the system locale
# (cp1252 on US/UK installs), which cannot encode em-dashes.  Python then
# tries to decode the file as UTF-8 when importing it (PEP 3120), fails,
# and the sandbox aborts with:
#
#     SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x97
#                  in position N: invalid start byte
#
# This was the *second* Windows-specific bug (WinError 10106 was the first).
# The fix is to always pass ``encoding="utf-8"`` when writing Python source.


class TestSandboxWritesUtf8:
    """Verify the file-write call sites use UTF-8 explicitly, not the
    platform default.  We check the source of ``execute_code`` rather
    than spawning a real sandbox because the latter needs a full agent
    context — but the code inspection is deterministic and fast."""

    def test_stub_and_script_writes_specify_utf8(self):
        """Both ``hermes_tools.py`` and ``script.py`` writes in
        ``_execute_local`` must pass ``encoding="utf-8"``."""
        import tools.code_execution_tool as cet
        src = open(cet.__file__, encoding="utf-8").read()

        # There should be no ``open(path, "w")`` without encoding= for
        # the two staging files.  Grep-style check: find every write of
        # a .py file inside tmpdir and assert the line also contains
        # ``encoding="utf-8"`` within a short window.
        import re
        pattern = re.compile(
            r'open\(\s*os\.path\.join\(\s*tmpdir\s*,\s*"[^"]+\.py"\s*\)\s*,\s*"w"[^)]*\)'
        )
        for match in pattern.finditer(src):
            line = match.group(0)
            assert 'encoding="utf-8"' in line or "encoding='utf-8'" in line, (
                f"Sandbox file write missing encoding=\"utf-8\" on Windows: {line!r}"
            )

    def test_file_rpc_stub_uses_utf8(self):
        """The file-based RPC transport stub (used by remote backends)
        reads/writes JSON response files.  Those must also specify UTF-8
        so non-ASCII tool results survive the round-trip intact."""
        from tools.code_execution_tool import generate_hermes_tools_module
        stub = generate_hermes_tools_module(["terminal"], transport="file")
        # The generated stub should open response + request files as UTF-8.
        assert 'encoding="utf-8"' in stub, (
            "File-based RPC stub does not specify encoding=\"utf-8\" — "
            "will corrupt non-ASCII tool results on non-UTF-8 locales."
        )

    def test_stub_source_roundtrips_through_utf8(self):
        """Concrete regression: write the generated stub to a temp file
        using ``encoding="utf-8"``, then parse it.  This is what the
        sandbox does, and it must succeed even when the stub contains
        em-dashes (which it does — check the transport-header docstring).
        """
        from tools.code_execution_tool import generate_hermes_tools_module
        import tempfile, ast
        stub = generate_hermes_tools_module(
            ["terminal", "read_file", "write_file"], transport="uds"
        )
        # Sanity: stub actually contains a non-ASCII character, otherwise
        # this test wouldn't prove anything meaningful.
        non_ascii = [c for c in stub if ord(c) > 127]
        assert non_ascii, (
            "Generated stub is pure ASCII — test is meaningless.  If the "
            "stub's docstrings have lost their em-dashes, update this "
            "assertion, but be aware the original regression is no longer "
            "covered."
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(stub)
            tmp_path = f.name

        try:
            # Re-read and parse exactly like the child Python would.
            with open(tmp_path, encoding="utf-8") as fh:
                round_tripped = fh.read()
            assert round_tripped == stub, "UTF-8 round-trip corrupted the stub"
            ast.parse(round_tripped)  # must not raise SyntaxError
        finally:
            os.unlink(tmp_path)

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="cp1252 default-encoding regression is Windows-specific",
    )
    def test_windows_default_encoding_would_have_failed(self):
        """Negative control: prove that on Windows, writing the stub
        *without* ``encoding="utf-8"`` would corrupt the file.  If this
        test ever starts failing (i.e. default write succeeds), it means
        Python's default encoding has changed and the explicit UTF-8
        requirement may be obsolete — reconsider the fix."""
        from tools.code_execution_tool import generate_hermes_tools_module
        import tempfile

        stub = generate_hermes_tools_module(["terminal"], transport="uds")
        # Find a non-ASCII character we can use to prove the corruption.
        non_ascii = [c for c in stub if ord(c) > 127]
        if not non_ascii:
            pytest.skip("stub has no non-ASCII chars — nothing to corrupt")

        # Write with default encoding (simulating the old buggy code).
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            try:
                f.write(stub)
                tmp_path = f.name
                wrote_successfully = True
            except UnicodeEncodeError:
                # Default encoding can't even encode it — that's the bug
                # in a different form.  Still proves the point.
                tmp_path = f.name
                wrote_successfully = False

        try:
            if not wrote_successfully:
                # Default-encoding write raised outright.  The bug is real.
                return

            # Read back as UTF-8 (what Python does on import).
            with open(tmp_path, encoding="utf-8") as fh:
                try:
                    fh.read()
                    # If this succeeds on Windows, the platform default is
                    # already UTF-8 (e.g. Python 3.15 with UTF-8 mode on).
                    # In that case the explicit encoding= is belt-and-
                    # suspenders but no longer strictly required.  Skip.
                    pytest.skip(
                        "Default text-file encoding is UTF-8-compatible on "
                        "this Windows build — explicit encoding= is no "
                        "longer load-bearing, but keep it for belt-and-"
                        "suspenders."
                    )
                except UnicodeDecodeError:
                    # Exactly the failure mode that motivated the fix.
                    pass
        finally:
            os.unlink(tmp_path)
