"""Tests for the Windows / Git Bash MSYS-path normalization in
``LocalEnvironment``.

Background
----------
On Windows, ``pwd -P`` inside Git Bash emits paths like
``/c/Users/NVIDIA``. ``subprocess.Popen(..., cwd=...)`` only accepts
native Windows paths (``C:\\Users\\NVIDIA``), and the validation done
by ``_resolve_safe_cwd`` was also checking the MSYS form against
``os.path.isdir``, which returns ``False`` on Windows. The combined
effect was a warning logged on every single terminal call:

    LocalEnvironment cwd '/c/Users/NVIDIA' is missing on disk;
    falling back to '/' so terminal commands keep working.

Platform gating
---------------
These tests used to fake Windows on Linux CI by patching
``local_mod._IS_WINDOWS`` (and sometimes ``os.path.isdir``) so an MSYS
path tested as "missing" exactly like on the real OS. That inverted the
thing under test: the bug was that ``os.path.isdir("/c/Users/x")`` is
False *on Windows*, and the fake had to recreate that condition by hand
on a host where the path semantics, the drive letters, the path
separator, and Git Bash itself are all absent.

So the Windows-behaviour tests are ``platforms("windows")`` and run on the
Windows CI job against a real Git Bash layout. The "no-op off Windows"
cases assert genuine POSIX behaviour and are ``platforms("linux")`` — on that
host ``_IS_WINDOWS`` is already False, so no patching is needed at all.
"""

import os
import subprocess
import time
from unittest.mock import patch

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments import local as local_mod
from tools.environments.local import (
    LocalEnvironment,
    _bash_safe_path,
    _git_bash_bin_dirs,
    _make_run_env,
    _msys_to_windows_path,
    _quote_bash_path,
    _resolve_safe_cwd,
    _sanitize_subprocess_env,
    _windows_to_msys_path,
    hermes_subprocess_env,
)


# ---------------------------------------------------------------------------
# _msys_to_windows_path — pure-function unit tests
# ---------------------------------------------------------------------------

class TestMsysToWindowsPath:
    @pytest.mark.platforms("linux")
    def test_noop_on_non_windows(self):
        # On a non-Windows host the function must never rewrite the path
        # — POSIX-style paths are real paths there.
        assert _msys_to_windows_path("/c/Users/NVIDIA") == "/c/Users/NVIDIA"
        assert _msys_to_windows_path("/home/teknium") == "/home/teknium"

    @pytest.mark.platforms("windows")
    def test_translates_drive_path(self):
        assert _msys_to_windows_path("/c/Users/NVIDIA") == r"C:\Users\NVIDIA"
        assert _msys_to_windows_path("/d/Projects/foo bar") == r"D:\Projects\foo bar"

    @pytest.mark.platforms("windows")
    def test_empty_string(self):
        assert _msys_to_windows_path("") == ""


# ---------------------------------------------------------------------------
# _windows_to_msys_path — reverse translation for bash builtin cd
# ---------------------------------------------------------------------------

class TestWindowsToMsysPath:
    @pytest.mark.platforms("linux")
    def test_noop_on_non_windows(self):
        assert _windows_to_msys_path(r"C:\Users\NVIDIA") == r"C:\Users\NVIDIA"

    @pytest.mark.platforms("windows")
    def test_does_not_translate_non_drive_path(self):
        assert _windows_to_msys_path("/tmp/foo") == "/tmp/foo"
        assert _windows_to_msys_path(r"\\server\share") == r"\\server\share"


# ---------------------------------------------------------------------------
# _bash_safe_path / _quote_bash_path — shell-script interpolation
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestBashSafePath:
    def test_native_windows_path_becomes_msys(self):
        assert _bash_safe_path(r"C:\Users\alice\notes.txt") == "/c/Users/alice/notes.txt"

    def test_quote_bash_path_quotes_mixed_windows_path(self):
        quoted = _quote_bash_path(
            r"C:\Users\Alexander\AppData\Local\Temp\hermes-snap-abc.sh"
        )
        assert "/c/Users/Alexander/AppData/Local/Temp/hermes-snap-abc.sh" in quoted
        assert "\\" not in quoted


# ---------------------------------------------------------------------------
# _resolve_safe_cwd — Windows fast path
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestResolveSafeCwdWindows:
    def test_msys_path_resolves_to_native_when_native_exists(self, tmp_path):
        """The whole point of this fix: a Git Bash ``/c/Users/x`` value
        should resolve to its native equivalent if that native dir exists,
        WITHOUT falling back to the temp dir.

        ``tmp_path`` is a real native directory on the Windows runner, so
        its MSYS spelling is a genuine round-trip rather than a stubbed
        translation.
        """
        native = str(tmp_path)
        msys = _windows_to_msys_path(native)
        assert msys != native, "expected a drive-letter path to translate"
        assert _resolve_safe_cwd(msys) == native


# ---------------------------------------------------------------------------
# End-to-end: _update_cwd via stdout marker
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestUpdateCwdWindowsMsys:
    def test_marker_output_msys_path_stored_in_native_form(self, tmp_path):
        """When Git Bash emits ``/c/Users/x`` in the cwd marker on Windows,
        ``_update_cwd`` must translate to native form before
        validating and storing — otherwise ``os.path.isdir`` rejects a
        perfectly real directory."""
        original = tmp_path / "starting"
        original.mkdir()

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        new_dir = tmp_path / "next"
        new_dir.mkdir()
        marker = env._cwd_marker
        # The real MSYS spelling of a real native dir — what Git Bash
        # actually writes into the marker.
        msys_new = _windows_to_msys_path(str(new_dir))

        env._update_cwd(
            {
                "output": f"x\n{marker}{msys_new}{marker}\n",
                "returncode": 0,
            }
        )

        assert env.cwd == str(new_dir)


# ---------------------------------------------------------------------------
# End-to-end: _extract_cwd_from_output rollback when marker is invalid
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestExtractCwdFromOutputWindowsMsys:
    def test_stale_msys_marker_does_not_clobber_cwd(self, tmp_path):
        """When the cwd marker in stdout points at a non-existent path,
        ``LocalEnvironment._extract_cwd_from_output`` must roll back to
        the previous cwd instead of propagating a bad value."""
        original = tmp_path / "starting"
        original.mkdir()

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        gone = _windows_to_msys_path(str(tmp_path / "definitely-does-not-exist"))
        result = {
            "output": f"some command output\n{marker}{gone}{marker}\n",
            "returncode": 0,
        }

        env._extract_cwd_from_output(result)

        assert env.cwd == str(original)

    def test_valid_msys_marker_normalized_to_native(self, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        msys_new = _windows_to_msys_path(str(new_dir))
        result = {
            "output": f"x\n{marker}{msys_new}{marker}\n",
            "returncode": 0,
        }

        env._extract_cwd_from_output(result)

        assert env.cwd == str(new_dir)


# ---------------------------------------------------------------------------
# MSYS_NO_PATHCONV — native Windows command flags (#56700)
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestWindowsMsysPathconvDefaults:
    def test_make_run_env_sets_msys_no_pathconv_on_windows(self):
        run_env = _make_run_env({})
        assert run_env.get("MSYS_NO_PATHCONV") == "1"

    def test_sanitize_subprocess_env_sets_msys_no_pathconv_on_windows(self):
        env = _sanitize_subprocess_env({})
        assert env.get("MSYS_NO_PATHCONV") == "1"

    def test_hermes_subprocess_env_sets_msys_no_pathconv_on_windows(self):
        env = hermes_subprocess_env()
        assert env.get("MSYS_NO_PATHCONV") == "1"

    def test_msys2_arg_conv_excl_respects_user_override(self):
        run_env = _make_run_env({"MSYS2_ARG_CONV_EXCL": "/custom"})
        assert run_env.get("MSYS2_ARG_CONV_EXCL") == "/custom"


# ---------------------------------------------------------------------------
# Git Bash coreutils on PATH — non-login ``bash -c`` fallback (empty
# write_file error / terminal exit 127 when login bash is broken)
# ---------------------------------------------------------------------------

class TestGitBashCoreutilsOnPath:
    def _fake_isdir(self, existing):
        existing = {e.replace("\\", "/") for e in existing}
        return lambda p: p.replace("\\", "/") in existing

    @pytest.mark.platforms("windows")
    def test_derives_dirs_from_portablegit_layout(self, monkeypatch):
        """The PortableGit layout probe, run on the real OS.

        ``_find_bash`` and ``os.path.isdir`` are still stubbed — the point of
        this test is the *derivation* (which sibling dirs we compute from a
        bash path, and in what order), and hard-coding a fake tree keeps it
        independent of which Git flavour the runner happens to have installed.
        What is no longer faked is the host: the ``_IS_WINDOWS`` gate this
        function opens with is genuinely True here.
        """
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        monkeypatch.setattr(local_mod, "_find_bash", lambda: "/pg/bin/bash.exe")
        existing = {"/pg/mingw64/bin", "/pg/usr/bin", "/pg/bin"}
        monkeypatch.setattr(local_mod.os.path, "isdir", self._fake_isdir(existing))

        dirs = _git_bash_bin_dirs()

        # Compare separator-agnostically: the derivation uses os.path.join, so
        # on real Windows these come back with backslashes ("/pg\\usr\\bin").
        # The subject is WHICH dirs are derived and in what ORDER, not which
        # separator the host's os.path uses.
        norm = [d.replace("\\", "/") for d in dirs]

        # usr/bin is the load-bearing coreutils dir; mingw64 precedes it.
        assert "/pg/usr/bin" in norm
        assert norm.index("/pg/mingw64/bin") < norm.index("/pg/usr/bin")
        # Non-existent dirs (mingw32, usr/local/bin) are excluded.
        assert "/pg/mingw32/bin" not in norm

    @pytest.mark.platforms("linux")
    def test_empty_off_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        assert _git_bash_bin_dirs() == []

    @pytest.mark.platforms("linux")
    def test_make_run_env_noop_on_posix(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        run_env = _make_run_env({"PATH": "/usr/bin:/bin"})
        # No Windows git dirs injected on POSIX.
        assert "mingw64" not in run_env["PATH"]


# ---------------------------------------------------------------------------
# Command wrapping — native Windows cwd must be Git Bash-friendly for cd
# ---------------------------------------------------------------------------

@pytest.mark.platforms("windows")
class TestWrapCommandWindowsNativeCwd:
    def test_wrap_command_converts_native_cwd_for_builtin_cd(self):
        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=r"C:\Users\liush", timeout=10)

        env._snapshot_ready = True
        wrapped = env._wrap_command("pwd", r"C:\Users\liush")

        assert "builtin cd -- /c/Users/liush || exit 126" in wrapped
        assert r"builtin cd -- C:\Users\liush || exit 126" not in wrapped

    def test_init_session_bootstrap_rewrites_backslash_snapshot_paths(self, monkeypatch):
        captured = {}

        def fake_run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured.setdefault("script", cmd_string)  # bootstrap only; ignore the failure-path probe
            raise RuntimeError("stop after capturing bootstrap")

        monkeypatch.setattr(LocalEnvironment, "_run_bash", fake_run_bash)

        snap = r"C:\Users\Alexander\AppData\Local\Temp\hermes-snap-deadbeef.sh"
        with patch.object(LocalEnvironment, "__init__", lambda self, **kw: None):
            env = LocalEnvironment.__new__(LocalEnvironment)
            BaseEnvironment.__init__(
                env,
                cwd=r"C:\Users\Alexander\Documents",
                timeout=10,
            )
            env._snapshot_path = snap
            env._cwd_file = snap + ".cwd"
            env.init_session()

        script = captured["script"]
        assert "/c/Users/Alexander/AppData/Local/Temp/hermes-snap-deadbeef.sh" in script
        assert r"C:\Users\Alexander\AppData" not in script


def test_git_bash_dirs_stay_first_when_hermes_tools_move_behind_the_users(monkeypatch):
    """Git Bash's coreutils are store dirs too, but bash must find them before System32's."""
    import pm.install
    from tools.environments import local as local_mod

    bash_dir, node_dir, system32 = "/store/git/usr/bin", "/store/node", "/windows/system32"
    monkeypatch.setattr(pm.install, "_store_path_dirs", lambda: [bash_dir, node_dir])
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [bash_dir])
    path = os.pathsep.join([bash_dir, node_dir, system32])
    assert local_mod._user_tools_first(path).split(os.pathsep) == [bash_dir, system32, node_dir]


# ---------------------------------------------------------------------------
# _run_bash script transport — #152919
#
# MSYS rewrites the argv it is handed before bash ever reads it, halving every
# backslash: an inline ``-c`` command carrying ``\\2026`` reached bash as
# ``\2026`` and printf emitted byte 0x82 — wrong output, exit code 0. The command
# has to travel as a file, and the exit/stderr/stdin contract must not change
# with it.
# ---------------------------------------------------------------------------

class TestRunBashScriptTransport:
    @pytest.mark.platforms("windows")
    def test_double_backslashes_reach_bash_unchanged(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute(r"""printf '%s' 'A\\B\\C\\2026'""")
        assert result["returncode"] == 0
        assert result["output"] == "A\\\\B\\\\C\\\\2026"

    @pytest.mark.platforms("windows")
    def test_exit_code_and_stderr_survive_the_file_hop(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute("printf 'boom' >&2; exit 7")
        assert result["returncode"] == 7
        assert "boom" in result["output"]

    @pytest.mark.platforms("windows")
    def test_stdin_data_still_reaches_the_command(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute("cat", stdin_data="payload\n")
        assert result["output"].strip() == "payload"

    @pytest.mark.platforms("windows")
    def test_script_file_is_removed_after_the_command_exits(self, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        env.execute(r"""printf '%s' "$0" > script_path.txt""")
        script_path = (tmp_path / "script_path.txt").read_text(encoding="utf-8").strip()
        assert script_path.endswith(".sh"), script_path
        deadline = time.monotonic() + 5
        while os.path.exists(script_path) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not os.path.exists(script_path), script_path

    @pytest.mark.platforms("windows")
    def test_bash_receives_a_script_path_not_an_inline_c(self, tmp_path, monkeypatch):
        seen = []
        real_popen = local_mod.subprocess.Popen

        def spy(args, *rest, **kwargs):
            seen.append(list(args))
            return real_popen(args, *rest, **kwargs)

        monkeypatch.setattr(local_mod.subprocess, "Popen", spy)
        LocalEnvironment(cwd=str(tmp_path), timeout=10).execute("true")
        assert seen, "no command was spawned"
        assert "-c" not in seen[-1]
        assert seen[-1][-1].endswith(".sh")

    @pytest.mark.platforms("linux")
    def test_posix_keeps_the_inline_c_form(self, tmp_path, monkeypatch):
        seen = []
        real_popen = local_mod.subprocess.Popen

        def spy(args, *rest, **kwargs):
            seen.append(list(args))
            return real_popen(args, *rest, **kwargs)

        monkeypatch.setattr(local_mod.subprocess, "Popen", spy)
        LocalEnvironment(cwd=str(tmp_path), timeout=10).execute("true")
        assert seen, "no command was spawned"
        assert "-c" in seen[-1]
        assert not seen[-1][-1].endswith(".sh")

    @pytest.mark.platforms("linux")
    def test_script_writer_preserves_terminal_backslash_at_eof(self, tmp_path, monkeypatch):
        """The file transport must match bash -c for a trailing unquoted escape."""
        env = LocalEnvironment(cwd=str(tmp_path), timeout=120)
        monkeypatch.setattr(env, "get_temp_dir", lambda: str(tmp_path))
        command = "printf '%s' data\\"
        path = env._write_run_script(command)
        try:
            inline = subprocess.run(["bash", "-c", command], capture_output=True, check=True)
            from_file = subprocess.run(["bash", path], capture_output=True, check=True)
            assert inline.stdout == b"data\\"
            assert from_file.stdout == inline.stdout
        finally:
            os.unlink(path)
