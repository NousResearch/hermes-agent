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

These tests fake the Windows env on Linux CI by patching ``_IS_WINDOWS``
and ``os.path.isdir`` so the MSYS path tests as "missing" exactly like
on the real OS.
"""

import os
import shlex
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
    _prepend_git_bash_dirs,
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
    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        # On a non-Windows host the function must never rewrite the path
        # — POSIX-style paths are real paths there.
        assert _msys_to_windows_path("/c/Users/NVIDIA") == "/c/Users/NVIDIA"
        assert _msys_to_windows_path("/home/teknium") == "/home/teknium"

    def test_translates_drive_path(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path("/c/Users/NVIDIA") == r"C:\Users\NVIDIA"
        assert _msys_to_windows_path("/d/Projects/foo bar") == r"D:\Projects\foo bar"


    def test_empty_string(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _msys_to_windows_path("") == ""


# ---------------------------------------------------------------------------
# _windows_to_msys_path — reverse translation for bash builtin cd
# ---------------------------------------------------------------------------

class TestWindowsToMsysPath:
    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        assert _windows_to_msys_path(r"C:\Users\NVIDIA") == r"C:\Users\NVIDIA"


    def test_does_not_translate_non_drive_path(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _windows_to_msys_path("/tmp/foo") == "/tmp/foo"
        assert _windows_to_msys_path(r"\\server\share") == r"\\server\share"


# ---------------------------------------------------------------------------
# _bash_safe_path / _quote_bash_path — shell-script interpolation
# ---------------------------------------------------------------------------

class TestBashSafePath:
    def test_native_windows_path_becomes_msys(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _bash_safe_path(r"C:\Users\alice\notes.txt") == "/c/Users/alice/notes.txt"


    def test_quote_bash_path_quotes_mixed_windows_path(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        quoted = _quote_bash_path(
            r"C:\Users\Alexander\AppData\Local\Temp\hermes-snap-abc.sh"
        )
        assert "/c/Users/Alexander/AppData/Local/Temp/hermes-snap-abc.sh" in quoted
        assert "\\" not in quoted


# ---------------------------------------------------------------------------
# _resolve_safe_cwd — Windows fast path
# ---------------------------------------------------------------------------

class TestResolveSafeCwdWindows:
    def test_msys_path_resolves_to_native_when_native_exists(
        self, monkeypatch, tmp_path,
    ):
        """The whole point of this fix: a Git Bash ``/c/Users/x`` value
        should resolve to its native equivalent if that native dir exists,
        WITHOUT falling back to the temp dir."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        # tmp_path is a real native dir on the test host. Build a fake
        # MSYS form pointing at it and prove the resolver finds it.
        native = str(tmp_path)
        # Construct a synthetic MSYS form for whatever tmp_path is.
        # On Linux CI tmp_path is /tmp/... ; the resolver shouldn't even
        # try to translate that (regex won't match), so emulate the
        # mapping by pointing the translator at the real native dir.
        with patch.object(
            local_mod, "_msys_to_windows_path", return_value=native
        ):
            assert _resolve_safe_cwd("/c/whatever") == native


# ---------------------------------------------------------------------------
# End-to-end: _update_cwd via stdout marker (Windows simulation)
# ---------------------------------------------------------------------------

class TestUpdateCwdWindowsMsys:
    def test_marker_output_msys_path_stored_in_native_form(
        self, monkeypatch, tmp_path,
    ):
        """When Git Bash emits ``/c/Users/x`` in the cwd marker on Windows,
        ``_update_cwd`` must translate to native form before
        validating and storing — otherwise ``os.path.isdir`` rejects a
        perfectly real directory."""
        original = tmp_path / "starting"
        original.mkdir()

        # Fake Windows for the test
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        # Pretend Git Bash wrote an MSYS path that maps to tmp_path/"next"
        new_dir = tmp_path / "next"
        new_dir.mkdir()
        marker = env._cwd_marker

        # Translate the synthetic MSYS marker path to the real native dir.
        def fake_translate(p):
            if p == "/c/whatever/from/bash":
                return str(new_dir)
            return p

        with patch.object(local_mod, "_msys_to_windows_path", side_effect=fake_translate):
            env._update_cwd(
                {
                    "output": f"x\n{marker}/c/whatever/from/bash{marker}\n",
                    "returncode": 0,
                }
            )

        assert env.cwd == str(new_dir)


# ---------------------------------------------------------------------------
# End-to-end: _extract_cwd_from_output rollback when marker is invalid
# ---------------------------------------------------------------------------

class TestExtractCwdFromOutputWindowsMsys:
    def test_stale_msys_marker_does_not_clobber_cwd(self, monkeypatch, tmp_path):
        """When the cwd marker in stdout points at a non-existent path,
        ``LocalEnvironment._extract_cwd_from_output`` must roll back to
        the previous cwd instead of propagating a bad value."""
        original = tmp_path / "starting"
        original.mkdir()

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        result = {
            "output": f"some command output\n{marker}/c/no/such/path{marker}\n",
            "returncode": 0,
        }

        # Translation produces a path that doesn't exist on disk → rollback.
        with patch.object(
            local_mod,
            "_msys_to_windows_path",
            return_value=str(tmp_path / "definitely-does-not-exist"),
        ):
            env._extract_cwd_from_output(result)

        assert env.cwd == str(original)

    def test_valid_msys_marker_normalized_to_native(self, monkeypatch, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        marker = env._cwd_marker
        result = {
            "output": f"x\n{marker}/c/whatever{marker}\n",
            "returncode": 0,
        }

        with patch.object(local_mod, "_msys_to_windows_path", return_value=str(new_dir)):
            env._extract_cwd_from_output(result)

        assert env.cwd == str(new_dir)


# ---------------------------------------------------------------------------
# MSYS_NO_PATHCONV — native Windows command flags (#56700)
# ---------------------------------------------------------------------------

class TestWindowsMsysPathconvDefaults:
    def test_make_run_env_sets_msys_no_pathconv_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        run_env = _make_run_env({})
        assert run_env.get("MSYS_NO_PATHCONV") == "1"

    def test_sanitize_subprocess_env_sets_msys_no_pathconv_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env = _sanitize_subprocess_env({})
        assert env.get("MSYS_NO_PATHCONV") == "1"

    def test_hermes_subprocess_env_sets_msys_no_pathconv_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env = hermes_subprocess_env()
        assert env.get("MSYS_NO_PATHCONV") == "1"


    def test_msys2_arg_conv_excl_respects_user_override(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
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

    def test_derives_dirs_from_portablegit_layout(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        monkeypatch.setattr(local_mod, "_find_bash", lambda: "/pg/bin/bash.exe")
        existing = {"/pg/mingw64/bin", "/pg/usr/bin", "/pg/bin"}
        monkeypatch.setattr(local_mod.os.path, "isdir", self._fake_isdir(existing))

        dirs = _git_bash_bin_dirs()

        # usr/bin is the load-bearing coreutils dir; mingw64 precedes it.
        assert "/pg/usr/bin" in dirs
        assert dirs.index("/pg/mingw64/bin") < dirs.index("/pg/usr/bin")
        # Non-existent dirs (mingw32, usr/local/bin) are excluded.
        assert "/pg/mingw32/bin" not in dirs


    def test_empty_off_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        assert _git_bash_bin_dirs() == []


    def test_make_run_env_noop_on_posix(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs_cache", None)
        run_env = _make_run_env({"PATH": "/usr/bin:/bin"})
        # No Windows git dirs injected on POSIX.
        assert "mingw64" not in run_env["PATH"]


# ---------------------------------------------------------------------------
# Command wrapping — native Windows cwd must be Git Bash-friendly for cd
# ---------------------------------------------------------------------------

class TestWrapCommandWindowsNativeCwd:
    def test_wrap_command_converts_native_cwd_for_builtin_cd(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        with patch.object(
            LocalEnvironment, "init_session", autospec=True, return_value=None
        ):
            env = LocalEnvironment(cwd=r"C:\Users\liush", timeout=10)

        env._snapshot_ready = True
        wrapped = env._wrap_command("pwd", r"C:\Users\liush")

        assert "builtin cd -- /c/Users/liush || exit 126" in wrapped
        assert r"builtin cd -- C:\Users\liush || exit 126" not in wrapped


    def test_init_session_bootstrap_rewrites_backslash_snapshot_paths(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

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


# ---------------------------------------------------------------------------
# PowerShell re-quoting for the outer Bash wrapper
# ---------------------------------------------------------------------------
# BaseEnvironment.execute() feeds _prepare_command()'s output to
# _wrap_command(), which evaluates it through Bash. Without re-quoting, Bash
# expands PowerShell's own `$_.Path` / `$env:TEMP` before PowerShell ever sees
# them. _quote_windows_powershell_command() rebuilds a *standalone* pwsh
# invocation with Bash-safe quoting, and deliberately declines (returns None)
# on anything it can't prove is simple, leaving the pre-existing Bash path
# untouched.


class TestWindowsPowerShellQuoting:
    """_quote_windows_powershell_command: round-trip, switches, refusals."""

    @staticmethod
    def _roundtrip(quoted: str) -> list[str]:
        """Re-parse the rebuilt string the way the outer Bash would."""
        return shlex.split(quoted, posix=True)

    @pytest.mark.parametrize("payload", [
        "Get-ChildItem | ForEach-Object { $_.Path }",
        "echo $env:TEMP",
        "$_.Name",
        # `$(...)` *inside the payload* is PowerShell subexpression syntax and
        # must be protected, not refused — it only disqualifies the command
        # when it appears before the -Command marker (see the refusal tests).
        "echo $(Get-Date)",
    ])
    def test_dollar_payloads_survive_the_bash_layer_literally(self, payload):
        """$_ / $env: must reach PowerShell unexpanded — the whole point."""
        quoted = local_mod._quote_windows_powershell_command(
            f'pwsh -Command "{payload}"'
        )
        assert quoted is not None
        # The payload token comes back byte-identical after Bash re-parses it.
        assert self._roundtrip(quoted)[-1] == payload

    @pytest.mark.parametrize("switch", [
        "-c", "-co", "-com", "-comm", "-comma", "-comman", "-command",
        "/c", "/command", "-Command", "-COMMAND",
    ])
    def test_command_switch_abbreviations_are_recognized(self, switch):
        """PowerShell accepts any unambiguous prefix of -Command, and the
        Windows-style `/c` form; all must be treated as the payload marker."""
        quoted = local_mod._quote_windows_powershell_command(
            f'powershell {switch} "echo $env:TEMP"'
        )
        assert quoted is not None, switch
        assert self._roundtrip(quoted)[-1] == "echo $env:TEMP"

    @pytest.mark.parametrize("mode_flag", ["-File", "-EncodedCommand"])
    def test_non_command_modes_are_declined(self, mode_flag):
        """-File and -EncodedCommand are not -Command payloads; the quoter
        must not claim them (an encoded payload is already Bash-inert, and a
        script path is handled by the normal path logic)."""
        assert local_mod._quote_windows_powershell_command(
            rf'pwsh {mode_flag} C:\tmp\script.ps1'
        ) is None

    @pytest.mark.parametrize("command", [
        "ls -la",
        "python -c 'print(1)'",
        "git status",
        "notpwsh -Command \"echo $env:TEMP\"",
    ])
    def test_non_powershell_commands_pass_through_untouched(self, command):
        """Anything that isn't a pwsh/powershell executable is None so the
        caller leaves the command exactly as-is."""
        assert local_mod._quote_windows_powershell_command(command) is None

    @pytest.mark.parametrize("command", [
        'pwsh -Command "echo hi" | grep hi',      # pipeline
        'pwsh -Command "echo hi" && echo done',   # operator
        'pwsh $(echo -Command) "echo hi"',        # expansion BEFORE the payload
        'pwsh -Command "echo hi"; rm -rf /tmp/x', # statement separator
        'pwsh -Command "unterminated',            # unbalanced quote
        'pwsh -Command "a" > out.txt',            # redirection
    ])
    def test_ambiguous_shell_syntax_is_refused(self, command):
        """Conservative by design: real Bash syntax around the invocation
        means the existing Bash path must keep handling it."""
        assert local_mod._quote_windows_powershell_command(command) is None

    def test_multiline_input_is_refused(self):
        assert local_mod._quote_windows_powershell_command(
            'pwsh -Command "echo a"\necho b'
        ) is None

    def test_empty_input_is_refused(self):
        assert local_mod._quote_windows_powershell_command("   ") is None

    def test_quoted_executable_path_with_spaces_is_preserved(self):
        """A quoted pwsh path must stay a single token after re-quoting."""
        quoted = local_mod._quote_windows_powershell_command(
            r'"C:\Program Files\PowerShell\7\pwsh.exe" -Command "echo $env:TEMP"'
        )
        assert quoted is not None
        tokens = self._roundtrip(quoted)
        assert tokens[0] == r"C:\Program Files\PowerShell\7\pwsh.exe"
        assert tokens[-1] == "echo $env:TEMP"


class TestPrepareCommandWindowsGate:
    """LocalEnvironment._prepare_command applies the rewrite only on Windows."""

    def _prepare(self, monkeypatch, *, is_windows):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", is_windows)
        monkeypatch.setattr(
            BaseEnvironment, "_prepare_command",
            lambda self, command: (command, None),
        )
        env = LocalEnvironment.__new__(LocalEnvironment)
        return env._prepare_command('pwsh -Command "echo $env:TEMP"')

    def test_rewrites_on_windows(self, monkeypatch):
        exec_command, sudo_stdin = self._prepare(monkeypatch, is_windows=True)
        assert sudo_stdin is None
        assert exec_command != 'pwsh -Command "echo $env:TEMP"'
        assert shlex.split(exec_command, posix=True)[-1] == "echo $env:TEMP"

    def test_leaves_command_untouched_on_posix(self, monkeypatch):
        """The quoter is a Windows/Git-Bash workaround; on POSIX the command
        must reach the shell exactly as the caller wrote it."""
        exec_command, _ = self._prepare(monkeypatch, is_windows=False)
        assert exec_command == 'pwsh -Command "echo $env:TEMP"'


class TestMsysControlsNotLeakedToPosix:
    """MSYS_NO_PATHCONV / MSYS2_ARG_CONV_EXCL are Windows-only controls."""

    def test_apply_defaults_strips_controls_on_posix(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        env = {"MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*", "PATH": "/usr/bin"}
        local_mod._apply_windows_msys_bash_env_defaults(env)
        assert "MSYS_NO_PATHCONV" not in env
        assert "MSYS2_ARG_CONV_EXCL" not in env
        assert env["PATH"] == "/usr/bin", "unrelated vars must be left alone"

    def test_apply_defaults_sets_controls_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env: dict = {}
        local_mod._apply_windows_msys_bash_env_defaults(env)
        assert env.get("MSYS_NO_PATHCONV") == "1"
        assert env.get("MSYS2_ARG_CONV_EXCL") == "*"

    def test_make_run_env_does_not_leak_inherited_controls_on_posix(self, monkeypatch):
        """Even when the parent process has them set, a POSIX subprocess env
        must come out clean."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        monkeypatch.setattr(
            local_mod.os, "environ",
            {"MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*", "HOME": "/home/x"},
        )
        result = local_mod._make_run_env({})
        assert "MSYS_NO_PATHCONV" not in result
        assert "MSYS2_ARG_CONV_EXCL" not in result
