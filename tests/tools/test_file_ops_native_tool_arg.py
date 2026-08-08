"""Tests for ``ShellFileOperations._escape_native_tool_arg`` and the
ripgrep command construction on Windows.

Background
----------
On Windows, ``_escape_shell_arg`` rewrites drive paths to the Git Bash
``/c/...`` form — required for MSYS tools (grep, find, test, bash
builtins). Native binaries are different: Hermes runs bash with
``MSYS_NO_PATHCONV=1``/``MSYS2_ARG_CONV_EXCL=*`` (see
``tools.environments.local._apply_windows_msys_bash_env_defaults``), so
argv reaches them verbatim, and a native ripgrep build (e.g. the WinGet
MSVC package) cannot resolve ``/c/...``:

    rg: /c/Users/x: IO error for operation on /c/Users/x:
    The system cannot find the path specified. (os error 3)

Every explicit-path content/file search failed in that configuration;
only the default relative root (``.``) worked. The fix feeds native
tools the ``C:/...`` form via ``_escape_native_tool_arg``.

These tests fake Windows on POSIX CI by patching ``_IS_WINDOWS``.
"""

from unittest.mock import patch

from tools.environments import local as local_mod
from tools.file_operations import ShellFileOperations


def _make_ops() -> ShellFileOperations:
    """Bare instance for command-building tests (no terminal_env needed)."""
    return ShellFileOperations.__new__(ShellFileOperations)


class TestEscapeNativeToolArgWindows:
    def test_msys_path_converted(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            assert _make_ops()._escape_native_tool_arg("/e/HermesWork/proj") == "'E:/HermesWork/proj'"

    def test_backslash_native_converted(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            assert _make_ops()._escape_native_tool_arg(r"E:\HermesWork\proj") == "'E:/HermesWork/proj'"

    def test_forward_slash_native_idempotent(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            assert _make_ops()._escape_native_tool_arg("E:/HermesWork/proj") == "'E:/HermesWork/proj'"

    def test_relative_and_plain_posix_untouched(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            ops = _make_ops()
            assert ops._escape_native_tool_arg(".") == "'.'"
            assert ops._escape_native_tool_arg("/home/user/proj") == "'/home/user/proj'"

    def test_single_quote_escaped(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            assert _make_ops()._escape_native_tool_arg("E:/it's/x") == "'E:/it'\"'\"'s/x'"

    def test_empty_passthrough(self):
        with patch.object(local_mod, "_IS_WINDOWS", True):
            assert _make_ops()._escape_native_tool_arg("") == ""


class TestEscapeNativeToolArgNonWindows:
    def test_noop_beyond_quoting(self):
        with patch.object(local_mod, "_IS_WINDOWS", False):
            ops = _make_ops()
            assert ops._escape_native_tool_arg("/e/HermesWork/proj") == "'/e/HermesWork/proj'"
            assert ops._escape_native_tool_arg(r"E:\HermesWork\proj") == r"'E:\HermesWork\proj'"


class _ExecResult:
    exit_code = 0
    stdout = ""
    stderr = ""


class TestRipgrepGetsNativePath:
    """The path argument handed to rg must be in native form on Windows."""

    def _capture(self, method, *args):
        ops = _make_ops()
        ops._has_command = lambda _cmd: True
        captured = []
        ops._exec = lambda cmd, timeout=None: captured.append(cmd) or _ExecResult()
        with patch.object(local_mod, "_IS_WINDOWS", True):
            method(ops, *args)
        return captured

    def test_search_with_rg(self):
        captured = self._capture(
            ShellFileOperations._search_with_rg,
            "pattern", r"E:\HermesWork\proj", None, 50, 0, "files_only", 0,
        )
        assert captured, "rg was never invoked"
        assert "'E:/HermesWork/proj'" in captured[0]

    def test_search_files_rg(self):
        captured = self._capture(
            ShellFileOperations._search_files_rg,
            "*foo*", r"E:\HermesWork\proj", 50, 0,
        )
        assert captured, "rg was never invoked"
        assert "'E:/HermesWork/proj'" in captured[0]
