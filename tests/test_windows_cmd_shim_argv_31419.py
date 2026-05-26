"""Regression for #31419: argv metacharacters reach cmd.exe re-parse.

On Windows, ``subprocess.Popen([target, *args])`` where ``target``
ends in ``.cmd`` / ``.bat`` quietly routes through ``cmd.exe /c``,
which **re-parses** the generated command line with its own
metacharacter rules (``|``, ``&``, ``<``, ``>``, ``^``, ``"``,
``(``, ``)``, ``%``, ``!``).  Python's ``list2cmdline`` only handles
the standard CRT quoting pass — it doesn't know about the cmd.exe
re-parse — so an argv element like ``a|b`` (no whitespace, no quotes)
slips through ``list2cmdline`` untouched and gets interpreted by
cmd.exe as a pipeline.  At best the child sees a truncated argv;
at worst, untrusted argv content (LLM-supplied prompts, JSON quoted
from chat, Markdown with ``&``, …) becomes shell injection.

These tests cover:
* :func:`hermes_cli._subprocess_compat.is_cmd_or_bat_shim` predicate.
* :func:`hermes_cli._subprocess_compat.quote_for_cmd_shim` per-arg quoting
  against a known-good test-vector table (CRT pass + cmd.exe meta pass).
* :func:`hermes_cli._subprocess_compat.build_cmd_shim_command_line`
  composition.
* :func:`hermes_cli._subprocess_compat.safe_subprocess_argv` end-to-end:
  passthrough on POSIX / non-shim targets, full string output on
  Windows shim targets, idempotent on the happy path.
* Source-level guards pinning the wiring at the npm install / agent-browser
  install call sites, so a future refactor can't silently strip the helper.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli._subprocess_compat import (
    build_cmd_shim_command_line,
    is_cmd_or_bat_shim,
    quote_for_cmd_shim,
    safe_subprocess_argv,
)


# ---------------------------------------------------------------------------
# is_cmd_or_bat_shim — predicate
# ---------------------------------------------------------------------------


class TestIsCmdOrBatShim:
    @pytest.mark.parametrize(
        "path",
        [
            r"C:\Users\u\AppData\Roaming\npm\npm.cmd",
            r"C:\tools\npm.CMD",
            r"D:\bin\install.bat",
            r"D:\bin\install.BAT",
        ],
    )
    def test_true_for_cmd_or_bat_on_windows(self, path):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            assert is_cmd_or_bat_shim(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            r"C:\Users\u\AppData\Roaming\npm\npm.exe",
            r"C:\tools\python.exe",
            "/usr/bin/npm",
            "/usr/local/bin/git",
        ],
    )
    def test_false_for_other_extensions(self, path):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            assert is_cmd_or_bat_shim(path) is False

    def test_no_op_on_posix_even_for_cmd_path(self):
        """On POSIX a path with .cmd suffix is just a regular file —
        there's no cmd.exe re-parse to defend against, so the predicate
        must return False so callers don't accidentally CRT-quote a
        Linux argv.
        """
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", False):
            assert is_cmd_or_bat_shim(r"C:\foo\bar.cmd") is False

    def test_handles_empty_and_none_safely(self):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            assert is_cmd_or_bat_shim("") is False


# ---------------------------------------------------------------------------
# quote_for_cmd_shim — per-arg escaping
# ---------------------------------------------------------------------------


class TestQuoteForCmdShim:
    """Pin the quoting against a known-good test-vector table.

    Each row is ``(input, expected_output)``.  The output is what
    cmd.exe must see on the command line so that, after both its own
    metacharacter parse and the .cmd shim's CRT argv parse, the script
    receives the original *input* as a single literal argv element.

    Hand-derived from the documented CRT + cmd.exe rules:
      * https://learn.microsoft.com/cpp/cpp/main-function-command-line-args
      * https://ss64.com/nt/syntax-esc.html
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Empty stays "" — explicit empty argv element.
            ("", '""'),
            # No metas, no spaces — passthrough (avoid gratuitous noise on
            # the static argv elements that dominate our spawn sites).
            ("install", "install"),
            ("--silent", "--silent"),
            # Spaces only → CRT-quote AND caret-escape the outer quotes.
            # cmd.exe technically treats unescaped ``"`` correctly inside
            # a balanced quoted token, but always caret-escaping the
            # outer wrappers is the BatBadBut-recommended belt-and-
            # suspenders posture: nested cmd invocations and
            # ``call`` / ``for`` re-parses can re-interpret an
            # unescaped ``"`` boundary.
            ("hello world", '^"hello world^"'),
            # Single cmd.exe meta — must be wrapped AND caret-escaped so
            # cmd.exe's re-parse pass treats the metas as literal.
            ("a|b", '^"a^|b^"'),
            ("foo&bar", '^"foo^&bar^"'),
            ("a>b", '^"a^>b^"'),
            ("a<b", '^"a^<b^"'),
            ("foo(bar)", '^"foo^(bar^)^"'),
            ("100%", '^"100^%^"'),
            # Embedded double quote → CRT escape (\") then caret-escape
            # both the embedded \" and the outer wrappers.
            ('a"b', '^"a\\^"b^"'),
            # Backslash-quote sequence (CVE-2024-24576-style).
            ('\\"', '^"\\\\\\^"^"'),
            # Trailing backslashes inside a token that DOES go through the
            # escape path (here: contains a space) must be doubled so the
            # closing CRT quote isn't itself escaped.  ``path with\`` →
            # CRT ``"path with\\"`` (one trailing ``\`` doubled to two)
            # → caret-escape the outer quotes.
            ("path with\\", '^"path with\\\\^"'),
            # Combination.
            ('say "hi & bye"', '^"say \\^"hi ^& bye\\^"^"'),
        ],
    )
    def test_known_vectors(self, raw, expected):
        assert quote_for_cmd_shim(raw) == expected, (
            f"quote_for_cmd_shim({raw!r}) → {quote_for_cmd_shim(raw)!r}, "
            f"expected {expected!r}"
        )

    def test_idempotent_on_safe_static_args(self):
        """Static args (most of Hermes's spawn sites) must round-trip
        unchanged so we don't mangle ``--silent`` into ``"--silent"``.
        """
        for arg in [
            "install",
            "--silent",
            "--no-fund",
            "--no-audit",
            "--prefix",
            "agent-browser",
            "@vue/language-server",
            "typescript-language-server",
        ]:
            assert quote_for_cmd_shim(arg) == arg


# ---------------------------------------------------------------------------
# build_cmd_shim_command_line — composition
# ---------------------------------------------------------------------------


class TestBuildCmdShimCommandLine:
    def test_target_path_is_quoted_when_it_contains_spaces(self):
        cmdline = build_cmd_shim_command_line(
            r"C:\Program Files\nodejs\npm.cmd",
            ["install"],
        )
        # Always caret-escape the outer wrappers (BatBadBut-recommended
        # belt-and-suspenders), so the prefix is ``^"...^"`` not ``"..."``.
        assert cmdline.startswith('^"C:\\Program Files\\nodejs\\npm.cmd^"'), cmdline
        assert cmdline.endswith(" install")

    def test_each_arg_is_individually_quoted(self):
        cmdline = build_cmd_shim_command_line(
            r"C:\npm.cmd",
            ["install", "--prefix", r"C:\Users\u\.hermes\lsp", "pkg|name"],
        )
        # Sanity: tokens are space-separated.
        tokens = cmdline.split(" ")
        # Static tokens preserved.
        assert "install" in tokens
        assert "--prefix" in tokens
        # Path with no metas / spaces is unchanged.
        assert r"C:\Users\u\.hermes\lsp" in tokens
        # The pipe-bearing token must be caret-escaped + quoted.
        assert any("^|" in t and '^"' in t for t in tokens), (
            f"expected an escaped token containing ^| and ^\"; got tokens={tokens}"
        )

    def test_empty_args_list_yields_just_quoted_target(self):
        cmdline = build_cmd_shim_command_line(r"C:\foo.cmd", [])
        assert cmdline == r"C:\foo.cmd"


# ---------------------------------------------------------------------------
# safe_subprocess_argv — top-level wrapper
# ---------------------------------------------------------------------------


class TestSafeSubprocessArgv:
    def test_passthrough_on_posix(self):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", False):
            argv = ["/usr/bin/npm", "install", "--prefix", "/tmp/foo"]
            result = safe_subprocess_argv(argv)
            assert result == argv
            assert isinstance(result, list)

    def test_passthrough_on_windows_exe_target(self):
        """An .exe target on Windows doesn't go through cmd.exe so the
        helper must keep it as a list (otherwise list2cmdline gets
        bypassed and we lose its CRT quoting work).
        """
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            argv = [r"C:\Python\python.exe", "-c", "print(1)"]
            result = safe_subprocess_argv(argv)
            assert result == argv
            assert isinstance(result, list)

    def test_returns_string_on_windows_cmd_target(self):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            argv = [r"C:\npm.cmd", "install", "a|b"]
            result = safe_subprocess_argv(argv)
            assert isinstance(result, str)
            # Target preserved.
            assert result.startswith(r"C:\npm.cmd")
            # Pipe got escaped — both as a meta and as part of a CRT-quoted token.
            assert "^|" in result
            assert "^\"" in result

    def test_returns_string_on_windows_bat_target(self):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            argv = [r"D:\hooks\install.bat", "deploy"]
            result = safe_subprocess_argv(argv)
            assert isinstance(result, str)
            assert result == r"D:\hooks\install.bat deploy"

    def test_empty_argv_returns_empty_list(self):
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", True):
            assert safe_subprocess_argv([]) == []

    def test_idempotent_on_already_safe_argv(self):
        """Running twice (e.g., a defensive double-wrap) must yield the
        same result — important so callers can apply the helper without
        worrying about whether something upstream already did."""
        with patch("hermes_cli._subprocess_compat.IS_WINDOWS", False):
            argv = ["/usr/bin/git", "log", "--oneline"]
            once = safe_subprocess_argv(argv)
            # On POSIX the helper returns a list — running it again
            # against the same argv must keep it stable.
            twice = safe_subprocess_argv(once)
            assert once == twice == argv


# ---------------------------------------------------------------------------
# Call-site guards — pin the wiring so a future refactor can't drop it.
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCallSiteWiring:
    def test_lsp_npm_install_uses_safe_subprocess_argv(self):
        from agent.lsp.install import _install_npm

        src = inspect.getsource(_install_npm)
        assert "safe_subprocess_argv" in src, (
            "agent/lsp/install.py::_install_npm must wrap its npm install "
            "call with safe_subprocess_argv so cmd.exe's re-parse on the "
            "npm.cmd shim can't reinterpret pkg names / paths as shell "
            "metacharacters. See #31419."
        )
        # Helper must wrap the actual subprocess.run argv, not just be imported.
        # Anchor: the call is `subprocess.run(safe_subprocess_argv([...`.
        assert re.search(r"subprocess\.run\(\s*safe_subprocess_argv\(", src), (
            "_install_npm must call ``subprocess.run(safe_subprocess_argv("
            "...))`` — the helper has to wrap the argv passed to "
            "subprocess.run, not float around as an unused import."
        )

    def test_tools_config_npm_install_uses_safe_subprocess_argv(self):
        text = (_REPO_ROOT / "hermes_cli" / "tools_config.py").read_text()
        # The post_setup npm install for browser tools.
        assert "safe_subprocess_argv([npm_bin, " in text, (
            "hermes_cli/tools_config.py:_run_post_setup must wrap the "
            "[npm_bin, 'install', ...] call with safe_subprocess_argv "
            "(npm_bin resolves to npm.cmd on Windows). See #31419."
        )
        # And the agent-browser install call further down.
        assert "safe_subprocess_argv(install_cmd)" in text, (
            "hermes_cli/tools_config.py:_run_post_setup must wrap the "
            "agent-browser install_cmd with safe_subprocess_argv "
            "(local_ab is .cmd on Windows). See #31419."
        )
