"""Regression tests for the Windows ``.cmd``-shim URL arg-splitting bug.

``tools/browser_tool.py`` resolves the agent-browser CLI to an npm ``.cmd``
batch shim on Windows (``node_modules/.bin/agent-browser.cmd``, a global-prefix
``agent-browser.cmd``, or ``npx.cmd`` for the npx fallback).  Those shims run
under ``cmd.exe`` and end their dispatch line with an unquoted ``%*``, so any
argument containing a cmd metacharacter — notably ``&`` in a real search URL
like ``https://x.com/search?q=from%3ANousResearch&src=typed_query&f=live`` —
gets re-parsed by cmd.exe into separate "commands", silently breaking
navigation (``'src' is not recognized as an internal or external command``).

The fix (``_bypass_windows_cmd_shim`` / ``_windows_cmd_shim_node_target``)
rewrites a recognized ``.cmd`` shim prefix to a direct invocation of the shim's
own dispatch target — ``node.exe <entry>.js`` for npm's Node shim layouts, or
the native ``.exe`` that agent-browser's own postinstall writes into its
optimized global shim — taking cmd.exe out of the spawn chain so
``subprocess.Popen`` hands the argv to the child verbatim.  Unrecognized or
semantically richer shims (extra dispatch arguments) are refused: the original
prefix is kept, never rewritten lossily.

Nearly everything here is native-Windows behavior, so per AGENTS.md ("Don't
fake the host OS") the classes carry ``@pytest.mark.windows_only`` and run on
the Windows CI lane (``tests-os.yml`` discovers this file via the marker and
selects with ``-m windows_only``).  The real-node classes layer a
``node.exe``-availability skip on top of the marker.  The one genuinely
portable test — the POSIX no-op contract — stays unmarked for the Linux lane.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import tools.browser_tool as browser_tool


X_SEARCH_URL = "https://x.com/search?q=from%3ANousResearch&src=typed_query&f=live"

# The npx fallback's real production prefix tail (after the npx launcher
# itself) — mirrored in the trailing-args test below.  Note the version spec
# contains ``^``, cmd.exe's escape character: through an un-bypassed .cmd hop
# even THIS argument would be silently rewritten (range -> exact pin).
_NPX_PREFIX_TAIL = [
    "--ignore-scripts",
    "--prefer-offline",
    "-y",
    browser_tool.AGENT_BROWSER_NPX_SPEC,
]

# A JS entrypoint that echoes its received argv as JSON, so a real node spawn can
# report exactly what reached the child process.
_ARGV_ECHO_JS = "console.log(JSON.stringify(process.argv.slice(2)));\n"


# npm cmd-shim package format — what ``node_modules/.bin/agent-browser.cmd``
# actually contains.  The JS entry is referenced literally on the dispatch line
# via ``%dp0%`` (relative to the shim's own dir).
_CMD_SHIM_TEMPLATE = (
    "@ECHO off\r\n"
    "GOTO start\r\n"
    ":find_dp0\r\n"
    "SET dp0=%~dp0\r\n"
    "EXIT /b\r\n"
    ":start\r\n"
    "SETLOCAL\r\n"
    "CALL :find_dp0\r\n"
    "\r\n"
    'IF EXIST "%dp0%\\node.exe" (\r\n'
    '  SET "_prog=%dp0%\\node.exe"\r\n'
    ") ELSE (\r\n"
    '  SET "_prog=node"\r\n'
    "  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n"
    ")\r\n"
    "\r\n"
    "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "
    '"%_prog%"  "%dp0%\\..\\agent-browser\\bin\\agent-browser.js" %*\r\n'
)

# npm's own npx.cmd — the REAL npm 11.12.1 bin/npx.cmd content verbatim,
# including the FOR/IF global-prefix override block.  The JS path is indirected
# through a SET'd variable rather than written literally on the dispatch line,
# and the file also references a *different* %~dp0 .js (npm-prefix.js) the
# resolver must NOT pick.
_NPX_SHIM_TEMPLATE = (
    ":: Created by npm, please don't edit manually.\r\n"
    "@ECHO OFF\r\n"
    "\r\n"
    "SETLOCAL\r\n"
    "\r\n"
    'SET "NODE_EXE=%~dp0\\node.exe"\r\n'
    'IF NOT EXIST "%NODE_EXE%" (\r\n'
    '  SET "NODE_EXE=node"\r\n'
    ")\r\n"
    "\r\n"
    'SET "NPM_PREFIX_JS=%~dp0\\node_modules\\npm\\bin\\npm-prefix.js"\r\n'
    'SET "NPX_CLI_JS=%~dp0\\node_modules\\npm\\bin\\npx-cli.js"\r\n'
    "FOR /F \"delims=\" %%F IN ('CALL \"%NODE_EXE%\" \"%NPM_PREFIX_JS%\"') DO (\r\n"
    '  SET "NPM_PREFIX_NPX_CLI_JS=%%F\\node_modules\\npm\\bin\\npx-cli.js"\r\n'
    ")\r\n"
    'IF EXIST "%NPM_PREFIX_NPX_CLI_JS%" (\r\n'
    '  SET "NPX_CLI_JS=%NPM_PREFIX_NPX_CLI_JS%"\r\n'
    ")\r\n"
    "\r\n"
    '"%NODE_EXE%" "%NPX_CLI_JS%" %*\r\n'
)

# agent-browser's own "optimized" global shim — scripts/postinstall.js
# (fixWindowsShims) overwrites the npm-generated global agent-browser.cmd with
# a direct native-binary dispatch. Still an unquoted %* through cmd.exe, so
# still vulnerable to '&' splitting until bypassed.
_NATIVE_EXE_SHIM_TEMPLATE = (
    "@ECHO off\r\n"
    '"%~dp0node_modules\\agent-browser\\bin\\agent-browser-win32-x64.exe" %*\r\n'
)


def _make_agent_browser_shim(tmp_path: Path):
    """Create node_modules/.bin/agent-browser.cmd + its .js entry + node.exe.

    A colocated ``node.exe`` is dropped next to the shim so resolution is
    deterministic (exercises the preferred colocated-node branch, mirroring the
    shim's own ``IF EXIST "%dp0%\\node.exe"`` probe) without depending on node
    being on the runner's PATH.
    """
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    cmd = bin_dir / "agent-browser.cmd"
    cmd.write_text(_CMD_SHIM_TEMPLATE, encoding="utf-8")
    (bin_dir / "node.exe").write_text("", encoding="utf-8")
    entry = tmp_path / "node_modules" / "agent-browser" / "bin" / "agent-browser.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    return cmd, entry


def _make_npx_shim(tmp_path: Path):
    """Create npx.cmd + its npx-cli.js entry + a colocated node.exe."""
    node_dir = tmp_path / "nodejs"
    node_dir.mkdir()
    cmd = node_dir / "npx.cmd"
    cmd.write_text(_NPX_SHIM_TEMPLATE, encoding="utf-8")
    (node_dir / "node.exe").write_text("", encoding="utf-8")
    entry = node_dir / "node_modules" / "npm" / "bin" / "npx-cli.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// npx cli\n", encoding="utf-8")
    return cmd, entry


def _make_native_exe_shim(tmp_path: Path, *, create_exe: bool = True):
    """Create a global-prefix agent-browser.cmd in agent-browser's native form."""
    prefix_dir = tmp_path / "npm-global"
    exe = prefix_dir / "node_modules" / "agent-browser" / "bin" / "agent-browser-win32-x64.exe"
    exe.parent.mkdir(parents=True)
    if create_exe:
        exe.write_text("", encoding="utf-8")
    cmd = prefix_dir / "agent-browser.cmd"
    cmd.write_text(_NATIVE_EXE_SHIM_TEMPLATE, encoding="utf-8")
    return cmd, exe


@pytest.mark.windows_only
class TestCmdShimBypass:
    def test_agent_browser_cmd_resolves_to_node_and_js(self, tmp_path):
        cmd, entry = _make_agent_browser_shim(tmp_path)
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target is not None
        # cmd.exe is out of the chain: the executable is node, not the .cmd.
        assert not target[0].lower().endswith(".cmd")
        assert os.path.basename(target[0]).lower().startswith("node")
        assert os.path.normpath(target[1]) == os.path.normpath(str(entry))

    def test_npx_cmd_resolves_to_npx_cli_js(self, tmp_path):
        cmd, entry = _make_npx_shim(tmp_path)
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target is not None
        assert not target[0].lower().endswith(".cmd")
        # Must pick npx-cli.js, NOT the npm-prefix.js that also appears in the
        # shim text.
        assert os.path.normpath(target[1]) == os.path.normpath(str(entry))

    def test_native_exe_shim_resolves_to_exe(self, tmp_path):
        # agent-browser's postinstall-optimized global shim dispatches straight
        # to a native binary; the bypass must launch that .exe itself.
        cmd, exe = _make_native_exe_shim(tmp_path)
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target == [os.path.normpath(str(exe))]

    def test_native_exe_shim_missing_binary_refused(self, tmp_path):
        # Fail-safe: shim advertises a binary that isn't there -> keep the .cmd.
        cmd, _exe = _make_native_exe_shim(tmp_path, create_exe=False)
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_extra_dispatch_args_refused(self, tmp_path):
        # cmd-shim emits shebang interpreter args between the program and the
        # script ('#!/usr/bin/env node --enable-source-maps'). Rewriting such a
        # shim would silently drop those args — it must be refused instead.
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        cmd = bin_dir / "with-args.cmd"
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                '"%_prog%"  "%dp0%\\..\\agent-browser\\bin\\agent-browser.js" %*',
                '"%_prog%" --enable-source-maps "%dp0%\\..\\agent-browser\\bin\\agent-browser.js" %*',
            ),
            encoding="utf-8",
        )
        entry = tmp_path / "node_modules" / "agent-browser" / "bin" / "agent-browser.js"
        entry.parent.mkdir(parents=True)
        entry.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None
        assert browser_tool._bypass_windows_cmd_shim([str(cmd)]) == [str(cmd)]

    def test_child_env_setup_shim_refused(self, tmp_path):
        # A wrapper that SETs a variable its own plumbing never reads is doing
        # environment setup for the CHILD; a rewrite would silently drop it.
        # (An otherwise byte-perfect cmd-shim body with one extra SET line.)
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                "SETLOCAL\r\n", "SET FOO=required\r\nSETLOCAL\r\n", 1
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_self_referential_env_mutation_refused(self, tmp_path):
        # `SET PATH=<extra>;%PATH%` is consumed only by its own assignment —
        # that's an environment mutation for the child, not shim plumbing.
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                "SETLOCAL\r\n", 'SET "PATH=C:\\extra;%PATH%"\r\nSETLOCAL\r\n', 1
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_extra_command_line_shim_refused(self, tmp_path):
        # Any command line outside the recognized plumbing grammar means the
        # shim does real work beyond dispatching — refuse, don't skip the work.
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                "SETLOCAL\r\n", 'MKDIR "%~dp0cache" 2>NUL\r\nSETLOCAL\r\n', 1
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_native_exe_wrong_path_refused(self, tmp_path):
        # The native branch is restricted to agent-browser's own binary path;
        # an arbitrary %~dp0-relative .exe wrapper is not rewritten even when
        # the target exists.
        prefix_dir = tmp_path / "npm-global"
        exe = prefix_dir / "other" / "tool.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
        cmd = prefix_dir / "tool.cmd"
        cmd.write_text('@ECHO off\r\n"%~dp0other\\tool.exe" %*\r\n', encoding="utf-8")
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_native_exe_shim_with_extra_setup_refused(self, tmp_path):
        # Even the exact native dispatch is refused when the file carries
        # unconsumed setup around it.
        cmd, _exe = _make_native_exe_shim(tmp_path)
        cmd.write_text(
            _NATIVE_EXE_SHIM_TEMPLATE.replace(
                "@ECHO off\r\n", "@ECHO off\r\nSET FOO=required\r\n", 1
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_dispatch_line_prefix_command_refused(self, tmp_path):
        # A command chained BEFORE an otherwise-perfect dispatch tail must
        # refuse the shim — a rewrite would silently skip that command.
        # (Only cmd-shim's exact endLocal/goto/title preamble is allowed.)
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & ",
                'ECHO required>"%dp0%marker.txt" & '
                "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & ",
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_set_consumed_only_in_comment_refused(self, tmp_path):
        # A comment mentioning %FOO% must not count as plumbing consumption.
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(
            _CMD_SHIM_TEMPLATE.replace(
                "SETLOCAL\r\n",
                "SET FOO=required\r\n:: shim consumes %FOO% (it does not)\r\nSETLOCAL\r\n",
                1,
            ),
            encoding="utf-8",
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_trailing_line_after_dispatch_refused(self, tmp_path):
        # The dispatch must be the file's final action; a trailing EXIT /b 7
        # would override the child's exit status, which the direct spawn can't
        # reproduce. Applies to both layouts.
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        cmd.write_text(_CMD_SHIM_TEMPLATE + "EXIT /b 7\r\n", encoding="utf-8")
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

        native_cmd, _exe = _make_native_exe_shim(tmp_path)
        native_cmd.write_text(
            _NATIVE_EXE_SHIM_TEMPLATE + "EXIT /b 7\r\n", encoding="utf-8"
        )
        assert browser_tool._windows_cmd_shim_node_target(str(native_cmd)) is None

    def test_literal_prog_dispatch_refused(self, tmp_path):
        # Conscious scope decision: dispatch lines whose program is a literal
        # path rather than a %VAR% (some pnpm/legacy shim generations) are NOT
        # rewritten — Hermes installs agent-browser via npm, whose shims use
        # %_prog%. Refusal is the fail-safe: the .cmd keeps working as before.
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        cmd = bin_dir / "literal.cmd"
        cmd.write_text(
            '@ECHO off\r\n"%~dp0\\node.exe"  "%~dp0\\..\\agent-browser\\bin\\agent-browser.js" %*\r\n',
            encoding="utf-8",
        )
        (bin_dir / "node.exe").write_text("", encoding="utf-8")
        entry = tmp_path / "node_modules" / "agent-browser" / "bin" / "agent-browser.js"
        entry.parent.mkdir(parents=True)
        entry.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None

    def test_bypass_preserves_trailing_prefix_args(self, tmp_path):
        # The npx fallback prefix is [npx.cmd, --ignore-scripts,
        # --prefer-offline, -y, agent-browser@^0.26.0]; every trailing sub-arg
        # must survive the rewrite byte-for-byte (including the ^ in the spec,
        # which cmd.exe would have eaten as its escape character).
        cmd, entry = _make_npx_shim(tmp_path)
        rewritten = browser_tool._bypass_windows_cmd_shim([str(cmd)] + _NPX_PREFIX_TAIL)
        assert rewritten[-len(_NPX_PREFIX_TAIL):] == _NPX_PREFIX_TAIL
        assert not rewritten[0].lower().endswith(".cmd")
        assert os.path.normpath(rewritten[1]) == os.path.normpath(str(entry))

    def test_node_resolved_via_path_when_not_colocated(self, tmp_path, monkeypatch):
        # Drop the colocated node.exe so the merged-PATH fallback runs. The fix
        # resolves "node.exe" specifically (a bare "node" lookup honours PATHEXT
        # and could resolve node.js / node.cmd), so the mock only answers to it.
        cmd, entry = _make_agent_browser_shim(tmp_path)
        (cmd.parent / "node.exe").unlink()
        fake_node = tmp_path / "node.exe"
        fake_node.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            browser_tool.shutil, "which",
            lambda name, path=None: str(fake_node) if name == "node.exe" else None,
        )
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target == [str(fake_node), os.path.normpath(str(entry))]

    def test_bare_node_name_not_used(self, tmp_path, monkeypatch):
        # Resolution must NOT fall through to a bare "node" (PATHEXT could turn
        # that into node.js/node.cmd). If only "node" resolves and "node.exe"
        # does not, there is no target.
        cmd, _entry = _make_agent_browser_shim(tmp_path)
        (cmd.parent / "node.exe").unlink()
        monkeypatch.setattr(
            browser_tool.shutil, "which",
            lambda name, path=None: r"C:\stray\node.js" if name == "node" else None,
        )
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None


@pytest.mark.windows_only
class TestAmpersandUrlNotArgSplit:
    """Unit-level check that the bypass builds a cmd.exe-free argv with the
    '&'-URL intact. The end-to-end proof — a REAL node.exe actually receiving the
    URL as one argument — is in :class:`TestRealNodeArgvRoundTrip` below."""

    def test_ampersand_url_stays_single_arg_after_bypass(self, tmp_path):
        cmd, _entry = _make_agent_browser_shim(tmp_path)

        # Pre-fix prefix is exactly what browser_tool builds: the raw .cmd path.
        buggy_prefix = [str(cmd)]
        # Sanity: the un-bypassed prefix WOULD route through cmd.exe (.cmd) —
        # that's what splits the URL on '&'.  Proves the test is meaningful.
        assert buggy_prefix[0].lower().endswith(".cmd")

        fixed_prefix = browser_tool._bypass_windows_cmd_shim(buggy_prefix)
        # cmd.exe is no longer the spawn target.
        assert not fixed_prefix[0].lower().endswith(".cmd")

        # Reproduce the production argv construction (cmd_prefix + … + args).
        cmd_parts = fixed_prefix + ["--json", "open", X_SEARCH_URL]

        # The '&'-bearing URL is intact and is a single argv element.
        assert cmd_parts.count(X_SEARCH_URL) == 1
        # No fragment of the URL leaked as a separate element — the cmd.exe
        # failure mode would surface 'src=typed_query' / 'f=live' as siblings.
        assert "src=typed_query" not in cmd_parts
        assert "f=live" not in cmd_parts
        # Nothing in the spawn argv is a .cmd shim.
        assert not any(part.lower().endswith(".cmd") for part in cmd_parts)


@pytest.mark.windows_only
class TestWarmCacheSpawnWiring:
    """Production-wiring proof for the warm-cache npx site: the REAL
    ``warm_agent_browser_npx_cache`` must hand ``Popen`` a bypassed argv with
    the ``^`` version spec intact.  Removing the bypass call at that spawn site
    fails this test."""

    def test_warm_cache_spawns_bypassed_node_with_caret_spec(self, tmp_path, monkeypatch):
        cmd, entry = _make_npx_shim(tmp_path)
        monkeypatch.setattr(browser_tool, "_resolve_npx_bin", lambda: str(cmd))

        captured = {}

        class _FakeProc:
            returncode = 0

            def communicate(self, timeout=None):
                return ("0.26.0", "")

        def fake_popen(argv, **kwargs):
            captured["argv"] = list(argv)
            return _FakeProc()

        monkeypatch.setattr(browser_tool.subprocess, "Popen", fake_popen)
        assert browser_tool.warm_agent_browser_npx_cache(timeout=5) is True

        argv = captured["argv"]
        # The .cmd shim is out of the spawn chain...
        assert not argv[0].lower().endswith(".cmd")
        assert os.path.normpath(argv[1]) == os.path.normpath(str(entry))
        # ...and the ^range reaches npx verbatim instead of being eaten by
        # cmd.exe into an exact pin.
        assert browser_tool.AGENT_BROWSER_NPX_SPEC in argv


@pytest.mark.windows_only
@pytest.mark.skipif(
    not shutil.which("node.exe"),
    reason="needs a real Windows node.exe to prove the &-URL argv round-trip",
)
class TestRealNodeArgvRoundTrip:
    """End-to-end proof (the evidence the resolver unit tests above can't give):
    spawn a REAL node.exe through the bypassed prefix exactly as production does
    and confirm the '&'-URL arrives as ONE unmodified ``process.argv`` element."""

    def _make_real_shim(self, tmp_path):
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        cmd = bin_dir / "agent-browser.cmd"
        cmd.write_text(_CMD_SHIM_TEMPLATE, encoding="utf-8")
        entry = tmp_path / "node_modules" / "agent-browser" / "bin" / "agent-browser.js"
        entry.parent.mkdir(parents=True)
        entry.write_text(_ARGV_ECHO_JS, encoding="utf-8")
        return cmd, entry

    def test_ampersand_url_survives_real_node_spawn(self, tmp_path):
        cmd, _entry = self._make_real_shim(tmp_path)
        # Build the argv exactly as _run_browser_command does: bypass, then args.
        argv = browser_tool._bypass_windows_cmd_shim([str(cmd)]) + ["--json", "open", X_SEARCH_URL]
        assert not argv[0].lower().endswith(".cmd")  # cmd.exe is out of the chain
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        seen = json.loads(result.stdout.strip())
        # node received the '&'-URL as exactly one, unmodified argv element.
        assert seen == ["--json", "open", X_SEARCH_URL]
        assert seen.count(X_SEARCH_URL) == 1
        assert not any(frag in seen for frag in ("src=typed_query", "f=live"))

    def test_native_exe_shim_real_argv_round_trip(self, tmp_path):
        # The native-binary variant of the same proof: the bypassed target IS
        # the .exe, and args reach it verbatim. A real node.exe stands in for
        # agent-browser's native binary (it is one), with an argv-echo script
        # as its first argument so the child can report what it received.
        cmd, exe = _make_native_exe_shim(tmp_path, create_exe=False)
        shutil.copyfile(shutil.which("node.exe"), str(exe))
        echo_js = tmp_path / "echo.js"
        echo_js.write_text(_ARGV_ECHO_JS, encoding="utf-8")

        argv = browser_tool._bypass_windows_cmd_shim([str(cmd)])
        assert argv == [os.path.normpath(str(exe))]

        result = subprocess.run(
            argv + [str(echo_js), "--json", "open", X_SEARCH_URL],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        seen = json.loads(result.stdout.strip())
        assert seen == ["--json", "open", X_SEARCH_URL]

    def test_raw_cmd_control_splits_the_url(self, tmp_path):
        # Negative control: the UN-bypassed .cmd (the status quo ante) mis-splits
        # the URL under cmd.exe — proving the fix addresses a real failure, not a
        # no-op. The shim's own node probe falls back to `node` on PATH, which
        # exists whenever node.exe does (this class's skip gate), so the control
        # must produce parseable output — inconclusive is a failure.
        cmd, _entry = self._make_real_shim(tmp_path)
        raw = subprocess.run(
            [str(cmd), "--json", "open", X_SEARCH_URL], capture_output=True, text=True
        )
        out = raw.stdout.strip()
        assert out.startswith("["), (
            f"negative control inconclusive: stdout={out!r} stderr={raw.stderr!r}"
        )
        seen = json.loads(out)
        # The URL was torn at '&': the full URL is NOT present intact.
        assert X_SEARCH_URL not in seen


@pytest.mark.windows_only
class TestWindowsFailSafes:
    """Do-no-harm paths on the genuine host: anything the bypass cannot
    faithfully rewrite keeps its original prefix."""

    def test_non_cmd_prefix_unchanged(self):
        prefix = ["/opt/agent-browser/bin/agent-browser"]
        assert browser_tool._bypass_windows_cmd_shim(prefix) == prefix

    def test_empty_prefix_unchanged(self):
        assert browser_tool._bypass_windows_cmd_shim([]) == []

    def test_missing_entry_js_falls_back_to_original(self, tmp_path, caplog):
        # A .cmd shim whose referenced .js doesn't exist must NOT be rewritten
        # (fail-safe: keep the original prefix rather than spawn a bogus path) —
        # and the un-bypassable shortfall must be logged, not hidden.
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        cmd = bin_dir / "agent-browser.cmd"
        cmd.write_text(_CMD_SHIM_TEMPLATE, encoding="utf-8")
        # NOTE: intentionally do not create the agent-browser.js entry.
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None
        browser_tool._warned_unbypassable_shims.discard(str(cmd))
        with caplog.at_level(logging.WARNING):
            assert browser_tool._bypass_windows_cmd_shim([str(cmd)]) == [str(cmd)]
        assert any("could not bypass" in r.getMessage() for r in caplog.records)


class TestPosixNoop:
    """The one genuinely portable contract: off Windows the bypass is a strict
    no-op.  Runs unmarked so the Linux lane exercises the real posix
    early-return (no host faking — AGENTS.md)."""

    def test_posix_is_noop(self, tmp_path):
        prefix = [str(tmp_path / "node_modules" / ".bin" / "agent-browser.cmd")]
        assert browser_tool._bypass_windows_cmd_shim(prefix) == prefix
        assert browser_tool._windows_cmd_shim_node_target(prefix[0]) is None
