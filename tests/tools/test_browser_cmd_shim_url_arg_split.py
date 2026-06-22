"""Regression tests for the Windows ``.cmd``-shim URL arg-splitting bug.

``tools/browser_tool.py`` resolves the agent-browser CLI to an npm ``.cmd``
batch shim on Windows (``node_modules/.bin/agent-browser.cmd`` or, for the npx
fallback, ``npx.cmd``).  Those shims run under ``cmd.exe`` and end their
dispatch line with an unquoted ``%*``, so any argument containing a cmd
metacharacter — notably ``&`` in a real search URL like
``https://x.com/search?q=from%3ACityBureaucrat&src=typed_query&f=live`` — gets
re-parsed by cmd.exe into separate "commands", silently breaking navigation
(``'src' is not recognized as an internal or external command``).

The fix (``_bypass_windows_cmd_shim`` / ``_windows_cmd_shim_node_target``)
rewrites a ``.cmd`` prefix to a direct ``node.exe <entry>.js`` invocation,
taking cmd.exe out of the spawn chain so ``subprocess.Popen`` hands the argv to
the child verbatim.

The functional resolution tests are Windows-gated: they read real on-disk shim
fixtures whose paths use Windows backslash separators, which only resolve under
``ntpath`` (i.e. on a real Windows host).  Portable no-op tests and a
source-level guard keep the fix covered on the Linux CI runner too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tools.browser_tool as browser_tool


X_SEARCH_URL = "https://x.com/search?q=from%3ACityBureaucrat&src=typed_query&f=live"

_WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt",
    reason="exercises the real ntpath-coupled .cmd-shim resolver (Windows-only bug)",
)


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

# npm's own npx.cmd format — the JS path is indirected through a SET'd variable
# rather than written literally on the dispatch line.  Note it also references a
# *different* %~dp0 .js (npm-prefix.js) the resolver must NOT pick.
_NPX_SHIM_TEMPLATE = (
    ":: Created by npm, please don't edit manually.\r\n"
    "@ECHO OFF\r\n"
    "SETLOCAL\r\n"
    'SET "NODE_EXE=%~dp0\\node.exe"\r\n'
    'IF NOT EXIST "%NODE_EXE%" (\r\n'
    '  SET "NODE_EXE=node"\r\n'
    ")\r\n"
    'SET "NPM_PREFIX_JS=%~dp0\\node_modules\\npm\\bin\\npm-prefix.js"\r\n'
    'SET "NPX_CLI_JS=%~dp0\\node_modules\\npm\\bin\\npx-cli.js"\r\n'
    '"%NODE_EXE%" "%NPX_CLI_JS%" %*\r\n'
)


@pytest.fixture
def _force_windows(monkeypatch):
    """Force the ``os.name == "nt"`` branch (no-op on a real Windows host)."""
    monkeypatch.setattr(browser_tool.os, "name", "nt")


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


@_WINDOWS_ONLY
class TestCmdShimBypass:
    def test_agent_browser_cmd_resolves_to_node_and_js(self, tmp_path, _force_windows):
        cmd, entry = _make_agent_browser_shim(tmp_path)
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target is not None
        # cmd.exe is out of the chain: the executable is node, not the .cmd.
        assert not target[0].lower().endswith(".cmd")
        assert os.path.basename(target[0]).lower().startswith("node")
        assert os.path.normpath(target[1]) == os.path.normpath(str(entry))

    def test_npx_cmd_resolves_to_npx_cli_js(self, tmp_path, _force_windows):
        cmd, entry = _make_npx_shim(tmp_path)
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target is not None
        assert not target[0].lower().endswith(".cmd")
        # Must pick npx-cli.js, NOT the npm-prefix.js that also appears in the
        # shim text.
        assert os.path.normpath(target[1]) == os.path.normpath(str(entry))

    def test_bypass_preserves_trailing_prefix_args(self, tmp_path, _force_windows):
        # The npx fallback prefix is [npx.cmd, "agent-browser"]; the literal
        # "agent-browser" sub-arg must survive the rewrite.
        cmd, entry = _make_npx_shim(tmp_path)
        rewritten = browser_tool._bypass_windows_cmd_shim([str(cmd), "agent-browser"])
        assert rewritten[-1] == "agent-browser"
        assert not rewritten[0].lower().endswith(".cmd")
        assert os.path.normpath(rewritten[1]) == os.path.normpath(str(entry))

    def test_node_resolved_via_path_when_not_colocated(self, tmp_path, _force_windows, monkeypatch):
        # Drop the colocated node.exe so the PATH fallback (shutil.which) runs.
        cmd, entry = _make_agent_browser_shim(tmp_path)
        (cmd.parent / "node.exe").unlink()
        fake_node = tmp_path / "node.exe"
        fake_node.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            browser_tool.shutil, "which",
            lambda name, path=None: str(fake_node) if name == "node" else None,
        )
        target = browser_tool._windows_cmd_shim_node_target(str(cmd))
        assert target == [str(fake_node), os.path.normpath(str(entry))]


@_WINDOWS_ONLY
class TestAmpersandUrlNotArgSplit:
    """The actual bug: an '&'-bearing URL must reach the child as ONE argv
    element, with cmd.exe out of the spawn chain."""

    def test_ampersand_url_stays_single_arg_after_bypass(self, tmp_path, _force_windows):
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


class TestPosixAndFallbacks:
    """Portable across platforms — guards the do-no-harm / fail-safe paths."""

    def test_posix_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(browser_tool.os, "name", "posix")
        prefix = [str(tmp_path / "node_modules" / ".bin" / "agent-browser.cmd")]
        assert browser_tool._bypass_windows_cmd_shim(prefix) == prefix
        assert browser_tool._windows_cmd_shim_node_target(prefix[0]) is None

    def test_non_cmd_prefix_unchanged(self, _force_windows):
        prefix = ["/opt/agent-browser/bin/agent-browser"]
        assert browser_tool._bypass_windows_cmd_shim(prefix) == prefix

    def test_empty_prefix_unchanged(self, _force_windows):
        assert browser_tool._bypass_windows_cmd_shim([]) == []

    def test_missing_entry_js_falls_back_to_original(self, tmp_path, _force_windows):
        # A .cmd shim whose referenced .js doesn't exist must NOT be rewritten
        # (fail-safe: keep the original prefix rather than spawn a bogus path).
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        cmd = bin_dir / "agent-browser.cmd"
        cmd.write_text(_CMD_SHIM_TEMPLATE, encoding="utf-8")
        # NOTE: intentionally do not create the agent-browser.js entry.
        assert browser_tool._windows_cmd_shim_node_target(str(cmd)) is None
        assert browser_tool._bypass_windows_cmd_shim([str(cmd)]) == [str(cmd)]


class TestBypassWiredAtSpawnSites:
    """Source-level guard (runs on Linux CI too): the bypass must stay applied
    to ``cmd_prefix`` at BOTH agent-browser spawn sites so a future refactor
    can't silently re-expose the &-URL splitting bug."""

    def test_both_spawn_prefixes_apply_bypass(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "tools" / "browser_tool.py").read_text(encoding="utf-8")
        assert src.count("_bypass_windows_cmd_shim(cmd_prefix)") >= 2, (
            "both browser_tool spawn sites must run cmd_prefix through "
            "_bypass_windows_cmd_shim() before building the argv"
        )
