"""Pre-flight guard for the cua-driver MCP server.

Card kn716mkjxbyzzxgsnf0a7hvexn87dbc4. Structural follow-on to the
SOUL.md Rule #2 discipline patch.

When the agent calls ``mcp_cua_driver_launch_app`` with:
  - a ``urls`` argument containing a path that doesn't exist on disk, or
  - a ``bundle_id`` / ``name`` that doesn't resolve to an installed app,

macOS NSWorkspace pops a persistent Finder / TextEdit "file not found"
dialog (or a launch-failure dialog) that the user has to dismiss
manually. Hermes intercepts these in ``tools/mcp_tool.py`` and returns a
structured error WITHOUT invoking the MCP server, so no dialog appears.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def _install_stub_server(mcp_tool_module, name, call_tool_impl):
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session
    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name):
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)


def test_launch_app_missing_file_returns_structured_error(monkeypatch, tmp_path):
    """launch_app(urls=['/tmp/does-not-exist.md']) must short-circuit
    before reaching the MCP server, so macOS never sees the bad path
    and never pops a file-not-found dialog.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_should_not_run(*a, **kw):  # pragma: no cover
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    _install_stub_server(mcp_tool, "cua-driver", _call_tool_should_not_run)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("cua-driver", "launch_app", 10.0)
        bogus = str(tmp_path / "does-not-exist.md")
        result = handler({"urls": [bogus]})
        parsed = json.loads(result)
        assert parsed.get("error") == "FILE_NOT_FOUND", parsed
        assert parsed.get("path") == bogus, parsed
        assert call_count["n"] == 0, (
            "MCP server must NOT be invoked when pre-flight catches a "
            "missing path. NSWorkspace dialog suppression depends on it."
        )

        # Sanity: a real path (tmp_path itself exists) still dispatches.
        result = handler({"urls": [str(tmp_path)]})
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        assert call_count["n"] == 1, parsed
    finally:
        _cleanup(mcp_tool, "cua-driver")


def test_launch_app_bogus_bundle_id_returns_structured_error(monkeypatch, tmp_path):
    """launch_app(bundle_id='com.nonexistent.fake') must short-circuit
    before reaching the MCP server.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_should_not_run(*a, **kw):  # pragma: no cover
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    # Force mdfind to return "no installed app" deterministically.
    class _FakeResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, *a, **kw):
        assert cmd[0] == "mdfind"
        return _FakeResult()

    monkeypatch.setattr(mcp_tool.subprocess, "run", _fake_run)

    _install_stub_server(mcp_tool, "cua-driver", _call_tool_should_not_run)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("cua-driver", "launch_app", 10.0)
        result = handler({"bundle_id": "com.nonexistent.hermes-test"})
        parsed = json.loads(result)
        assert parsed.get("error") == "APP_NOT_INSTALLED", parsed
        assert parsed.get("bundle_id") == "com.nonexistent.hermes-test", parsed
        assert call_count["n"] == 0, (
            "MCP server must NOT be invoked when pre-flight catches an "
            "uninstalled bundle_id."
        )
    finally:
        _cleanup(mcp_tool, "cua-driver")


def test_focus_app_bogus_bundle_id_returns_structured_error(monkeypatch, tmp_path):
    """focus_app(bundle_id='com.nonexistent.fake') must short-circuit
    before reaching the MCP server.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_should_not_run(*a, **kw):  # pragma: no cover
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    class _FakeResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, *a, **kw):
        assert cmd[0] == "mdfind"
        return _FakeResult()

    monkeypatch.setattr(mcp_tool.subprocess, "run", _fake_run)

    _install_stub_server(mcp_tool, "cua-driver", _call_tool_should_not_run)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("cua-driver", "focus_app", 10.0)
        result = handler({"bundle_id": "com.nonexistent.hermes-test"})
        parsed = json.loads(result)
        assert parsed.get("error") == "APP_NOT_INSTALLED", parsed
        assert parsed.get("bundle_id") == "com.nonexistent.hermes-test", parsed
        assert call_count["n"] == 0, (
            "MCP server must NOT be invoked when focus_app pre-flight "
            "catches an uninstalled bundle_id."
        )
    finally:
        _cleanup(mcp_tool, "cua-driver")


def test_launch_app_http_urls_pass_through(monkeypatch, tmp_path):
    """URLs (http/https) must bypass the path-existence check and
    dispatch to the MCP server normally — that's the happy browser path.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_ok(*a, **kw):
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structuredContent = None
        return result

    _install_stub_server(mcp_tool, "cua-driver", _call_tool_ok)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("cua-driver", "launch_app", 10.0)
        result = handler({"urls": ["https://example.com", "about:blank"]})
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        assert call_count["n"] == 1
    finally:
        _cleanup(mcp_tool, "cua-driver")


def test_preflight_does_not_fire_for_other_servers(monkeypatch, tmp_path):
    """The guard is scoped to server_name == 'cua-driver'. Other MCP
    servers must not be affected even if a tool happens to be named
    launch_app.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_ok(*a, **kw):
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structuredContent = None
        return result

    _install_stub_server(mcp_tool, "other-server", _call_tool_ok)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("other-server", "launch_app", 10.0)
        bogus = str(tmp_path / "does-not-exist.md")
        result = handler({"urls": [bogus]})
        parsed = json.loads(result)
        assert "error" not in parsed, parsed
        assert call_count["n"] == 1
    finally:
        _cleanup(mcp_tool, "other-server")
