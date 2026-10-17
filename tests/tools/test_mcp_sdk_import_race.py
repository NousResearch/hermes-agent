"""Regression for #124357: MCP SDK fast path must wait for HTTP flags."""

from __future__ import annotations

import importlib
import threading
import time
import types

from tools import mcp_tool

_SDK_GLOBALS = (
    "_MCP_SDK_IMPORT_ATTEMPTED",
    "_MCP_AVAILABLE",
    "ClientSession",
    "_MCP_HTTP_AVAILABLE",
    "_MCP_NEW_HTTP",
    "_MCP_LEGACY_HTTP",
)


def _patch_sdk_globals(monkeypatch, **values):
    for name in _SDK_GLOBALS:
        if name in values:
            monkeypatch.setattr(mcp_tool, name, values[name])


def test_ensure_mcp_sdk_fast_path_waits_for_http_flags(monkeypatch):
    """A second thread must block on the lock until HTTP/SSE flags are set."""
    _patch_sdk_globals(
        monkeypatch,
        _MCP_SDK_IMPORT_ATTEMPTED=False,
        _MCP_AVAILABLE=True,
        ClientSession=None,
        _MCP_HTTP_AVAILABLE=False,
        _MCP_NEW_HTTP=False,
        _MCP_LEGACY_HTTP=False,
    )

    real_import = importlib.import_module
    started = threading.Event()
    release = threading.Event()

    def slow_import(name, package=None):
        if name == "mcp.client.streamable_http":
            started.set()
            if not release.wait(5):
                raise TimeoutError("HTTP import not released")
        return real_import(name, package)

    monkeypatch.setattr(
        mcp_tool,
        "importlib",
        types.SimpleNamespace(import_module=slow_import, util=importlib.util),
    )

    holder_result = {}
    waiter_result = {}
    holder_thread = threading.Thread(
        target=lambda: holder_result.update(
            available=mcp_tool._ensure_mcp_sdk(),
            http=mcp_tool._MCP_HTTP_AVAILABLE,
        )
    )
    waiter_thread = threading.Thread(
        target=lambda: waiter_result.update(
            available=mcp_tool._ensure_mcp_sdk(),
            http=mcp_tool._MCP_HTTP_AVAILABLE,
        )
    )

    try:
        holder_thread.start()
        assert started.wait(5), "holder never reached HTTP import"

        waiter_thread.start()
        time.sleep(0.2)
        assert waiter_thread.is_alive(), "waiter took the unlocked ClientSession fast path"
        assert mcp_tool._MCP_HTTP_AVAILABLE is False
        assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False
    finally:
        release.set()
        holder_thread.join(timeout=5)
        waiter_thread.join(timeout=5)

    assert not holder_thread.is_alive()
    assert not waiter_thread.is_alive()

    assert holder_result["available"] is True
    assert holder_result["http"] is True
    assert waiter_result["available"] is True
    assert waiter_result["http"] is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is True


def test_ensure_mcp_sdk_skips_reimport_for_preinstalled_mock(monkeypatch):
    """Tests that bind ClientSession before first use must not re-import."""
    _patch_sdk_globals(
        monkeypatch,
        _MCP_SDK_IMPORT_ATTEMPTED=False,
        _MCP_AVAILABLE=True,
        ClientSession=object(),
    )
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False
