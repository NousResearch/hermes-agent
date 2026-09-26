"""Regression for #124357: MCP SDK fast path must wait for HTTP flags."""

from __future__ import annotations

import importlib
import threading
import time
import types

from tools import mcp_tool


def test_ensure_mcp_sdk_fast_path_waits_for_http_flags(monkeypatch):
    """A second thread must block on the lock until HTTP/SSE flags are set."""
    mcp_tool._MCP_SDK_IMPORT_ATTEMPTED = False
    mcp_tool._MCP_AVAILABLE = True
    mcp_tool.ClientSession = None
    mcp_tool._MCP_HTTP_AVAILABLE = False
    mcp_tool._MCP_NEW_HTTP = False
    mcp_tool._MCP_LEGACY_HTTP = False

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

    def holder():
        holder_result["available"] = mcp_tool._ensure_mcp_sdk()
        holder_result["http"] = mcp_tool._MCP_HTTP_AVAILABLE

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert started.wait(5), "holder never reached HTTP import"

    waiter_result = {}

    def waiter():
        waiter_result["available"] = mcp_tool._ensure_mcp_sdk()
        waiter_result["http"] = mcp_tool._MCP_HTTP_AVAILABLE

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    time.sleep(0.2)
    assert waiter_thread.is_alive(), "waiter took the unlocked ClientSession fast path"
    assert mcp_tool._MCP_HTTP_AVAILABLE is False
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False

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


def test_ensure_mcp_sdk_skips_reimport_for_preinstalled_mock():
    """Tests that bind ClientSession before first use must not re-import."""
    mcp_tool._MCP_SDK_IMPORT_ATTEMPTED = False
    mcp_tool._MCP_AVAILABLE = True
    mcp_tool.ClientSession = object()
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False
