"""The lazy MCP SDK probe never reports HTTP missing while the SDK is installed.

Regression for #124357 and #134933. ``_ensure_mcp_sdk()`` imports the SDK on first use
and publishes the transport flags. Two faults made a healthy install fail every HTTP MCP
server with ``requires HTTP transport but mcp.client.streamable_http is not available``:

* its unlocked fast path returned "ready" as soon as ``ClientSession`` was bound, before
  the lock holder had set ``_MCP_HTTP_AVAILABLE``, so a second thread connected with the
  pre-import default (False);
* a transport import that failed once (a racing import thread, a dependency mid-swap) was
  cached as "unavailable" for the process lifetime, so no reconnect could recover.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
import types

import pytest

from hermes_agent.tools import mcp_tool

# What CPython raises when two threads import the SDK's circular modules at once; a
# RuntimeError, not an ImportError.
_DeadlockError = importlib._bootstrap._DeadlockError

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp SDK not installed")

_PROBE_STATE = ("_MCP_SDK_IMPORT_ATTEMPTED", "_MCP_AVAILABLE", "ClientSession", "_MCP_HTTP_AVAILABLE",
                "_MCP_NEW_HTTP", "_MCP_LEGACY_HTTP", "sse_client", "_MCP_SDK_PROBED_SESSION",
                "_MCP_HTTP_IMPORT_ERROR", "_MCP_SDK_IMPORT_ERROR", "LATEST_HANDSHAKE_VERSION",
                "_MCP_SAMPLING_TYPES", "_MCP_ELICITATION_TYPES", "_MCP_NOTIFICATION_TYPES")


@pytest.fixture
def cold_probe(monkeypatch):
    """Reset the probe to its never-ran state; monkeypatch restores the real values after."""
    for name in _PROBE_STATE:
        monkeypatch.setattr(mcp_tool, name, getattr(mcp_tool, name), raising=False)
    monkeypatch.setattr(mcp_tool, "_MCP_SDK_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "ClientSession", None)
    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", False)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", False)
    monkeypatch.setattr(mcp_tool, "_MCP_LEGACY_HTTP", False)
    monkeypatch.setattr(mcp_tool, "sse_client", None)
    monkeypatch.setattr(mcp_tool, "_MCP_SDK_PROBED_SESSION", None)
    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_IMPORT_ERROR", None)
    monkeypatch.setattr(mcp_tool, "_MCP_SDK_IMPORT_ERROR", None)
    return monkeypatch


def _patch_import(monkeypatch, hook):
    """Route the probe's ``importlib.import_module`` through ``hook(name, real_import)``. Only
    ``hermes_agent.tools.mcp_tool``'s own ``importlib`` name is swapped, never the shared module."""
    real_import = importlib.import_module
    monkeypatch.setattr(mcp_tool, "importlib", types.SimpleNamespace(
        util=importlib.util,
        import_module=lambda name, package=None: hook(name, lambda: real_import(name, package))))


def test_concurrent_caller_sees_http_flag_set_by_lock_holder(cold_probe):
    """A thread that calls the probe while another is mid-import must not be told
    "ready" before the HTTP flag is published."""
    in_http_import, release = threading.Event(), threading.Event()
    # Set once the second caller either reaches the probe lock or returns without it. Releasing
    # the holder only after that makes the race deterministic on a busy runner.
    waiter_progressed = threading.Event()
    real_lock = mcp_tool._MCP_SDK_IMPORT_LOCK

    class _ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "waiter":
                waiter_progressed.set()
            return real_lock.__enter__()

        def __exit__(self, *exc_info):
            return real_lock.__exit__(*exc_info)

    def hook(name, real):
        if name == "mcp.client.streamable_http" and threading.current_thread().name == "holder":
            in_http_import.set()
            assert release.wait(5), "test never released the HTTP import"
        return real()

    _patch_import(cold_probe, hook)
    cold_probe.setattr(mcp_tool, "_MCP_SDK_IMPORT_LOCK", _ObservedLock())
    holder = threading.Thread(target=mcp_tool._ensure_mcp_sdk, name="holder")
    holder.start()
    assert in_http_import.wait(5)

    seen = {}
    def waiter():
        try:
            seen["ready"] = mcp_tool._ensure_mcp_sdk()
            seen["http"] = mcp_tool._MCP_HTTP_AVAILABLE
        finally:
            waiter_progressed.set()
    waiter_thread = threading.Thread(target=waiter, name="waiter")
    waiter_thread.start()
    assert waiter_progressed.wait(5), "second caller never reached the probe"
    release.set()
    holder.join(5)
    waiter_thread.join(5)

    assert not holder.is_alive() and not waiter_thread.is_alive()
    assert seen == {"ready": True, "http": True}


def test_racing_transport_import_is_retried_in_place(cold_probe):
    """A concurrent import's _DeadlockError clears within milliseconds: the probe retries it
    in the same call, so the caller that hit the race still sees HTTP available."""
    failures = {"left": 2}

    def hook(name, real):
        if name == "mcp.client.streamable_http" and failures["left"]:
            failures["left"] -= 1
            raise _DeadlockError("deadlock detected by _ModuleLock('mcp.client')")
        return real()

    _patch_import(cold_probe, hook)
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is True


def test_failed_http_transport_import_is_retried(cold_probe):
    """A transport import that fails for a reason other than the SDK lacking it (a dependency
    mid-swap during an update) must not latch HTTP off: the next probe imports it again."""
    broken = {"on": True}

    def hook(name, real):
        if name == "mcp.client.streamable_http" and broken["on"]:
            raise ModuleNotFoundError("No module named 'httpx2'", name="httpx2")
        return real()

    _patch_import(cold_probe, hook)
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is False
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False

    broken["on"] = False
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
    assert mcp_tool._MCP_NEW_HTTP is True
    assert mcp_tool._MCP_HTTP_IMPORT_ERROR is None


def test_racing_core_import_is_retried_in_place(cold_probe):
    """A concurrent import's _DeadlockError on the core ``mcp`` import clears within
    milliseconds: the probe retries it in the same call instead of reporting no SDK (which
    would leave a gateway that lost the startup race with zero MCP tools)."""
    failures = {"left": 1}

    def hook(name, real):
        if name == "mcp" and failures["left"]:
            failures["left"] -= 1
            raise _DeadlockError("deadlock detected by _ModuleLock('mcp')")
        return real()

    _patch_import(cold_probe, hook)
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is True


def test_persistent_core_import_failure_warns_and_recovers(cold_probe, caplog):
    """A core import failure that is not definitive reports the SDK unusable for that call
    (with a warning, not a silent debug line) and is probed again on the next call."""
    broken = {"on": True}

    def hook(name, real):
        if name == "mcp" and broken["on"]:
            raise ModuleNotFoundError("No module named 'pydantic_core'", name="pydantic_core")
        return real()

    _patch_import(cold_probe, hook)
    with caplog.at_level("WARNING", logger="hermes_agent.tools.mcp_tool"):
        assert mcp_tool._ensure_mcp_sdk() is False
    assert "pydantic_core" in caplog.text
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False

    broken["on"] = False
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool.ClientSession is not None
    assert mcp_tool._MCP_HTTP_AVAILABLE is True


def test_attribute_missing_on_partially_initialized_module_is_retried(cold_probe):
    """A deadlocked import hands back the partially initialized module; a name missing from
    it is transient even if the other thread finishes initializing before classification."""
    real_import = importlib.import_module
    calls = {"n": 0}

    class _Spec:
        _initializing = True

    class _PartialModule(types.ModuleType):
        def __getattr__(self, attr):
            # The other thread completes the module right after this lookup fails.
            self.__spec__._initializing = False
            raise AttributeError(attr)

    def hook(name, real):
        if name == "mcp.client.streamable_http":
            calls["n"] += 1
            if calls["n"] == 1:
                partial = _PartialModule(name)
                partial.__spec__ = _Spec()
                return partial
        return real()

    _patch_import(cold_probe, hook)
    cold_probe.setitem(importlib.sys.modules, "mcp.client.streamable_http",
                       real_import("mcp.client.streamable_http"))
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False, "a partial-init miss was latched as definitive"

    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
    assert mcp_tool._MCP_NEW_HTTP is True


def test_reprobe_never_downgrades_a_value_an_earlier_pass_bound(cold_probe):
    """An unsettled probe re-runs; a transient failure on that later pass must not reset a
    transport an earlier pass already loaded."""
    sse_broken = {"on": True}
    http_broken = {"on": False}

    def hook(name, real):
        if name == "mcp.client.sse" and sse_broken["on"]:
            raise ModuleNotFoundError("No module named 'sse_starlette'", name="sse_starlette")
        if name == "mcp.client.streamable_http" and http_broken["on"]:
            raise ModuleNotFoundError("No module named 'httpx2'", name="httpx2")
        return real()

    _patch_import(cold_probe, hook)
    assert mcp_tool._ensure_mcp_sdk() is True
    handshake = mcp_tool.LATEST_HANDSHAKE_VERSION
    assert mcp_tool._MCP_HTTP_AVAILABLE is True and mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False

    sse_broken["on"], http_broken["on"] = False, True
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
    assert mcp_tool._MCP_NEW_HTTP is True
    assert mcp_tool.sse_client is not None
    assert mcp_tool.LATEST_HANDSHAKE_VERSION == handshake


@pytest.mark.parametrize("runner, config", [
    ("_run_http", {"url": "https://example.com/mcp"}),
    ("_run_stdio", {"command": "true"}),
])
def test_transport_error_names_a_failed_core_import(cold_probe, runner, config):
    """When the core SDK import fails, the connect error says so and carries the cause instead
    of claiming a missing HTTP transport or an uninstalled SDK."""
    import asyncio

    def hook(name, real):
        if name == "mcp":
            raise ModuleNotFoundError("No module named 'pydantic_core'", name="pydantic_core")
        return real()

    _patch_import(cold_probe, hook)
    server = mcp_tool.MCPServerTask("remote")
    with pytest.raises(ImportError) as excinfo:
        asyncio.run(getattr(server, runner)(config))
    message = str(excinfo.value)
    assert "pydantic_core" in message
    assert "requires HTTP transport" not in message
    assert "not installed" not in message


def test_settled_probe_does_not_reimport(cold_probe):
    """Once every transport has loaded, later calls take the fast path (no import cost)."""
    assert mcp_tool._ensure_mcp_sdk() is True
    calls = []
    _patch_import(cold_probe, lambda name, real: calls.append(name) or real())
    for _ in range(3):
        assert mcp_tool._ensure_mcp_sdk() is True
    assert calls == []


def test_sdk_httpx_survives_a_racing_transport_import(monkeypatch):
    """``sdk_httpx()`` must not raise on a concurrent import's _DeadlockError, nor cache the
    fallback it picked then: the transport module's own httpx wins once it loads."""
    monkeypatch.setattr(mcp_tool, "_SDK_HTTPX_MOD", None)
    real_import = importlib.import_module
    failures = {"left": 1}

    def fake_import(name, *args, **kwargs):
        if name == "mcp.client.streamable_http" and failures["left"]:
            failures["left"] -= 1
            raise _DeadlockError("deadlock detected by _ModuleLock('mcp.client')")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(mcp_tool, "importlib", types.SimpleNamespace(util=importlib.util, import_module=fake_import))
    transport_httpx = real_import("mcp.client.streamable_http").httpx2

    assert mcp_tool.sdk_httpx() is real_import("httpx2")
    assert mcp_tool._SDK_HTTPX_MOD is None
    assert mcp_tool.sdk_httpx() is transport_httpx
    assert mcp_tool._SDK_HTTPX_MOD is transport_httpx


def test_http_unavailable_error_names_the_import_failure(cold_probe):
    """The "requires HTTP transport" error carries the import's own error, not only advice to
    upgrade a package that is already installed."""
    import asyncio

    def hook(name, real):
        if name == "mcp.client.streamable_http":
            raise ModuleNotFoundError("No module named 'httpx2'", name="httpx2")
        return real()

    _patch_import(cold_probe, hook)
    server = mcp_tool.MCPServerTask("remote")
    with pytest.raises(ImportError, match=r"No module named 'httpx2'"):
        asyncio.run(server._run_http({"url": "https://example.com/mcp"}))


def test_attribute_error_raised_while_importing_is_retried(cold_probe):
    """An AttributeError from the transport module's own body (a dependency mid-swap during an
    update) is not "this SDK build lacks the name": the next probe imports it again."""
    broken = {"on": True}

    def hook(name, real):
        if name == "mcp.client.streamable_http" and broken["on"]:
            raise AttributeError("module 'httpx2' has no attribute 'AsyncClient'")
        return real()

    _patch_import(cold_probe, hook)
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False

    broken["on"] = False
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_HTTP_AVAILABLE is True
