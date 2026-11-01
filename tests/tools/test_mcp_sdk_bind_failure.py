"""A broken MCP SDK import chain must surface its real cause and stay retryable (#137438).

``import mcp`` failing inside its dependency chain (e.g. a compiled wheel built for another
interpreter) used to be swallowed into a misleading ``AttributeError: module 'tools.mcp_tool'
has no attribute ...`` logged at debug level, and ``_MCP_SDK_IMPORT_ATTEMPTED`` latched the
failure for the process lifetime so every stdio MCP server parked permanently at boot. The
bind must instead warn once with the real traceback and retry on next use.
"""

import logging

import pytest

from tools import mcp_tool


@pytest.fixture
def broken_sdk_import(monkeypatch):
    """Point the SDK bind at an interpreter whose ``mcp`` import chain is broken.

    The autouse conftest fixture materializes the real SDK first (``ClientSession`` bound,
    ``_MCP_SDK_IMPORT_ATTEMPTED`` True), which would short-circuit ``_ensure_mcp_sdk`` before
    the import is even attempted; reset that state and record import calls. Everything is
    monkeypatched, so teardown restores the successfully-bound module state.
    """
    if not mcp_tool._MCP_AVAILABLE:
        pytest.skip("mcp SDK not installed")
    real_import_module = mcp_tool.importlib.import_module
    calls = []
    state = {"broken": True}

    def fake_import_module(module):
        calls.append(module)
        if state["broken"]:
            raise ImportError("No module named 'pydantic_core._pydantic_core'")
        return real_import_module(module)

    # Patch the import first: delattr's existence probe below would otherwise go through
    # the PEP-562 __getattr__ and rebind the real SDK before the fake import is in place.
    monkeypatch.setattr(mcp_tool.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(mcp_tool, "ClientSession", None)
    monkeypatch.setattr(mcp_tool, "_MCP_SDK_IMPORT_ATTEMPTED", False)
    monkeypatch.setattr(mcp_tool, "_MCP_SDK_IMPORT_ERROR", None)
    for name in mcp_tool._MCP_SDK_LAZY_SYMBOLS:  # force the PEP-562 __getattr__ path
        if name in mcp_tool.__dict__:
            monkeypatch.delattr(mcp_tool, name)
    return calls, state


def test_broken_import_warns_once_with_real_cause(broken_sdk_import, caplog):
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        assert mcp_tool._ensure_mcp_sdk() is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "MCP SDK import failed" in message
    assert "pydantic_core" in message  # the real cause, not the absent-attribute symptom
    assert warnings[0].exc_info is not None  # with its traceback
    assert isinstance(mcp_tool._MCP_SDK_IMPORT_ERROR, ImportError)


def test_broken_import_does_not_latch_and_retries_quietly(broken_sdk_import, caplog):
    calls, _state = broken_sdk_import
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        mcp_tool._ensure_mcp_sdk()
        imports_after_first = len(calls)
        mcp_tool._ensure_mcp_sdk()
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is False  # not latched: bind stays retryable
    assert len(calls) > imports_after_first  # second use actually retried the import
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1  # no spam


def test_getattr_chains_the_real_import_error(broken_sdk_import):
    mcp_tool._ensure_mcp_sdk()
    with pytest.raises(AttributeError) as excinfo:
        mcp_tool.StdioServerParameters
    assert "pydantic_core" in str(excinfo.value.__cause__)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_retry_after_fix_rebinds_and_clears_the_error(broken_sdk_import):
    calls, state = broken_sdk_import
    assert mcp_tool._ensure_mcp_sdk() is False
    state["broken"] = False  # environment fixed (correct interpreter, repaired wheel)
    assert mcp_tool._ensure_mcp_sdk() is True
    assert mcp_tool._MCP_SDK_IMPORT_ERROR is None
    assert mcp_tool.ClientSession is not None
    assert mcp_tool._MCP_SDK_IMPORT_ATTEMPTED is True
    assert len(calls) > 1  # and it genuinely was a retry, not a first-time import
