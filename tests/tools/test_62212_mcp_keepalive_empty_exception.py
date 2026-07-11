"""
Regression test for issue #62212 - MCP stdio keepalive probe triggers
infinite reconnect loop on empty exceptions.

The bug: _is_method_not_found_error() returned False on empty str(exc),
causing empty exceptions (silent pipe closure, server doesn't implement
ping, CancelledError with no args) to propagate and trigger an immediate
reconnect. Reconnect spawns new process, also fails, also reconnects —
infinite loop.

The fix: when str(exc) is empty, treat the exception as "method not found"
so the caller falls back to list_tools. If list_tools ALSO fails with empty
message, that triggers a real reconnect (after list_tools has confirmed
the server doesn't speak JSON-RPC for the simple path).

Tests (static + behavioral):
  1. test_static_treats_empty_exception_as_method_not_found:
     Source tripwire - verifies the empty-msg branch returns True.
  2. test_behavioral_empty_exception_falls_back_to_list_tools:
     Behavioral - construct an Exception with no message, verify that
     _is_method_not_found_error returns True.
"""

import re
from pathlib import Path


def test_static_treats_empty_exception_as_method_not_found():
    """Static tripwire: the empty-msg branch in _is_method_not_found_error
    must return True (treating it as method-not-found, falling back to
    list_tools). Fails on unfixed code."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    src = (worktree / "tools" / "mcp_tool.py").read_text()

    # Find _is_method_not_found_error function
    m = re.search(r"def _is_method_not_found_error.*?(?=^def |\Z)",
                  src, re.MULTILINE | re.DOTALL)
    assert m, "_is_method_not_found_error function not found"
    body = m.group(0)

    # Find the empty-msg branch
    assert "if not msg:" in body, (
        "#62212 regression: empty-msg branch missing in _is_method_not_found_error. "
        "When str(exc) is empty (silent pipe closure, server doesn't implement "
        "ping, CancelledError with no args), the exception must NOT propagate "
        "to trigger reconnect — it should fall back to list_tools."
    )

    # The empty-msg branch must return True (fall through to list_tools fallback)
    # Look for the structure: "if not msg:\n... return True"
    pattern = r"if not msg:.*?return True"
    assert re.search(pattern, body, re.DOTALL), (
        "#62212: the empty-msg branch in _is_method_not_found_error must "
        "return True (treating it as method-not-found → list_tools fallback). "
        "Returning False causes the empty exception to propagate and "
        "trigger immediate reconnect → infinite loop."
    )


def test_behavioral_empty_exception_falls_back_to_list_tools():
    """Behavioral: construct an Exception with no message, verify that
    _is_method_not_found_error returns True. The fix should make empty
    exceptions fall back to list_tools instead of triggering reconnect."""
    import sys
    sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")

    from tools.mcp_tool import _is_method_not_found_error

    # Empty exception
    exc = Exception()
    assert str(exc) == ""
    assert _is_method_not_found_error(exc) is True, (
        f"#62212: _is_method_not_found_error(Exception()) returned "
        f"{_is_method_not_found_error(exc)!r}, expected True. "
        f"Empty exception should be treated as method-not-found → list_tools fallback."
    )

    # Exception with explicit empty args
    exc2 = Exception("")
    assert str(exc2) == ""
    assert _is_method_not_found_error(exc2) is True

    # CancelledError with no message (typical in stdio pipe close)
    import asyncio
    exc3 = asyncio.CancelledError()
    assert str(exc3) == ""
    assert _is_method_not_found_error(exc3) is True, (
        "#62212: empty CancelledError should also fall back to list_tools."
    )

    # Real exception with message: should NOT return True
    exc4 = Exception("connection refused")
    assert _is_method_not_found_error(exc4) is False, (
        "#62212: exception WITH a message should NOT be treated as method-not-found. "
        "Only EMPTY messages should fall through to list_tools."
    )

    # Real "method not found" exception: SHOULD return True (existing behavior)
    exc5 = Exception("Method not found: ping")
    assert _is_method_not_found_error(exc5) is True