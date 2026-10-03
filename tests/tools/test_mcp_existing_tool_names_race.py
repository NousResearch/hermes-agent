"""Race regression: MCP registry mutation while `_existing_tool_names` iterates unscoped.

The unscoped branch of ``_existing_tool_names`` iterated ``mcp_tool._servers`` without
``mcp_tool._lock`` while every other reader (the scoped branch, teardown, adoption)
mutates it under the lock. A server disconnecting mid-iteration raised
``RuntimeError: dictionary changed size during iteration`` out of
``register_mcp_servers``, which the agent loop surfaced as a discovery failure even
though the tools of every surviving server were registered fine.

Regression contract: a concurrent ``_servers.pop`` during the unscoped enumeration
must not raise — the surviving servers' tool names are returned (a torn-down server's
tools disappearing from the result is expected).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from tools import mcp_tool_registration as _mcp_registration


def test_unscoped_existing_tool_names_survives_concurrent_server_teardown():
    import tools.mcp_tool as mcp_tool

    reached_doomed = threading.Event()
    may_continue = threading.Event()

    class _PausingServer(SimpleNamespace):
        """Its ``_registered_tool_names`` read pauses the reader mid-iteration."""

        @property
        def _registered_tool_names(self):  # noqa: N805 - SimpleNamespace field shadow
            reached_doomed.set()
            may_continue.wait(timeout=5)
            return ["mcp__doomed__tool"]

    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_lazy = dict(mcp_tool._lazy_server_tool_names)
        mcp_tool._servers.clear()
        mcp_tool._lazy_server_tool_names.clear()
        # Two live servers: popping the doomed one mid-iteration must not break the reader.
        mcp_tool._servers["survivor"] = SimpleNamespace(  # type: ignore[assignment]
            name="survivor", _registered_tool_names=["mcp__survivor__tool"]
        )
        mcp_tool._servers["doomed"] = _PausingServer(name="doomed")  # type: ignore[assignment]

    try:
        result: list[str] = []
        error: list[BaseException] = []

        def _reader():
            try:
                result.extend(_mcp_registration._existing_tool_names())
            except BaseException as exc:
                error.append(exc)

        reader = threading.Thread(target=_reader)
        reader.start()
        assert reached_doomed.wait(timeout=5), "reader never reached the doomed server"
        # Teardown's under-lock pop of the server the reader is parked on.
        with mcp_tool._lock:
            mcp_tool._servers.pop("doomed", None)
        may_continue.set()
        reader.join(timeout=5)
        assert not reader.is_alive()

        assert not error, f"unscoped enumeration raised: {error[0]!r}"
        assert "mcp__survivor__tool" in result
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._lazy_server_tool_names.clear()
            mcp_tool._lazy_server_tool_names.update(saved_lazy)
