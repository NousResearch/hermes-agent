"""Regression tests for MCP stdio aggregate liveness (#94335 / #94637).

The #81995 fast-fail gate consumes ``_stdio_children_dead`` as a boolean
state machine: True means every tracked child is gone; False means at least
one child is alive or liveness is unknown. The live-child branch was inverted,
so healthy stdio RPCs were cancelled while their subprocesses were still alive.

Watcher-consumer cases are distilled from #94521. Dependency/probe fail-open
cases are distilled from #94661 into the canonical #94339 carrier.
"""

import asyncio
import builtins
from unittest.mock import patch

import pytest

from tools.mcp_tool import MCPServerTask

def _task_with_pids(pids, *, http=False):
    task = object.__new__(MCPServerTask)
    task._stdio_child_pids = pids
    task._config = {"url": "http://example.invalid"} if http else {"command": "x"}
    return task

def test_live_child_reports_not_dead():
    """The reported bug: an alive tracked pid must NOT report all-dead."""
    with patch("psutil.pid_exists", return_value=True):
        assert _task_with_pids([60634])._stdio_children_dead() is False

def test_all_children_dead_reports_dead():
    with patch("psutil.pid_exists", return_value=False):
        assert _task_with_pids([111, 222])._stdio_children_dead() is True

def test_mixed_liveness_reports_not_dead():
    """One live sibling is enough — dead others must not flip the verdict."""
    with patch("psutil.pid_exists", side_effect=lambda pid: pid != 111):
        assert _task_with_pids([111, 222])._stdio_children_dead() is False

def test_no_captured_pids_stays_fail_open():
    """Unknown (no tracked pids / HTTP transport) must not fail fast."""
    assert _task_with_pids([])._stdio_children_dead() is False
    assert _task_with_pids([1], http=True)._stdio_children_dead() is False

def test_psutil_unavailable_stays_fail_open():
    """Missing probe support is unknown, never proof of child death."""
    real_import = builtins.__import__

    def _without_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil unavailable")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_without_psutil):
        assert _task_with_pids([1])._stdio_children_dead() is False

def test_pid_probe_error_stays_fail_open():
    """A failed probe cannot authorize the destructive fast-fail."""
    with patch("psutil.pid_exists", side_effect=OSError("probe failed")):
        assert _task_with_pids([1])._stdio_children_dead() is False

def test_watcher_does_not_resolve_while_a_child_is_alive():
    """The watcher must not cancel an RPC while any child is still live."""

    async def _run():
        with patch("psutil.pid_exists", return_value=True):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    _task_with_pids([60634])._watch_stdio_children(),
                    timeout=0.05,
                )

    asyncio.run(_run())

def test_watcher_resolves_when_all_children_are_dead():
    """The watcher completes only when the aggregate verdict is all-dead."""

    async def _run():
        with patch("psutil.pid_exists", return_value=False):
            await asyncio.wait_for(
                _task_with_pids([111, 222])._watch_stdio_children(),
                timeout=0.1,
            )

    asyncio.run(_run())


def test_fast_fail_probe_never_calls_the_watcher_on_the_plain_path():
    """The fast-fail gate must inspect the watcher, not call it (#96044).

    The old probe, ``inspect.isawaitable(_watch_children())``, created a fresh
    watcher coroutine on every stdio tool call and dropped it unawaited on the
    plain-await path ("coroutine ... was never awaited" under -W error, GC churn).
    Drive the real call path with a stubbed (non-coroutine) session result and
    assert the async watcher is never invoked.
    """
    import gc
    import warnings
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tools.mcp_tool_handlers import _call_tool_racing_stdio_death

    # AsyncMock qualifies as a coroutine function (like the real async method)
    # and counts calls even when the returned coroutine is never awaited.
    watcher = AsyncMock()
    server = SimpleNamespace(
        session=SimpleNamespace(call_tool=lambda name, arguments: {"ok": name}),
        _watch_stdio_children=watcher,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(_call_tool_racing_stdio_death(server, "srv", "tool", {}))
        gc.collect()

    assert result == {"ok": "tool"}
    assert watcher.call_count == 0, "fast-fail probe called the watcher (leaks a coroutine per call)"
    assert not [w for w in caught if "never awaited" in str(w.message)]
