"""Per-call cancellation, recovery and trust controls for overlapping HTTP RPCs.

Only external recovery is replaced: the real handler and in-flight bookkeeping
still drive a real loopback SDK session. Transport rebuild itself is covered by
this suite's lifecycle tests and the existing session-expiry/401 area suites.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tools.test_mcp_http_parallel_calls import connected as _connected, _pair
from tests.waiting import wait_until

connected = _connected


def test_canceling_one_http_call_does_not_cancel_its_sibling(connected):
    from tools.mcp_tool_loop import _run_on_mcp_loop

    http, connect = connected
    server, handler = connect()
    http.hold = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handler, {"label": label}) for label in ("a", "b")]
        try:
            wait_until(lambda: http.active == 2, timeout=10, what="two requests on the HTTP server")
            async def cancel_one():
                assert len(server._inflight_tasks) == 2
                next(iter(server._inflight_tasks)).cancel()
            _run_on_mcp_loop(cancel_one)
            wait_until(lambda: len(server._inflight_tasks) == 1, timeout=10, what="only the canceled RPC settling")
            assert not server._reconnecting
            http.loop.call_soon_threadsafe(http.release.set)
            results = [json.loads(f.result(15)) for f in futures]
            assert sum("result" in result for result in results) == 1, results
            assert sum("error" in result for result in results) == 1, results
            wait_until(lambda: not server._inflight_tasks, timeout=10, what="sibling completion")
            assert sorted(http.received) == ["a", "b"]
        finally:
            http.loop.call_soon_threadsafe(http.release.set)


@pytest.mark.parametrize("failure", ["401", "expired-read", "expired-write"])
def test_each_overlapping_call_keeps_its_replay_safety(connected, monkeypatch, failure):
    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool as core
    from tools import mcp_tool_loop as loop
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    http, connect = connected
    server, handler = connect()
    core._tool_read_only_hints["parallel-probe"]["probe"] = failure == "expired-read"
    entered = asyncio.Event()
    attempts = {}
    real_call = server.session.call_tool
    recovery = []

    async def call_tool(name, arguments, **kwargs):
        label = arguments["label"]
        attempts[label] = attempts.get(label, 0) + 1
        if attempts[label] != 1:
            return await real_call(name, arguments=arguments, **kwargs)
        if failure == "expired-write":
            # The server completed the write but the response path then broke.
            await real_call(name, arguments=arguments, **kwargs)
        else:
            if len(attempts) == 2:
                entered.set()
            await asyncio.wait_for(entered.wait(), 5)
        if failure == "401":
            raise OAuthFlowError("injected pre-dispatch 401")
        raise RuntimeError("Session expired")

    async def recover_auth(*args):
        recovery.append("auth")
        return True

    def recover_transport(*args, **kwargs):
        recovery.append("transport")
        return True

    reset_manager_for_tests()
    monkeypatch.setattr(get_manager(), "handle_401", recover_auth)
    monkeypatch.setattr(loop, "_signal_reconnect_and_wait", recover_transport)
    monkeypatch.setattr(server.session, "call_tool", call_tool)
    try:
        results = _pair(handler)
        if failure == "expired-write":
            assert attempts == {"a": 1, "b": 1}
            assert all(result.get("outcome_uncertain") is True for result in results), results
        else:
            assert results == [{"result": "answer:a"}, {"result": "answer:b"}]
            assert attempts == {"a": 2, "b": 2}
        assert sorted(http.received) == ["a", "b"]
        assert http.peak == 2
        assert not server._inflight_tasks
        assert recovery.count("transport") == 2
        assert recovery.count("auth") == (2 if failure == "401" else 0)
    finally:
        reset_manager_for_tests()


def test_parallel_http_admission_waits_for_discovery_before_trust_recheck(connected, monkeypatch):
    from tools import mcp_tool as core
    from tools import mcp_tool_loop as loop

    http, connect = connected
    server, handler = connect()
    core._server_trust_levels["parallel-probe"] = "untrusted"
    core._tool_read_only_hints["parallel-probe"]["probe"] = True
    listing, scheduled = threading.Event(), threading.Event()
    release = asyncio.Event()
    real_run = loop._run_on_mcp_loop

    async def rediscover():
        async with server._rpc_lock:
            listing.set()
            await release.wait()
            core._tool_read_only_hints["parallel-probe"]["probe"] = False

    def run(coro, timeout=30):
        scheduled.set()
        return real_run(coro, timeout=timeout)

    pending = asyncio.run_coroutine_threadsafe(rediscover(), core._mcp_loop)
    assert listing.wait(10)
    monkeypatch.setattr(loop, "_run_on_mcp_loop", run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(handler, {"label": "unapproved"})
        try:
            assert scheduled.wait(10)
            wait_until(lambda: bool(server._rpc_lock._waiters), timeout=10,
                       what="call admission waiting behind discovery")
            core._mcp_loop.call_soon_threadsafe(release.set)
            pending.result(10)
            result = json.loads(future.result(10))
            assert "NOT run without approval" in result["error"], result
            assert http.received == []
            assert not server._inflight_tasks
        finally:
            core._mcp_loop.call_soon_threadsafe(release.set)
