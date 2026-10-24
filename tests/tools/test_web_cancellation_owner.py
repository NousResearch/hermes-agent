"""Web fallback and cache boundaries honor the originating tool's cancellation."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.interrupt import acting_for_tid, set_interrupt
from tools import web_tools, web_tools_extract, web_tools_rescue, web_result_cache
from plugins.web import keyless_mcp


@pytest.mark.parametrize("operation, outcome", [
    ("search", "error"), ("search", "raise"), ("search", "success"),
    ("extract", "error"), ("extract", "raise"), ("extract", "partial"), ("extract", "short"),
    ("ring_search", "error"), ("ring_search", "success"),
    ("ring_extract", "error"), ("ring_extract", "partial"),
    ("rescue_extract", "partial"), ("rescue_extract", "redirect"),
])
@pytest.mark.parametrize("when", ["before", "during"])
def test_cancelled_origin_stops_fallback_and_cache(monkeypatch, operation, outcome, when):
    from model_tools import _run_async

    owner = threading.get_ident()
    urls = ["https://example.invalid/completed", "https://example.invalid/pending"]
    page = {"url": urls[0], "content": "completed page", "error": ""}
    failures = [{"url": url, "error": "429 rate limited"} for url in urls]
    policy_refusal = {"url": urls[0], "error": "Blocked by website policy", "blocked_by_policy": True}
    if operation == "rescue_extract":
        page["url"] = urls[1] if outcome == "partial" else "https://redirect.invalid/completed"
    observed = []
    primary_calls = []
    managed = Mock(return_value={"success": True, "data": {"web": []}})
    rescue_search = Mock(return_value={"success": True, "data": {"web": []}})
    rescue_extract = Mock(return_value=[page])
    cache_write = Mock()
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"cache_enabled": True, "extract_timeout": 5})
    monkeypatch.setattr(web_tools, "_managed_search_fallback", managed)
    monkeypatch.setattr(web_tools, "_rescue_search", rescue_search)
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda _provider: True)
    monkeypatch.setattr(web_tools_extract, "_rescue_eligible", lambda _provider: True)
    monkeypatch.setattr(web_tools_extract, "_rescue_extract", rescue_extract)
    memo = web_result_cache.SearchMemo()
    memo.store = cache_write
    monkeypatch.setattr(web_result_cache, "search_memo", memo)
    monkeypatch.setattr(web_result_cache, "extract_cache_put", cache_write)

    def primary(*_args, **_kwargs):
        primary_calls.append(True)
        observed.append((threading.get_ident(), acting_for_tid.get()))
        set_interrupt(True, thread_id=owner)
        if outcome == "raise":
            raise RuntimeError("provider failed after interruption")
        if "extract" in operation:
            if operation == "rescue_extract":
                return [page, failures[1]] if outcome == "redirect" else [page]
            if outcome == "short":
                return [page]
            return [page, failures[1]] if outcome == "partial" else failures
        return {"success": outcome == "success", "data": {"web": []}, "error": "429 rate limited"}

    provider = SimpleNamespace(name="example-provider", search=primary, extract=primary)
    next_vendor = Mock(return_value=[page] if "extract" in operation else {"success": True, "data": {"web": []}})
    monkeypatch.setattr(keyless_mcp, "_ring_order", lambda _name: ["exa", "parallel"])
    callbacks = keyless_mcp._KEYLESS_EXTRACTORS if "extract" in operation else keyless_mcp._KEYLESS_SEARCHERS
    monkeypatch.setitem(callbacks, "exa", primary)
    monkeypatch.setitem(callbacks, "parallel", next_vendor)

    async def tool_call():
        if operation == "search":
            return await asyncio.to_thread(web_tools._memoized_search, provider, "synthetic query", 2)
        if operation == "extract":
            return await web_tools_extract._dispatch_extract(provider, urls, "markdown")
        if operation == "ring_search":
            return await asyncio.to_thread(keyless_mcp.search_with_failover, "exa", "synthetic query", 2)
        if operation == "rescue_extract":
            return await asyncio.to_thread(web_tools_rescue._rescue_extract, "example-provider", urls,
                                           [policy_refusal, failures[1]])
        return await asyncio.to_thread(keyless_mcp.extract_with_failover, "exa", urls)

    async def origin():
        # Exercise the sync bridge while its caller already has a running loop.
        return _run_async(tool_call())

    try:
        if when == "before":
            set_interrupt(True, thread_id=owner)
        result = asyncio.run(origin())
        assert len(primary_calls) == (0 if when == "before" else 1)
        if observed:
            assert observed[0][0] != owner
            assert observed[0][1] == owner
        managed.assert_not_called()
        rescue_search.assert_not_called()
        rescue_extract.assert_not_called()
        next_vendor.assert_not_called()
        cache_write.assert_not_called()
        if "extract" in operation:
            if outcome in ("partial", "short", "redirect") and when == "during":
                assert any(entry is page for entry in result)
                assert page["content"] == "completed page"
            if operation == "rescue_extract":
                assert result[0] is policy_refusal
            if not (operation == "rescue_extract" and outcome == "partial" and when == "during"):
                assert any(entry.get("error") == "Interrupted" for entry in result)
        else:
            assert result["success"] is False
            assert result["error"] == "Interrupted"
    finally:
        set_interrupt(False, thread_id=owner)


@pytest.mark.parametrize("running_loop", [False, True])
def test_async_bridge_owner_scope_is_restored_and_unrelated_session_is_not_cancelled(running_loop):
    from concurrent.futures import ThreadPoolExecutor
    from model_tools import _run_async
    from tools.interrupt import is_interrupted

    owner = threading.get_ident()
    previous = acting_for_tid.get()
    async def observe():
        return await asyncio.to_thread(lambda: (acting_for_tid.get(), is_interrupted()))
    async def with_running_loop():
        return _run_async(observe())
    result = asyncio.run(with_running_loop()) if running_loop else _run_async(observe())
    assert result == (owner, False)
    assert acting_for_tid.get() == previous
    try:
        set_interrupt(True, thread_id=owner)
        def independent_session():
            independent_owner = threading.get_ident()
            return independent_owner, _run_async(observe())
        with ThreadPoolExecutor(max_workers=1) as pool:
            independent_owner, independent_result = pool.submit(independent_session).result(timeout=10)
        assert independent_owner != owner
        assert independent_result == (independent_owner, False)
    finally:
        set_interrupt(False, thread_id=owner)
