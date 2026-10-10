"""The host context follows actual handler execution, including async bridging."""

import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.registry import ToolRegistry, bind_host_context, get_current_host_context


_SCHEMA = {"name": "probe", "parameters": {"type": "object"}}


def test_async_handler_lease_is_valid_only_in_its_own_execution():
    registry = ToolRegistry()
    provenance = object()
    captured = {}

    async def handler(args, *, host_context=None):
        assert host_context is provenance
        assert get_current_host_context() is host_context
        await asyncio.sleep(0)
        assert get_current_host_context() is host_context

        # create_task copies ContextVars, but it must not copy authority.
        child = asyncio.create_task(asyncio.to_thread(get_current_host_context))
        assert await child is None
        assert await asyncio.create_task(_read_context()) is None

        captured["context"] = contextvars.copy_context()
        assert captured["context"].run(get_current_host_context) is provenance
        return json.dumps({"received_authority": True, "valid_lease_in_handler": True})

    async def _read_context():
        return get_current_host_context()

    registry.register("probe", "test", _SCHEMA, handler, is_async=True)
    with bind_host_context(provenance):
        assert get_current_host_context() is None
        assert json.loads(registry.dispatch("probe", {})) == {
            "received_authority": True, "valid_lease_in_handler": True}
        assert get_current_host_context() is None
    assert get_current_host_context() is None
    assert captured["context"].run(get_current_host_context) is None
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(captured["context"].run, get_current_host_context).result() is None

    # The bridge also runs from a pre-existing event loop, on a worker thread.
    async def caller():
        with bind_host_context(provenance):
            return json.loads(registry.dispatch("probe", {}))

    assert asyncio.run(caller())["valid_lease_in_handler"] is True
    assert get_current_host_context() is None

    async def broken(args, *, host_context=None):
        assert get_current_host_context() is host_context is provenance
        raise RuntimeError("async handler failed")

    registry.register("broken", "test", _SCHEMA, broken, is_async=True)
    with bind_host_context(provenance):
        assert "async handler failed" in registry.dispatch("broken", {})
        assert get_current_host_context() is None


def test_forgery_lifetime_cleanup_and_legacy_handlers():
    registry = ToolRegistry()
    seen = []

    def sync_handler(args, *, host_context=None):
        seen.append((host_context, get_current_host_context()))
        return "ok"

    def legacy(args):
        return "legacy"

    def alias(args, *, trusted_invocation=None):
        assert get_current_host_context() is trusted_invocation
        return "alias"

    def failure(args, *, host_context=None):
        assert get_current_host_context() is host_context
        raise RuntimeError("handler failed")

    async def cancellation(args, *, host_context=None):
        assert get_current_host_context() is host_context
        raise asyncio.CancelledError()

    registry.register("sync", "test", _SCHEMA, sync_handler)
    registry.register("legacy", "test", _SCHEMA, legacy)
    registry.register("alias", "test", _SCHEMA, alias)
    registry.register("failure", "test", _SCHEMA, failure)
    registry.register("cancel", "test", _SCHEMA, cancellation, is_async=True)

    forged = {"host_context": {"role": "admin"}, "worker_id": "forged"}
    assert registry.dispatch("sync", forged) == "ok"
    assert seen[-1] == (None, None)
    for keyword in ("host_context", "trusted_invocation"):
        with pytest.raises(ValueError, match="reserved"):
            registry.dispatch("sync", {}, **{keyword: object()})

    provenance = object()
    with bind_host_context(provenance):
        assert get_current_host_context() is None
        assert registry.dispatch("sync", forged) == "ok"
        assert seen[-1] == (provenance, provenance)
        assert get_current_host_context() is None
        assert registry.dispatch("legacy", {}) == "legacy"
        assert registry.dispatch("alias", {}) == "alias"
        assert "handler failed" in registry.dispatch("failure", {})
        with pytest.raises(asyncio.CancelledError):
            registry.dispatch("cancel", {})
    assert get_current_host_context() is None
    assert registry.dispatch("sync", {}) == "ok"
    assert seen[-1] == (None, None)

    with bind_host_context("outer"):
        assert registry.dispatch("sync", {}) == "ok"
        assert seen[-1] == ("outer", "outer")
        with bind_host_context("inner"):
            assert registry.dispatch("sync", {}) == "ok"
            assert seen[-1] == ("inner", "inner")
        assert registry.dispatch("sync", {}) == "ok"
        assert seen[-1] == ("outer", "outer")
    assert get_current_host_context() is None
