"""Async registration dispatch contracts, including the running-loop thread path."""
import asyncio
import functools
import json

import pytest

from tools.registry import ToolRegistry


async def handler(args, **kwargs):
    await asyncio.sleep(0)
    return json.dumps({"value": args["value"]})


class Handler:
    async def run(self, args, **kwargs):
        return await handler(args, **kwargs)


def dispatch(fn, *, explicit=False):
    registry = ToolRegistry()
    registry.register(name="probe", toolset="core", schema={"name": "probe", "parameters": {}},
                      handler=fn, is_async=explicit)
    result = registry.dispatch("probe", {"value": 7})
    assert isinstance(result, str)
    return json.loads(result)


@pytest.mark.parametrize("fn", [handler, Handler().run, functools.partial(handler)])
def test_declared_async_handler_is_awaited(fn):
    assert dispatch(fn) == {"value": 7}


@pytest.mark.asyncio
async def test_declared_async_handler_dispatch_inside_running_loop():
    assert dispatch(handler) == {"value": 7}


def test_sync_wrapper_explicit_flag_and_sync_handler_unchanged():
    assert dispatch(lambda args, **kw: handler(args, **kw), explicit=True) == {"value": 7}
    assert dispatch(lambda args, **kw: json.dumps({"value": args["value"]})) == {"value": 7}


def test_async_errors_keep_registry_error_contract():
    async def broken(args, **kwargs):
        raise ValueError("synthetic failure")
    assert "synthetic failure" in dispatch(broken)["error"]
