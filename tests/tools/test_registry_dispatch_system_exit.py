"""A tool handler's ``SystemExit`` is a tool failure, not a request to stop Hermes.

``ToolRegistry.dispatch`` promises that every exception comes back as ``{"error": ...}``. A plugin
handler, or a library it calls (argparse or click rejecting arguments, a client signalling an auth
failure with ``sys.exit()``), can raise ``SystemExit``, which is not an ``Exception``: it escaped
dispatch, unwound the turn and, on the gateway, stopped the event loop. Plugin hooks contain it
the same way (#117303). ``KeyboardInterrupt`` and ``GeneratorExit`` stay control flow.
"""

import json
import logging

import pytest

from tools.registry import ToolRegistry


def _registry(**handlers):
    reg = ToolRegistry()
    for name, (handler, is_async) in handlers.items():
        reg.register(
            name=name, toolset="core", handler=handler, is_async=is_async,
            schema={"name": name, "description": name, "parameters": {"type": "object", "properties": {}}},
        )
    return reg


def _exit_sync(code):
    def handler(args, **kw):
        raise SystemExit(code)
    return handler


def _exit_async(code):
    async def handler(args, **kw):
        raise SystemExit(code)
    return handler


async def _ok_async(args, **kw):
    return json.dumps({"ok": True})


@pytest.mark.parametrize("make_handler,is_async", [(_exit_sync, False), (_exit_async, True)], ids=["sync", "async"])
@pytest.mark.parametrize("code", ["auth failed", 2])
def test_system_exit_from_a_handler_is_returned_as_a_tool_error(make_handler, is_async, code, caplog):
    reg = _registry(exits=(make_handler(code), is_async), ok=(_ok_async, True))
    with caplog.at_level(logging.ERROR, logger="tools.registry"):
        result = json.loads(reg.dispatch("exits", {}))
    assert "SystemExit" in result["error"] and str(code) in result["error"]
    assert any("exits" in r.getMessage() for r in caplog.records), "the contained exit must be logged"
    # The async bridge's loop survives the contained exit: the next async tool still runs.
    assert json.loads(reg.dispatch("ok", {})) == {"ok": True}


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, GeneratorExit])
def test_keyboard_interrupt_and_generator_exit_still_propagate(exc_type):
    def handler(args, **kw):
        raise exc_type()
    reg = _registry(control=(handler, False))
    with pytest.raises(exc_type):
        reg.dispatch("control", {})
