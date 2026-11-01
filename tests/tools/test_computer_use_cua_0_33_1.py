"""Regression coverage for NousResearch/hermes-agent#132724's blocker: cua-driver
0.33+ serves ``click``/``scroll``/``set_value`` with ``additionalProperties: false``
and no longer lists ``element_index`` in the schema — only ``element_token``. Hermes
always builds ``element_index`` first (``cua_backend_input.py``), so a stale schema
gate would make every element click/scroll/set_value call fail closed with a
``additionalProperties`` validation error from cua-driver itself.

The fixture here is a real ``tools/list`` capture taken live from the pinned
cua-driver 0.33.1 Windows binary (see tests/fixtures/cua_driver_0_33_1_tools_list.json),
not a hand-written approximation — it is the literal schema the driver serves.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest


FIXTURE_0_33_1 = Path(__file__).parents[1] / "fixtures" / "cua_driver_0_33_1_tools_list.json"


@pytest.fixture(autouse=True)
def _reset_computer_use_state():
    from tools.computer_use.tool import reset_backend_for_tests

    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


def _load_fixture_tools(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    return [
        SimpleNamespace(
            name=item["name"],
            capabilities=item["capabilities"],
            inputSchema=item["inputSchema"],
            model_extra={},
        )
        for item in fixture["tools"]
    ]


def _populate_session_from_fixture(path: Path):
    """Build a real ``_CuaDriverSession`` whose capability maps came from an
    actual ``tools/list`` response captured against the fixture's driver build."""
    from tools.computer_use.cua_backend_session import _CuaDriverSession

    listed = _load_fixture_tools(path)

    class _McpSession:
        async def list_tools(self):
            return SimpleNamespace(tools=listed, model_extra={})

    session = _CuaDriverSession.__new__(_CuaDriverSession)
    session._capabilities = {}
    session._input_properties = {}
    session._capability_version = ""
    asyncio.run(session._populate_capabilities(_McpSession()))
    return session


def _make_backend(session):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "hermes-session"
    backend._snapshot_tokens = {5: "s00000001:5"}
    backend._active_pid = 111
    backend._active_window_id = 222
    return backend


class _RecordingSession:
    """Minimal session double wrapping a real populated session's schema maps
    (so `supports_input_property`/`_tool_schemas` come from the live capture)
    while keeping `call_tool` a plain recorder."""

    def __init__(self, real_session):
        self._real = real_session
        self._tool_schemas = real_session._tool_schemas
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def supports_input_property(self, tool: str, prop: str) -> bool:
        return self._real.supports_input_property(tool, prop)

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return self._real.supports_capability(capability, tool=tool)

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}


@pytest.mark.parametrize("action_name,call", [
    ("click", lambda b: b.click(element=5, button="left")),
    ("scroll", lambda b: b.scroll(direction="down", element=5)),
    ("set_value", lambda b: b.set_value("hello", element=5)),
])
def test_element_index_is_dropped_against_live_0_33_1_schema(action_name, call):
    """Against the real 0.33.1 schema (additionalProperties: false, only
    element_token), element_index must never ride along or cua-driver rejects
    the call outright."""
    real_session = _populate_session_from_fixture(FIXTURE_0_33_1)
    session = _RecordingSession(real_session)
    backend = _make_backend(session)

    result = call(backend)

    assert result.ok is True
    name, args = session.calls[0]
    assert name == action_name
    assert args["element_token"] == "s00000001:5"
    assert "element_index" not in args, (
        f"{action_name}: element_index must be stripped for a schema with "
        "additionalProperties:false and no element_index property"
    )


def test_legacy_capability_only_schema_keeps_element_index_fallback():
    """Pre-0.21 drivers advertise capabilities[] (no live input schema map for
    the tool) — the index must still be sent since the token is only a bonus,
    not a replacement, for drivers that never gated on the live schema."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "data": "ok", "images": [], "image_mime_types": [],
        "structuredContent": None, "isError": False,
    }
    capabilities = {"click": {"input.pointer.click", "accessibility.element_tokens"}}
    backend._session.supports_capability = (
        lambda cap, tool=None: cap in capabilities.get(tool, set()) if tool else
        any(cap in caps for caps in capabilities.values())
    )
    backend._session.supports_input_property = lambda tool, prop: False
    backend._session._tool_schemas = {}
    backend._active_pid = 111
    backend._active_window_id = 222
    backend._snapshot_tokens = {5: "s0001:5"}

    backend.click(element=5, button="left")

    name, args = backend._session.call_tool.call_args.args
    assert name == "click"
    assert args["element_index"] == 5
    assert args["element_token"] == "s0001:5"
