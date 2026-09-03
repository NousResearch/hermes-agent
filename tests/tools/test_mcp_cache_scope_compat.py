"""Compatibility tests for legacy empty MCP list-result cache scopes."""

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from tools import mcp_tool


class FakeDispatcher:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def send_raw_request(self, method, params, opts=None):
        self.calls.append((method, params, opts))
        if isinstance(self.results, Mapping):
            return self.results
        return self.results[method]


class FakeSession:
    def __init__(self, dispatcher):
        self._dispatcher = dispatcher


def _sdk_session(dispatcher):
    assert mcp_tool._ensure_mcp_sdk()
    session = mcp_tool.ClientSession(dispatcher=dispatcher)
    return mcp_tool._install_list_cache_scope_compat(session)


def _list_payload(method, **updates):
    item_key = {
        "tools/list": "tools",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
        "prompts/list": "prompts",
    }[method]
    payload = {item_key: [], **updates}
    return payload


@pytest.mark.parametrize(
    ("method", "call_name"),
    [
        ("tools/list", "list_tools"),
        ("resources/list", "list_resources"),
        ("resources/templates/list", "list_resource_templates"),
        ("prompts/list", "list_prompts"),
    ],
)
def test_empty_cache_scope_is_omitted_before_real_sdk_validation(method, call_name):
    raw = _list_payload(method, cacheScope="")
    session = _sdk_session(FakeDispatcher(raw))

    result = asyncio.run(getattr(session, call_name)())

    assert mcp_tool.mcp_field(result, "cache_scope", "cacheScope") == "private"
    assert raw["cacheScope"] == ""


@pytest.mark.parametrize("cache_scope", ["public", "private"])
def test_valid_cache_scope_is_preserved_without_copy(cache_scope):
    raw = _list_payload("tools/list", cacheScope=cache_scope)
    dispatcher = FakeDispatcher(raw)
    session = _sdk_session(dispatcher)

    returned = asyncio.run(
        session._dispatcher.send_raw_request("tools/list", None, {})
    )
    result = asyncio.run(session.list_tools())

    assert returned is raw
    assert mcp_tool.mcp_field(result, "cache_scope", "cacheScope") == cache_scope


def test_missing_cache_scope_uses_sdk_private_default_without_copy():
    raw = _list_payload("tools/list")
    dispatcher = FakeDispatcher(raw)
    session = _sdk_session(dispatcher)

    returned = asyncio.run(
        session._dispatcher.send_raw_request("tools/list", None, {})
    )
    result = asyncio.run(session.list_tools())

    assert returned is raw
    assert mcp_tool.mcp_field(result, "cache_scope", "cacheScope") == "private"


def test_session_without_v2_dispatcher_is_left_untouched():
    session = object()

    assert mcp_tool._install_list_cache_scope_compat(session) is session


@pytest.mark.parametrize("bad_scope", [" ", "shared", None, 1, [], {}])
def test_other_invalid_cache_scopes_still_fail_sdk_validation(bad_scope):
    session = _sdk_session(
        FakeDispatcher(_list_payload("tools/list", cacheScope=bad_scope))
    )

    with pytest.raises(ValidationError):
        asyncio.run(session.list_tools())


@pytest.mark.parametrize(
    "malformed",
    [
        {"ttlMs": "forever"},
        {"ttlMs": None},
        {"tools": "not-a-list"},
    ],
)
def test_other_malformed_list_fields_still_fail_sdk_validation(malformed):
    raw = _list_payload("tools/list", cacheScope="", **malformed)
    session = _sdk_session(FakeDispatcher(raw))

    with pytest.raises(ValidationError):
        asyncio.run(session.list_tools())


def test_shim_is_narrow_non_mutating_and_shallow():
    nested = {"keep": "same-object"}
    raw = {"cacheScope": "", "nested": nested}
    dispatcher = FakeDispatcher(raw)
    session = FakeSession(dispatcher)
    mcp_tool._install_list_cache_scope_compat(session)

    normalized = asyncio.run(
        session._dispatcher.send_raw_request("tools/list", None, {})
    )

    assert normalized == {"nested": nested}
    assert normalized is not raw
    assert normalized["nested"] is nested
    assert raw == {"cacheScope": "", "nested": nested}


@pytest.mark.parametrize(
    "method",
    ["tools/call", "resources/read", "prompts/get", "initialize", "custom/list"],
)
def test_non_list_methods_are_never_rewritten(method):
    raw = {"cacheScope": ""}
    session = FakeSession(FakeDispatcher(raw))
    mcp_tool._install_list_cache_scope_compat(session)

    returned = asyncio.run(session._dispatcher.send_raw_request(method, None, {}))

    assert returned is raw
