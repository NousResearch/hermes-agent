"""A SEP-2549 ``ttlMs`` hint counts only when the server actually sent one."""

from __future__ import annotations

import asyncio

import pytest

from tools.mcp_tool import _paginate_full_list

types = pytest.importorskip("mcp.types")


def _drain(result) -> dict:
    """Run one full ``tools/list`` pagination and return the cache metadata."""

    async def list_method(**_kwargs):
        return result

    meta: dict = {}
    asyncio.run(_paginate_full_list(list_method, "tools", "srv", cache_meta_out=meta))
    return meta


def test_absent_hint_is_not_recorded_as_a_ttl():
    """The SDK defaults ``ttl_ms`` to 0, which must not read as a real TTL.

    Recording it made ``mcp_schema_cache`` stamp a ``written_at``, and
    ``(now - written_at) * 1000 >= 0`` always holds — so the entry was expired
    the instant it was written and a ``lazy`` server was re-probed on every
    startup. The stray stamp also defeated the byte-identical write-through
    skip, rewriting the whole cache file on every registration.
    """
    meta = _drain(types.ListToolsResult.model_validate({"tools": []}))
    assert "ttl_ms" not in meta


def test_explicit_zero_hint_is_preserved():
    """``ttlMs: 0`` is a server saying "do not serve this from cache".

    Same value as the default, opposite meaning — so it must survive
    extraction and expire the entry, exactly as the SDK's own caching client
    treats a zero TTL.
    """
    meta = _drain(types.ListToolsResult.model_validate({"tools": [], "ttlMs": 0}))
    assert meta["ttl_ms"] == 0


def test_positive_hint_is_preserved():
    meta = _drain(types.ListToolsResult.model_validate({"tools": [], "ttlMs": 60_000}))
    assert meta["ttl_ms"] == 60_000


def test_cache_scope_still_extracted_without_a_ttl():
    """The TTL gate must not swallow the neighbouring ``cacheScope`` hint."""
    meta = _drain(
        types.ListToolsResult.model_validate({"tools": [], "cacheScope": "private"})
    )
    assert meta["cache_scope"] == "private"
    assert "ttl_ms" not in meta


def test_result_without_field_tracking_falls_back_to_the_value():
    """Older SDKs and test doubles expose no ``model_fields_set``.

    They also never carry a defaulted ``ttl_ms``, so trusting the value is
    both safe and the behaviour this code had before the gate.
    """

    class _Legacy:
        tools = []
        ttl_ms = 60_000
        next_cursor = None

    assert _drain(_Legacy())["ttl_ms"] == 60_000
