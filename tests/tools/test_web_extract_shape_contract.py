"""``extract()``'s return contract is a bare ``list[dict]``, and a wrong shape fails by name.

``search()`` answers with a ``{"success": ..., "data": ...}`` envelope; ``extract()`` does not — the
dispatcher iterates its return value and reads each entry with ``.get()``. A provider written to the
envelope (the shape the module docstring and the plugin guide used to document) had its own string
keys iterated here and died on ``'str' object has no attribute 'get'``, which names neither the
provider nor the shape. These cases pin both halves: the documented list is consumed, anything else
is refused with the provider and the expected type in the message.
"""
from __future__ import annotations

import asyncio

import pytest

from tools import web_result_cache
from tools import web_tools_extract as wte

URLS = ["https://example.com/a", "https://example.com/b"]


class _DictProvider:
    """Returns the old envelope, byte-for-byte as the guide documented it."""

    name = "envelope-dict"

    async def extract(self, urls, format=None):
        return {"success": True, "data": [{"url": u, "title": "t", "content": "c"} for u in urls]}


class _BareListProvider:
    name = "bare-list"

    async def extract(self, urls, format=None):
        return [{"url": u, "title": "t", "content": f"content for {u}", "raw_content": "raw"}
                for u in urls]


class _StringEntryProvider:
    name = "string-entries"

    async def extract(self, urls, format=None):
        return [f"content for {u}" for u in urls]


class _NoneProvider:
    name = "returns-none"

    async def extract(self, urls, format=None):
        return None


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr(wte, "_extract_timeout_seconds", lambda: 0.0)
    monkeypatch.setattr(wte, "_rescue_eligible", lambda provider: False)
    monkeypatch.setattr(web_result_cache, "extract_cache_put", lambda *a, **k: None)


def test_documented_bare_list_passes_through():
    results = asyncio.run(wte._dispatch_extract(_BareListProvider(), URLS, None))

    assert [r["url"] for r in results] == URLS
    assert results[0]["content"] == f"content for {URLS[0]}"


@pytest.mark.parametrize(
    "provider, expected",
    [
        (_DictProvider(), "returned a success/data envelope; expected list[dict]"),
        (_StringEntryProvider(), "returned a list containing str; expected list[dict]"),
        (_NoneProvider(), "returned NoneType; expected list[dict]"),
    ],
    ids=["envelope", "string-entries", "none"],
)
def test_wrong_shape_is_refused_by_name(provider, expected):
    with pytest.raises(TypeError) as excinfo:
        asyncio.run(wte._dispatch_extract(provider, URLS, None))

    message = str(excinfo.value)
    assert provider.name in message
    assert expected in message


def test_envelope_message_points_at_the_fix():
    """The error has to be actionable on its own: it is all the tool result shows."""
    with pytest.raises(TypeError) as excinfo:
        asyncio.run(wte._dispatch_extract(_DictProvider(), URLS, None))

    assert "extract() has no envelope" in str(excinfo.value)
