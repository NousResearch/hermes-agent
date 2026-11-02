"""web_extract must run plugin discovery BEFORE resolving the extract backend.

A user-installed plugin provider (e.g. trafilatura) registers in
``agent.web_search_registry`` only after plugin discovery. When
``web_extract_tool`` resolved ``_get_extract_backend()`` first, the
availability probe inside the autodetect ladder saw nothing registered and
silently fell through to the legacy fallback (ddgs). The search path already
loads plugins first; this pins the same order for extract.
"""
import asyncio
import json
from unittest.mock import patch

import tools.web_tools as wt


class _AsyncTrue:
    """Async callable that always returns True (re-awaitable per call)."""

    async def __call__(self, *a, **k):
        return True


class _FakeProvider:
    name = "fake"
    display_name = "Fake"

    def supports_extract(self):
        return True

    async def extract(self, urls, **kwargs):
        return [
            {"url": urls[0], "title": "T", "content": "hello",
             "raw_content": "hello", "metadata": {}}
        ]


def test_web_extract_loads_plugins_before_resolving_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = []

    with patch.object(wt, "_ensure_web_plugins_loaded",
                      side_effect=lambda: calls.append("load_plugins")), \
         patch.object(wt, "_get_extract_backend",
                      side_effect=lambda: (calls.append("resolve_backend"), "fake")[1]), \
         patch.object(wt, "async_is_safe_url", new=_AsyncTrue()), \
         patch("agent.web_search_registry.get_provider", return_value=_FakeProvider()):
        result = json.loads(asyncio.new_event_loop().run_until_complete(
            wt.web_extract_tool(["https://example.com"])
        ))

    assert result["results"][0]["content"] == "hello"
    assert calls == ["load_plugins", "resolve_backend"], (
        "web_extract must run plugin discovery before backend resolution, "
        "or the autodetect ladder cannot see user-installed providers"
    )