import json

import pytest

from tools import research_trace


def test_trace_context_emits_bounded_decision_without_sensitive_fields():
    events = []
    with research_trace.trace_context(events.append):
        research_trace.emit_decision("select_source", {"url": "https://example.test", "reason": "best match"})

    assert events == [{
        "type": "research.decision",
        "decision": "select_source",
        "details": {"url": "https://example.test", "reason": "best match"},
    }]
    assert len(json.dumps(events)) < 2000


def test_trace_context_is_noop_without_sink():
    research_trace.emit("research.query", query="ignored", provider="test")


@pytest.mark.asyncio
async def test_web_search_emits_query_provider_and_source_metadata(monkeypatch):
    from tools import web_tools

    class Provider:
        name = "fixture"

        def supports_search(self):
            return True

        def search(self, query, limit):
            return {"success": True, "data": {"web": [{
                "title": "Title", "url": "https://example.test/a", "description": "Description",
            }]}}

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "fixture")
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: Provider())
    events = []
    with research_trace.trace_context(events.append):
        result = web_tools.web_search_tool("climate policy", limit=1)

    assert json.loads(result)["success"] is True
    assert [event["type"] for event in events] == ["research.query", "research.sources"]
    assert events[0]["provider"] == "fixture"
    assert events[1]["sources"][0] == {"title": "Title", "url": "https://example.test/a", "description": "Description", "position": 1}


@pytest.mark.asyncio
async def test_web_extract_emits_status_without_content(monkeypatch):
    from tools import web_tools

    class Provider:
        name = "fixture"

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fixture")
    async def _extract(provider, urls, format):
        return [{"url": urls[0], "title": "Private title", "content": "must not be traced", "error": None}]

    monkeypatch.setattr(web_tools, "_extract_safe_urls", _extract)
    monkeypatch.setattr(web_tools, "_resolve_extract_provider", lambda backend: (Provider(), None))
    monkeypatch.setattr(web_tools, "async_is_safe_url", lambda url: _true_async())
    monkeypatch.setattr(web_tools, "_effective_char_limit", lambda value: 1000)
    events = []
    with research_trace.trace_context(events.append):
        result = await web_tools.web_extract_tool(["https://example.test/a"])

    assert json.loads(result)["results"][0]["content"] == "must not be traced"
    assert events == [{"type": "research.extraction", "provider": "fixture", "results": [
        {"url": "https://example.test/a", "status": "ok", "error": None}
    ]}]


async def _true_async():
    return True
