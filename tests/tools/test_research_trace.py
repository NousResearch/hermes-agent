import json

import pytest

from tools import research_trace


def test_trace_context_minimizes_free_text_from_query_decision_and_sources():
    events = []
    raw_query = "private prompt: plan my secret acquisition"
    raw_result = "untrusted result body and exception text"
    with research_trace.trace_context(events.append):
        research_trace.emit_query(raw_query, "fixture", 999)
        research_trace.emit_sources([{
            "title": raw_result,
            "url": "https://alice:password@example.test/a?token=opaque#fragment",
            "description": raw_result,
        }])
        research_trace.emit_extraction("fixture", [{
            "url": "https://alice:password@example.test/a?token=opaque#fragment",
            "error": raw_result,
            "content": raw_result,
        }])
        research_trace.emit_decision("select_source", raw_result)

    encoded = json.dumps(events)
    assert raw_query not in encoded
    assert raw_result not in encoded
    assert "password" not in encoded
    assert "opaque" not in encoded
    assert "fragment" not in encoded
    assert events == [
        {"type": "research.query", "provider": "fixture", "limit": 20},
        {"type": "research.sources", "sources": [{"url": "https://example.test/a", "position": 1}]},
        {"type": "research.extraction", "provider": "fixture", "results": [{"url": "https://example.test/a", "status": "error"}]},
        {"type": "research.decision", "decision": "select_source"},
    ]


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
    assert events[1]["sources"][0] == {"url": "https://example.test/a", "position": 1}


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
        {"url": "https://example.test/a", "status": "ok"}
    ]}]


async def _true_async():
    return True
