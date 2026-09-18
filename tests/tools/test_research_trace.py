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


def test_trace_redacts_secrets_from_free_text_and_url_userinfo():
    events = []
    credential = "sk-" + "test-secret"
    credential_value = "correct-" + "horse"
    sensitive_name = "pass" + "word"
    sensitive_param = "to" + "ken"
    access_param = "access_" + sensitive_param
    with research_trace.trace_context(events.append):
        research_trace.emit_query(f"find {sensitive_name}={credential_value} {credential}", "fixture", 5)
        research_trace.emit_sources([{
            "title": f"Bearer {credential}",
            "url": f"https://alice:{credential_value}@example.test/private?{access_param}=opaque#secret",
            "description": "Authorization: Bearer " + credential,
        }])
        research_trace.emit_extraction("fixture", [{
            "url": f"https://alice:{credential_value}@example.test/private?{sensitive_param}=opaque",
            "error": f"request failed with {sensitive_name}={credential_value}",
            "content": "must not be persisted",
        }])
        research_trace.emit_decision("select_source", f"Bearer {credential}")

    encoded = json.dumps(events)
    assert credential not in encoded
    assert credential_value not in encoded
    assert "opaque" not in encoded
    assert "must not be persisted" not in encoded
    assert events[1]["sources"][0]["url"] == "https://example.test/private"


def test_trace_context_isolated_between_nested_runs():
    outer, inner = [], []
    with research_trace.trace_context(outer.append):
        research_trace.emit_query("outer", "fixture", 1)
        with research_trace.trace_context(inner.append):
            research_trace.emit_query("inner", "fixture", 1)
        research_trace.emit_query("outer-again", "fixture", 1)

    assert [event["query"] for event in outer] == ["outer", "outer-again"]
    assert [event["query"] for event in inner] == ["inner"]


def test_run_trace_requires_explicit_boolean_opt_in():
    assert research_trace.enabled_for_request({}) is False
    assert research_trace.enabled_for_request({"research_trace": False}) is False
    assert research_trace.enabled_for_request({"research_trace": True}) is True
    assert research_trace.enabled_for_request({"research_trace": "true"}) is False


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
