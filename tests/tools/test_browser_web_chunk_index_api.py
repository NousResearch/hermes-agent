import asyncio
import json


def test_web_extract_schema_exposes_chunk_index_and_find_tool():
    from tools.web_tools import WEB_EXTRACT_SCHEMA, WEB_EXTRACT_FIND_SCHEMA

    props = WEB_EXTRACT_SCHEMA["parameters"]["properties"]
    assert "chunk_index" in props
    assert "offset" not in props
    assert "limit" not in props
    assert "chunk_index" in WEB_EXTRACT_SCHEMA["description"]
    assert "LLM" not in WEB_EXTRACT_SCHEMA["description"]

    find_props = WEB_EXTRACT_FIND_SCHEMA["parameters"]["properties"]
    assert "query" in find_props
    assert "urls" in find_props
    assert "chunk_index" not in find_props
    assert WEB_EXTRACT_FIND_SCHEMA["parameters"]["required"] == ["query"]


def test_browser_snapshot_schema_exposes_chunk_index_and_find_tool():
    from tools.browser_tool import BROWSER_TOOL_SCHEMAS

    snapshot = next(schema for schema in BROWSER_TOOL_SCHEMAS if schema["name"] == "browser_snapshot")
    props = snapshot["parameters"]["properties"]
    assert "chunk_index" in props
    assert "offset" not in props
    assert "limit" not in props
    assert "LLM" not in snapshot["description"]

    find = next(schema for schema in BROWSER_TOOL_SCHEMAS if schema["name"] == "browser_find")
    find_props = find["parameters"]["properties"]
    assert "query" in find_props
    assert "chunk_index" not in find_props
    assert find["parameters"]["required"] == ["query"]


def test_web_extract_chunks_source_and_reuses_cache_by_chunk_index(monkeypatch):
    import tools.web_tools as wt
    from tools.chunked_content import DEFAULT_SOURCE_CHUNK_SIZE

    wt._ensure_web_extract_cache().clear()
    raw = "A" * DEFAULT_SOURCE_CHUNK_SIZE + "SECOND WEB CHUNK"
    calls = []

    async def fake_parallel_extract(urls):
        calls.append(tuple(urls))
        return [{"url": urls[0], "title": "Web", "content": raw, "raw_content": raw}]

    monkeypatch.setattr(wt, "_get_extract_backend", lambda: "parallel")
    monkeypatch.setattr(wt, "is_safe_url", lambda url: True)
    monkeypatch.setattr(wt, "check_website_access", lambda url: None)
    monkeypatch.setattr(wt, "_parallel_extract", fake_parallel_extract)

    first = json.loads(asyncio.get_event_loop().run_until_complete(
        wt.web_extract_tool(["https://x.test"], use_llm_processing=False, chunk_index=1, format="markdown")
    ))
    cached = json.loads(asyncio.get_event_loop().run_until_complete(
        wt.web_extract_tool(["https://x.test"], use_llm_processing=False, chunk_index=0, format="markdown")
    ))
    out_of_range = json.loads(asyncio.get_event_loop().run_until_complete(
        wt.web_extract_tool(["https://x.test"], use_llm_processing=False, chunk_index=99, format="markdown")
    ))

    assert calls == [("https://x.test",)]
    assert first["results"][0]["chunk_index"] == 1
    assert first["results"][0]["chunk_count"] == 2
    assert first["results"][0]["content"] == "SECOND WEB CHUNK"
    assert cached["results"][0]["chunk_index"] == 0
    assert cached["results"][0]["next_chunk"] == 1
    assert out_of_range["results"][0]["error"] == "chunk_index 99 is out of range for 2 chunks"
    assert "raw_content" not in json.dumps(first)


def test_web_extract_find_searches_cached_source_chunks_and_suggests_followups():
    import tools.web_tools as wt

    cache = wt._ensure_web_extract_cache()
    cache.clear()
    cache[("https://a.test", "markdown", False, "", 5000)] = {
        "url": "https://a.test",
        "title": "A",
        "content": "alpha\nneedle here\nomega",
        "raw_content": "alpha\nneedle here\nomega",
        "input_chunks": ["alpha", "needle here\nomega"],
    }

    out = json.loads(wt.web_extract_find("needle", urls=["https://a.test", "https://missing.test"]))

    assert out["success"] is True
    assert out["searched_urls"] == ["https://a.test"]
    assert out["missing_cache_urls"] == ["https://missing.test"]
    assert out["match_count"] == 1
    assert out["results"][0]["chunk_index"] == 1
    assert "needle here" in out["results"][0]["snippet"]
    assert out["next_actions"] == [
        {"tool": "web_extract", "args": {"urls": ["https://a.test"], "chunk_index": 1}},
        {"tool": "web_extract", "args": {"urls": ["https://missing.test"], "chunk_index": 0}},
    ]


def test_browser_snapshot_full_chunks_source_without_llm_summary(monkeypatch):
    import tools.browser_tool as bt
    from tools.chunked_content import DEFAULT_SOURCE_CHUNK_SIZE

    raw = "A" * DEFAULT_SOURCE_CHUNK_SIZE + "SECOND BROWSER CHUNK [e2]"

    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt, "_extract_relevant_content", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not summarize full chunked snapshots")))
    monkeypatch.setattr(bt, "_truncate_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not truncate full chunked snapshots")))
    monkeypatch.setattr(bt, "_run_browser_command", lambda task_id, command, args: {
        "success": True,
        "data": {"snapshot": raw, "refs": {"e2": {}}, "url": "https://x.test", "title": "X"},
    })

    out = json.loads(bt.browser_snapshot(full=True, chunk_index=1, task_id="snap"))

    assert out["success"] is True
    assert out["chunk_index"] == 1
    assert out["chunk_count"] == 2
    assert out["next_chunk"] is None
    assert out["has_more"] is False
    assert out["snapshot"] == "SECOND BROWSER CHUNK [e2]"


def test_browser_find_searches_cached_full_snapshot_chunks(monkeypatch):
    import tools.browser_tool as bt

    cache_entry = {
        "chunks": ["alpha", "needle target [e7]\n- /url: https://x.test/needle"],
        "url": "https://x.test/page",
        "title": "X",
        "element_count": 1,
    }
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt, "_get_browser_snapshot_source_cache", lambda task_id, full=True: cache_entry)

    out = json.loads(bt.browser_find("needle", task_id="find"))

    assert out["success"] is True
    assert out["chunk_count"] == 2
    assert out["searched_chunks"] == [0, 1]
    assert out["match_count"] == 1
    assert out["results"][0]["chunk_index"] == 1
    assert "needle target" in out["results"][0]["snippet"]
    assert out["next_actions"]["full_snapshot"]["args"] == {"full": True, "chunk_index": 1}


def test_toolsets_and_execute_code_stubs_include_find_and_chunk_index():
    import toolsets
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, _TOOL_STUBS, _TOOL_DOC_LINES

    assert "web_extract_find" in toolsets.TOOLSETS["web"]["tools"]
    assert "browser_find" in toolsets.TOOLSETS["browser"]["tools"]
    assert "web_extract_find" in SANDBOX_ALLOWED_TOOLS
    assert "browser_find" in SANDBOX_ALLOWED_TOOLS
    assert "chunk_index" in _TOOL_STUBS["web_extract"][1]
    assert "chunk_index" in _TOOL_STUBS["browser_snapshot"][1]
    docs = "\n".join(doc for _, doc in _TOOL_DOC_LINES)
    assert "web_extract_find" in docs
    assert "browser_find" in docs
