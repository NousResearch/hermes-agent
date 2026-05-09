import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_build_firecrawl_search_kwargs_accepts_safe_filters():
    from tools.web_tools import _build_firecrawl_search_kwargs

    kwargs = _build_firecrawl_search_kwargs(
        query="firecrawl docs",
        limit=12,
        sources=["web", "images", "rss", "news"],
        categories=["github", "research", "unsafe", "pdf"],
        include_domains=["https://docs.firecrawl.dev/api-reference/endpoint/search"],
        tbs="qdr:w",
        location="San Francisco,California,United States",
    )

    assert kwargs == {
        "query": "firecrawl docs",
        "limit": 12,
        "sources": ["web", "images", "news"],
        "categories": ["github", "research", "pdf"],
        "include_domains": ["docs.firecrawl.dev"],
        "tbs": "qdr:w",
        "location": "San Francisco,California,United States",
    }


def test_build_firecrawl_scrape_kwargs_uses_safe_documented_options():
    from tools.web_tools import _build_firecrawl_scrape_kwargs

    kwargs = _build_firecrawl_scrape_kwargs(
        format="raw_html",
        formats=["markdown", "summary", "unsafe"],
        only_main_content=False,
        only_clean_content=True,
        include_tags=["main", "article"],
        exclude_tags=["nav"],
        max_age=172800000,
        min_age=1,
        wait_for=500,
        mobile=True,
        timeout=3000,
        pdf_parser_mode="ocr",
        pdf_max_pages=5,
        actions=[
            {"type": "wait", "milliseconds": 250},
            {"type": "click", "selector": "#accept"},
            {"type": "press", "key": "Enter"},
            {"type": "write", "text": "not exposed"},
            {"type": "executeJavascript", "script": "alert(1)"},
        ],
    )

    assert kwargs == {
        "formats": ["markdown", "summary"],
        "only_main_content": False,
        "only_clean_content": True,
        "include_tags": ["main", "article"],
        "exclude_tags": ["nav"],
        "max_age": 172800000,
        "min_age": 1,
        "wait_for": 500,
        "mobile": True,
        "timeout": 3000,
        "parsers": [{"type": "pdf", "mode": "ocr", "max_pages": 5}],
        "actions": [
            {"type": "wait", "milliseconds": 250},
            {"type": "click", "selector": "#accept"},
            {"type": "press", "key": "Enter"},
        ],
    }


def test_web_extract_passes_firecrawl_scrape_kwargs_without_live_api():
    client = MagicMock()
    client.scrape.return_value = {
        "markdown": "# Example",
        "metadata": {"sourceURL": "https://example.com", "title": "Example"},
    }

    with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
         patch("tools.web_tools._get_firecrawl_client", return_value=client), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch("tools.web_tools.check_website_access", return_value=None), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        from tools.web_tools import web_extract_tool

        result = json.loads(_run(web_extract_tool(
            ["https://example.com"],
            format="raw_html",
            use_llm_processing=False,
            include_tags=["article"],
            only_main_content=False,
            wait_for=250,
            pdf_parser_mode="fast",
        )))

    assert result["results"][0]["url"] == "https://example.com"
    assert client.scrape.call_args.kwargs == {
        "url": "https://example.com",
        "formats": ["rawHtml"],
        "only_main_content": False,
        "include_tags": ["article"],
        "wait_for": 250,
        "parsers": [{"type": "pdf", "mode": "fast"}],
    }


def test_web_extract_returns_requested_links_images_and_screenshot_content():
    from tools.web_tools import web_extract_tool

    for requested_format, payload, expected in [
        ("links", {"links": ["https://example.com/a", "https://example.com/b"]}, "https://example.com/a"),
        ("images", {"images": [{"url": "https://example.com/image.png", "alt": "Example"}]}, "image.png"),
        ("screenshot", {"screenshot": "data:image/png;base64,AAAA"}, "[BASE64_IMAGE_REMOVED]"),
    ]:
        client = MagicMock()
        client.scrape.return_value = {
            **payload,
            "metadata": {"sourceURL": "https://example.com", "title": "Example"},
        }

        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_firecrawl_client", return_value=client), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            result = json.loads(_run(web_extract_tool(
                ["https://example.com"],
                format=requested_format,
                use_llm_processing=False,
            )))

        content = result["results"][0]["content"]
        assert f"Firecrawl {requested_format}" in content
        assert expected in content
        assert client.scrape.call_args.kwargs["formats"] == [requested_format]


def test_web_crawl_returns_requested_nested_links_content():
    client = MagicMock()
    client.crawl.return_value = SimpleNamespace(
        status="completed",
        data=[
            {
                "links": ["https://example.com/docs/api"],
                "metadata": {"sourceURL": "https://example.com/docs", "title": "Docs"},
            }
        ],
    )

    with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
         patch.dict("os.environ", {"FIRECRAWL_API_KEY": "fc-test"}), \
         patch("tools.web_tools._get_firecrawl_client", return_value=client), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch("tools.web_tools.check_website_access", return_value=None), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        from tools.web_tools import web_crawl_tool

        result = json.loads(_run(web_crawl_tool(
            "https://example.com",
            scrape_options={"format": "links"},
            use_llm_processing=False,
        )))

    assert "Firecrawl links" in result["results"][0]["content"]
    assert "https://example.com/docs/api" in result["results"][0]["content"]
    assert client.crawl.call_args.kwargs["scrape_options"] == {"formats": ["links"]}


def test_build_firecrawl_crawl_kwargs_uses_prompt_and_safe_options():
    from tools.web_tools import _build_firecrawl_crawl_kwargs

    kwargs = _build_firecrawl_crawl_kwargs(
        instructions="Find API reference pages",
        limit=7,
        include_paths=["api-reference/.*"],
        exclude_paths=["blog/.*"],
        max_discovery_depth=2,
        sitemap="only",
        ignore_query_parameters=True,
        crawl_entire_domain=True,
        allow_subdomains=True,
        allow_external_links=False,
        delay=0.5,
        max_concurrency=2,
        scrape_options={"format": "markdown", "only_main_content": False, "min_age": 1},
    )

    assert kwargs == {
        "prompt": "Find API reference pages",
        "limit": 7,
        "include_paths": ["api-reference/.*"],
        "exclude_paths": ["blog/.*"],
        "max_discovery_depth": 2,
        "sitemap": "only",
        "ignore_query_parameters": True,
        "crawl_entire_domain": True,
        "allow_external_links": False,
        "allow_subdomains": True,
        "delay": 0.5,
        "max_concurrency": 2,
        "scrape_options": {
            "formats": ["markdown"],
            "only_main_content": False,
            "min_age": 1,
        },
    }


def test_web_crawl_passes_instructions_as_firecrawl_prompt_without_live_api():
    client = MagicMock()
    client.crawl.return_value = SimpleNamespace(
        status="completed",
        data=[
            {
                "markdown": "Docs content",
                "metadata": {"sourceURL": "https://example.com/docs", "title": "Docs"},
            }
        ],
    )

    with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
         patch.dict("os.environ", {"FIRECRAWL_API_KEY": "fc-test"}), \
         patch("tools.web_tools._get_firecrawl_client", return_value=client), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch("tools.web_tools.check_website_access", return_value=None), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        from tools.web_tools import web_crawl_tool

        result = json.loads(_run(web_crawl_tool(
            "https://example.com",
            instructions="Find docs",
            limit=3,
            include_paths=["docs/.*"],
            use_llm_processing=False,
        )))

    assert result["results"][0]["title"] == "Docs"
    assert client.crawl.call_args.kwargs["url"] == "https://example.com"
    assert client.crawl.call_args.kwargs["prompt"] == "Find docs"
    assert client.crawl.call_args.kwargs["limit"] == 3
    assert client.crawl.call_args.kwargs["include_paths"] == ["docs/.*"]
    assert client.crawl.call_args.kwargs["scrape_options"] == {"formats": ["markdown"]}


def test_web_crawl_is_registered():
    import tools.web_tools  # noqa: F401
    from tools.registry import registry

    assert "web_crawl" in registry.get_all_tool_names()
    schema = registry.get_entry("web_crawl").schema
    assert schema["parameters"]["properties"]["limit"]["default"] == 20
    assert "include_paths" in schema["parameters"]["properties"]
