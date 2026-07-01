import asyncio
import json


async def _allow_all_urls(url):
    return True


def test_extract_x_status_id_accepts_supported_hosts():
    from tools import web_tools

    assert (
        web_tools._extract_x_status_id(
            "https://x.com/davidondrej1/status/2062546475888845157?s=12"
        )
        == "2062546475888845157"
    )
    assert (
        web_tools._extract_x_status_id(
            "https://twitter.com/AnthropicAI/status/2062568862479208923"
        )
        == "2062568862479208923"
    )
    assert web_tools._extract_x_status_id("https://x.com/home") is None
    assert web_tools._extract_x_status_id("https://example.com/i/status/12345") is None


def test_web_extract_uses_fxtwitter_for_x_status(monkeypatch):
    from tools import web_tools

    def fail_backend():
        raise AssertionError("backend should not be used for X status URLs")

    monkeypatch.setattr(web_tools, "_get_extract_backend", fail_backend)
    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all_urls)
    monkeypatch.setattr(
        web_tools,
        "_fetch_fxtwitter_status",
        lambda status_id: {
            "code": 200,
            "tweet": {
                "text": "A test post about local model security.",
                "created_at": "Thu Jun 04 20:12:13 +0000 2026",
                "views": 845,
                "likes": 5,
                "retweets": 0,
                "replies": 0,
                "bookmarks": 0,
                "quotes": 1,
                "author": {"name": "Matthew Berman", "screen_name": "MatthewBerman"},
                "card": {
                    "url": "https://briefing.forwardfuture.ai/p/local-llms",
                    "title": "Local LLMs",
                    "description": "Why running local AI models matters.",
                },
            },
        },
    )

    result_str = asyncio.get_event_loop().run_until_complete(
        web_tools.web_extract_tool(
            ["https://x.com/MatthewBerman/status/2062628512243347566?s=12"],
        )
    )
    result = json.loads(result_str)

    assert result["results"][0]["title"] == "X post 2062628512243347566"
    assert "A test post about local model security." in result["results"][0]["content"]
    assert "Card URL: https://briefing.forwardfuture.ai/p/local-llms" in result["results"][0]["content"]
    assert result["results"][0]["error"] is None


def test_web_extract_reports_fxtwitter_error(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "unused")
    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all_urls)
    monkeypatch.setattr(
        web_tools,
        "_fetch_fxtwitter_status",
        lambda status_id: {"code": 404, "message": "not found"},
    )

    result_str = asyncio.get_event_loop().run_until_complete(
        web_tools.web_extract_tool(
            ["https://x.com/nope/status/2062546475888845157"],
        )
    )
    result = json.loads(result_str)

    assert result["results"][0]["content"] == ""
    assert result["results"][0]["error"] == "not found"
