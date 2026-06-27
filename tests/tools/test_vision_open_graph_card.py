import json

import pytest

from tools import vision_tools


HTML = """
<!doctype html>
<html>
  <head>
    <title>Fallback title</title>
    <meta property="og:title" content="OG Title">
    <meta property="og:description" content="A useful summary.">
    <meta property="og:image" content="/card.png">
    <meta property="og:url" content="https://example.com/story">
    <meta property="og:type" content="article">
  </head>
  <body>Not an image.</body>
</html>
"""


def test_extract_open_graph_card_resolves_relative_image_url():
    card = vision_tools._extract_open_graph_card(HTML, "https://example.com/path/page")

    assert card == {
        "title": "OG Title",
        "description": "A useful summary.",
        "image": "https://example.com/card.png",
        "url": "https://example.com/story",
        "type": "article",
    }


@pytest.mark.asyncio
async def test_vision_analyze_returns_open_graph_card_for_html_url(monkeypatch):
    async def fake_download(_url, destination, max_retries=3):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(HTML, encoding="utf-8")
        return destination

    monkeypatch.setattr(vision_tools, "_download_image", fake_download)

    result = json.loads(
        await vision_tools.vision_analyze_tool(
            "https://example.com/path/page",
            "what is this?",
        )
    )

    assert result["success"] is True
    assert result["open_graph"]["title"] == "OG Title"
    assert result["open_graph"]["image"] == "https://example.com/card.png"
    assert "did not resolve to image bytes" in result["analysis"]


@pytest.mark.asyncio
async def test_native_vision_returns_open_graph_card_for_html_url(monkeypatch):
    async def fake_download(_url, destination, max_retries=3):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(HTML, encoding="utf-8")
        return destination

    monkeypatch.setattr(vision_tools, "_download_image", fake_download)

    result = json.loads(
        await vision_tools._vision_analyze_native(
            "https://example.com/path/page",
            "what is this?",
        )
    )

    assert result["success"] is True
    assert result["open_graph"]["title"] == "OG Title"
    assert result["open_graph"]["url"] == "https://example.com/story"
