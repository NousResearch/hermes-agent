"""Exa's free MCP tier throttles a client with HTTP 200 and an ordinary text item (#146853).

No JSON-RPC ``error``, no ``result.isError`` — so ``_parse_mcp_body`` used to hand the throttle
notice back as if it were content: ``web_search`` reported ``success: true`` with zero rows (so the
keyless ring never failed over) and ``web_extract`` returned the notice *as the page*.
"""

import json

import pytest

from plugins.web import keyless_mcp

_THROTTLE = (
    "You've hit Exa's free MCP rate limit. To continue using without limits, create your own "
    "Exa API key.\n\nFix: Create API key at https://dashboard.exa.ai/api-keys , and then update "
    "Exa MCP URL to this https://mcp.exa.ai/mcp?exaApiKey=YOUR_EXA_API_KEY"
)


class _Response:
    """Minimal ``requests.Response`` surface: ``_response_text`` reads headers + content only."""

    status_code = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, text: str) -> None:
        envelope = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}
        self.content = f"data: {json.dumps(envelope)}\n\n".encode()


def _serve(text: str, monkeypatch) -> None:
    monkeypatch.setattr("requests.post", lambda *a, **k: _Response(text))


def test_throttle_notice_raises_instead_of_becoming_content(monkeypatch):
    _serve(_THROTTLE, monkeypatch)
    with pytest.raises(keyless_mcp.KeylessMCPError) as err:
        keyless_mcp.mcp_call(keyless_mcp.EXA_MCP_URL, "web_search_exa", {"query": "x", "numResults": 5})
    assert "rate limit" in str(err.value).lower()


def test_keyless_exa_search_reports_a_rate_limit_so_the_ring_can_advance(monkeypatch):
    _serve(_THROTTLE, monkeypatch)
    result = keyless_mcp.exa_search_keyless("anything")
    assert result["success"] is False
    assert "rate limit" in result["error"].lower()
    # This is the contract the ring's `_throttled` predicate depends on.
    assert keyless_mcp._is_search_failover_eligible(result["error"]) is True


def test_search_fails_over_to_the_next_vendor_instead_of_returning_zero_rows(monkeypatch):
    _serve(_THROTTLE, monkeypatch)
    row = {"url": "https://real.example/hit", "title": "hit", "description": "", "position": 1}
    searchers = dict(keyless_mcp._KEYLESS_SEARCHERS)
    searchers["exa"] = keyless_mcp.exa_search_keyless
    for other in ("parallel", "firecrawl", "keenable"):
        searchers[other] = lambda query, limit, _o=other: keyless_mcp.search_ok([dict(row, url=f"https://{_o}.example/hit")])
    monkeypatch.setattr(keyless_mcp, "_KEYLESS_SEARCHERS", searchers)
    # Pin the walk so Exa is provably tried first: the ring otherwise starts at a rotating cursor,
    # and a vendor that answers before Exa would make this test pass without the fix.
    monkeypatch.setattr(keyless_mcp, "_ring_order", lambda name: ["exa", "parallel", "firecrawl", "keenable"])

    result = keyless_mcp.search_with_failover("exa", "anything")

    assert result["success"] is True
    assert result["data"]["web"], "a throttled Exa must not end the walk with zero rows"
    assert result["data"]["web"][0]["url"] != "https://exa.example/hit"
    assert result["data"]["served_by"] != "exa"


def test_extract_fails_over_instead_of_returning_the_notice_as_the_page(monkeypatch):
    _serve(_THROTTLE, monkeypatch)
    url = "https://real.example/page"
    extractors = dict(keyless_mcp._KEYLESS_EXTRACTORS)
    extractors["exa"] = keyless_mcp.exa_extract_keyless
    for other in ("parallel", "firecrawl", "keenable"):
        extractors[other] = lambda urls, _o=other: [
            keyless_mcp._page(u, "real", f"{_o} served this page") for u in urls
        ]
    monkeypatch.setattr(keyless_mcp, "_KEYLESS_EXTRACTORS", extractors)
    monkeypatch.setattr(keyless_mcp, "_ring_order", lambda name: ["exa", "parallel", "firecrawl", "keenable"])

    results = keyless_mcp.extract_with_failover("exa", [url])

    assert len(results) == 1
    assert "rate limit" not in results[0]["content"].lower()
    assert results[0]["content"].endswith("served this page")


def test_a_real_page_that_mentions_the_phrase_is_not_mistaken_for_a_throttle(monkeypatch):
    page = "Exa free MCP rate limit explained\n\n" + ("Body text about the free MCP rate limit. " * 40)
    _serve(page, monkeypatch)
    text = keyless_mcp.mcp_call(keyless_mcp.EXA_MCP_URL, "web_fetch_exa", {"urls": ["https://exa.ai/docs"]})
    assert text.startswith("Exa free MCP rate limit explained")


def test_other_vendors_are_untouched_by_the_exa_guard(monkeypatch):
    _serve(_THROTTLE, monkeypatch)  # a throttle-shaped payload from somebody else's endpoint
    text = keyless_mcp.mcp_call("https://search.parallel.ai/mcp", "web_search", {"objective": "x"})
    assert "free MCP rate limit" in text
