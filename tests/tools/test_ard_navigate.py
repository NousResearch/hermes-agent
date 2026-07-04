from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_navigate.py"
spec = importlib.util.spec_from_file_location("ard_navigate", SCRIPT)
assert spec is not None and spec.loader is not None
ard_navigate = importlib.util.module_from_spec(spec)
sys.modules["ard_navigate"] = ard_navigate
spec.loader.exec_module(ard_navigate)


def _resp(payload: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = "{}"
    return r


def test_normalize_search_url_avoids_double_search() -> None:
    assert ard_navigate.normalize_search_url("https://example.com") == "https://example.com/search"
    assert ard_navigate.normalize_search_url("https://example.com/") == "https://example.com/search"
    assert ard_navigate.normalize_search_url("https://example.com/search") == "https://example.com/search"
    assert ard_navigate.normalize_search_url("https://example.com/api/search") == "https://example.com/api/search"


def test_well_known_url_from_domain_or_catalog() -> None:
    assert ard_navigate.well_known_url("https://example.com") == "https://example.com/.well-known/ai-catalog.json"
    assert ard_navigate.well_known_url("https://example.com/a/b") == "https://example.com/.well-known/ai-catalog.json"
    assert ard_navigate.well_known_url("https://example.com/.well-known/ai-catalog.json") == "https://example.com/.well-known/ai-catalog.json"


@patch("ard_navigate.check_website_access", return_value=None)
@patch("ard_navigate.is_safe_url", return_value=True)
@patch("ard_navigate.httpx.get")
@patch("ard_navigate.httpx.post")
def test_navigate_fetches_well_known_searches_and_follows_referrals(mock_post, mock_get, _safe, _policy) -> None:
    mock_get.return_value = _resp(
        {
            "specVersion": "1.0",
            "host": {"displayName": "Primary"},
            "entries": [
                {
                    "identifier": "urn:ai:primary:skill:web",
                    "displayName": "Web Tool",
                    "type": "application/ai-skill",
                    "description": "extract web pages",
                    "data": {"name": "web"},
                }
            ],
            "search_endpoint": {"url": "/search"},
            "referrals": [{"url": "https://secondary.example.com"}],
        }
    )
    first_search = _resp(
        {
            "results": [],
            "referrals": [{"url": "https://third.example.com/search"}],
        }
    )
    second_search = _resp(
        {
            "results": [
                {
                    "identifier": "urn:ai:secondary:mcp:nmap",
                    "displayName": "Nmap MCP",
                    "type": "application/mcp-server-card+json",
                    "description": "port scanning",
                    "url": "https://secondary.example.com/mcp/server.json",
                }
            ]
        }
    )
    third_search = _resp({"results": []})
    mock_post.side_effect = [first_search, second_search, third_search]

    result = ard_navigate.navigate("https://primary.example.com", "port scan", page_size=5, max_depth=2)

    assert result["ok"] is True
    assert result["visited"] == [
        "https://primary.example.com/search",
        "https://secondary.example.com/search",
        "https://third.example.com/search",
    ]
    assert any(e["identifier"] == "urn:ai:secondary:mcp:nmap" for e in result["entries"])
    body = mock_post.call_args_list[0].kwargs["json"]
    assert body["federation"] == "referrals"
    assert body["query"]["filter"]["type"] == [
        "application/ai-skill",
        "application/mcp-server-card+json",
        "application/a2a-agent-card+json",
        "application/vnd.huggingface.space+json",
    ]


@patch("ard_navigate.check_website_access", return_value=None)
@patch("ard_navigate.is_safe_url", return_value=True)
@patch("ard_navigate.httpx.get")
@patch("ard_navigate.httpx.post")
def test_navigate_deduplicates_entries_and_referrals(mock_post, mock_get, _safe, _policy) -> None:
    mock_get.return_value = _resp({"entries": [], "referrals": [{"url": "https://dup.example.com/search"}]})
    mock_post.side_effect = [
        _resp({"results": [{"identifier": "urn:ai:x", "displayName": "X", "type": "application/ai-skill", "data": {}}], "referrals": [{"url": "https://dup.example.com/search"}]}),
        _resp({"results": [{"identifier": "urn:ai:x", "displayName": "X2", "type": "application/ai-skill", "data": {}}]}),
    ]

    result = ard_navigate.navigate("https://primary.example.com", "x", page_size=5, max_depth=1)

    assert [e["identifier"] for e in result["entries"]] == ["urn:ai:x"]
    assert result["visited"] == ["https://primary.example.com/search", "https://dup.example.com/search"]
