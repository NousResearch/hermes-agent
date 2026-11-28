"""openai-native web provider: client-side search + extract through the Codex search endpoint.

Regression for #135684 — the provider used to be a marker that made the Codex transport
declare the hosted ``web_search`` built-in, which fails the turn with ``server_error`` on the
Codex backend. It now serves ``web_search`` / ``web_extract`` itself via ``{base}/alpha/search``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

_BASE = "https://chatgpt.com/backend-api/codex"
_CITE = "cite"


def _resp(payload):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = payload
    return m


def test_search_and_extract_go_through_codex_search_endpoint(monkeypatch):
    """Both capabilities hit the same token's host at ``alpha/search`` with Hermes' own Codex
    identity, and the Codex citation markup never reaches the model."""
    import agent.auxiliary_client as aux
    from plugins.web.openai_native.provider import OpenAINativeWebSearchProvider

    monkeypatch.setattr(aux, "_resolve_codex_credential_and_base", lambda: ("tok", _BASE))
    search_payload = {"output": "…", "results": [
        {"type": "text_result", "title": "Changelog", "url": "https://obsidian.md/changelog/",
         "snippet": f"Latest {_CITE}turn0search0 release"},
        {"type": "text_result", "title": "", "domain": "github.com", "url": "https://github.com/o/r"},
    ]}
    open_payload = {"output": (
        "Changelog - Obsidian (https://obsidian.md/changelog/)\n"
        f"{_CITE}turn0view0 [wordlim: 200] Crawled: today; Total lines: 3\n"
        f"L0: {_CITE}0†Download L1: # Changelog\nL2: v1.14.4")}

    provider = OpenAINativeWebSearchProvider()
    with patch("httpx.post", side_effect=[_resp(search_payload), _resp(open_payload)]) as post:
        found = provider.search("obsidian version", limit=5)
        pages = provider.extract(["https://obsidian.md/changelog/"])

    for call, command in zip(post.call_args_list, ("search_query", "open")):
        assert call.args[0] == f"{_BASE}/alpha/search"
        assert call.kwargs["headers"]["Authorization"] == "Bearer tok"
        assert call.kwargs["headers"]["originator"] == "hermes-agent"
        assert command in call.kwargs["json"]["commands"]
        assert call.kwargs["json"]["model"]
    assert found["success"] is True
    assert [(r["title"], r["url"], r["position"]) for r in found["data"]["web"]] == [
        ("Changelog", "https://obsidian.md/changelog/", 1), ("github.com", "https://github.com/o/r", 2)]
    assert found["data"]["web"][0]["description"] == "Latest  release"
    assert pages[0]["title"] == "Changelog - Obsidian"
    assert pages[0]["content"] == "Download\n# Changelog\nv1.14.4"
    assert not any("" in str(v) for v in (found, pages))
