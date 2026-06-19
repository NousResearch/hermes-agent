from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mcp_registry_to_ard_cache.py"
spec = importlib.util.spec_from_file_location("mcp_registry_to_ard_cache", SCRIPT)
assert spec is not None and spec.loader is not None
mcp_registry_to_ard_cache = importlib.util.module_from_spec(spec)
sys.modules["mcp_registry_to_ard_cache"] = mcp_registry_to_ard_cache
spec.loader.exec_module(mcp_registry_to_ard_cache)


def _registry_item(name: str = "ac.example/tool", *, latest: bool = True) -> dict:
    return {
        "server": {
            "name": name,
            "title": "Example Tool",
            "description": "Example MCP server",
            "version": "1.2.3",
            "repository": {"url": "https://github.com/example/tool", "source": "github"},
            "websiteUrl": "https://example.com",
            "remotes": [{"type": "streamable-http", "url": "https://example.com/mcp"}],
        },
        "_meta": {
            "io.modelcontextprotocol.registry/official": {
                "status": "active",
                "isLatest": latest,
                "publishedAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-02T00:00:00Z",
            }
        },
    }


def test_registry_item_to_ard_entry_prefers_remote_url_and_preserves_metadata() -> None:
    entry = mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item())
    assert entry is not None
    assert entry["identifier"] == "urn:ai:registry.modelcontextprotocol.io:mcp:ac.example:tool"
    assert entry["displayName"] == "Example Tool"
    assert entry["type"] == "application/mcp-server-card+json"
    assert entry["url"] == "https://example.com/mcp"
    assert entry["metadata"]["mcpRegistry"]["name"] == "ac.example/tool"
    assert entry["metadata"]["mcpRegistry"]["version"] == "1.2.3"
    assert entry["metadata"]["mcpRegistry"]["isLatest"] is True
    assert entry["metadata"]["transport"] == "streamable-http"


def test_registry_item_to_ard_entry_skips_old_versions_by_default() -> None:
    assert mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item(latest=False)) is None
    assert mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item(latest=False), include_non_latest=True) is not None


def test_registry_item_to_ard_entry_skips_entries_without_remote_url() -> None:
    item = _registry_item()
    item["server"]["remotes"] = []
    assert mcp_registry_to_ard_cache.registry_item_to_ard_entry(item) is None


def test_build_cache_deduplicates_and_writes_ard_cache(tmp_path: Path) -> None:
    entries = [
        mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item("ac.example/tool")),
        mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item("ac.example/tool")),
        mcp_registry_to_ard_cache.registry_item_to_ard_entry(_registry_item("ac.example/other")),
    ]
    output = tmp_path / "ard-cache.json"
    meta = tmp_path / "ard-cache.meta.json"
    result = mcp_registry_to_ard_cache.write_ard_cache([e for e in entries if e], output, meta)
    assert result["total_entries"] == 2
    data = json.loads(output.read_text())
    assert data["version"] == 1
    assert data["specVersion"] == "1.0"
    assert data["host"]["identifier"] == "did:web:registry.modelcontextprotocol.io"
    assert len(data["entries"]) == 2
    meta_data = json.loads(meta.read_text())
    assert meta_data["sources"]["mcp_registry"]["entries"] == 2


@patch("mcp_registry_to_ard_cache.httpx.get")
def test_fetch_registry_page_uses_cursor_and_limit(mock_get) -> None:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"servers": [_registry_item()], "metadata": {"nextCursor": "next"}}
    mock_get.return_value = r
    page = mcp_registry_to_ard_cache.fetch_registry_page("https://registry.example/v0.1/servers", limit=2, cursor="abc")
    assert page["next_cursor"] == "next"
    assert len(page["items"]) == 1
    assert mock_get.call_args.kwargs["params"] == {"limit": 2, "cursor": "abc"}
