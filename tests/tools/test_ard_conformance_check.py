from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_conformance_check.py"
spec = importlib.util.spec_from_file_location("ard_conformance_check", SCRIPT)
assert spec is not None and spec.loader is not None
ard_conformance_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ard_conformance_check)


def _valid_catalog() -> dict:
    return {
        "specVersion": "1.0",
        "host": {"displayName": "Hermes", "identifier": "did:web:hermes.local"},
        "entries": [
            {
                "identifier": "urn:ai:hermes.local:skill:webclaw",
                "displayName": "Webclaw",
                "type": "application/ai-skill",
                "description": "Web extraction",
                "data": {"name": "webclaw"},
            },
            {
                "identifier": "urn:ai:hermes.local:mcp:remote",
                "displayName": "Remote MCP",
                "type": "application/mcp-server-card+json",
                "description": "Remote MCP server",
                "url": "https://example.com/mcp/server.json",
            },
        ],
    }


def test_validate_catalog_accepts_url_xor_data_entries() -> None:
    report = ard_conformance_check.validate_catalog(_valid_catalog(), visibility="public")
    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_catalog_rejects_url_and_data_or_empty_url() -> None:
    catalog = _valid_catalog()
    catalog["entries"].append(
        {
            "identifier": "urn:ai:bad:both",
            "displayName": "Bad both",
            "type": "application/ai-skill",
            "url": "https://example.com/a",
            "data": {"name": "bad"},
        }
    )
    catalog["entries"].append(
        {
            "identifier": "urn:ai:bad:empty",
            "displayName": "Bad empty",
            "type": "application/mcp-server-card+json",
            "url": "",
        }
    )

    report = ard_conformance_check.validate_catalog(catalog, visibility="public")

    assert report["ok"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert "entry_source_xor_violation" in codes
    assert "entry_empty_url" in codes


def test_public_catalog_rejects_stdio_local_paths_and_secret_markers() -> None:
    catalog = _valid_catalog()
    catalog["entries"] = [
        {
            "identifier": "urn:ai:bad:stdio",
            "displayName": "Local stdio",
            "type": "application/mcp-server-card+json",
            "url": "stdio:security-tools-mcp",
            "metadata": {"workdir": "/home/ameobius/private", "env": {"API_TOKEN": "secret"}},
        }
    ]

    report = ard_conformance_check.validate_catalog(catalog, visibility="public")

    assert report["ok"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert "public_stdio_url" in codes
    assert "public_sensitive_marker" in codes


def test_private_catalog_allows_stdio_but_not_secret_values() -> None:
    catalog = _valid_catalog()
    catalog["entries"] = [
        {
            "identifier": "urn:ai:local:mcp:stdio",
            "displayName": "Local stdio",
            "type": "application/mcp-server-card+json",
            "url": "stdio:security-tools-mcp",
            "metadata": {"transport": "stdio"},
        }
    ]
    report = ard_conformance_check.validate_catalog(catalog, visibility="private")
    assert report["ok"] is True

    catalog["entries"][0]["metadata"]["env"] = {"API_TOKEN": "secret"}
    report = ard_conformance_check.validate_catalog(catalog, visibility="private")
    assert report["ok"] is False
    assert {issue["code"] for issue in report["errors"]} == {"private_sensitive_marker"}


def test_validate_search_exchange_checks_root_federation_and_referrals() -> None:
    request = {
        "query": {"text": "port scan", "filter": {"type": ["application/mcp-server-card+json"]}},
        "federation": "referrals",
        "pageSize": 5,
    }
    response = {
        "results": [],
        "referrals": [{"url": "https://registry.example/search", "type": "application/ai-registry+json"}],
    }
    report = ard_conformance_check.validate_search_exchange(request, response)
    assert report["ok"] is True

    bad_request = {"query": {"text": "port scan"}, "pageSize": 5}
    bad_response = {"results": [], "federation": {"referrals": ["https://legacy.example/search"]}}
    report = ard_conformance_check.validate_search_exchange(bad_request, bad_response)
    assert report["ok"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert "request_missing_root_federation" in codes
    assert "response_legacy_nested_referrals" in codes
