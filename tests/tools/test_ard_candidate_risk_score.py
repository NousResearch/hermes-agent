from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_candidate_risk_score.py"
spec = importlib.util.spec_from_file_location("ard_candidate_risk_score", SCRIPT)
assert spec is not None and spec.loader is not None
ard_candidate_risk_score = importlib.util.module_from_spec(spec)
sys.modules["ard_candidate_risk_score"] = ard_candidate_risk_score
spec.loader.exec_module(ard_candidate_risk_score)


def _remote_entry() -> dict:
    return {
        "identifier": "urn:ai:mcp:safe",
        "displayName": "Safe MCP",
        "type": "application/mcp-server-card+json",
        "url": "https://example.com/mcp",
        "metadata": {"transport": "streamable-http", "mcpRegistry": {"isLatest": True, "status": "active"}},
    }


def test_score_allows_https_remote_registry_mcp() -> None:
    result = ard_candidate_risk_score.score_entry(_remote_entry())
    assert result["decision"] == "allow"
    assert result["risk"] == "low"
    assert result["reasons"] == []


def test_score_reviews_stdio_and_env_requirements() -> None:
    entry = _remote_entry()
    entry["url"] = "stdio:npx -y dangerous-mcp"
    entry["metadata"]["requiresEnv"] = ["API_TOKEN"]
    result = ard_candidate_risk_score.score_entry(entry)
    assert result["decision"] == "review"
    assert result["risk"] == "high"
    assert "local_command_transport" in result["reasons"]
    assert "requires_env_secrets" in result["reasons"]


def test_score_denies_public_catalog_secret_leak() -> None:
    entry = _remote_entry()
    entry["metadata"]["api_key"] = "sk-1234567890abcdef"
    result = ard_candidate_risk_score.score_entry(entry, visibility="public")
    assert result["decision"] == "deny"
    assert "public_secret_or_path_leak" in result["reasons"]


def test_plan_install_returns_no_side_effect_plan() -> None:
    plan = ard_candidate_risk_score.plan_install(_remote_entry())
    assert plan["side_effects"] == []
    assert plan["next_action"] == "manual_register_after_review"
    assert plan["rollback"] == "remove MCP server/card registration from Hermes config/cache"


def test_cli_scores_catalog_entry(tmp_path: Path, capsys) -> None:
    catalog = {"entries": [_remote_entry()]}
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(catalog))
    rc = ard_candidate_risk_score.main(["--catalog", str(p), "--identifier", "urn:ai:mcp:safe", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "allow"


def test_find_entry_in_catalogs_searches_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"entries": []}))
    second.write_text(json.dumps({"entries": [_remote_entry()]}))
    entry = ard_candidate_risk_score.find_entry_in_catalogs("urn:ai:mcp:safe", [first, second])
    assert entry["displayName"] == "Safe MCP"
