"""Wire-level Skills conformance through the real client, discovery and registry dispatch."""

import json
from pathlib import Path
import sys
from unittest.mock import Mock

from tools import approval_prompt
from tools.approval import clear_session
from tools.approval_context import _approval_session_key
from tools.mcp_tool_discovery import register_mcp_servers
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.registry import registry


def test_remote_skill_load_uses_modern_metadata_and_lazy_verified_reads(tmp_path, monkeypatch):
    log = tmp_path / "wire.jsonl"
    fixture = Path(__file__).parent / "fixtures" / "mcp_skills_server.py"
    token = _approval_session_key.set("skills-stdio")
    consent = Mock(return_value="accept")
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", consent)
    try:
        names = register_mcp_servers({"skill_peer": {
            "command": sys.executable, "args": [str(fixture), str(log)],
            "protocol": "stateless", "keepalive_interval": 0,
        }})
        assert "mcp__skill_peer__skill_view" in names
        assert "mcp__skill_peer__list_skills" in names
        def messages():
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert not any(m["method"].startswith(("skills/", "resources/read")) for m in messages())
        listed = json.loads(registry.dispatch("mcp__skill_peer__list_skills", {}))
        uri = listed["skills"][0]["uri"]
        assert not any(m["method"] == "resources/read" for m in messages())
        loaded = json.loads(registry.dispatch("mcp__skill_peer__skill_view", {"uri": uri}))
        assert loaded["success"] and loaded["server"] == "skill_peer"
        assert loaded["content"].startswith("---\nname: review\n")
        reference = json.loads(registry.dispatch("mcp__skill_peer__skill_view", {
            "uri": uri, "revision": loaded["revision"], "file_path": "references/check.md"}))
        assert reference["content"] == "Check correctness.\n"
        assert consent.call_count == 1
        requests = [m for m in messages() if m["method"] in ("skills/list", "skills/get", "resources/read")]
        assert [m["method"] for m in requests] == ["skills/list", "skills/get", "resources/read", "resources/read"]
        for request in requests:
            meta = request["params"]["_meta"]
            assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
            assert meta["io.modelcontextprotocol/clientInfo"]["name"] == "hermes-agent"
            assert "io.modelcontextprotocol/clientCapabilities" in meta
    finally:
        shutdown_mcp_servers()
        clear_session("skills-stdio")
        _approval_session_key.reset(token)
