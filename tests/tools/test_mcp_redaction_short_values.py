"""Exact-value redaction must not shred ordinary diagnostic text with trivial config values.

A server's resolving snapshot holds every env_file value and every env/header literal, not
only credentials: ``DEBUG=1`` or ``REGION=eu`` sit next to the real token. Matching such
values as substrings rewrites status codes and numbers in the tool error the model reads.
"""
import json
from types import SimpleNamespace

from tools import mcp_tool_config as config
from tools import mcp_tool_handlers as handlers


def _resolved(tmp_path, monkeypatch):
    monkeypatch.delenv("PROJECT_MCP_TOKEN", raising=False)
    env_file = tmp_path / "project.env"
    env_file.write_text("PROJECT_MCP_TOKEN=opaqueProjectCredential42\nREGION=eu\n", encoding="utf-8")
    return config._resolve_mcp_server_config({
        "command": "node", "args": ["server.js"], "env_file": str(env_file),
        "env": {"DEBUG": "1", "API_TOKEN": "${PROJECT_MCP_TOKEN}", "AREA": "${REGION}"},
    })


def test_tool_error_keeps_ordinary_text_while_redacting_the_credential(tmp_path, monkeypatch):
    values = config._mcp_redaction_values(_resolved(tmp_path, monkeypatch))
    result = SimpleNamespace(isError=True, content=[SimpleNamespace(
        type="text", text="HTTP 401 for user 123 in europe: rejected opaqueProjectCredential42")])

    rendered = json.loads(handlers._render_call_tool_result(result, "project", values))["error"]

    assert rendered == "HTTP 401 for user 123 in europe: rejected [REDACTED]"
