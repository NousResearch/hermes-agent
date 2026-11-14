"""URL-mode elicitation over a real stdio MCP server: the SDK advertises URL-mode support whenever an
elicitation callback is set, so a server's ``ctx.elicit_url`` must reach the user's consent prompt
with the full URL, and only an explicit accept goes back to the server."""

import json
import sys
import textwrap

import pytest

SERVER = textwrap.dedent('''\
    import sys
    from mcp.server.mcpserver import MCPServer, Context
    app = MCPServer("auth")

    @app.tool()
    async def login(ctx: Context) -> str:
        result = await ctx.elicit_url(message="Sign in to continue", url=sys.argv[1], elicitation_id="e1")
        return f"action={result.action}"

    app.run()
''')


def _call_login(monkeypatch, tmp_path, url, answer):
    """Connect the server, dispatch ``login``; returns (tool result, consent calls)."""
    pytest.importorskip("mcp.server.mcpserver")
    home = tmp_path / "home"
    home.mkdir()
    (home / "server.py").write_text(SERVER, encoding="utf-8")
    (home / "config.yaml").write_text(
        f"mcp_servers:\n  auth:\n    command: {json.dumps(sys.executable)}\n"
        f"    args: [{json.dumps(str(home / 'server.py'))}, {json.dumps(url)}]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    asked = []

    def consent(message, description, **kwargs):  # the user's answer; the handler under test is real
        asked.append(description)
        return answer

    monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", consent)
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.registry import registry
    try:
        assert "mcp__auth__login" in disc.discover_mcp_tools()
        return registry.dispatch("mcp__auth__login", {}), asked
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        _loop._stop_mcp_loop()


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_url_elicitation_shows_the_full_url_and_returns_the_users_accept(monkeypatch, tmp_path):
    url = "https://auth.example.com/oauth/authorize?client_id=hermes&state=xyz"
    result, asked = _call_login(monkeypatch, tmp_path, url, "accept")

    assert "action=accept" in result
    assert len(asked) == 1 and url in asked[0] and "auth.example.com" in asked[0]


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_url_elicitation_with_a_non_http_scheme_declines_without_asking(monkeypatch, tmp_path):
    result, asked = _call_login(monkeypatch, tmp_path, "javascript:alert(document.cookie)", "accept")

    assert "action=decline" in result
    assert asked == []
