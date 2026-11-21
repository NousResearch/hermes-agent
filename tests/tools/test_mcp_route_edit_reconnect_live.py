"""End-to-end with a real stdio child and a real ``config.yaml`` on disk: an edit to what a LIVE
server is opened with replaces that connection (the credential half of the
url / headers / auth family).

``test_mcp_route_edit_reconnects.py`` pins the verdict against stand-in connections. This one pins
that the verdict reaches the real transport: a credential edit on disk must close the running child,
open a new one with the new value, and leave the profile's tool registered and answering. The unit
tests keep the url / header wording; this is the same identity comparison, proven through the SDK,
the registry and the process teardown.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

SERVER = textwrap.dedent('''\
    import os
    from mcp.server.mcpserver import MCPServer

    app = MCPServer("cred")

    @app.tool()
    def key() -> str:
        return os.environ.get("DEMO_KEY", "")

    app.run()
''')


def _write_config(home, key: str) -> None:
    path = home / "config.yaml"
    path.write_text(
        "mcp_servers:\n  cred:\n"
        f"    command: {json.dumps(sys.executable)}\n"
        f"    args: [{json.dumps(str(home / 'server.py'))}]\n"
        "    env:\n"
        f"      DEMO_KEY: {json.dumps(key)}\n",
        encoding="utf-8")
    # The config read is mtime-keyed; a rewrite inside the same mtime tick must still be seen.
    import os
    stamp = path.stat().st_mtime + 2
    os.utime(path, (stamp, stamp))


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_a_credential_edit_replaces_a_live_stdio_server(monkeypatch, tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    home = tmp_path / "home"
    home.mkdir()
    (home / "server.py").write_text(SERVER, encoding="utf-8")
    _write_config(home, "first")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)

    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.mcp_tool_common import _core
    from tools.registry import registry

    try:
        assert "mcp__cred__key" in disc.discover_mcp_tools()
        key = (registry.current_scope_key(), "cred")
        before = _core._servers[key]
        first = registry.dispatch("mcp__cred__key", {})
        assert "first" in first and "second" not in first

        _write_config(home, "second")  # the edit: same url/command, new credential
        result = disc.reconcile_mcp_servers_with_config()

        assert result["added"] == ["cred"], result
        after = _core._servers.get(key)
        assert after is not None and after is not before, \
            "the live connection must be replaced, not left running on the old credential"
        assert after._config["env"]["DEMO_KEY"] == "second"
        second = registry.dispatch("mcp__cred__key", {})
        assert "second" in second and "first" not in second, \
            "the tool must answer from the new child, not the old one"
        assert registry.snapshot_registration("mcp__cred__key", scope=key[0]) is not None
        # The old child is closed, not leaked beside the new one.
        old_task = getattr(before, "_task", None)
        assert old_task is None or old_task.done()
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        _loop._stop_mcp_loop()
