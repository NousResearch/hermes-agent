"""Fresh-interpreter proof that a safe worker never spawns MCP servers; config-only keeps them."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

PEER = '''import json, os, sys
from pathlib import Path
Path(os.environ["SPAWNED"]).touch()
for line in sys.stdin:
    r = json.loads(line); method = r.get("method"); ident = r.get("id")
    if ident is None: continue
    if method == "initialize":
        result = {"protocolVersion": r["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "owned-peer", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "owned echo",
                             "inputSchema": {"type": "object", "properties": {}}}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)
'''


@pytest.mark.parametrize("mode", ["safe", "config"])
def test_safe_worker_never_spawns_mcp_servers(tmp_path, mode):
    home = tmp_path / mode
    home.mkdir()
    (home / "peer.py").write_text(PEER, encoding="utf-8")
    spawned = home / "spawned"
    server = {"command": sys.executable, "args": [str(home / "peer.py")],
              "env": {"SPAWNED": str(spawned)}}
    config = {"mcp_servers": {"owned": server}, "plugins": {"enabled": []}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), PROBE_MODE=mode)
    for key in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
        env.pop(key, None)
    script = home / "probe.py"
    script.write_text(f"import sys; sys.path.insert(0, {str(root)!r})\n" + textwrap.dedent(f'''
        import json, os
        from agent.safe_worker_policy import _bind_safe_worker_policy
        config = {config!r}
        _bind_safe_worker_policy(safe_mode=os.environ["PROBE_MODE"] == "safe",
                                 ignore_user_config=True, config=config)
        from tools.mcp_tool_config import _load_mcp_config
        from tools.mcp_tool_discovery import discover_mcp_tools, register_mcp_servers
        loaded = _load_mcp_config()
        explicit = register_mcp_servers({{"editor": {server!r}}})
        discovered = discover_mcp_tools()
        # Registration must not hand back already-existing names either.
        again = register_mcp_servers({{"editor": {server!r}}})
        print(json.dumps({{"loaded": sorted(loaded), "explicit": explicit,
                           "discovered": discovered, "again": again}}))
    '''), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    if mode == "safe":
        assert out == {"loaded": [], "explicit": [], "discovered": [], "again": []}, out
        assert not spawned.exists()
    else:
        assert out["loaded"] == ["owned"], out
        assert any(n.endswith("__echo") for n in out["explicit"]), out
        assert spawned.exists()
