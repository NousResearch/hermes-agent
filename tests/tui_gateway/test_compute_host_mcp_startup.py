"""Isolated turns discover configured MCP tools before their first tool snapshot."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading

import pytest
import hermes_yaml as yaml

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.tui_gateway._compute_host_frames import FrameSink


@pytest.mark.parametrize(("profiles", "explicit_profile_home"), [
    (("launch",), False),
    (("launch",), True),
    (("a", "b", "a"), True),
])
def test_first_isolated_turn_can_call_its_profiles_mcp(tmp_path, profiles, explicit_profile_home):
    peer = tmp_path / "peer.py"
    peer.write_text(textwrap.dedent("""\
        import os
        from mcp.server import MCPServer

        server = MCPServer("identity")

        @server.tool()
        def whoami() -> str:
            return os.environ["PROFILE_CREDENTIAL"]

        server.run(transport="stdio")
        """), encoding="utf-8")
    call = ToolCall("tool_call", {"calls": [{"name": "mcp__identity__whoami", "arguments": {}}]})
    with FakeLLMServer([call, Text("done")] * len(profiles)) as provider:
        launch = tmp_path / "launch"
        launch.mkdir()
        for name in set(profiles):
            home = tmp_path / name
            write_hermes_home(home, provider.base_url, extra_config=yaml.safe_dump({
                "mcp_discovery_timeout": 30,
                "mcp_single_query_discovery_timeout": 30,
                "tools": {"tool_search": {"enabled": True}},
                "mcp_servers": {"identity": {
                    "command": sys.executable,
                    "args": [str(peer)],
                    "env": {"PROFILE_CREDENTIAL": "${MCP_PROFILE_TOKEN}"},
                }},
            }))
            with (home / ".env").open("a", encoding="utf-8") as env_file:
                env_file.write(f"MCP_PROFILE_TOKEN=credential-{name}\n")

        env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
                   HERMES_HOME=str(launch), MCP_PROFILE_TOKEN="wrong-launch-credential",
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        frames = FrameSink()
        errors = []
        with subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.compute_host"],
            cwd=tmp_path, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            def read_frames():
                for line in process.stdout:
                    frames.write(line)

            stdout_reader = threading.Thread(target=read_frames, daemon=True)
            stderr_reader = threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True)
            stdout_reader.start()
            stderr_reader.start()
            try:
                hello = frames.wait_for(lambda f: f.get("type") == "hello", timeout=60)
                assert hello["host_pid"] == process.pid
                for index, name in enumerate(profiles):
                    sid = f"mcp-{index}"
                    frame = {
                        "type": "turn.start", "sid": sid, "session_key": sid,
                        "source": "desktop",
                        "cwd": str(tmp_path), "text": "Call the identity MCP tool.",
                    }
                    if explicit_profile_home:
                        frame["profile_home"] = str(tmp_path / name)
                    process.stdin.write(json.dumps(frame) + "\n")
                    process.stdin.flush()
                    end = frames.wait_for(
                        lambda f: f.get("sid") == sid and f.get("type") in {"turn.end", "turn.error"},
                        timeout=60,
                    )
                    assert end["type"] == "turn.end", (end, "".join(errors))
                    requests = provider.main_requests()
                    assert requests, "".join(errors)
                    results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
                    assert len(results) == 1, (results, "".join(errors))
                    assert f"credential-{name}" in results[0], (results[0], "".join(errors))
                    assert "wrong-launch-credential" not in results[0]
                    for other in set(profiles) - {name}:
                        assert f"credential-{other}" not in results[0]
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                stdout_reader.join(timeout=5)
                stderr_reader.join(timeout=5)
