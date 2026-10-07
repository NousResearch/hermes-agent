"""Integration coverage for profile-local MCP discovery in slash workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
import threading

import pytest
import hermes_yaml as yaml

_mcp_server_mod = pytest.importorskip("mcp.server")

if not hasattr(_mcp_server_mod, "MCPServer"):
    # `mcp.server.MCPServer` replaced `mcp.server.fastmcp.FastMCP` in mcp 2.0.
    # Skip rather than fail on a FastMCP-era SDK: the probe below is written
    # against the 2.x API, and the pinned version provides it.
    pytest.skip(
        "profile-local MCP discovery probe requires mcp >= 2.0 (MCPServer)",
        allow_module_level=True,
    )


def test_profile_local_mcp_tool_is_visible_in_slash_worker(tmp_path):
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    marker = "profile-local-61922"
    server = tmp_path / "mcp_probe.py"
    server.write_text(
        textwrap.dedent(
            f"""
            from mcp.server import MCPServer

            mcp = MCPServer("profileprobe")

            @mcp.tool()
            def hermes_61922_profile_probe() -> str:
                return {marker!r}

            if __name__ == "__main__":
                mcp.run(transport="stdio")
            """
        ),
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                # Let real discovery finish before the CLI snapshots its tools.
                "mcp_discovery_timeout": 30,
                "mcp_servers": {
                    "profileprobe": {
                        "enabled": True,
                        "command": sys.executable,
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HERMES_SLASH_WATCHDOG_GRACE_S"] = "0"
    env["HERMES_SLASH_WATCHDOG_POLL_S"] = "0.05"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            "agent:main:tui:dm:mcp-profile-test",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    output: queue.Queue[str] = queue.Queue()
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None

        def read_responses():
            for line in proc.stdout:
                output.put(line)
            output.put("")

        threading.Thread(target=read_responses, daemon=True).start()

        def request(request_id, command, timeout):
            proc.stdin.write(json.dumps({"id": request_id, "command": command}) + "\n")
            proc.stdin.flush()
            try:
                line = output.get(timeout=timeout)
            except queue.Empty:
                pytest.fail(f"slash worker did not answer {command} within {timeout}s")
            assert line, f"slash worker exited before answering {command}"
            response = json.loads(line)
            assert response["id"] == request_id
            assert response["ok"] is True, response
            return response

        # Cold imports and real MCP startup have their own budget; the warm
        # command must still answer promptly, with the tool already present.
        request(1, "/version", 60)
        response = request(2, "/tools", 10)
        assert "mcp__profileprobe__hermes_61922_profile_probe" in response["output"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
