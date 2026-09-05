"""Cross-process regression coverage for MCP discovery serialization.

Two independent Hermes processes can start MCP discovery at the same time
(dashboard and gateway startup).  The losing process must wait for the shared
lock and then perform its own local discovery; another process's registry is
not usable because ``_servers`` is process-local.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _wait_for_file(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_two_processes_each_complete_local_mcp_discovery(tmp_path):
    """A lock loser waits, acquires the lock, and builds its own registry."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()

    holder_ready = tmp_path / "holder-ready"
    release_holder = tmp_path / "release-holder"
    loser_started = tmp_path / "loser-started"
    holder_output = tmp_path / "holder.json"
    loser_output = tmp_path / "loser.json"
    child_script = tmp_path / "mcp-discovery-child.py"

    child_script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            import sys
            import time
            from types import SimpleNamespace

            repo_root, role, ready_arg, release_arg, started_arg, output_arg = sys.argv[1:]
            sys.path.insert(0, repo_root)

            import tools.mcp_tool as mcp_tool

            ready = Path(ready_arg)
            release = Path(release_arg)
            started = Path(started_arg)
            output = Path(output_arg)

            mcp_tool._MCP_AVAILABLE = True
            mcp_tool._MCP_DISCOVERY_LOCK_PATH = None
            mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES = 200
            mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S = 0.01
            mcp_tool._servers.clear()

            config = {
                "test_srv": {
                    "command": "fake",
                    "enabled": True,
                }
            }
            mcp_tool._load_mcp_config = lambda: config

            def fake_register_mcp_servers(servers):
                tool_name = "mcp__test_srv__ping"
                mcp_tool._servers["test_srv"] = SimpleNamespace(
                    _registered_tool_names=[tool_name],
                )

                if role == "holder":
                    ready.write_text("1", encoding="utf-8")
                    deadline = time.monotonic() + 10.0
                    while not release.exists():
                        if time.monotonic() >= deadline:
                            raise RuntimeError("holder release signal timed out")
                        time.sleep(0.01)

                return [tool_name]

            mcp_tool.register_mcp_servers = fake_register_mcp_servers
            started.write_text("1", encoding="utf-8")

            result = mcp_tool.discover_mcp_tools()
            server = mcp_tool._servers.get("test_srv")
            output.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "result": result,
                        "server_present": server is not None,
                        "registered_tools": (
                            list(server._registered_tool_names)
                            if server is not None
                            else []
                        ),
                    }
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)

    holder_started = tmp_path / "holder-started"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(child_script),
            str(_REPO_ROOT),
            "holder",
            str(holder_ready),
            str(release_holder),
            str(holder_started),
            str(holder_output),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    loser = None

    try:
        _wait_for_file(holder_ready)

        loser = subprocess.Popen(
            [
                sys.executable,
                str(child_script),
                str(_REPO_ROOT),
                "loser",
                str(holder_ready),
                str(release_holder),
                str(loser_started),
                str(loser_output),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_file(loser_started)

        # The loser must not finish from an empty process-local registry while
        # the holder owns the lock.
        time.sleep(0.1)
        assert loser.poll() is None
        assert not loser_output.exists()

        release_holder.write_text("1", encoding="utf-8")

        holder_stdout, holder_stderr = holder.communicate(timeout=15)
        loser_stdout, loser_stderr = loser.communicate(timeout=15)
        assert holder.returncode == 0, holder_stdout + holder_stderr
        assert loser.returncode == 0, loser_stdout + loser_stderr
    finally:
        release_holder.touch(exist_ok=True)
        for process in (holder, loser):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    expected_tools = ["mcp__test_srv__ping"]
    holder_result = json.loads(holder_output.read_text(encoding="utf-8"))
    loser_result = json.loads(loser_output.read_text(encoding="utf-8"))

    assert holder_result["pid"] != loser_result["pid"]
    for result in (holder_result, loser_result):
        assert result["result"] == expected_tools
        assert result["server_present"] is True
        assert result["registered_tools"] == expected_tools
