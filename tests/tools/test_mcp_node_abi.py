"""A stdio MCP server whose native addon was built by another Node.js (#124264).

Hermes runs stdio servers on its own Node only. When such a server dies at startup with a
``NODE_MODULE_VERSION`` mismatch, the user must be told, with the rebuild under Hermes's Node,
instead of an opaque "Connection closed" and a silent park.
"""

import sys
from pathlib import Path

import pytest

# The remedy is rendered in the host shell's syntax; these pin the POSIX form.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell remedy")

_DIES_ON_ABI = """
import pathlib, sys
pathlib.Path(sys.argv[1]).open("a").write("spawn\\n")
addon = sys.argv[2]
sys.stderr.write(
    "node:internal/modules/cjs/loader:1921\\n  return process.dlopen(module, path.toNamespacedPath(filename));\\n\\n"
    f"Error: The module '{addon}'\\nwas compiled against a different Node.js version using\\n"
    "NODE_MODULE_VERSION 127. This version of Node.js requires\\nNODE_MODULE_VERSION 147. Please try "
    "re-compiling or re-installing\\n  code: 'ERR_DLOPEN_FAILED'\\n}\\n\\nNode.js v26.7.0\\n")
sys.exit(1)
"""


@pytest.fixture
def managed_node(tmp_path, monkeypatch):
    """Hermes's PM-installed node/npm, as PM would report them."""
    store = tmp_path / "tools"
    node, npm = store / "node-26.7.0-linux-x64" / "bin" / "node", store / "npm-12.0.2-linux-x64" / "bin" / "npm"
    from tools import mcp_tool_node_abi
    monkeypatch.setattr(mcp_tool_node_abi, "_managed",
                        lambda name: {"node": (node, "26.7.0"), "npm": (npm, "12.0.2")}[name])
    return node, npm


def test_stdio_server_dying_on_a_node_abi_mismatch_names_the_rebuild_under_hermes_node(tmp_path, monkeypatch,
                                                                                         managed_node):
    from hermes_cli.mcp_config import _probe_failure_reason, _probe_single_server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    node, npm = managed_node
    entry = tmp_path / ".npm" / "_npx" / "0e2eab1c"
    addon = entry / "node_modules" / "better-sqlite3" / "build" / "Release" / "better_sqlite3.node"
    spawns, server = tmp_path / "spawns.log", tmp_path / "server.py"
    server.write_text(_DIES_ON_ABI, encoding="utf-8")

    with pytest.raises(Exception) as caught:
        _probe_single_server("sqlite", {"command": sys.executable, "args": [str(server), str(spawns), str(addon)],
                                        "connect_timeout": 20})

    reason = _probe_failure_reason(caught.value)
    assert "NODE_MODULE_VERSION 127; Hermes's Node 26.7.0 needs 147" in reason
    assert f"rm -rf {entry}" in reason
    assert f"PATH={node.parent}:\"$PATH\" {npm} rebuild better-sqlite3 --prefix {entry}" in reason
    # Every retry loads the same binary: parked at once, not walked through the retry ladder.
    assert spawns.read_text().count("spawn") == 1
    # The shared stderr log still receives the child's output.
    assert "ERR_DLOPEN_FAILED" in (tmp_path / "home" / "logs" / "mcp-stderr.log").read_text()


def test_the_remedy_never_points_the_server_at_another_node(managed_node):
    """A module outside an npx cache gets the rebuild alone, still with Hermes's npm under Hermes's node,
    and nothing suggests pinning ``command:`` to the user's Node."""
    from tools.mcp_tool_node_abi import node_abi_error

    node, npm = managed_node
    pkg_root = Path("/usr/lib/node_modules/@acme/mcp-server")
    stderr = (f"Error: The module '{pkg_root}/node_modules/sharp/build/Release/sharp.node'\nwas compiled against a "
              "different Node.js version using\nNODE_MODULE_VERSION 127. This version of Node.js requires\n"
              "NODE_MODULE_VERSION 147.\n  code: 'ERR_DLOPEN_FAILED'\n")

    message = str(node_abi_error("imgs", stderr))

    assert f"{npm}" in message and f"{node.parent}" in message and " rebuild sharp --prefix " in message
    assert str(pkg_root) in message and "rm -rf" not in message and "command:" not in message
