"""MCP benchmark server: N dynamically generated echo tools over stdio.

Used by ``scripts/benchmark_utcp.py`` as the MCP baseline. Tool count comes from
``BENCH_MCP_TOOLS`` so one script serves every configuration.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

COUNT = int(os.environ.get("BENCH_MCP_TOOLS", "100"))

server = FastMCP("bench-mcp")


def _make(index: int):
    def _tool(message: str) -> str:
        """Echo a message back through the MCP benchmark server."""
        return f"mcp:{index}:{message}"

    _tool.__name__ = f"mcp_op_{index:04d}"
    return _tool


for _index in range(COUNT):
    server.add_tool(_make(_index), name=f"mcp_op_{_index:04d}", description=f"MCP bench op {_index}")


if __name__ == "__main__":
    server.run()
