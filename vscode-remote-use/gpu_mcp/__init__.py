"""gpu_mcp package — self-contained GPU-MCP server shipped with the Hermes Agent
VS Code extension. Exposes local CUDA + Rust/WASM hands over MCP (stdio).
Canonical source of truth lives in the hermes-fork repo (environments/gpu_mcp.py
and friends); this copy is bundled so the MCP server runs standalone.
"""
from .gpu_mcp import _main, TOOLS, _call

__all__ = ["_main", "TOOLS", "_call"]
