"""gpu_mcp.py — GPU-MCP server (self-contained plugin copy).

SELF-CONTAINED COPY bundled inside the Hermes Agent VS Code extension.
Canonical source of truth: environments/gpu_mcp.py (hermes-fork repo).

Canonical scheme: +æ://cc home://
  (+æ = sovereign agent protocol; cc = command & control; home = local node)

A zero-dependency (stdlib-only) Model Context Protocol server over stdio.
Exposes the local CUDA + Rust/WASM hands as MCP tools so any MCP client can
drive the local GPU / wasm toolchain. Launched by the VS Code extension's
contributes.mcp.servers entry, so it appears in the client's "MCP Servers" list.

Run (via extension):  python gpu_mcp.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_agent import GPUAgent  # noqa: E402
from rust_wasm_tools import build_wasm, run_wasm  # noqa: E402

_GPU = GPUAgent()


def _tool(name, description, props, required=None):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": props,
            "required": required or [],
        },
    }


TOOLS = [
    _tool("probe_gpu", "Read live NVIDIA GPU telemetry (nvidia-smi) on the local machine.", {}),
    _tool("compile_kernel", "Compile a CUDA matmul kernel with nvcc (via MSVC vcvars).",
          {"name": {"type": "string", "description": "kernel name", "default": "matmul"}}, ["name"]),
    _tool("run_kernel", "Run a previously compiled CUDA kernel on the local GPU.",
          {"name": {"type": "string", "description": "kernel name", "default": "matmul"}}, ["name"]),
    _tool("rust_build_wasm", "Compile a Rust crate to wasm32-unknown-unknown (local).",
          {"crate_dir": {"type": "string", "description": "path to cargo crate"}}, ["crate_dir"]),
    _tool("rust_run_wasm", "Run a .wasm via wasmtime if present; else report capability.",
          {"wasm_path": {"type": "string", "description": "path to .wasm"}}, ["wasm_path"]),
]


def _call(name, args):
    args = args or {}
    if name == "probe_gpu":
        return _GPU.probe_gpu()
    if name == "compile_kernel":
        return _GPU.compile_kernel(args.get("name", "matmul"))
    if name == "run_kernel":
        return _GPU.run_kernel(args.get("name", "matmul"))
    if name == "rust_build_wasm":
        return build_wasm(args["crate_dir"])
    if name == "rust_run_wasm":
        return run_wasm(args["wasm_path"])
    return {"error": f"unknown tool {name}"}


def _respond(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        req_id = msg.get("id")
        if method == "initialize":
            sys.stdout.write(json.dumps(_respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "gpu-mcp", "version": "0.1.0"},
            })) + "\n")
            sys.stdout.flush()
        elif method == "tools/list":
            sys.stdout.write(json.dumps(_respond(req_id, {"tools": TOOLS})) + "\n")
            sys.stdout.flush()
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            out = _call(name, args)
            sys.stdout.write(json.dumps(_respond(req_id, {
                "content": [{"type": "text", "text": json.dumps(out, default=str)}],
                "isError": bool(out.get("error") or out.get("ok") is False),
            })) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    _main()
