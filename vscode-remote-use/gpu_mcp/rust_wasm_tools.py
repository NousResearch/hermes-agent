"""rust_wasm_tools.py — Rust/WASM hands for the +æ^glocal host loop.

SELF-CONTAINED COPY bundled inside the Hermes Agent VS Code extension.
Canonical source of truth: environments/rust_wasm_tools.py (hermes-fork repo).
This copy ships with the plugin so the GPU-MCP server runs standalone.

Tools (all local, air-gappable):
  rust_build_wasm : cargo build --target wasm32-unknown-unknown (or wasm-pack build)
  rust_run_wasm   : execute a .wasm via wasmtime if present; else report capability.
"""
from __future__ import annotations
import json, os, shutil, subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _wasmtime() -> Optional[str]:
    return shutil.which("wasmtime")


def _rustup_cargo() -> Optional[str]:
    """Prefer the rustup-managed cargo shim (knows about wasm32 std in RUSTUP_HOME)
    over a standalone `rustc` install on PATH that may lack the target std."""
    home = os.environ.get("RUSTUP_HOME", os.path.join(os.path.expanduser("~"), ".rustup"))
    shim = os.path.join(home, "toolchains", "stable-x86_64-pc-windows-msvc", "bin", "cargo.exe")
    if os.path.exists(shim):
        return shim
    return shutil.which("cargo")


def _wasm_std_present() -> bool:
    home = os.environ.get("RUSTUP_HOME", os.path.join(os.path.expanduser("~"), ".rustup"))
    d = os.path.join(home, "toolchains", "stable-x86_64-pc-windows-msvc",
                     "lib", "rustlib", "wasm32-unknown-unknown", "lib")
    if not os.path.isdir(d):
        return False
    return any(f.startswith("libstd") and f.endswith(".rlib") for f in os.listdir(d))


def _rustup_env() -> Dict[str, str]:
    """Return an env dict that puts rustup's bin first, so cargo->rustc both
    resolve to the rustup shim (not a standalone Rust install that lacks wasm32 std)."""
    home = os.environ.get("RUSTUP_HOME", os.path.join(os.path.expanduser("~"), ".rustup"))
    bin_dir = os.path.join(home, "toolchains", "stable-x86_64-pc-windows-msvc", "bin")
    cargo_home = os.environ.get("CARGO_HOME", os.path.join(os.path.expanduser("~"), ".cargo", "bin"))
    env = dict(os.environ)
    env["PATH"] = bin_dir + os.pathsep + cargo_home + os.pathsep + env.get("PATH", "")
    return env


def build_wasm(crate_dir: str, use_pack: bool = False, release: bool = True) -> Dict[str, Any]:
    crate = Path(crate_dir)
    if not (crate / "Cargo.toml").exists():
        return {"ok": False, "error": f"no Cargo.toml at {crate}"}
    if not _wasm_std_present():
        return {"ok": False, "error": "wasm32-unknown-unknown std not installed "
                                      "(run: rustup target add wasm32-unknown-unknown)"}
    cargo = _rustup_cargo()
    if not cargo:
        return {"ok": False, "error": "cargo not found on PATH"}
    cmd = [cargo, "build", "--target", "wasm32-unknown-unknown"]
    if release:
        cmd.append("--release")
    if use_pack:
        cmd = ["wasm-pack", "build", "--target", "web"] + (["--release"] if release else [])
    env = _rustup_env()
    r = subprocess.run(cmd, cwd=str(crate), capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout)[-400]}
    cand = sorted(crate.glob("target/wasm32-unknown-unknown/**/*.wasm"), reverse=True) or \
           sorted(crate.glob("pkg/**/*.wasm"), reverse=True)
    if not cand:
        return {"ok": False, "error": "build succeeded but no .wasm found"}
    wasm = cand[0]
    size = wasm.stat().st_size
    return {"ok": True, "wasm": str(wasm), "bytes": size,
            "tool": "wasm-pack" if use_pack else "cargo"}


def run_wasm(wasm_path: str, args: Optional[list[str]] = None) -> Dict[str, Any]:
    wasm = Path(wasm_path)
    if not wasm.exists():
        return {"ok": False, "error": f"no such wasm: {wasm}"}
    wt = _wasmtime()
    if not wt:
        return {"ok": False, "reason": "no_runtime",
                "detail": "wasmtime not on PATH; install via `cargo install wasmtime` to execute locally",
                "wasm": str(wasm), "bytes": wasm.stat().st_size}
    cmd = [wt, "run", str(wasm)] + (args or [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return {"ok": r.returncode == 0, "stdout": r.stdout[:400], "stderr": r.stderr[:200],
            "wasm": str(wasm), "bytes": wasm.stat().st_size}


def toolset(crate_dir: str) -> Dict[str, Any]:
    crate = Path(crate_dir)

    def _run(_arg=None) -> Dict[str, Any]:
        built = build_wasm(str(crate), use_pack=False)
        if not built.get("ok"):
            return built
        return run_wasm(built["wasm"])

    return {
        "rust_build_wasm": lambda _a: build_wasm(str(crate), use_pack=False),
        "rust_run_wasm": _run,
    }
