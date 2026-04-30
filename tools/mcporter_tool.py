"""Lazy first-class bridge for mcporter-served MCP tools."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.registry import registry

_TOOLSET = "mcporter"
_DEFAULT_TIMEOUT = 60
_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"(Authorization\s*[:=]\s*)[^\s,;}]+", re.IGNORECASE),
    re.compile(r"\b(?:sk|ghp|gho|github_pat)_[A-Za-z0-9_\-]{8,}\b"),
]


def _sanitize_error(text: str) -> str:
    safe = str(text or "")
    for pat in _SECRET_PATTERNS:
        safe = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", safe)
    return safe[-4000:]


def _load_mcporter_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        loaded = load_config()
        cfg = (loaded.get("mcporter") or {}) if isinstance(loaded, dict) else {}
    except Exception:
        cfg = {}

    config_path = os.environ.get("MCPORTER_CONFIG") or cfg.get("config_path")
    resolved_config = _resolve_config_path(config_path)
    return {
        "enabled": cfg.get("enabled", "auto"),
        "command": cfg.get("command") or os.environ.get("MCPORTER_COMMAND") or "npx",
        "args": [str(a) for a in (cfg.get("args") or ["-y", "mcporter"])],
        "timeout": int(cfg.get("timeout", os.environ.get("MCPORTER_TIMEOUT", _DEFAULT_TIMEOUT)) or _DEFAULT_TIMEOUT),
        "config_path": str(resolved_config) if resolved_config else "",
    }


def _resolve_config_path(config_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    candidates.extend([
        Path.cwd() / "config" / "mcporter.json",
        Path(__file__).resolve().parents[1] / "config" / "mcporter.json",
        Path.home() / ".config" / "mcporter" / "mcporter.json",
        Path.home() / ".config" / "mcporter" / "config.json",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip().lower() == "auto":
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _check_requirements() -> bool:
    cfg = _load_mcporter_config()
    return _is_enabled(cfg.get("enabled")) and bool(cfg.get("config_path")) and bool(shutil.which(cfg["command"]))


def _run_mcporter(extra_args: list[str], timeout: int | None = None) -> dict[str, Any]:
    cfg = _load_mcporter_config()
    if not _is_enabled(cfg.get("enabled")):
        return {"error": "mcporter bridge is disabled"}
    if not cfg.get("config_path"):
        return {"error": "mcporter config not found; set mcporter.config_path or MCPORTER_CONFIG"}
    if not shutil.which(cfg["command"]):
        return {"error": f"mcporter command not found: {cfg['command']}"}

    command = [cfg["command"], *cfg.get("args", []), "--config", cfg["config_path"], *extra_args]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout or cfg["timeout"],
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"mcporter command timed out after {timeout or cfg['timeout']}s"}
    except Exception as exc:
        return {"error": _sanitize_error(f"mcporter failed: {type(exc).__name__}: {exc}")}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {"error": _sanitize_error(stderr or stdout or f"mcporter exited {proc.returncode}"), "exit_code": proc.returncode}
    return {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode}


def _json_or_text(payload: str) -> Any:
    text = payload.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def _list_handler(args: dict, **kwargs) -> str:
    extra = ["list"]
    if bool(args.get("schema", False)):
        extra.append("--schema")
    timeout = args.get("timeout")
    result = _run_mcporter(extra, timeout=int(timeout) if timeout else None)
    if "error" in result:
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"result": _json_or_text(result["stdout"])}, ensure_ascii=False)


def _schema_handler(args: dict, **kwargs) -> str:
    server = str(args.get("server") or "").strip()
    if not server:
        return json.dumps({"error": "Missing required parameter: server"})
    timeout = args.get("timeout")
    result = _run_mcporter(["list", server, "--schema"], timeout=int(timeout) if timeout else None)
    if "error" in result:
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"server": server, "schema": _json_or_text(result["stdout"])}, ensure_ascii=False)


def _call_handler(args: dict, **kwargs) -> str:
    server = str(args.get("server") or "").strip()
    tool = str(args.get("tool") or "").strip()
    arguments = args.get("arguments") or {}
    timeout = args.get("timeout")
    if not server:
        return json.dumps({"error": "Missing required parameter: server"})
    if not tool:
        return json.dumps({"error": "Missing required parameter: tool"})
    if not isinstance(arguments, dict):
        return json.dumps({"error": "arguments must be an object"})

    result = _run_mcporter(
        ["call", f"{server}.{tool}", "--args", json.dumps(arguments, ensure_ascii=False), "--output", "json"],
        timeout=int(timeout) if timeout else None,
    )
    if "error" in result:
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"server": server, "tool": tool, "result": _json_or_text(result["stdout"])}, ensure_ascii=False)


registry.register(
    name="mcporter_list",
    toolset=_TOOLSET,
    schema={
        "name": "mcporter_list",
        "description": "List configured mcporter MCP servers. Use before mcporter_schema or mcporter_call when you need to discover available servers.",
        "parameters": {"type": "object", "properties": {"schema": {"type": "boolean", "description": "Include schema summary for all servers; slower."}, "timeout": {"type": "integer", "description": "Optional command timeout in seconds."}}},
    },
    handler=_list_handler,
    check_fn=_check_requirements,
    description="List configured mcporter MCP servers",
    emoji="🔌",
)
registry.register(
    name="mcporter_schema",
    toolset=_TOOLSET,
    schema={
        "name": "mcporter_schema",
        "description": "Inspect tool schemas for one configured mcporter server. Call this before mcporter_call unless you already know the tool arguments.",
        "parameters": {"type": "object", "properties": {"server": {"type": "string", "description": "Configured mcporter server name."}, "timeout": {"type": "integer", "description": "Optional command timeout in seconds."}}, "required": ["server"]},
    },
    handler=_schema_handler,
    check_fn=_check_requirements,
    description="Inspect mcporter server tool schemas",
    emoji="🔎",
)
registry.register(
    name="mcporter_call",
    toolset=_TOOLSET,
    schema={
        "name": "mcporter_call",
        "description": "Call a tool exposed by a configured mcporter MCP server. Prefer this for lightweight retrieval tools served by mcporter before using heavier browser automation.",
        "parameters": {"type": "object", "properties": {"server": {"type": "string", "description": "Configured mcporter server name."}, "tool": {"type": "string", "description": "Tool name on that server."}, "arguments": {"type": "object", "description": "JSON object of tool arguments.", "additionalProperties": True}, "timeout": {"type": "integer", "description": "Optional command timeout in seconds."}}, "required": ["server", "tool", "arguments"]},
    },
    handler=_call_handler,
    check_fn=_check_requirements,
    description="Call a mcporter-served MCP tool",
    emoji="🔌",
)
