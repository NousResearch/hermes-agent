#!/usr/bin/env python3
"""MiniMax CLI tools.

Small, provider-native wrapper around the official ``mmx`` CLI from
MiniMax's Token Plan docs.  The CLI owns auth/region/quota semantics; the
Hermes tool only validates inputs, chooses safe output paths, and returns
compact JSON.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error


ASPECT_RATIOS = {
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
    "1:1": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
}


def _cache_dir() -> Path:
    path = get_hermes_home() / "cache" / "minimax-cli"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime = get_hermes_home() / "runtime-secrets.env"
    if runtime.exists():
        for raw in runtime.read_text(errors="ignore").splitlines():
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if key and key not in env:
                env[key] = value.strip().strip('"').strip("'")
    return env


def _run_mmx(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    if not shutil.which("mmx"):
        raise RuntimeError("MiniMax CLI `mmx` is not installed. Install with: npm install -g mmx-cli")
    return subprocess.run(
        ["mmx", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_runtime_env(),
        check=False,
    )


def _json_or_text(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _completed_or_error(proc: subprocess.CompletedProcess[str], command: list[str]) -> None:
    if proc.returncode == 0:
        return
    stderr = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
    safe_command = "mmx " + " ".join(shlex.quote(p) for p in command)
    raise RuntimeError(f"{safe_command} failed: {stderr[:1200]}")


def minimax_cli_requirements() -> bool:
    return bool(shutil.which("mmx"))


def minimax_cli_status(task_id: str | None = None) -> str:
    """Return auth status and Token Plan quota from the official mmx CLI."""
    try:
        auth_cmd = ["auth", "status"]
        quota_cmd = ["quota"]
        auth = _run_mmx(auth_cmd, timeout=60)
        _completed_or_error(auth, auth_cmd)
        quota = _run_mmx(quota_cmd, timeout=120)
        _completed_or_error(quota, quota_cmd)
        return json.dumps(
            {
                "ok": True,
                "provider": "minimax-cli",
                "mmx_path": shutil.which("mmx"),
                "auth": _json_or_text(auth.stdout),
                "quota": _json_or_text(quota.stdout),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            # Deliberately ignore MiniMax thinking blocks; tools should not
            # surface hidden reasoning back into the agent context.
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part.get("content"), str):
                texts.append(part["content"])
    return "\n".join(t for t in texts if t).strip()


def minimax_text_chat(
    message: str,
    system: str | None = None,
    model: str = "MiniMax-M2.7-highspeed",
    max_tokens: int = 4096,
    task_id: str | None = None,
) -> str:
    """Run a MiniMax text chat completion through ``mmx text chat``."""
    if not message or not message.strip():
        return tool_error("message is required")
    max_tokens = max(1, min(int(max_tokens or 4096), 32768))
    cmd = [
        "text",
        "chat",
        "--model",
        model,
        "--message",
        message,
        "--max-tokens",
        str(max_tokens),
        "--output",
        "json",
    ]
    if system:
        cmd.extend(["--system", system])
    try:
        proc = _run_mmx(cmd, timeout=600)
        _completed_or_error(proc, cmd)
        payload = _json_or_text(proc.stdout)
        text = _extract_text(payload.get("content")) if isinstance(payload, dict) else str(payload or "")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        model_used = payload.get("model") if isinstance(payload, dict) else model
        return json.dumps(
            {
                "ok": True,
                "provider": "minimax-cli",
                "model": model_used,
                "text": text,
                "usage": usage,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


def minimax_image_generate(
    prompt: str,
    aspect_ratio: str = "square",
    output_path: str | None = None,
    prompt_optimizer: bool = False,
    seed: int | None = None,
    task_id: str | None = None,
) -> str:
    """Generate an image with MiniMax image-01 through ``mmx image generate``."""
    if not prompt or not prompt.strip():
        return tool_error("prompt is required")
    ratio = ASPECT_RATIOS.get(aspect_ratio, aspect_ratio)
    if ratio not in {"1:1", "16:9", "9:16", "4:3", "3:4"}:
        return tool_error("aspect_ratio must be one of square, landscape, portrait, 1:1, 16:9, 9:16, 4:3, 3:4")
    if output_path:
        out = Path(output_path).expanduser()
        if not out.is_absolute():
            out = _cache_dir() / out
    else:
        out = _cache_dir() / f"minimax-image-01-{uuid.uuid4().hex[:12]}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "image",
        "generate",
        "--prompt",
        prompt,
        "--aspect-ratio",
        ratio,
        "--out",
        str(out),
        "--output",
        "json",
    ]
    if prompt_optimizer:
        cmd.append("--prompt-optimizer")
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    try:
        proc = _run_mmx(cmd, timeout=900)
        _completed_or_error(proc, cmd)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("mmx completed but did not write an image file")
        return json.dumps(
            {
                "ok": True,
                "provider": "minimax-cli",
                "model": "image-01",
                "path": str(out),
                "bytes": out.stat().st_size,
                "aspect_ratio": ratio,
                "raw": _json_or_text(proc.stdout),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


registry.register(
    name="minimax_cli_status",
    toolset="minimax_cli",
    schema={
        "name": "minimax_cli_status",
        "description": "Check official MiniMax CLI auth status and Token Plan quota via `mmx auth status` and `mmx quota`.",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: minimax_cli_status(task_id=kw.get("task_id")),
    check_fn=minimax_cli_requirements,
    emoji="📊",
)

registry.register(
    name="minimax_text_chat",
    toolset="minimax_cli",
    schema={
        "name": "minimax_text_chat",
        "description": "Run MiniMax text generation through the official Token Plan CLI (`mmx text chat`).",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "User message to send."},
                "system": {"type": "string", "description": "Optional system prompt."},
                "model": {"type": "string", "default": "MiniMax-M2.7-highspeed"},
                "max_tokens": {"type": "integer", "default": 4096},
            },
            "required": ["message"],
        },
    },
    handler=lambda args, **kw: minimax_text_chat(
        message=args.get("message", ""),
        system=args.get("system"),
        model=args.get("model") or "MiniMax-M2.7-highspeed",
        max_tokens=args.get("max_tokens") or 4096,
        task_id=kw.get("task_id"),
    ),
    check_fn=minimax_cli_requirements,
    emoji="🧠",
)

registry.register(
    name="minimax_image_generate",
    toolset="minimax_cli",
    schema={
        "name": "minimax_image_generate",
        "description": "Generate an image with MiniMax image-01 via the official Token Plan CLI. Returns a local image path suitable for MEDIA delivery.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image prompt."},
                "aspect_ratio": {"type": "string", "enum": ["square", "landscape", "portrait", "1:1", "16:9", "9:16", "4:3", "3:4"], "default": "square"},
                "output_path": {"type": "string", "description": "Optional absolute or cache-relative output path."},
                "prompt_optimizer": {"type": "boolean", "default": False},
                "seed": {"type": "integer", "description": "Optional deterministic seed."},
            },
            "required": ["prompt"],
        },
    },
    handler=lambda args, **kw: minimax_image_generate(
        prompt=args.get("prompt", ""),
        aspect_ratio=args.get("aspect_ratio") or "square",
        output_path=args.get("output_path"),
        prompt_optimizer=bool(args.get("prompt_optimizer", False)),
        seed=args.get("seed"),
        task_id=kw.get("task_id"),
    ),
    check_fn=minimax_cli_requirements,
    emoji="🖼️",
)
