"""Hermes agent tool: OpenClaw LINE/Telegram channel readiness (no secret leakage)."""

from __future__ import annotations

import json
from pathlib import Path

from tools.openclaw.channel_readiness import build_channel_readiness
from tools.openclaw.paths import default_openclaw_config_path, default_openclaw_state_root
from tools.registry import registry


def channel_readiness_check(config_path: str = "") -> str:
    if config_path:
        cfg = Path(config_path).expanduser()
        state_root = cfg.parent
    else:
        cfg = default_openclaw_config_path()
        state_root = cfg.parent if cfg.is_file() else default_openclaw_state_root()
    if not cfg.is_file() and not config_path:
        cfg = state_root / "openclaw.json"
    result = build_channel_readiness(cfg, state_root)
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="channel_readiness_check",
    toolset="openclaw",
    schema={
        "name": "channel_readiness_check",
        "description": (
            "Diagnose whether OpenClaw LINE/Telegram channels are configured for live "
            "round-trip messaging. Reports credential presence and candidate targets "
            "without echoing secrets. Uses OPENCLAW_CONFIG or ~/.openclaw/openclaw.json."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Optional path to openclaw.json (default: auto-detect).",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: channel_readiness_check(args.get("config_path", "")),
    emoji="📡",
)
