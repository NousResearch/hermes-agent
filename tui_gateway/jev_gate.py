"""JEV pre-turn gate — an optional, fully detachable speed layer.

Before a user turn runs, this asks an external JEV (TypeSafe "System One")
classifier whether the turn needs any tool. When it is confident the turn needs
NO tools, the caller runs that single turn with tools disabled, so plainly
conversational messages skip the agentic tool-search loop entirely.

Design goals:
  * OPT-IN — off unless `jev.gate_enabled: true` in config.yaml, or env
    `HERMES_JEV_GATE_ENABLED=true`. Off = zero behaviour change.
  * FAIL-OPEN — any missing key / HTTP error / timeout / slow response returns
    False, i.e. the turn runs normally with tools intact. JEV can only ever make
    a turn cheaper, never block one.
  * DETACHABLE — everything lives in this file plus one small guarded block in
    `prompt_turn.py::run_body` (search "JEV pre-turn gate"). Delete both to remove.

Config keys (all optional, under a top-level `jev:` section):
    jev:
      gate_enabled: true          # master switch
      api_key: "apikey_..."       # or env TYPESAFE_API_KEY
      model: "jev-latest"
      timeout_seconds: 5.0
      skip_confidence: 0.7        # only skip tools when P(no-tools) >= this
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() == "true"


def _config() -> Dict[str, Any]:
    """Read gate config, falling back to pure-env when config.yaml is unavailable."""
    try:
        from hermes_cli.config import load_config_readonly, cfg_get

        cfg = load_config_readonly()
        return {
            "enabled": bool(cfg_get(cfg, "jev", "gate_enabled", default=False))
            or _env_true("HERMES_JEV_GATE_ENABLED"),
            "api_key": cfg_get(cfg, "jev", "api_key", default="")
            or os.environ.get("TYPESAFE_API_KEY", ""),
            "model": cfg_get(cfg, "jev", "model", default="jev-latest") or "jev-latest",
            "timeout": float(cfg_get(cfg, "jev", "timeout_seconds", default=5.0) or 5.0),
            "skip_confidence": float(
                cfg_get(cfg, "jev", "skip_confidence", default=0.7) or 0.7
            ),
        }
    except Exception:
        return {
            "enabled": _env_true("HERMES_JEV_GATE_ENABLED"),
            "api_key": os.environ.get("TYPESAFE_API_KEY", ""),
            "model": os.environ.get("JEV_MODEL", "jev-latest"),
            "timeout": 5.0,
            "skip_confidence": 0.7,
        }


def is_enabled() -> bool:
    """True only when the gate is switched on AND an API key is present."""
    c = _config()
    return bool(c["enabled"] and c["api_key"])


def should_disable_tools(text: Any) -> bool:
    """Ask JEV if this turn needs no tools. Returns True only on a confident
    "no tools" verdict; fail-open to False on anything uncertain or broken."""
    if not isinstance(text, str) or not text.strip():
        return False
    c = _config()
    if not c["api_key"]:
        return False
    try:
        import requests

        resp = requests.post(
            _JEV_ENDPOINT,
            headers={
                "Authorization": f"Bearer {c['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "state": text,
                "model": c["model"],
                "questions": {
                    "needs_tools": {
                        "type": "noul",
                        "instructions": (
                            "Does answering this message require an external tool: looking "
                            "up channels, fetching data from another system, creating or "
                            "updating a ticket, or sending / posting / scheduling a message "
                            "or marketing broadcast to an outside platform? Answer true if "
                            "ANY external tool or action is needed (including any request to "
                            "send, broadcast, or schedule something for customers, or to run "
                            "a campaign), and false only if it can be answered directly from "
                            "the conversation."
                        ),
                        "criteria": {
                            "true": "Needs an external tool, lookup, ticket, or outbound/broadcast",
                            "false": "Can be answered directly, no tools",
                        },
                    }
                },
            },
            timeout=c["timeout"],
        )
        if resp.status_code != 200:
            logger.warning("[jev-gate] HTTP %s: %s", resp.status_code, resp.text[:300])
            return False
        answer = (resp.json().get("answers") or {}).get("needs_tools") or {}
        # noul is P("true") = P(needs tools); default to 1.0 (safe = keep tools).
        p_needs = float(answer.get("noul", 1.0))
        disable = p_needs <= (1.0 - c["skip_confidence"])
        logger.info(
            "[jev-gate] p_needs_tools=%.2f disable_tools=%s msg=%r",
            p_needs,
            disable,
            text[:80],
        )
        return disable
    except Exception as e:
        logger.debug("[jev-gate] failed (non-fatal): %s", e)
        return False
