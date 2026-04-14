#!/usr/bin/env python3
"""Skill candidate lifecycle tool.

Exposes layered-memory skill candidates as a formal Hermes tool so the agent can
list, inspect, approve, and reject candidates through the standard tool-calling
surface instead of relying only on CLI slash commands.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from hermes_constants import get_hermes_home
from plugins.memory import load_memory_provider
from tools.registry import registry


def _with_provider():
    hermes_home = get_hermes_home()
    provider = load_memory_provider("layered")
    provider.initialize(session_id="skill-candidates-tool", hermes_home=str(hermes_home), platform="cli")
    return provider


def skill_candidates(args: Dict[str, Any], **kwargs) -> str:
    action = str(args.get("action", "")).strip().lower()
    name = str(args.get("name", "")).strip()
    reason = str(args.get("reason", "manual_reject")).strip() or "manual_reject"

    provider = _with_provider()
    try:
        if action == "list":
            return json.dumps({"success": True, "candidates": provider.list_skill_candidates()}, ensure_ascii=False)

        if action == "inspect":
            if not name:
                return json.dumps({"success": False, "error": "name is required for inspect"}, ensure_ascii=False)
            candidate = provider.inspect_skill_candidate(name)
            if not candidate:
                return json.dumps({"success": False, "error": f"No candidate found for {name}"}, ensure_ascii=False)
            return json.dumps({"success": True, "candidate": candidate}, ensure_ascii=False)

        if action == "approve":
            if not name:
                return json.dumps({"success": False, "error": "name is required for approve"}, ensure_ascii=False)
            installed_skill_path = provider.approve_skill_candidate(name)
            candidate = provider.inspect_skill_candidate(name)
            return json.dumps(
                {
                    "success": True,
                    "strategy": candidate.get("approval_strategy", provider.decide_install_strategy(name)),
                    "installed_skill_path": installed_skill_path,
                    "candidate": candidate,
                },
                ensure_ascii=False,
            )

        if action == "reject":
            if not name:
                return json.dumps({"success": False, "error": "name is required for reject"}, ensure_ascii=False)
            provider.reject_skill_candidate(name, reason=reason)
            candidate = provider.inspect_skill_candidate(name)
            return json.dumps(
                {
                    "success": True,
                    "review_status": candidate.get("review_status"),
                    "review_gate_reason": candidate.get("review_gate_reason"),
                    "candidate": candidate,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success": False,
                "error": "Unknown action. Use one of: list, inspect, approve, reject",
            },
            ensure_ascii=False,
        )
    finally:
        try:
            provider.shutdown()
        except Exception:
            pass


registry.register(
    name="skill_candidates",
    toolset="skills",
    schema={
        "name": "skill_candidates",
        "description": "List, inspect, approve, or reject layered-memory skill candidates before they become installed Hermes skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "inspect", "approve", "reject"],
                    "description": "Which skill-candidate operation to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Candidate skill name. Required for inspect/approve/reject.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional rejection reason for action=reject.",
                },
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kwargs: skill_candidates(args, **kwargs),
    description="Manage layered-memory skill candidates.",
    emoji="🧠",
)
