"""Deliverable approval-box tools for Discord and Slack.

These tools post Willie/OpenClaw-style approval boxes with interactive buttons.
They are intentionally separate from shell command approvals: clicking a button
records an auditable decision and updates the platform message; it does not run
or unblock arbitrary code.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry, tool_error
from tools.approval_box_state import create_record

logger = logging.getLogger(__name__)


def _schema(name: str, platform_label: str, target_example: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": (
            f"Send an interactive {platform_label} approval box with Approve / Needs Work / Reject buttons. "
            f"Use for Willie/OpenClaw-style deliverable approval posts when the delivery target is {platform_label}. "
            f"Requires a target like {target_example}. Deliverable files must be uploaded to Google Drive first "
            "and passed as a Google Drive link; local Hermes paths are not final delivery."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": f"{platform_label} target. Example: {target_example}"},
                "title": {"type": "string", "description": "Approval box title"},
                "body": {"type": "string", "description": "Main readable approval body, including the actual draft text when approving communications"},
                "drive_link": {"type": "string", "description": "Required Google Drive deliverable link for files/artifacts"},
                "artifact_path": {"type": "string", "description": "Optional local artifact path for traceability only; not final delivery"},
                "approve_label": {"type": "string", "description": "Approve button label, default 'Approve'"},
                "revise_label": {"type": "string", "description": "Needs Work button label, default 'Needs Work'"},
                "reject_label": {"type": "string", "description": "Reject button label, default 'Reject'"},
            },
            "required": ["target", "title", "body"],
        },
    }


DISCORD_APPROVAL_BOX_SCHEMA = _schema(
    "discord_approval_box",
    "Discord",
    "'discord:1234567890', 'discord:1234567890:9876543210', or 'discord:#channel-name'",
)
SLACK_APPROVAL_BOX_SCHEMA = _schema(
    "slack_approval_box",
    "Slack",
    "'slack:C1234567890', 'slack:C1234567890:1712345678.000100', or 'slack:#engineering'",
)


def _parse_target(target: str, expected_platform: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = (target or "").split(":")
    platform = parts[0].strip().lower() if parts else ""
    if platform != expected_platform:
        return None, None, f"target must start with {expected_platform}:"
    if len(parts) < 2 or not parts[1].strip():
        return None, None, "target must include a channel/chat id or channel name"
    chat_ref = parts[1].strip()
    thread_id = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None

    # Resolve #channel names through the gateway channel directory when available.
    if chat_ref.startswith("#"):
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(expected_platform, chat_ref)
            if not resolved:
                return None, None, f"Could not resolve {chat_ref}; use send_message(action='list') or an explicit id"
            rparts = resolved.split(":")
            chat_ref = rparts[0]
            if len(rparts) > 1 and not thread_id:
                thread_id = rparts[1]
        except Exception as exc:
            return None, None, f"Could not resolve {chat_ref}: {exc}"
    return chat_ref, thread_id, None


def _check_gateway_or_platform() -> bool:
    try:
        from gateway.session_context import get_session_env
        if get_session_env("HERMES_SESSION_PLATFORM", "") not in ("", "local"):
            return True
    except Exception:
        pass
    try:
        from gateway.status import is_gateway_running
        return is_gateway_running()
    except Exception:
        return False


def _get_live_adapter(platform_name: str):
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
        if not runner:
            return None
        return runner.adapters.get(Platform(platform_name))
    except Exception:
        return None


def _build_record(args: Dict[str, Any], platform: str, target: str) -> Dict[str, Any]:
    return create_record(
        platform=platform,
        target=target,
        title=(args.get("title") or "Approval requested").strip(),
        body=(args.get("body") or "").strip(),
        drive_link=(args.get("drive_link") or "").strip(),
        artifact_path=(args.get("artifact_path") or "").strip(),
    )


def _result(record: Dict[str, Any], send_result: Any) -> str:
    success = bool(getattr(send_result, "success", False)) if not isinstance(send_result, dict) else bool(send_result.get("success"))
    message_id = getattr(send_result, "message_id", "") if not isinstance(send_result, dict) else send_result.get("message_id", "")
    error = getattr(send_result, "error", "") if not isinstance(send_result, dict) else send_result.get("error", "")
    if message_id:
        record["message_id"] = message_id
        try:
            from tools.approval_box_state import save_record
            save_record(record)
        except Exception:
            pass
    payload = {
        "success": success,
        "approval_id": record.get("approval_id"),
        "message_id": message_id,
        "state_path": f"approval_boxes/{record.get('approval_id')}.json",
    }
    if error:
        payload["error"] = str(error)
    return json.dumps(payload)


def _send_discord_approval_box(args: Dict[str, Any], **kw) -> str:
    target = args.get("target", "")
    chat_id, thread_id, error = _parse_target(target, "discord")
    if error:
        return tool_error(error)
    if not args.get("body"):
        return tool_error("body is required")
    record = _build_record(args, "discord", target)
    adapter = _get_live_adapter("discord")
    if not adapter or not hasattr(adapter, "send_deliverable_approval"):
        return tool_error("No live Discord adapter with deliverable approval support. Restart the gateway after installing this change.")
    from model_tools import _run_async
    send_result = _run_async(adapter.send_deliverable_approval(
        chat_id=chat_id,
        thread_id=thread_id,
        record=record,
        approve_label=args.get("approve_label") or "Approve",
        revise_label=args.get("revise_label") or "Needs Work",
        reject_label=args.get("reject_label") or "Reject",
    ))
    return _result(record, send_result)


def _send_slack_approval_box(args: Dict[str, Any], **kw) -> str:
    target = args.get("target", "")
    chat_id, thread_id, error = _parse_target(target, "slack")
    if error:
        return tool_error(error)
    if not args.get("body"):
        return tool_error("body is required")
    record = _build_record(args, "slack", target)
    adapter = _get_live_adapter("slack")
    if not adapter or not hasattr(adapter, "send_deliverable_approval"):
        return tool_error("No live Slack adapter with deliverable approval support. Restart the gateway after installing this change.")
    from model_tools import _run_async
    send_result = _run_async(adapter.send_deliverable_approval(
        chat_id=chat_id,
        thread_ts=thread_id,
        record=record,
        approve_label=args.get("approve_label") or "Approve",
        revise_label=args.get("revise_label") or "Needs Work",
        reject_label=args.get("reject_label") or "Reject",
    ))
    return _result(record, send_result)


registry.register(
    name="discord_approval_box",
    toolset="messaging",
    schema=DISCORD_APPROVAL_BOX_SCHEMA,
    handler=_send_discord_approval_box,
    check_fn=_check_gateway_or_platform,
    emoji="✅",
)
registry.register(
    name="slack_approval_box",
    toolset="messaging",
    schema=SLACK_APPROVAL_BOX_SCHEMA,
    handler=_send_slack_approval_box,
    check_fn=_check_gateway_or_platform,
    emoji="✅",
)
