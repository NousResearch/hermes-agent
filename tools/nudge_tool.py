"""Nudge Tool -- inject messages into agent sessions from cronjobs or background processes.

A nudge allows a cronjob (or any isolated agent run) to signal the gateway
to inject a message into a running agent session or trigger a new agent run.
This enables "push" updates from background tasks without requiring the user
to send a message first.

How it works:
1. Cronjob calls nudge(action="send", target="discord:123", content="PR merged!")
2. Nudge is written to ~/.hermes/nudges/<session_key>.json
3. Gateway's cron ticker checks for pending nudges every 60 seconds
4. Gateway either:
   - Interrupts a running agent with the nudge (agent responds immediately)
   - Triggers a new agent run with the nudge as the user message

Usage from a cronjob:
    # Inject a nudge into the current origin session
    nudge(action="send", target="origin", content="PR #33 has been merged!")

    # List pending nudges
    nudge(action="list")

    # Clear all nudges for a target
    nudge(action="clear", target="discord:123456789")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_NUDGE_DIR = get_hermes_home() / "nudges"
_NUDGE_DIR.mkdir(parents=True, exist_ok=True)


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _success(message: str, data: dict = None) -> str:
    result = {"success": True, "message": message}
    if data:
        result.update(data)
    return json.dumps(result, ensure_ascii=False)


def _nudge_file_path(session_key: str) -> Path:
    """Get the path to the nudge file for a session key."""
    # Sanitize session key to be filesystem-safe
    safe_key = session_key.replace("/", "_").replace("\\", "_").replace(":", "-")
    return _NUDGE_DIR / f"{safe_key}.json"


def _read_nudges(session_key: str) -> list:
    """Read pending nudges for a session key."""
    path = _nudge_file_path(session_key)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("nudges", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read nudge file for %s: %s", session_key, e)
        return []


def _write_nudges(session_key: str, nudges: list) -> bool:
    """Write nudges to the file for a session key."""
    path = _nudge_file_path(session_key)
    try:
        with open(path, "w") as f:
            json.dump({"nudges": nudges}, f, indent=2)
        return True
    except OSError as e:
        logger.error("Failed to write nudge file for %s: %s", session_key, e)
        return False


def _get_origin_session() -> Optional[str]:
    """Resolve the origin session from environment/context variables.

    Cronjobs set session context via ContextVars or env vars during execution.
    """
    # Try context vars first (gateway cron execution)
    try:
        from gateway.session_context import _VAR_MAP
        platform = _VAR_MAP.get("platform", "")
        chat_id = _VAR_MAP.get("chat_id", "")
        thread_id = _VAR_MAP.get("thread_id", "")
        if platform and chat_id:
            key = f"{platform}:{chat_id}"
            if thread_id:
                key = f"{key}:{thread_id}"
            return key
    except Exception:
        pass

    # Fallback to env vars (set by cron scheduler for delivery targets)
    platform = os.environ.get("HERMES_CRON_AUTO_DELIVER_PLATFORM", "")
    chat_id = os.environ.get("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "")
    thread_id = os.environ.get("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "")
    if platform and chat_id:
        key = f"{platform}:{chat_id}"
        if thread_id:
            key = f"{key}:{thread_id}"
        return key

    return None


def nudge_tool(
    action: str = "send",
    target: Optional[str] = None,
    content: Optional[str] = None,
    **kwargs,  # Accept framework-injected params like task_id
) -> dict:
    """Manage nudges — inject messages into agent sessions.

    A nudge allows cronjobs and background processes to communicate with
    the main agent session. Nudges are queued and injected at the start
    of the next agent turn in the target session.

    Args:
        action: One of:
            - "send": Queue a nudge for the target session (requires target + content)
            - "list": List pending nudges (optionally for a specific target)
            - "clear": Clear all pending nudges for the target
        target: Session key to target. Format: "platform:chat_id" or
            "platform:chat_id:thread_id". Use "origin" to target the
            session that created this cronjob (resolved from cron context).
        content: The nudge message content (required for "send" action).
            This text will be injected into the agent's conversation as a
            system-level message before the next user turn.

    Returns:
        Success/error dict with details of the operation.
    """
    try:
        action = action.lower().strip()
    except AttributeError:
        return _error(f"Invalid action: {action!r}")

    if action not in ("send", "list", "clear"):
        return _error(f"Unknown action '{action}'. Use 'send', 'list', or 'clear'.")

    # ── SEND ──────────────────────────────────────────────────────────
    if action == "send":
        if not content:
            return _error("'content' is required for the 'send' action.")

        # Resolve target
        if target == "origin" or target is None:
            resolved = _get_origin_session()
            if not resolved:
                return _error(
                    "No origin session found. When not running as a cron job with "
                    "an origin, you must specify an explicit target like "
                    "'discord:123456789' or 'telegram:-1001234567890'."
                )
        else:
            resolved = target

        nudge = {
            "content": content,
            "timestamp": time.time(),
            "source": os.environ.get("HERMES_CRON_SESSION", "0") and "cronjob" or "agent",
        }

        existing = _read_nudges(resolved)
        existing.append(nudge)

        if _write_nudges(resolved, existing):
            logger.info("Nudge queued for session %s", resolved)
            return _success(f"Nudge queued for session '{resolved}'.", {
                "target": resolved,
                "nudge_count": len(existing),
            })
        else:
            return _error("Failed to write nudge file.")

    # ── LIST ──────────────────────────────────────────────────────────
    if action == "list":
        if target == "origin" or target is None:
            # List all pending nudges
            all_nudges = {}
            for f in _NUDGE_DIR.glob("*.json"):
                try:
                    with open(f, "r") as fh:
                        data = json.load(fh)
                    if data.get("nudges"):
                        key = f.stem.replace("-", ":", 1) if ":" not in f.stem else f.stem
                        # Reconstruct original key: we stored with ":" replaced by "-"
                        # But only the first ":" should be preserved
                        # Actually, let's just show the safe key
                        all_nudges[f.stem] = data["nudges"]
                except (json.JSONDecodeError, OSError):
                    pass
            if not all_nudges:
                return _success("No pending nudges.", {"nudges": {}})
            return _success(f"Found pending nudges for {len(all_nudges)} session(s).", {
                "nudges": all_nudges,
            })
        else:
            nudges = _read_nudges(target)
            if not nudges:
                return _success(f"No pending nudges for '{target}'.", {"nudges": []})
            return _success(f"Found {len(nudges)} pending nudge(s) for '{target}'.", {
                "nudges": nudges,
            })

    # ── CLEAR ─────────────────────────────────────────────────────────
    if action == "clear":
        if target == "origin" or target is None:
            resolved = _get_origin_session()
            if not resolved:
                return _error("No origin session found. Specify an explicit target.")
        else:
            resolved = target

        path = _nudge_file_path(resolved)
        if path.exists():
            try:
                path.unlink()
                return _success(f"Cleared nudges for '{resolved}'.")
            except OSError as e:
                return _error(f"Failed to clear nudges: {e}")
        return _success(f"No nudges to clear for '{resolved}'.")

    return _error(f"Unknown action '{action}'.")


# --- Registry ---
from tools.registry import registry, tool_error

NUDGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "nudge",
        "description": (
            "Inject a message (nudge) into an agent session. "
            "Nudges allow cronjobs and background processes to communicate with the main agent. "
            "The nudge content will be injected into the target session's next turn as a system-level message. "
            "Use action='send' with target and content to queue a nudge. "
            "Use 'origin' as target when running in a cronjob to nudge the session that created it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list", "clear"],
                    "description": "Action to perform: 'send' to queue a nudge, 'list' to see pending nudges, 'clear' to remove them."
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Session key to target, e.g. 'discord:123456789' or 'telegram:-1001234567890:17585'. "
                        "Use 'origin' to target the session that created this cronjob. "
                        "Required for 'send' and 'clear' (defaults to 'origin' if omitted in a cronjob context)."
                    )
                },
                "content": {
                    "type": "string",
                    "description": "The nudge message content. Required for 'send' action. This text will be injected into the agent's conversation."
                }
            },
            "required": ["action"],
        },
    },
}


def _check_nudge() -> bool:
    """Nudge tool is always available."""
    return True


registry.register(
    name="nudge",
    toolset="nudge",
    schema=NUDGE_SCHEMA,
    handler=nudge_tool,
    check_fn=_check_nudge,
    emoji="👉",
)
