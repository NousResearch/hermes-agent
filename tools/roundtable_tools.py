"""Hermes Agent adapter for Roundtable.

Drop-in replacement for hermes tools/roundtable_tools.py that delegates
to the independent roundtable library. Registers all 7 tools with the
Hermes tool registry.

Usage in Hermes:
    - Enable the ``roundtable`` toolset in profile config
    - This module auto-registers when imported by the tool discovery system
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from roundtable.core import RoundtableCore
from roundtable.exceptions import RoundtableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification send callback
# ---------------------------------------------------------------------------


def _hermes_send_fn(platform: str, chat_id: str, message: str) -> None:
    """Deliver a notification message to a messaging platform.

    Called by the Notifier when a discussion event fires.
    Must never raise — exceptions are caught and logged.
    """
    try:
        if platform == "feishu":
            profile = os.environ.get("HERMES_PROFILE", "default")
            script = os.path.expanduser("~/.hermes/scripts/feishu-send.py")
            subprocess.run(
                ["python3", script, profile, chat_id, message],
                capture_output=True,
                timeout=15,
            )
        else:
            logger.warning("Unsupported notification platform: %s", platform)
    except Exception as e:
        logger.warning("Notification send failed (platform=%s, chat=%s): %s", platform, chat_id, e)


# ---------------------------------------------------------------------------
# Lazy core singleton
# ---------------------------------------------------------------------------

_core: RoundtableCore | None = None


def _get_core() -> RoundtableCore:
    global _core
    if _core is None:
        _core = RoundtableCore(send_fn=_hermes_send_fn)
    return _core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def _handle(args: dict[str, Any], method: str, **extra: Any) -> str:
    """Generic handler: call a RoundtableCore method, catch errors."""
    try:
        core = _get_core()
        fn = getattr(core, method)
        result = fn(**{**args, **extra})
        return json.dumps(result)
    except (ValueError, RoundtableError) as e:
        return _err(str(e))
    except Exception as e:
        logger.exception(f"roundtable_{method} failed")
        return _err(f"roundtable_{method}: {e}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_init(args: dict[str, Any], **kw: Any) -> str:
    # Default web=True so discussion mode always opens WebViewer
    args.setdefault("web", True)
    return _handle(args, "create_discussion")


def _handle_speak(args: dict[str, Any], **kw: Any) -> str:
    return _handle(args, "speak")


def _handle_read(args: dict[str, Any], **kw: Any) -> str:
    return _handle(args, "read")


def _handle_status(args: dict[str, Any], **kw: Any) -> str:
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return _err("discussion_id is required")
    return _handle({"discussion_id": discussion_id}, "status")


def _handle_summarize(args: dict[str, Any], **kw: Any) -> str:
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return _err("discussion_id is required")
    compact = args.get("compact", False)
    return _handle({"discussion_id": discussion_id, "compact": compact}, "summarize")


def _handle_end(args: dict[str, Any], **kw: Any) -> str:
    return _handle(args, "end_discussion")


def _handle_list(args: dict[str, Any], **kw: Any) -> str:
    return _handle(args, "list_discussions")


def _handle_advance(args: dict[str, Any], **kw: Any) -> str:
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return _err("discussion_id is required")
    return _handle({"discussion_id": discussion_id}, "advance")


def _handle_notify(args: dict[str, Any], **kw: Any) -> str:
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return _err("discussion_id is required")
    event = args.get("event", "").strip()
    if not event:
        return _err("event is required")
    extra = {k: v for k, v in args.items() if k not in ("discussion_id", "event")}
    return _handle({"discussion_id": discussion_id, "event": event, **extra}, "notify")


# ---------------------------------------------------------------------------
# Tool schemas (identical to original)
# ---------------------------------------------------------------------------

ROUNDTABLE_INIT_SCHEMA = {
    "name": "roundtable_init",
    "description": (
        "Create a new roundtable discussion with a topic and participants. "
        "Each participant is an agent profile that will take turns speaking. "
        "Returns the discussion_id for subsequent calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The discussion topic"},
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string", "description": "Agent profile name"},
                        "role": {"type": "string", "description": "Role description"},
                        "perspective": {"type": "string", "description": "Role perspective hint"},
                        "display_name": {"type": "string", "description": "Display name"},
                    },
                    "required": ["profile"],
                },
                "description": "List of participant profiles (min 2)",
            },
            "context": {"type": "string", "description": "Background context"},
            "max_rounds": {"type": "integer", "description": "Max rounds (default: 5)", "default": 5},
            "speech_order": {
                "type": "string",
                "enum": ["fixed", "random", "priority", "free"],
                "description": "Speech order strategy (default: fixed)",
                "default": "fixed",
            },
            "output_path": {"type": "string", "description": "Path to save conclusion"},
            "created_by": {"type": "string", "description": "Creator profile name"},
            "web": {
                "type": "boolean",
                "description": "Start a WebPublisher HTTP server for live viewing in browser (default: false)",
                "default": False,
            },
            "web_port": {
                "type": "integer",
                "description": "Port for the WebPublisher server (default: 8765)",
                "default": 8765,
            },
            "notifications": {
                "type": "object",
                "description": "Notification config for real-time push to messaging channels",
                "properties": {
                    "enabled": {"type": "boolean", "description": "Enable notifications"},
                    "channels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "platform": {"type": "string", "description": "Platform (e.g. 'feishu')"},
                                "chat_id": {"type": "string", "description": "Chat/channel ID"},
                            },
                            "required": ["chat_id"],
                        },
                        "description": "Target channels",
                    },
                    "events": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["round_start", "speech", "round_end", "concluded"]},
                        "description": "Events to subscribe (default: all)",
                    },
                },
            },
        },
        "required": ["topic", "participants"],
    },
}

ROUNDTABLE_SPEAK_SCHEMA = {
    "name": "roundtable_speak",
    "description": (
        "Record a participant's speech in a roundtable discussion. "
        "Automatically tracks rounds and advances when all participants have spoken."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
            "participant": {"type": "string", "description": "Profile name of the speaker"},
            "content": {"type": "string", "description": "Speech content (Markdown supported)"},
            "reply_to": {"type": "integer", "description": "Optional: ID of a speech being referenced"},
        },
        "required": ["discussion_id", "participant", "content"],
    },
}

ROUNDTABLE_READ_SCHEMA = {
    "name": "roundtable_read",
    "description": (
        "Read the discussion history — all speeches or filtered by round/participant. "
        "Returns both structured data and a formatted history string."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
            "since_round": {"type": "integer", "description": "Only speeches from this round onwards"},
            "participant": {"type": "string", "description": "Only speeches from this participant"},
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_STATUS_SCHEMA = {
    "name": "roundtable_status",
    "description": (
        "Get discussion status including current round, convergence score, "
        "consensus/disagreement points, and next speaker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_SUMMARIZE_SCHEMA = {
    "name": "roundtable_summarize",
    "description": (
        "Generate summary data for a conclusion document. Returns structured_summary "
        "(compact Markdown, <5KB) plus metadata. Use compact=true to omit raw speech data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
            "compact": {
                "type": "boolean",
                "description": "If true, return only structured_summary without raw rounds data (default: false)",
                "default": False,
            },
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_END_SCHEMA = {
    "name": "roundtable_end",
    "description": (
        "End a roundtable discussion. By default, marks it as concluded. Use force=true to cancel instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
            "force": {"type": "boolean", "description": "Cancel instead of conclude", "default": False},
            "conclusion": {"type": "string", "description": "Optional: conclusion text"},
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_LIST_SCHEMA = {
    "name": "roundtable_list",
    "description": "List roundtable discussions with optional status filter.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "concluded", "cancelled"],
                "description": "Filter by status (omit for all)",
            },
            "limit": {"type": "integer", "description": "Max results (default: 50)", "default": 50},
        },
    },
}

ROUNDTABLE_ADVANCE_SCHEMA = {
    "name": "roundtable_advance",
    "description": (
        "Explicitly advance to the next round. Use when auto-advance "
        "doesn't trigger. If max_rounds is exceeded, the discussion "
        "is automatically concluded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_NOTIFY_SCHEMA = {
    "name": "roundtable_notify",
    "description": (
        "Manually trigger a notification for a discussion event. "
        "Valid events: round_start, speech, round_end, concluded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {"type": "string", "description": "Discussion ID (rt_xxxxxxxx)"},
            "event": {
                "type": "string",
                "enum": ["round_start", "speech", "round_end", "concluded"],
                "description": "Event type to notify",
            },
            "round_num": {"type": "integer", "description": "Round number"},
            "participant": {"type": "string", "description": "Participant name"},
            "content": {"type": "string", "description": "Speech content"},
            "conclusion": {"type": "string", "description": "Conclusion text"},
        },
        "required": ["discussion_id", "event"],
    },
}


# ---------------------------------------------------------------------------
# Registration function — called by Hermes tool discovery
# ---------------------------------------------------------------------------


def register_roundtable_tools(registry: Any, *, check_fn: Any = None) -> None:
    """Register all 8 roundtable tools with a Hermes tool registry.

    Args:
        registry: The Hermes tools.registry object.
        check_fn: Optional gating function. If None, always enabled.
    """
    tools = [
        ("roundtable_init", ROUNDTABLE_INIT_SCHEMA, _handle_init, "🎯"),
        ("roundtable_speak", ROUNDTABLE_SPEAK_SCHEMA, _handle_speak, "💬"),
        ("roundtable_read", ROUNDTABLE_READ_SCHEMA, _handle_read, "📖"),
        ("roundtable_status", ROUNDTABLE_STATUS_SCHEMA, _handle_status, "📊"),
        ("roundtable_summarize", ROUNDTABLE_SUMMARIZE_SCHEMA, _handle_summarize, "📝"),
        ("roundtable_end", ROUNDTABLE_END_SCHEMA, _handle_end, "🏁"),
        ("roundtable_list", ROUNDTABLE_LIST_SCHEMA, _handle_list, "📋"),
        ("roundtable_advance", ROUNDTABLE_ADVANCE_SCHEMA, _handle_advance, "⏭️"),
        ("roundtable_notify", ROUNDTABLE_NOTIFY_SCHEMA, _handle_notify, "🔔"),
    ]
    for name, schema, handler, emoji in tools:
        registry.register(
            name=name,
            toolset="roundtable",
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            emoji=emoji,
        )


# ---------------------------------------------------------------------------
# Auto-registration when imported by Hermes tool discovery
# ---------------------------------------------------------------------------


def _auto_register() -> None:
    """Try to auto-register with Hermes if available."""
    try:
        from tools.registry import registry

        def _check_roundtable_enabled() -> bool:
            try:
                from hermes_cli.config import load_config

                cfg = load_config()
                return "roundtable" in cfg.get("toolsets", [])
            except Exception:
                return False

        register_roundtable_tools(registry, check_fn=_check_roundtable_enabled)
    except ImportError:
        # Not running inside Hermes — that's fine, the library works standalone
        pass


_auto_register()
