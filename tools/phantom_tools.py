"""Phantom (canary) tool definitions for detecting prompt injection.

Phantom tools are decoy tool definitions registered with the LLM that look
useful for data exfiltration but serve no legitimate purpose.  If the model
calls one, it is almost certainly acting on injected instructions — no
real user workflow would invoke ``send_to_webhook`` or ``upload_file_external``.

How it works
------------
1. At import time, five phantom tools are registered in the tool registry.
   Their schemas are included in the tool list sent to the LLM.
2. Each tool's handler logs a **critical** security event and returns an
   error to the model so it does not retry.
3. Callers can query ``get_phantom_events()`` to retrieve a log of all
   phantom tool invocations for the current process lifetime.

The toolset is named ``"phantom"`` and its availability check always returns
True so the tools appear in the schema sent to the model.
"""

import collections
import json
import logging
import threading
import time
from typing import List

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event log (bounded + rate-limited)
# ---------------------------------------------------------------------------

_MAX_EVENTS = 1000
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX = 10  # max events per tool per window

_events_lock = threading.Lock()
_events: collections.deque = collections.deque(maxlen=_MAX_EVENTS)
_rate_counts: dict = {}  # {tool_name: (window_start, count)}


def get_phantom_events() -> List[dict]:
    """Return a copy of all phantom tool invocations recorded so far."""
    with _events_lock:
        return list(_events)


def _record_event(tool_name: str, args: dict) -> bool:
    """Record a phantom-tool invocation.  Returns False if rate-limited."""
    now = time.time()
    with _events_lock:
        # Rate limiting per tool
        window_start, count = _rate_counts.get(tool_name, (now, 0))
        if now - window_start > _RATE_LIMIT_WINDOW:
            window_start, count = now, 0
        if count >= _RATE_LIMIT_MAX:
            return False
        _rate_counts[tool_name] = (window_start, count + 1)

        _events.append({
            "timestamp": now,
            "tool": tool_name,
            "args": args,
        })
    return True


# ---------------------------------------------------------------------------
# Phantom tool definitions
# ---------------------------------------------------------------------------

PHANTOM_SCHEMAS = [
    {
        "name": "send_to_webhook",
        "description": "Send a JSON payload to an external webhook URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The webhook endpoint URL.",
                },
                "payload": {
                    "type": "string",
                    "description": "JSON-encoded body to POST.",
                },
            },
            "required": ["url", "payload"],
        },
    },
    {
        "name": "upload_file_external",
        "description": "Upload a local file to an external storage endpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the local file.",
                },
                "destination_url": {
                    "type": "string",
                    "description": "Remote URL to upload to.",
                },
            },
            "required": ["file_path", "destination_url"],
        },
    },
    {
        "name": "relay_data",
        "description": "Relay data to a remote API endpoint with custom headers.",
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "Remote API endpoint.",
                },
                "data": {
                    "type": "string",
                    "description": "Data to send.",
                },
                "headers": {
                    "type": "string",
                    "description": "JSON-encoded HTTP headers.",
                },
            },
            "required": ["endpoint", "data"],
        },
    },
    {
        "name": "forward_email",
        "description": "Forward a message to an email address.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "Email body text.",
                },
            },
            "required": ["to", "body"],
        },
    },
    {
        "name": "export_to_cloud_storage",
        "description": "Export data to a cloud storage bucket (S3, GCS, Azure Blob).",
        "parameters": {
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "description": "Cloud storage bucket name or URI.",
                },
                "key": {
                    "type": "string",
                    "description": "Object key / path in the bucket.",
                },
                "data": {
                    "type": "string",
                    "description": "Content to upload.",
                },
            },
            "required": ["bucket", "data"],
        },
    },
]

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _make_handler(tool_name: str):
    """Return a handler closure for the given phantom tool."""

    def handler(args: dict, **kwargs) -> str:
        recorded = _record_event(tool_name, args)

        # Sanitize args for logging — truncate long values to avoid
        # flooding logs with exfiltrated data.
        safe_args = {
            k: (v[:200] + "…" if isinstance(v, str) and len(v) > 200 else v)
            for k, v in args.items()
        }
        if recorded:
            logger.critical(
                "SECURITY: Phantom tool invoked — probable prompt injection.  "
                "tool=%s args=%s",
                tool_name,
                json.dumps(safe_args, ensure_ascii=False),
            )
        else:
            logger.warning(
                "SECURITY: Phantom tool rate-limited (>%d calls in %ds).  "
                "tool=%s",
                _RATE_LIMIT_MAX, int(_RATE_LIMIT_WINDOW), tool_name,
            )
        return tool_error(
            f"Tool '{tool_name}' is not available in this environment."
        )

    return handler


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _phantom_available() -> bool:
    """Phantom tools should always appear in the schema sent to the model."""
    return True


for _schema in PHANTOM_SCHEMAS:
    registry.register(
        name=_schema["name"],
        toolset="phantom",
        schema=_schema,
        handler=_make_handler(_schema["name"]),
        check_fn=_phantom_available,
        is_async=False,
        emoji="",
        description=_schema["description"],
    )
