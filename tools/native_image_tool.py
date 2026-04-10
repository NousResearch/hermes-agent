#!/usr/bin/env python3
"""Native image attachment tool schema.

This tool does not analyze images itself. Instead, it tells the agent loop to
queue a local image file for native multimodal input on the next model call.
"""

from __future__ import annotations

import json

from tools.registry import registry


def check_attach_image_requirements() -> bool:
    return True


ATTACH_IMAGE_SCHEMA = {
    "name": "attach_image",
    "description": (
        "Attach a local image file so the current model can inspect it natively on the next step. "
        "Use this instead of vision analysis when the active runtime already supports native multimodal input. "
        "Pass a filesystem path to an image and an optional note describing what to inspect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative filesystem path to a local image file.",
            },
            "note": {
                "type": "string",
                "description": "Optional short note describing what to inspect in the image.",
            },
        },
        "required": ["path"],
    },
}


def _attach_image_stub(args, **kwargs) -> str:
    return json.dumps({"error": "attach_image must be handled by the agent loop"})


registry.register(
    name="attach_image",
    toolset="vision",
    schema=ATTACH_IMAGE_SCHEMA,
    handler=_attach_image_stub,
    check_fn=check_attach_image_requirements,
    emoji="📎",
)
