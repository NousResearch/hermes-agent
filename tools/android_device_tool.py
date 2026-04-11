#!/usr/bin/env python3
"""Android device helper tools for the Hermes Android app runtime."""

from __future__ import annotations

import json
import os

from hermes_android.device_bridge import read_device_capabilities
from tools.registry import registry


def check_requirements() -> bool:
    return bool(os.getenv("HERMES_ANDROID_BOOTSTRAP", "").strip())


def android_device_status_tool(task_id: str | None = None) -> str:
    del task_id  # unused
    return json.dumps(read_device_capabilities(), ensure_ascii=False)


registry.register(
    name="android_device_status",
    toolset="hermes-android-app",
    schema={
        "name": "android_device_status",
        "description": (
            "Inspect Hermes Android workspace and device capabilities. Returns the workspace_path "
            "for Android file tools, imported workspace files, shared-folder grant status, and "
            "accessibility-control availability."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_device_status_tool(task_id=kwargs.get("task_id")),
    check_fn=check_requirements,
    description="Inspect Hermes Android workspace and capability status",
    emoji="📱",
)
