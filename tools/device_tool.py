"""Ask the session's client for a device capability it advertised.

A phone that connects over the gateway ``/api/ws`` sidecar advertises what it can serve — camera,
microphone, location, screen capture, notifications, photo library, files — in
``client.capabilities {device_capabilities: [...]}``. This tool is the agent's side of that
advertisement: it sends ``device.request`` and hands the answer back.

The capability vocabulary is closed and mirrored from
``tui_gateway/contracts/liveness.py::DeviceCapability`` so the model is offered exactly what a
client can answer; a client that advertised nothing (a desktop window, the TUI) is refused before a
frame is written, so the tool reports that instead of stalling the turn for the deadline.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

CAPABILITIES = ("camera", "microphone", "location", "screen_capture", "notifications",
                "photo_library", "files")

CAPABILITY_HINTS = {
    "camera": "one still frame from the phone's camera (JPEG, base64)",
    "microphone": "a short audio recording from the phone's microphone",
    "location": "the phone's current GPS/Wi-Fi position",
    "screen_capture": "a still frame of the phone's screen",
    "notifications": "the phone's recent notifications (permission-gated)",
    "photo_library": "one image picked from the phone's photo library",
    "files": "one file picked from the phone's storage",
}


def device_request_tool(capability: str = "", options: Optional[dict] = None,
                        callback: Optional[Callable] = None) -> str:
    """Ask the session's client for *capability* and return its answer as JSON text."""
    if callback is None:
        return tool_error("device_request needs a client that connects over the Hermes gateway.")
    verb = (capability or "").strip().lower()
    if verb not in CAPABILITIES:
        return tool_error(f"capability must be one of: {', '.join(CAPABILITIES)}.")
    if options is not None and not isinstance(options, dict):
        return tool_error("options must be an object of capability-specific settings.")
    try:
        raw = callback({"capability": verb, **({"options": options} if options else {})})
    except Exception as exc:
        return tool_error(f"device request failed: {exc}")
    if not raw:
        return tool_error("The device request timed out or no client answered it.")
    return json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)


_DEVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "capability": {
            "type": "string",
            "enum": list(CAPABILITIES),
            "description": "What to ask the connected phone for. The client must have advertised it.",
        },
        "options": {
            "type": "object",
            "description": "Capability-specific settings (for example a camera quality or a geo timeout).",
        },
    },
    "required": ["capability"],
}


def check_device_clients() -> bool:
    """Offered whenever the gateway is reachable: a client that advertised no device capability
    answers with a clear refusal, so the tool is never dead weight on a desktop-only setup."""
    return True


registry.register(
    name="device_request", toolset="hermes-gateway", schema=_DEVICE_SCHEMA, check_fn=check_device_clients,
    handler=lambda args, **kw: device_request_tool(
        capability=args.get("capability", ""), options=args.get("options"),
        callback=kw.get("callback")),
    emoji="📱")
