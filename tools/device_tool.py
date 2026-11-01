"""Ask the session's client for a device capability it advertised.

A phone that connects over the gateway ``/api/ws`` sidecar advertises what it can serve — camera,
microphone, location, screen capture, notifications, photo library, files — in
``client.capabilities {device_capabilities: [...]}``. This tool is the agent's side of that
advertisement: it sends ``device.request`` and hands the answer back.

The capability vocabulary is closed and mirrored from
``tui_gateway/contracts/liveness.py::DeviceCapability`` so the model is offered exactly what a
client can answer; a client that advertised nothing (a desktop window, the TUI) is refused before a
frame is written, so the tool reports that instead of stalling the turn for the deadline.

Two things this tool owns, not the wire:

* **Privacy.** Every capability that reaches a human's camera, microphone, position, screen,
  notifications or files asks the shared approval gate first (``device.approval.<capability>``:
  ``ask`` — the default — ``allow``, ``deny``). A grant is one store entry (once / session /
  always), like any terminal pattern.
* **Token cost.** A capture is megabytes; a tool result is re-sent on every later API call of the
  conversation. So captures are written to the profile's media cache
  (``$HERMES_HOME/cache/device/``) and the result carries a handle — path, MIME type, size — never
  the bytes. ``vision_analyze`` reads the path like any other screenshot, on purpose and once
  (#11911).
"""

import base64
import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

CAPABILITIES = ("camera", "microphone", "location", "screen_capture", "notifications",
                "photo_library", "files")

CAPABILITY_HINTS = {
    "camera": "one still frame from the phone's camera (JPEG)",
    "microphone": "a short audio recording from the phone's microphone",
    "location": "the phone's current GPS/Wi-Fi position",
    "screen_capture": "a still frame of the phone's screen",
    "notifications": "the phone's recent notifications (permission-gated)",
    "photo_library": "one image picked from the phone's photo library",
    "files": "one file picked from the phone's storage",
}

#: Wire MIME type -> cache file extension (no dot: ``provider_media.save_bytes`` adds it). A capture
#: whose type is not listed is refused rather than guessed: the bytes arrive from a client, and the
#: extension decides how they are read later.
_CAPTURE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp",
                       "audio/wav": "wav", "audio/mpeg": "mp3", "audio/webm": "webm"}

#: Longest ``value`` text that may reach a tool result. A non-binary capability (a location fix, a
#: notification list) is small by nature; a client that puts a base64 body here is truncated instead
#: of being allowed to multiply every later API call.
_MAX_VALUE_CHARS = 4000

_DEFAULT_MAX_CAPTURE_BYTES = 8 * 1024 * 1024
_APPROVAL_POLICIES = ("ask", "allow", "deny")


def _device_cfg(*keys, default=None):
    """``device.<keys>`` from config.yaml; ``default`` when config is unavailable."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), "device", *keys, default=default)
    except Exception:
        return default


def approval_policy(capability: str) -> str:
    """``device.approval.<capability>``: ``ask`` (default), ``allow`` or ``deny``.

    An unreadable value is ``ask``: a privacy setting fails toward the prompt, never toward the
    capture.
    """
    policy = str(_device_cfg("approval", capability, default="ask") or "ask").strip().lower()
    return policy if policy in _APPROVAL_POLICIES else "ask"


def _max_capture_bytes() -> int:
    """``device.max_capture_bytes`` clamped to 64 KiB..32 MiB (8 MiB default)."""
    raw = _device_cfg("max_capture_bytes", default=_DEFAULT_MAX_CAPTURE_BYTES)
    try:
        limit = int(raw if raw is not None else _DEFAULT_MAX_CAPTURE_BYTES)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CAPTURE_BYTES
    return min(max(limit, 64 * 1024), 32 * 1024 * 1024)


def _approval_refusal(capability: str) -> Optional[str]:
    """``None`` when this capability may be requested, else the refusal text for the model.

    ``allow`` skips the gate, ``deny`` never asks, ``ask`` runs the same human gate as a dangerous
    shell pattern (yolo / mode-off bypass, once/session/always grants, gateway card, cron and
    unattended policy, fail-closed with nobody to ask).
    """
    policy = approval_policy(capability)
    if policy == "allow":
        return None
    if policy == "deny":
        return (f"device.approval.{capability} is deny — this profile does not allow asking the device "
                f"for {capability}.")
    from tools.approval import _run_approval_gate

    result = _run_approval_gate(
        pattern_key=f"device:{capability}",
        description=f"Allow the connected device to hand over `{capability}` ({CAPABILITY_HINTS[capability]})?",
        display_target=f"device_request: {capability}",
        subject=f"device_request `{capability}` requires approval",
        noun="device captures",
        advice=("Ask the user for the information instead, or set device.approval."
                f"{capability} to allow once you trust this client."),
        autoapprove_log_prefix="device_request in non-interactive non-gateway context",
        fail_closed_when_no_human=True,
        no_human_block_message=(f"BLOCKED: device_request `{capability}` requires approval but no interactive "
                                "user or gateway is present to approve it."),
    )
    if result.get("approved"):
        return None
    return result.get("message") or f"the user declined the {capability} request"


def _trim_value(value: str) -> str:
    """*value* capped at ``_MAX_VALUE_CHARS``, with a note naming what was cut."""
    if len(value) <= _MAX_VALUE_CHARS:
        return value
    return (value[:_MAX_VALUE_CHARS]
            + f"\n[truncated: {len(value) - _MAX_VALUE_CHARS} more characters; a binary body belongs in "
              "`captures`, not `value`]")


def _materialise_captures(capability: str, captures) -> tuple[list[dict], list[str]]:
    """Write every capture into the media cache; return ``(handles, notes)``.

    A handle is ``{path, mime_type, bytes, label?}``. Anything unusable (unknown media type,
    malformed base64, over ``device.max_capture_bytes``) becomes a note instead of an exception, so
    one bad body never loses the rest of the answer.
    """
    handles: list[dict] = []
    notes: list[str] = []
    limit = _max_capture_bytes()
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            notes.append(f"capture {index}: not an object")
            continue
        mime = str(capture.get("mime_type") or "").split(";", 1)[0].strip().lower()
        extension = _CAPTURE_EXTENSIONS.get(mime)
        if extension is None:
            notes.append(f"capture {index}: unsupported media type {mime or '(missing)'}")
            continue
        data = capture.get("data")
        if not isinstance(data, str) or not data:
            notes.append(f"capture {index}: no data")
            continue
        # Base64 is ~4/3 of the payload: refuse an oversize body before decoding it, then again on the
        # decoded length (a client can pad).
        if len(data) > limit * 4 // 3 + 4:
            notes.append(f"capture {index}: larger than device.max_capture_bytes ({limit})")
            continue
        try:
            raw = base64.b64decode(data, validate=True)
        except (TypeError, ValueError) as exc:
            notes.append(f"capture {index}: not valid base64 ({exc})")
            continue
        if len(raw) > limit:
            notes.append(f"capture {index}: larger than device.max_capture_bytes ({limit})")
            continue
        from agent.provider_media import save_bytes
        path = save_bytes("device", raw, prefix=capability, extension=extension)
        handle = {"path": str(path), "mime_type": mime, "bytes": len(raw)}
        if capture.get("label"):
            handle["label"] = str(capture["label"])
        handles.append(handle)
    return handles, notes


def result_payload(capability: str, raw: dict) -> dict:
    """The tool result: the capability's small text answer plus capture HANDLES, never bytes."""
    payload: dict = {"capability": capability, "success": bool(raw.get("success", True))}
    if raw.get("denied"):
        payload.update(success=False, denied=True, reason=raw.get("reason") or "the user declined the request")
        return payload
    value = raw.get("value") or ""
    if isinstance(value, str) and value:
        payload["value"] = _trim_value(value)
    handles, notes = _materialise_captures(capability, raw.get("captures") or [])
    if handles:
        payload["captures"] = handles
        payload["hint"] = ("Each capture is a file on this machine — call vision_analyze with its "
                           "`path` to look at it, or share the path with the user.")
    if notes:
        payload["notes"] = notes
    return payload


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
    refusal = _approval_refusal(verb)
    if refusal is not None:
        return json.dumps({"capability": verb, "success": False, "denied": True, "reason": refusal},
                          ensure_ascii=False)
    try:
        raw = callback({"capability": verb, **(dict(options) if options else {})})
    except Exception as exc:
        return tool_error(f"device request failed: {exc}")
    if not raw:
        return tool_error("The device request timed out or no client answered it.")
    if isinstance(raw, str):
        return json.dumps({"capability": verb, "success": True, "value": _trim_value(raw)}, ensure_ascii=False)
    return json.dumps(result_payload(verb, raw), ensure_ascii=False)


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
    """Offered whenever the gateway bridge is wired: the SESSION gate is the toolset, not this probe.

    ``mobile_device`` is folded into a session only when a client on it advertised a device
    capability (``tui_gateway/agent_factory.py``), so a desktop or CLI session never sees the tool.
    ``check_fn`` answers reachability and is cached process-wide, so it must not try to answer
    "does this session have a phone" — it cannot (``tools/AGENTS.md``).
    """
    return True


registry.register(
    name="device_request", toolset="mobile_device", schema=_DEVICE_SCHEMA, check_fn=check_device_clients,
    handler=lambda args, **kw: device_request_tool(
        capability=args.get("capability", ""), options=args.get("options"),
        callback=kw.get("callback")),
    emoji="📱")
