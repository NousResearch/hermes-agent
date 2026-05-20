"""VRChat OSC tools.

These tools talk only to VRChat's official local OSC interface. They do not
use VRChat account credentials, client modification, or remote asset loading.

VRChat OSC defaults:
  - send host: 127.0.0.1
  - send port: 9000
  - receive port: 9001

Install manually with:
  uv pip install "hermes-agent[vrchat]"
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from tools.registry import registry, tool_result

logger = logging.getLogger(__name__)

_VRCHAT_FEATURE = "tool.vrchat"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_SEND_PORT = 9000
_DEFAULT_RECV_PORT = 9001
_CHATBOX_MAX_CHARS = 144
_MAX_OSC_ADDRESS_CHARS = 256

_client_cache: dict[tuple[str, int], Any] = {}
_client_lock = threading.Lock()


def _parse_port(value: str | None, default: int, name: str) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if port <= 0 or port > 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


def _endpoint() -> tuple[str, int, int]:
    host = os.environ.get("VRCHAT_OSC_HOST", _DEFAULT_HOST).strip() or _DEFAULT_HOST
    send_port = _parse_port(
        os.environ.get("VRCHAT_OSC_SEND_PORT"),
        _DEFAULT_SEND_PORT,
        "VRCHAT_OSC_SEND_PORT",
    )
    recv_port = _parse_port(
        os.environ.get("VRCHAT_OSC_RECV_PORT"),
        _DEFAULT_RECV_PORT,
        "VRCHAT_OSC_RECV_PORT",
    )
    return host, send_port, recv_port


def _import_udp_client():
    try:
        from pythonosc import udp_client

        return udp_client
    except ImportError:
        pass

    try:
        from tools.lazy_deps import FeatureUnavailable, ensure

        ensure(_VRCHAT_FEATURE, prompt=False)
    except FeatureUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"python-osc is not available: {exc}") from exc

    try:
        from pythonosc import udp_client

        return udp_client
    except ImportError as exc:
        raise RuntimeError(
            "python-osc is required for VRChat OSC. "
            "Install with: uv pip install 'hermes-agent[vrchat]'"
        ) from exc


def _python_osc_available() -> bool:
    try:
        import pythonosc  # noqa: F401

        return True
    except ImportError:
        return False


def _get_client(host: str, port: int):
    key = (host, port)
    with _client_lock:
        client = _client_cache.get(key)
        if client is None:
            udp_client = _import_udp_client()
            client = udp_client.SimpleUDPClient(host, port)
            _client_cache[key] = client
        return client


def _validate_osc_address(address: str) -> str:
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address cannot be empty")
    address = address.strip()
    if len(address) > _MAX_OSC_ADDRESS_CHARS:
        raise ValueError(f"address exceeds {_MAX_OSC_ADDRESS_CHARS} characters")
    if not address.startswith("/"):
        raise ValueError("address must start with '/'")
    if any(ch.isspace() for ch in address):
        raise ValueError("address must not contain whitespace")
    return address


def _validate_osc_arg(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("OSC args must be strings, booleans, integers, floats, or null")


def vrchat_chatbox(text: str, immediate: bool = True, notify: bool = False) -> dict[str, Any]:
    """Send a message to the VRChat chatbox via official OSC."""
    if not isinstance(text, str) or not text.strip():
        return {"success": False, "error": "text cannot be empty"}
    if len(text) > _CHATBOX_MAX_CHARS:
        return {
            "success": False,
            "error": f"text exceeds {_CHATBOX_MAX_CHARS} characters",
            "length": len(text),
        }

    try:
        host, send_port, _ = _endpoint()
        client = _get_client(host, send_port)
        client.send_message("/chatbox/input", [text, bool(immediate), bool(notify)])
        return {
            "success": True,
            "host": host,
            "send_port": send_port,
            "address": "/chatbox/input",
        }
    except Exception as exc:
        logger.debug("VRChat chatbox send failed", exc_info=True)
        return {"success": False, "error": str(exc)}


def vrchat_typing(is_typing: bool) -> dict[str, Any]:
    """Show or hide the VRChat chatbox typing indicator."""
    try:
        host, send_port, _ = _endpoint()
        client = _get_client(host, send_port)
        client.send_message("/chatbox/typing", bool(is_typing))
        return {
            "success": True,
            "host": host,
            "send_port": send_port,
            "address": "/chatbox/typing",
        }
    except Exception as exc:
        logger.debug("VRChat typing OSC send failed", exc_info=True)
        return {"success": False, "error": str(exc)}


def vrchat_avatar_param(name: str, value: bool | int | float) -> dict[str, Any]:
    """Set a VRChat avatar OSC parameter."""
    if not isinstance(name, str) or not name.strip():
        return {"success": False, "error": "name cannot be empty"}
    name = name.strip()
    if "/" in name or "\\" in name:
        return {"success": False, "error": "name must be a parameter name, not a path"}
    if not isinstance(value, (bool, int, float)):
        return {"success": False, "error": "value must be a boolean, integer, or float"}

    try:
        host, send_port, _ = _endpoint()
        address = f"/avatar/parameters/{name}"
        client = _get_client(host, send_port)
        client.send_message(address, value)
        return {
            "success": True,
            "host": host,
            "send_port": send_port,
            "address": address,
        }
    except Exception as exc:
        logger.debug("VRChat avatar parameter OSC send failed", exc_info=True)
        return {"success": False, "error": str(exc)}


def vrchat_send_osc(address: str, args: list[Any] | None = None) -> dict[str, Any]:
    """Send a raw OSC message to VRChat."""
    try:
        address = _validate_osc_address(address)
        values = [_validate_osc_arg(value) for value in (args or [])]
        host, send_port, _ = _endpoint()
        client = _get_client(host, send_port)
        payload: Any
        if not values:
            payload = None
        elif len(values) == 1:
            payload = values[0]
        else:
            payload = values
        client.send_message(address, payload)
        return {
            "success": True,
            "host": host,
            "send_port": send_port,
            "address": address,
            "arg_count": len(values),
        }
    except Exception as exc:
        logger.debug("VRChat raw OSC send failed", exc_info=True)
        return {"success": False, "error": str(exc)}


def vrchat_status() -> dict[str, Any]:
    """Return VRChat OSC configuration and local dependency status."""
    try:
        host, send_port, recv_port = _endpoint()
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "host": host,
        "send_port": send_port,
        "recv_port": recv_port,
        "python_osc_installed": _python_osc_available(),
        "note": "UDP OSC cannot confirm that VRChat consumed a packet; enable OSC in VRChat before sending.",
    }


registry.register(
    name="vrchat_chatbox",
    toolset="vrchat",
    schema={
        "name": "vrchat_chatbox",
        "description": (
            "Send text to VRChat's chatbox through the official local OSC interface. "
            "VRChat must be running with OSC enabled. Limited to 144 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Chatbox text to send. Max 144 characters.",
                },
                "immediate": {
                    "type": "boolean",
                    "description": "Whether VRChat should display the message immediately. Default: true.",
                },
                "notify": {
                    "type": "boolean",
                    "description": "Whether VRChat should play a chatbox notification. Default: false.",
                },
            },
            "required": ["text"],
        },
    },
    handler=lambda args, **kw: tool_result(
        vrchat_chatbox(
            text=args.get("text", ""),
            immediate=args.get("immediate", True),
            notify=args.get("notify", False),
        )
    ),
    emoji="VR",
)

registry.register(
    name="vrchat_typing",
    toolset="vrchat",
    schema={
        "name": "vrchat_typing",
        "description": "Show or hide the VRChat chatbox typing indicator through official OSC.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_typing": {
                    "type": "boolean",
                    "description": "true to show the typing indicator; false to hide it.",
                },
            },
            "required": ["is_typing"],
        },
    },
    handler=lambda args, **kw: tool_result(vrchat_typing(args.get("is_typing", False))),
    emoji="VR",
)

registry.register(
    name="vrchat_avatar_param",
    toolset="vrchat",
    schema={
        "name": "vrchat_avatar_param",
        "description": (
            "Set a VRChat avatar OSC parameter through /avatar/parameters/{name}. "
            "Use only parameters intentionally exposed by the user's avatar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Avatar parameter name, for example GestureLeft or a custom parameter.",
                },
                "value": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                    ],
                    "description": "Parameter value: boolean, integer, or float.",
                },
            },
            "required": ["name", "value"],
        },
    },
    handler=lambda args, **kw: tool_result(
        vrchat_avatar_param(args.get("name", ""), args.get("value"))
    ),
    emoji="VR",
)

registry.register(
    name="vrchat_send_osc",
    toolset="vrchat",
    schema={
        "name": "vrchat_send_osc",
        "description": (
            "Send a raw OSC message to VRChat's official local OSC endpoint. "
            "Use vrchat_chatbox or vrchat_avatar_param for common actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "OSC address starting with '/', for example /avatar/parameters/MyParam.",
                },
                "args": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "boolean"},
                            {"type": "integer"},
                            {"type": "number"},
                            {"type": "null"},
                        ]
                    },
                    "description": "OSC argument values.",
                },
            },
            "required": ["address"],
        },
    },
    handler=lambda args, **kw: tool_result(
        vrchat_send_osc(args.get("address", ""), args.get("args", []))
    ),
    emoji="VR",
)

registry.register(
    name="vrchat_status",
    toolset="vrchat",
    schema={
        "name": "vrchat_status",
        "description": "Show VRChat OSC endpoint configuration and python-osc availability.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: tool_result(vrchat_status()),
    emoji="VR",
)
