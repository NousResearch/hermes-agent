"""
hermes-android tool — registers android_* tools into hermes-agent registry.

Tools registered:
  - android_ping          check bridge connectivity
  - android_read_screen   get accessibility tree of current screen
  - android_tap           tap at coordinates or by node id
  - android_tap_text      tap element by visible text
  - android_type          type text into focused field
  - android_swipe         swipe gesture
  - android_open_app      launch app by package name
  - android_press_key     press hardware/software key (back, home, recents)
  - android_screenshot    capture screenshot as base64
  - android_scroll        scroll in direction
  - android_wait          wait for element to appear
  - android_get_apps      list installed apps
  - android_current_app   get foreground app package name
  - android_setup         configure bridge URL and pairing code
"""

import json
import os
import time
import requests
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

def _bridge_url() -> str:
    return os.getenv("ANDROID_BRIDGE_URL", "http://localhost:8765")

def _bridge_token() -> Optional[str]:
    return os.getenv("ANDROID_BRIDGE_TOKEN")

def _timeout() -> float:
    return float(os.getenv("ANDROID_BRIDGE_TIMEOUT", "10"))

def _auth_headers() -> dict:
    """Build auth headers with pairing code if configured."""
    token = _bridge_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def _check_requirements() -> bool:
    """Returns True only if the bridge is reachable."""
    try:
        r = requests.get(f"{_bridge_url()}/ping", headers=_auth_headers(), timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{_bridge_url()}{path}", json=payload,
                      headers=_auth_headers(), timeout=_timeout())
    r.raise_for_status()
    return r.json()

def _get(path: str) -> dict:
    r = requests.get(f"{_bridge_url()}{path}", headers=_auth_headers(),
                     timeout=_timeout())
    r.raise_for_status()
    return r.json()

# ── Tool implementations ───────────────────────────────────────────────────────

def android_ping() -> str:
    try:
        data = _get("/ping")
        return json.dumps({"status": "ok", "bridge": data})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def android_read_screen(include_bounds: bool = False) -> str:
    """
    Returns the accessibility tree of the current screen as JSON.
    Each node has: nodeId, text, contentDescription, className,
                   clickable, focusable, bounds (if include_bounds=True)
    """
    try:
        data = _get(f"/screen?bounds={str(include_bounds).lower()}")
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_tap(x: Optional[int] = None, y: Optional[int] = None,
                node_id: Optional[str] = None) -> str:
    """
    Tap at screen coordinates (x, y) or by accessibility node_id.
    Prefer node_id when available — it's more reliable than coordinates.
    """
    try:
        payload = {}
        if node_id:
            payload["nodeId"] = node_id
        elif x is not None and y is not None:
            payload["x"] = x
            payload["y"] = y
        else:
            return json.dumps({"error": "Provide either (x, y) or node_id"})
        data = _post("/tap", payload)
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_tap_text(text: str, exact: bool = False) -> str:
    """
    Tap the first element whose visible text matches `text`.
    exact=False uses contains matching. exact=True requires full match.
    Useful when you can see text on screen but don't have node IDs.
    """
    try:
        data = _post("/tap_text", {"text": text, "exact": exact})
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_type(text: str, clear_first: bool = False) -> str:
    """
    Type text into the currently focused input field.
    Set clear_first=True to clear existing content before typing.
    """
    try:
        data = _post("/type", {"text": text, "clearFirst": clear_first})
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_swipe(direction: str, distance: str = "medium") -> str:
    """
    Swipe in direction: up, down, left, right.
    distance: short, medium, long
    """
    try:
        data = _post("/swipe", {"direction": direction, "distance": distance})
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_open_app(package: str) -> str:
    """
    Launch an app by its package name.
    Common packages:
      com.ubercab        - Uber
      com.whatsapp       - WhatsApp
      com.spotify.music  - Spotify
      com.google.android.apps.maps - Google Maps
      com.android.chrome - Chrome
      com.google.android.gm - Gmail
    """
    try:
        data = _post("/open_app", {"package": package})
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_press_key(key: str) -> str:
    """
    Press a key. Supported keys:
      back, home, recents, power, volume_up, volume_down,
      enter, delete, tab, escape, search, notifications
    """
    try:
        data = _post("/press_key", {"key": key})
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_screenshot() -> str:
    """
    Capture a screenshot. Returns base64-encoded PNG.
    Use this when the accessibility tree doesn't give enough context.
    """
    try:
        data = _get("/screenshot")
        return json.dumps(data)  # { "image": "<base64>", "width": ..., "height": ... }
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_scroll(direction: str, node_id: Optional[str] = None) -> str:
    """
    Scroll within a scrollable element or the whole screen.
    direction: up, down, left, right
    """
    try:
        payload = {"direction": direction}
        if node_id:
            payload["nodeId"] = node_id
        data = _post("/scroll", payload)
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_wait(text: str = None, class_name: str = None,
                 timeout_ms: int = 5000) -> str:
    """
    Wait for an element to appear on screen.
    Polls every 500ms up to timeout_ms.
    Returns the matching node if found, error if timeout.
    """
    try:
        payload = {"timeoutMs": timeout_ms}
        if text:
            payload["text"] = text
        if class_name:
            payload["className"] = class_name
        data = _post("/wait", payload)
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_get_apps() -> str:
    """List all installed apps with their package names and labels."""
    try:
        data = _get("/apps")
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_current_app() -> str:
    """Get the package name and activity of the current foreground app."""
    try:
        data = _get("/current_app")
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})


def android_setup(bridge_url: str, pairing_code: str) -> str:
    """
    Configure the Android bridge connection. Saves ANDROID_BRIDGE_URL and
    ANDROID_BRIDGE_TOKEN to ~/.hermes/.env, then verifies the connection.

    Call this when the user provides their phone's IP and pairing code.
    Example: android_setup("http://192.168.1.50:8765", "K7V3NP")
    """
    try:
        # Validate URL format
        if not bridge_url.startswith("http"):
            bridge_url = f"http://{bridge_url}"
        if ":8765" not in bridge_url and ":" not in bridge_url.split("//")[1]:
            bridge_url = f"{bridge_url}:8765"

        # Save to ~/.hermes/.env using hermes-agent's config system
        try:
            from hermes_cli.config import save_env_value
            save_env_value("ANDROID_BRIDGE_URL", bridge_url)
            save_env_value("ANDROID_BRIDGE_TOKEN", pairing_code)
        except ImportError:
            # Fallback: write directly
            from pathlib import Path
            env_path = Path.home() / ".hermes" / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            _update_env_file(env_path, "ANDROID_BRIDGE_URL", bridge_url)
            _update_env_file(env_path, "ANDROID_BRIDGE_TOKEN", pairing_code)

        # Update current process env so subsequent calls work immediately
        os.environ["ANDROID_BRIDGE_URL"] = bridge_url
        os.environ["ANDROID_BRIDGE_TOKEN"] = pairing_code

        # Verify connection
        headers = {"Authorization": f"Bearer {pairing_code}"}
        r = requests.get(f"{bridge_url}/ping", headers=headers, timeout=5)
        ping_data = r.json()

        if r.status_code == 200 and ping_data.get("authenticated"):
            return json.dumps({
                "status": "ok",
                "message": "Android bridge configured and connected!",
                "bridge_url": bridge_url,
                "accessibility_service": ping_data.get("accessibilityService"),
                "authenticated": True,
            })
        elif r.status_code == 200:
            return json.dumps({
                "status": "partial",
                "message": "Bridge reachable but pairing code not accepted. Check the code in the Hermes Bridge app.",
                "bridge_url": bridge_url,
                "authenticated": False,
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"Bridge returned HTTP {r.status_code}",
            })

    except (requests.exceptions.ConnectionError, ConnectionError):
        return json.dumps({
            "status": "error",
            "message": f"Cannot reach bridge at {bridge_url}. Check that the Hermes Bridge app is running and the IP/port are correct.",
            "saved": True,
            "hint": "Config was saved — connection will work once the bridge is reachable.",
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _update_env_file(env_path, key: str, value: str):
    """Simple .env file updater (fallback when hermes_cli.config not available)."""
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")
    env_path.write_text("".join(lines), encoding="utf-8")


# ── Schema definitions ─────────────────────────────────────────────────────────

_SCHEMAS = {
    "android_ping": {
        "name": "android_ping",
        "description": "Check if the Android bridge is reachable. Call this first before any other android_ tools.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "android_read_screen": {
        "name": "android_read_screen",
        "description": "Get the accessibility tree of the current Android screen. Returns all visible UI nodes with text, class names, node IDs, and interactability. Use this to understand what's on screen before tapping.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_bounds": {
                    "type": "boolean",
                    "description": "Include pixel coordinates for each node. Default false.",
                    "default": False,
                }
            },
            "required": [],
        },
    },
    "android_tap": {
        "name": "android_tap",
        "description": "Tap a UI element by node_id (preferred) or by screen coordinates (x, y). Always prefer node_id over coordinates — it's more reliable. Get node_ids from android_read_screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate in pixels"},
                "y": {"type": "integer", "description": "Y coordinate in pixels"},
                "node_id": {"type": "string", "description": "Accessibility node ID from android_read_screen"},
            },
            "required": [],
        },
    },
    "android_tap_text": {
        "name": "android_tap_text",
        "description": "Tap the first visible UI element matching the given text. Useful when you see text on screen and want to tap it without needing node IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to find and tap"},
                "exact": {"type": "boolean", "description": "Exact match (true) or contains match (false, default)", "default": False},
            },
            "required": ["text"],
        },
    },
    "android_type": {
        "name": "android_type",
        "description": "Type text into the currently focused input field. Tap the field first using android_tap or android_tap_text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "clear_first": {"type": "boolean", "description": "Clear existing content before typing", "default": False},
            },
            "required": ["text"],
        },
    },
    "android_swipe": {
        "name": "android_swipe",
        "description": "Perform a swipe gesture on screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "distance": {"type": "string", "enum": ["short", "medium", "long"], "default": "medium"},
            },
            "required": ["direction"],
        },
    },
    "android_open_app": {
        "name": "android_open_app",
        "description": "Launch an Android app by its package name. Use android_get_apps to find package names.",
        "parameters": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "App package name e.g. com.ubercab"},
            },
            "required": ["package"],
        },
    },
    "android_press_key": {
        "name": "android_press_key",
        "description": "Press a hardware or software key.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": ["back", "home", "recents", "power", "volume_up", "volume_down", "enter", "delete", "tab", "escape", "search", "notifications"],
                }
            },
            "required": ["key"],
        },
    },
    "android_screenshot": {
        "name": "android_screenshot",
        "description": "Take a screenshot of the current Android screen. Returns base64 PNG. Use when the accessibility tree is missing context or the screen uses canvas/game rendering.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "android_scroll": {
        "name": "android_scroll",
        "description": "Scroll the screen or a specific scrollable element.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "node_id": {"type": "string", "description": "Node ID of scrollable container (optional, defaults to screen scroll)"},
            },
            "required": ["direction"],
        },
    },
    "android_wait": {
        "name": "android_wait",
        "description": "Wait for a UI element to appear on screen. Use after actions that trigger loading or navigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Wait for element with this text"},
                "class_name": {"type": "string", "description": "Wait for element of this class"},
                "timeout_ms": {"type": "integer", "description": "Max wait time in milliseconds", "default": 5000},
            },
            "required": [],
        },
    },
    "android_get_apps": {
        "name": "android_get_apps",
        "description": "List all installed apps on the Android device with their package names and display labels.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "android_current_app": {
        "name": "android_current_app",
        "description": "Get the package name and activity name of the currently active (foreground) Android app.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "android_setup": {
        "name": "android_setup",
        "description": "Configure the Android bridge connection. Use when the user provides their phone's IP address and pairing code from the Hermes Bridge app. Saves config to ~/.hermes/.env and verifies the connection.",
        "parameters": {
            "type": "object",
            "properties": {
                "bridge_url": {
                    "type": "string",
                    "description": "Bridge URL e.g. http://192.168.1.50:8765 or just 192.168.1.50",
                },
                "pairing_code": {
                    "type": "string",
                    "description": "6-character pairing code shown in the Hermes Bridge app",
                },
            },
            "required": ["bridge_url", "pairing_code"],
        },
    },
}

# ── Tool handlers map ──────────────────────────────────────────────────────────

_HANDLERS = {
    "android_ping":         lambda args, **kw: android_ping(),
    "android_read_screen":  lambda args, **kw: android_read_screen(**args),
    "android_tap":          lambda args, **kw: android_tap(**args),
    "android_tap_text":     lambda args, **kw: android_tap_text(**args),
    "android_type":         lambda args, **kw: android_type(**args),
    "android_swipe":        lambda args, **kw: android_swipe(**args),
    "android_open_app":     lambda args, **kw: android_open_app(**args),
    "android_press_key":    lambda args, **kw: android_press_key(**args),
    "android_screenshot":   lambda args, **kw: android_screenshot(),
    "android_scroll":       lambda args, **kw: android_scroll(**args),
    "android_wait":         lambda args, **kw: android_wait(**args),
    "android_get_apps":     lambda args, **kw: android_get_apps(),
    "android_current_app":  lambda args, **kw: android_current_app(),
    "android_setup":        lambda args, **kw: android_setup(**args),
}

# ── Registry registration ──────────────────────────────────────────────────────

try:
    from tools.registry import registry

    for tool_name, schema in _SCHEMAS.items():
        registry.register(
            name=tool_name,
            toolset="android",
            schema={"type": "function", "function": schema},
            handler=_HANDLERS[tool_name],
            # android_setup must work without a bridge connection (it creates the connection)
            check_fn=(lambda: True) if tool_name == "android_setup" else _check_requirements,
            requires_env=[],  # ANDROID_BRIDGE_URL has a default
        )
except ImportError:
    # Running outside hermes-agent context (e.g. tests)
    pass
