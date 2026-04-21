"""Minimal macOS desktop-control tool for Hermes Agent.

Provides a small, explicit action surface for controlling the local Mac:
- take screenshots
- activate an application
- open a file/folder/URL
- send keystrokes
- inspect the current frontmost app/window
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry

_REQUIRED_COMMANDS = ("osascript", "open", "screencapture")
_SPECIAL_KEY_CODES = {
    "return": 36,
    "enter": 76,
    "tab": 48,
    "space": 49,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "delete": 51,
    "forward_delete": 117,
    "home": 115,
    "end": 119,
    "page_up": 116,
    "page_down": 121,
}
_MODIFIER_ALIASES = {
    "cmd": "command",
    "command": "command",
    "ctrl": "control",
    "control": "control",
    "alt": "option",
    "opt": "option",
    "option": "option",
    "shift": "shift",
}
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

COMPUTER_CONTROL_SCHEMA = {
    "name": "computer_control",
    "description": "Control this macOS desktop in a minimal, explicit way: take a screenshot, activate an app, open a file/folder/URL, send keystrokes, inspect the current frontmost app/window, or perform basic pointer actions.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "activate_app", "open", "keystroke", "frontmost_app", "click", "scroll", "drag"],
                "description": "Desktop action to perform.",
            },
            "app_name": {
                "type": "string",
                "description": "Application name for action='activate_app' (for example 'Safari' or 'Finder').",
            },
            "target": {
                "type": "string",
                "description": "File path, folder path, or URL for action='open'.",
            },
            "output_path": {
                "type": "string",
                "description": "Optional output path for action='screenshot'. Defaults to a timestamped PNG under the Hermes home directory.",
            },
            "window_id": {
                "type": "integer",
                "description": "Optional macOS window id for action='screenshot'. When provided, capture that specific window instead of the whole display.",
            },
            "text": {
                "type": "string",
                "description": "Literal text to type for action='keystroke'.",
            },
            "key": {
                "type": "string",
                "description": "Named key or single character for action='keystroke' (for example 'return', 'tab', 'escape', 'l').",
            },
            "modifiers": {
                "type": "array",
                "description": "Optional modifier keys for action='keystroke'. Supported: command, control, option, shift.",
                "items": {
                    "type": "string",
                    "enum": ["command", "cmd", "control", "ctrl", "option", "opt", "alt", "shift"],
                },
            },
            "x": {
                "type": "integer",
                "description": "Screen x coordinate for pointer actions such as click or scroll.",
            },
            "y": {
                "type": "integer",
                "description": "Screen y coordinate for pointer actions such as click or scroll.",
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for action='click'.",
            },
            "click_count": {
                "type": "integer",
                "description": "Number of presses for action='click'. Defaults to 1.",
            },
            "delta_y": {
                "type": "integer",
                "description": "Vertical wheel delta for action='scroll'. Positive scrolls up, negative scrolls down.",
            },
            "start_x": {
                "type": "integer",
                "description": "Start x coordinate for action='drag'.",
            },
            "start_y": {
                "type": "integer",
                "description": "Start y coordinate for action='drag'.",
            },
            "end_x": {
                "type": "integer",
                "description": "End x coordinate for action='drag'.",
            },
            "end_y": {
                "type": "integer",
                "description": "End y coordinate for action='drag'.",
            },
        },
        "required": ["action"],
    },
}


def _check_computer_control_available() -> bool:
    if sys.platform != "darwin":
        return False
    return all(shutil.which(cmd) for cmd in _REQUIRED_COMMANDS)


def _default_screenshot_path() -> Path:
    root = get_hermes_home() / "computer-control"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"desktop-screenshot-{int(time.time())}.png"


def _escape_applescript_string(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return (proc.stdout or proc.stderr or "").strip()


def _run_osascript(script: str) -> str:
    return _run_command(["osascript", "-e", script])


def _modifier_clause(modifiers: list[str] | None) -> str:
    if not modifiers:
        return ""
    normalized: list[str] = []
    for raw in modifiers:
        name = _MODIFIER_ALIASES.get(str(raw).strip().lower())
        if not name:
            raise ValueError(f"Unsupported modifier: {raw}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        return ""
    return " using {" + ", ".join(f"{name} down" for name in normalized) + "}"


def _build_keystroke_script(*, text: str | None = None, key: str | None = None, modifiers: list[str] | None = None) -> str:
    clause = _modifier_clause(modifiers)
    if text:
        escaped = _escape_applescript_string(text)
        return f'tell application "System Events" to keystroke "{escaped}"{clause}'
    if not key:
        raise ValueError("keystroke action requires text or key")
    key_name = str(key).strip().lower()
    if key_name in _SPECIAL_KEY_CODES:
        return f'tell application "System Events" to key code {_SPECIAL_KEY_CODES[key_name]}{clause}'
    if len(str(key)) == 1:
        escaped = _escape_applescript_string(str(key))
        return f'tell application "System Events" to keystroke "{escaped}"{clause}'
    raise ValueError(f"Unknown action key: {key}")


def _frontmost_app_script() -> str:
    return """
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set winName to ""
    try
        if (count of windows of frontApp) > 0 then
            set winName to name of front window of frontApp
        end if
    end try
    return appName & linefeed & winName
end tell
""".strip()


def _window_helper_source() -> Path:
    return Path(__file__).with_name("mac_window_info.swift")


def _pointer_helper_source() -> Path:
    return Path(__file__).with_name("mac_pointer_action.swift")


def _window_helper_binary() -> Path:
    root = get_hermes_home() / "computer-control" / "bin"
    root.mkdir(parents=True, exist_ok=True)
    return root / "mac-window-info"


def _pointer_helper_binary() -> Path:
    root = get_hermes_home() / "computer-control" / "bin"
    root.mkdir(parents=True, exist_ok=True)
    return root / "mac-pointer-action"


def _ensure_window_helper_binary() -> Path:
    source = _window_helper_source()
    swiftc = shutil.which("swiftc")
    if not swiftc or not source.exists():
        raise RuntimeError("Window helper source or swiftc is unavailable.")
    binary = _window_helper_binary()
    needs_build = not binary.exists() or source.stat().st_mtime > binary.stat().st_mtime
    if needs_build:
        subprocess.run([swiftc, str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _ensure_pointer_helper_binary() -> Path:
    source = _pointer_helper_source()
    swiftc = shutil.which("swiftc")
    if not swiftc or not source.exists():
        raise RuntimeError("Pointer helper source or swiftc is unavailable.")
    binary = _pointer_helper_binary()
    needs_build = not binary.exists() or source.stat().st_mtime > binary.stat().st_mtime
    if needs_build:
        subprocess.run([swiftc, str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _run_pointer_command(args: list[str]) -> dict[str, object]:
    helper = _ensure_pointer_helper_binary()
    raw = _run_command([str(helper), *args])
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Pointer helper returned invalid payload")
    return payload


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_window_bounds(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    x = _coerce_int(value.get("x"))
    if x is None:
        x = _coerce_int(value.get("X"))
    y = _coerce_int(value.get("y"))
    if y is None:
        y = _coerce_int(value.get("Y"))
    width = _coerce_int(value.get("width"))
    if width is None:
        width = _coerce_int(value.get("Width"))
    height = _coerce_int(value.get("height"))
    if height is None:
        height = _coerce_int(value.get("Height"))
    if width is None or height is None:
        return None
    return {
        "x": x or 0,
        "y": y or 0,
        "width": width,
        "height": height,
    }


def _normalize_accessibility_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    return None


def _normalize_accessibility_scalar(value: object) -> object | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if value.strip() else None
    return None


def _normalize_accessibility_actions(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    actions: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        action = raw.strip()
        if action and action not in actions:
            actions.append(action)
    return actions


def _normalize_accessibility_node(value: object, next_index: list[int]) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    index = next_index[0]
    next_index[0] += 1
    node: dict[str, object] = {"index": index}

    role = str(value.get("role") or value.get("AXRole") or "").strip()
    if role:
        node["role"] = role
    subrole = str(value.get("subrole") or value.get("AXSubrole") or "").strip()
    if subrole:
        node["subrole"] = subrole
    title = str(value.get("title") or value.get("AXTitle") or "")
    if title.strip():
        node["title"] = title
    description = str(value.get("description") or value.get("AXDescription") or "")
    if description.strip():
        node["description"] = description

    normalized_value = _normalize_accessibility_scalar(value.get("value"))
    if normalized_value is not None:
        node["value"] = normalized_value

    enabled = _normalize_accessibility_bool(value.get("enabled"))
    if enabled is not None:
        node["enabled"] = enabled
    focused = _normalize_accessibility_bool(value.get("focused"))
    if focused is not None:
        node["focused"] = focused

    frame = _normalize_window_bounds(value.get("frame") or value.get("bounds"))
    if frame:
        node["frame"] = frame

    actions = _normalize_accessibility_actions(value.get("actions"))
    if actions:
        node["actions"] = actions

    children: list[dict[str, object]] = []
    raw_children = value.get("children")
    if isinstance(raw_children, list):
        for child in raw_children:
            normalized_child = _normalize_accessibility_node(child, next_index)
            if normalized_child:
                children.append(normalized_child)
    node["children"] = children

    if set(node.keys()) == {"index", "children"} and not children:
        return None
    return node


def _normalize_accessibility_tree(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        raw_nodes = [value]
    elif isinstance(value, list):
        raw_nodes = value
    else:
        return []
    next_index = [0]
    nodes: list[dict[str, object]] = []
    for raw in raw_nodes:
        normalized = _normalize_accessibility_node(raw, next_index)
        if normalized:
            nodes.append(normalized)
    return nodes


def _window_bounds_close(left: object, right: object, *, position_tolerance: int = 16, size_tolerance: int = 16) -> bool:
    left_bounds = _normalize_window_bounds(left)
    right_bounds = _normalize_window_bounds(right)
    if not left_bounds or not right_bounds:
        return False
    return (
        abs(left_bounds["x"] - right_bounds["x"]) <= position_tolerance
        and abs(left_bounds["y"] - right_bounds["y"]) <= position_tolerance
        and abs(left_bounds["width"] - right_bounds["width"]) <= size_tolerance
        and abs(left_bounds["height"] - right_bounds["height"]) <= size_tolerance
    )


def _accessibility_tree_matches_window(tree: object, *, window_title: object, window_bounds: object) -> bool:
    if not isinstance(tree, list) or not tree or not isinstance(tree[0], dict):
        return False
    root = tree[0]
    role = str(root.get("role") or "").strip()
    if role and role not in {"AXWindow", "AXSheet", "AXDialog", "AXPopover", "AXDrawer"}:
        return False

    root_title = str(root.get("title") or "")
    wanted_title = str(window_title or "")
    title_match = bool(root_title.strip() and wanted_title.strip() and root_title == wanted_title)
    if root_title.strip() and wanted_title.strip() and root_title != wanted_title:
        return False
    root_frame = _normalize_window_bounds(root.get("frame"))
    wanted_bounds = _normalize_window_bounds(window_bounds)
    frame_match = _window_bounds_close(root_frame, wanted_bounds)
    if root_frame and wanted_bounds:
        return frame_match
    if title_match:
        return True
    return frame_match


def _normalize_helper_window(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    window_id = _coerce_int(value.get("window_id"))
    if window_id is None:
        window_id = _coerce_int(value.get("id"))
    bounds = _normalize_window_bounds(value.get("window_bounds") or value.get("bounds"))
    if window_id is None or not bounds:
        return None
    return {
        "window_title": str(value.get("window_title") or value.get("title") or ""),
        "window_id": window_id,
        "window_bounds": bounds,
    }


def _window_area(window: dict[str, object]) -> int:
    bounds = window.get("window_bounds")
    if not isinstance(bounds, dict):
        return 0
    width = _coerce_int(bounds.get("width")) or 0
    height = _coerce_int(bounds.get("height")) or 0
    return max(width, 0) * max(height, 0)


def _window_looks_fragmentary(window: dict[str, object]) -> bool:
    bounds = window.get("window_bounds")
    if not isinstance(bounds, dict):
        return True
    width = _coerce_int(bounds.get("width")) or 0
    height = _coerce_int(bounds.get("height")) or 0
    area = _window_area(window)
    title = str(window.get("window_title") or "").strip()
    return not title and (width < 120 or height < 80 or area < 10_000)


def _select_frontmost_window(windows: object) -> dict[str, object] | None:
    if not isinstance(windows, list):
        return None
    candidates = [candidate for candidate in (_normalize_helper_window(item) for item in windows) if candidate]
    if not candidates:
        return None
    if not _window_looks_fragmentary(candidates[0]):
        return candidates[0]
    for candidate in candidates[1:]:
        if not _window_looks_fragmentary(candidate):
            return candidate
    titled = [candidate for candidate in candidates if str(candidate.get("window_title") or "").strip()]
    if titled:
        return titled[0]
    return candidates[0]


def _frontmost_window_info(*, require_helper_success: bool = False) -> dict[str, object]:
    try:
        helper = _ensure_window_helper_binary()
        raw = _run_command([str(helper)])
        data = json.loads(raw)
        if not isinstance(data, dict):
            if require_helper_success:
                raise RuntimeError("frontmost window helper returned invalid payload")
            raise ValueError("frontmost window helper returned invalid payload")
        info: dict[str, object] = {
            "app_name": str(data.get("app_name") or ""),
            "bundle_id": str(data.get("bundle_id") or ""),
            "bundle_name": str(data.get("bundle_name") or ""),
            "bundle_path": str(data.get("bundle_path") or ""),
            "process_id": _coerce_int(data.get("process_id")),
            "window_title": str(data.get("window_title") or ""),
            "window_id": _coerce_int(data.get("window_id")),
            "window_bounds": _normalize_window_bounds(data.get("window_bounds")),
            "accessibility_tree": _normalize_accessibility_tree(data.get("accessibility_tree")),
        }
        selected = _select_frontmost_window(data.get("windows"))
        if selected:
            info.update(selected)
        if info["accessibility_tree"] and not _accessibility_tree_matches_window(
            info["accessibility_tree"],
            window_title=info.get("window_title"),
            window_bounds=info.get("window_bounds"),
        ):
            info["accessibility_tree"] = []
        return info
    except Exception as e:
        if require_helper_success:
            raise RuntimeError(f"frontmost window helper failed: {e}") from e

    raw = _run_osascript(_frontmost_app_script())
    parts = raw.splitlines()
    app = parts[0].strip() if parts else ""
    window = parts[1].strip() if len(parts) > 1 else ""
    return {
        "app_name": app,
        "bundle_id": "",
        "bundle_name": "",
        "bundle_path": "",
        "process_id": None,
        "window_title": window,
        "window_id": None,
        "window_bounds": None,
        "accessibility_tree": [],
    }


def frontmost_window_state(*, include_accessibility: bool = True, require_helper_success: bool = False) -> dict[str, object]:
    info = dict(_frontmost_window_info(require_helper_success=require_helper_success))
    if not include_accessibility:
        info["accessibility_tree"] = []
    return info


def _screencapture_command(path: Path, *, window_id: int | None = None) -> list[str]:
    cmd = ["screencapture", "-x"]
    if window_id is not None:
        cmd.extend(["-o", f"-l{int(window_id)}"])
    cmd.append(str(path))
    return cmd


def _humanize_os_error(message: str) -> str:
    text = (message or "").strip()
    low = text.lower()
    if "-1743" in text or "not authorized" in low or "automation" in low:
        return "macOS blocked Automation/System Events access. Grant Hermes/Terminal the needed Automation or Accessibility permission in System Settings."
    if "could not create image from display" in low:
        return "macOS could not capture the display. This usually means Screen Recording permission is missing, the display is unavailable, or the current runtime cannot access the GUI session."
    if "could not create image from window" in low:
        return "macOS could not capture that specific window. The window may be unavailable, private, or unsupported for window-level capture from the current session."
    if "not permitted" in low or "screen recording" in low:
        return "macOS blocked screen capture. Grant Hermes/Terminal Screen Recording permission in System Settings."
    return text or "Desktop control command failed."


def computer_control(*, action: str, app_name: str | None = None, target: str | None = None,
                     output_path: str | None = None, window_id: int | None = None, text: str | None = None,
                     key: str | None = None, modifiers: list[str] | None = None,
                     x: int | None = None, y: int | None = None,
                     button: str | None = None, click_count: int | None = None,
                     delta_y: int | None = None,
                     start_x: int | None = None, start_y: int | None = None,
                     end_x: int | None = None, end_y: int | None = None) -> str:
    action_name = str(action or "").strip().lower()
    screenshot_path: Path | None = None
    try:
        if action_name == "screenshot":
            path = Path(output_path).expanduser() if output_path else _default_screenshot_path()
            screenshot_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized_window_id = int(window_id) if window_id is not None else None
            _run_command(_screencapture_command(path, window_id=normalized_window_id))
            payload = {"success": True, "action": action_name, "path": str(path)}
            if normalized_window_id is not None:
                payload["window_id"] = normalized_window_id
            if " " not in str(path):
                payload["media_tag"] = f"MEDIA:{path}"
            return json.dumps(payload, ensure_ascii=False)

        if action_name == "activate_app":
            if not app_name:
                raise ValueError("activate_app action requires app_name")
            _run_osascript(f'tell application "{_escape_applescript_string(app_name)}" to activate')
            return json.dumps({"success": True, "action": action_name, "app_name": app_name}, ensure_ascii=False)

        if action_name == "open":
            if not target:
                raise ValueError("open action requires target")
            resolved = str(Path(target).expanduser()) if not _URL_RE.match(str(target)) else str(target)
            _run_command(["open", resolved])
            return json.dumps({"success": True, "action": action_name, "target": resolved}, ensure_ascii=False)

        if action_name == "keystroke":
            script = _build_keystroke_script(text=text, key=key, modifiers=modifiers)
            _run_osascript(script)
            return json.dumps({
                "success": True,
                "action": action_name,
                "text": text,
                "key": key,
                "modifiers": modifiers or [],
            }, ensure_ascii=False)

        if action_name == "click":
            if x is None or y is None:
                raise ValueError("click action requires x and y")
            normalized_button = str(button or "left").strip().lower()
            if normalized_button not in {"left", "right", "middle"}:
                raise ValueError(f"Unsupported click button: {button}")
            normalized_count = int(click_count or 1)
            if normalized_count < 1:
                raise ValueError("click_count must be >= 1")
            helper_payload = _run_pointer_command([
                "click",
                "--x", str(int(x)),
                "--y", str(int(y)),
                "--button", normalized_button,
                "--count", str(normalized_count),
            ])
            return json.dumps({
                "success": True,
                "action": action_name,
                "x": int(x),
                "y": int(y),
                "button": normalized_button,
                "click_count": normalized_count,
                **helper_payload,
            }, ensure_ascii=False)

        if action_name == "scroll":
            if delta_y is None:
                raise ValueError("scroll action requires delta_y")
            args = ["scroll", "--delta-y", str(int(delta_y))]
            if x is not None:
                args.extend(["--x", str(int(x))])
            if y is not None:
                args.extend(["--y", str(int(y))])
            helper_payload = _run_pointer_command(args)
            return json.dumps({
                "success": True,
                "action": action_name,
                "x": int(x) if x is not None else None,
                "y": int(y) if y is not None else None,
                "delta_y": int(delta_y),
                **helper_payload,
            }, ensure_ascii=False)

        if action_name == "drag":
            if None in {start_x, start_y, end_x, end_y}:
                raise ValueError("drag action requires start_x, start_y, end_x, and end_y")
            helper_payload = _run_pointer_command([
                "drag",
                "--start-x", str(int(start_x)),
                "--start-y", str(int(start_y)),
                "--end-x", str(int(end_x)),
                "--end-y", str(int(end_y)),
            ])
            return json.dumps({
                "success": True,
                "action": action_name,
                "start_x": int(start_x),
                "start_y": int(start_y),
                "end_x": int(end_x),
                "end_y": int(end_y),
                **helper_payload,
            }, ensure_ascii=False)

        if action_name == "frontmost_app":
            info = frontmost_window_state(include_accessibility=False)
            return json.dumps({
                "success": True,
                "action": action_name,
                "app_name": str(info.get("app_name") or ""),
                "bundle_id": str(info.get("bundle_id") or ""),
                "bundle_name": str(info.get("bundle_name") or ""),
                "bundle_path": str(info.get("bundle_path") or ""),
                "process_id": info.get("process_id"),
                "window_title": str(info.get("window_title") or ""),
                "window_id": info.get("window_id"),
                "accessibility_tree": info.get("accessibility_tree") or [],
                "window_bounds": info.get("window_bounds"),
            }, ensure_ascii=False)

        raise ValueError(f"Unknown action: {action}")
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except subprocess.CalledProcessError as e:
        if screenshot_path and screenshot_path.exists() and screenshot_path.stat().st_size == 0:
            try:
                screenshot_path.unlink()
            except OSError:
                pass
        detail = _humanize_os_error((e.stderr or e.stdout or str(e)))
        return json.dumps({"error": detail}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Desktop control failed: {type(e).__name__}: {e}"}, ensure_ascii=False)


def _handle_computer_control(args, **_kw):
    return computer_control(
        action=args.get("action", ""),
        app_name=args.get("app_name"),
        target=args.get("target"),
        output_path=args.get("output_path"),
        window_id=args.get("window_id"),
        text=args.get("text"),
        key=args.get("key"),
        modifiers=args.get("modifiers"),
        x=args.get("x"),
        y=args.get("y"),
        button=args.get("button"),
        click_count=args.get("click_count"),
        delta_y=args.get("delta_y"),
        start_x=args.get("start_x"),
        start_y=args.get("start_y"),
        end_x=args.get("end_x"),
        end_y=args.get("end_y"),
    )


registry.register(
    name="computer_control",
    toolset="computer",
    schema=COMPUTER_CONTROL_SCHEMA,
    handler=_handle_computer_control,
    check_fn=_check_computer_control_available,
    emoji="🖥️",
    max_result_size_chars=100_000,
)
