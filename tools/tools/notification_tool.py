"""
notification_tool — Send desktop notifications and alert sounds when Hermes
finishes long-running tasks. Works locally and degrades gracefully over SSH.
"""

import shutil
import subprocess
import sys
from typing import Any

from tools.registry import registry


def _check_notify() -> bool:
    if sys.platform == "linux":
        return shutil.which("notify-send") is not None
    if sys.platform == "darwin":
        return True
    if sys.platform == "win32":
        return True
    return False


def _check_sound() -> bool:
    if sys.platform == "linux":
        return shutil.which("paplay") is not None or shutil.which("aplay") is not None
    if sys.platform == "darwin":
        return True
    if sys.platform == "win32":
        return True
    return False


def _applescript_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _handle_notify(args: dict[str, Any], **kwargs) -> dict[str, Any]:
    title   = str(args.get("title",   "Hermes"))
    message = str(args.get("message", "Task complete."))
    urgency = str(args.get("urgency", "normal"))

    if urgency not in ("low", "normal", "critical"):
        urgency = "normal"

    try:
        if sys.platform == "linux":
            if shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", "--urgency", urgency, "--", title, message],
                    check=True,
                    capture_output=True,
                )
            else:
                print("\a", end="", flush=True)
                return {"success": True, "backend": "terminal_bell",
                        "note": "notify-send not found; used terminal bell"}

        elif sys.platform == "darwin":
            script = (
                f'display notification {_applescript_quote(message)} '
                f'with title {_applescript_quote(title)}'
            )
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)

        elif sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n = New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon = [System.Drawing.SystemIcons]::Information;"
                "$n.Visible = $true;"
                "$n.ShowBalloonTip(5000, $args[0], $args[1], "
                "[System.Windows.Forms.ToolTipIcon]::None);"
                "Start-Sleep -Seconds 6; $n.Dispose()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script,
                 "-ArgumentList", title, message],
                check=True,
                capture_output=True,
            )

        else:
            print("\a", end="", flush=True)
            return {"success": True, "backend": "terminal_bell"}

        return {"success": True, "backend": sys.platform}

    except subprocess.CalledProcessError as e:
        print("\a", end="", flush=True)
        return {
            "success": True,
            "backend": "terminal_bell",
            "note": f"Primary backend failed ({e}); used terminal bell",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_notify_sound(args: dict[str, Any], **kwargs) -> dict[str, Any]:
    sound = str(args.get("sound", "default"))

    try:
        if sys.platform == "linux":
            candidates = [
                "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "/usr/share/sounds/freedesktop/stereo/message.oga",
                "/usr/share/sounds/alsa/Front_Center.wav",
            ]
            played = False
            for path in candidates:
                player = shutil.which("paplay") or shutil.which("aplay")
                if player:
                    try:
                        subprocess.run([player, path], check=True, capture_output=True)
                        played = True
                        break
                    except subprocess.CalledProcessError:
                        continue
            if not played:
                print("\a", end="", flush=True)
                return {"success": True, "backend": "terminal_bell"}

        elif sys.platform == "darwin":
            sound_map = {
                "default":  "Ping",
                "error":    "Basso",
                "complete": "Glass",
            }
            sound_name = sound_map.get(sound, "Ping")
            subprocess.run(
                ["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
                check=True,
                capture_output=True,
            )

        elif sys.platform == "win32":
            freq = {"default": 800, "error": 400, "complete": 1000}.get(sound, 800)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"[console]::beep({freq}, 400)"],
                check=True,
                capture_output=True,
            )

        else:
            print("\a", end="", flush=True)
            return {"success": True, "backend": "terminal_bell"}

        return {"success": True, "backend": sys.platform}

    except subprocess.CalledProcessError as e:
        print("\a", end="", flush=True)
        return {
            "success": True,
            "backend": "terminal_bell",
            "note": f"Primary backend failed ({e}); used terminal bell",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


registry.register(
    name="notify",
    toolset="notification",
    schema={
        "name": "notify",
        "description": (
            "Send a desktop notification to the user. "
            "Falls back to a terminal bell when running over SSH or "
            "when no display server is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notification title, e.g. 'Task Complete'",
                },
                "message": {
                    "type": "string",
                    "description": "Notification body text",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "critical"],
                    "description": "Urgency level (Linux only). Default: normal",
                    "default": "normal",
                },
            },
            "required": ["title", "message"],
        },
    },
    handler=lambda args, **kw: _handle_notify(args, **kw),
    check_fn=_check_notify,
)

registry.register(
    name="notify_sound",
    toolset="notification",
    schema={
        "name": "notify_sound",
        "description": (
            "Play a system alert sound. "
            "Falls back to a terminal bell on headless / SSH environments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sound": {
                    "type": "string",
                    "enum": ["default", "error", "complete"],
                    "description": "Sound style hint. Default: default",
                    "default": "default",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: _handle_notify_sound(args, **kw),
    check_fn=_check_sound,
)
