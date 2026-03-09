"""
notification_tool — Send desktop notifications and alert sounds when Hermes
finishes long-running tasks. Works locally and degrades gracefully over SSH.
"""

import base64
import os
import shutil
import subprocess
import sys
from typing import Any

from tools.registry import registry


def _is_headless() -> bool:
    """Return True when running over SSH or without a display server."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if sys.platform == "linux":
        return not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    return False


def _check_notify() -> bool:
    if _is_headless():
        return True  # terminal bell fallback always works
    if sys.platform == "linux":
        return shutil.which("notify-send") is not None
    if sys.platform in ("darwin", "win32"):
        return True
    return False


def _check_sound() -> bool:
    if sys.platform == "linux":
        return (
            shutil.which("paplay") is not None
            or shutil.which("aplay") is not None
            or True  # terminal bell fallback
        )
    if sys.platform in ("darwin", "win32"):
        return True
    return False


def _applescript_escape(value: str) -> str:
    """Escape a string for safe embedding in an AppleScript quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _handle_notify(args: dict[str, Any], **kwargs) -> dict[str, Any]:
    title   = str(args.get("title",   "Hermes"))
    message = str(args.get("message", "Task complete."))
    urgency = str(args.get("urgency", "normal"))

    if urgency not in ("low", "normal", "critical"):
        urgency = "normal"

    # SSH / headless — terminal bell only
    if _is_headless():
        print("\a", end="", flush=True)
        return {"success": True, "backend": "terminal_bell",
                "note": "SSH/headless session; used terminal bell"}

    try:
        if sys.platform == "linux":
            if shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", "--urgency", urgency, "--", title, message],
                    check=True, capture_output=True,
                )
            else:
                print("\a", end="", flush=True)
                return {"success": True, "backend": "terminal_bell",
                        "note": "notify-send not found; used terminal bell"}

        elif sys.platform == "darwin":
            # Pass via stdin to avoid any shell interpolation
            script = (
                f'display notification "{_applescript_escape(message)}" '
                f'with title "{_applescript_escape(title)}"'
            )
            subprocess.run(
                ["osascript"], input=script.encode(), check=True, capture_output=True
            )

        elif sys.platform == "win32":
            # Use -EncodedCommand so title/message never touch the PS command string
            ps_raw = (
                "param([string]$T,[string]$M);"
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n=New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon=[System.Drawing.SystemIcons]::Information;"
                "$n.Visible=$true;"
                "$n.ShowBalloonTip(5000,$T,$M,[System.Windows.Forms.ToolTipIcon]::None);"
                "Start-Sleep -Seconds 6;$n.Dispose()"
            )
            encoded = base64.b64encode(ps_raw.encode("utf-16-le")).decode("ascii")
            subprocess.run(
                ["powershell", "-NoProfile", "-EncodedCommand", encoded,
                 "-T", title, "-M", message],
                check=True, capture_output=True,
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
            player = shutil.which("paplay") or shutil.which("aplay")
            played = False
            if player:
                for path in candidates:
                    if not os.path.exists(path):
                        continue
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
            sound_map = {"default": "Ping", "error": "Basso", "complete": "Glass"}
            sound_name = sound_map.get(sound, "Ping")
            subprocess.run(
                ["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
                check=True, capture_output=True,
            )

        elif sys.platform == "win32":
            freq = {"default": 800, "error": 400, "complete": 1000}.get(sound, 800)
            ps_cmd = f"[System.Media.SystemSounds]::Beep.Play(); [console]::beep({freq},400)"
            encoded = base64.b64encode(ps_cmd.encode("utf-16-le")).decode("ascii")
            subprocess.run(
                ["powershell", "-NoProfile", "-EncodedCommand", encoded],
                check=True, capture_output=True,
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
