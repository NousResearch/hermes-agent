#!/usr/bin/env python3
"""
Desktop/System Notification Tool Module

Supports Linux (notify-send), macOS (osascript), Windows (PowerShell toast).
Falls back to terminal bell in SSH/headless environments.
"""

import json
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def _is_headless():
    if os.getenv("SSH_CONNECTION") or os.getenv("SSH_CLIENT"):
        return True
    if sys.platform.startswith("linux"):
        return not bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    return False


def _get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "darwin":
        return "macos"
    elif sys.platform == "win32":
        return "windows"
    return "unknown"


def _terminal_bell():
    sys.stdout.write("\a")
    sys.stdout.flush()
    return "terminal_bell"


def _notify_linux(title, message, urgency="normal"):
    if not shutil.which("notify-send"):
        return _terminal_bell()
    urgency_val = urgency if urgency in ("low", "normal", "critical") else "normal"
    subprocess.run(["notify-send", "--urgency", urgency_val, "--", title, message], timeout=10)
    return "notify-send"


def _notify_macos(title, message):
    if not shutil.which("osascript"):
        return _terminal_bell()
    script = "on run argv\n  display notification (item 2 of argv) with title (item 1 of argv)\nend run"
    subprocess.run(["osascript", "-e", script, title, message], timeout=10)
    return "osascript"


def _notify_windows(title, message):
    if not shutil.which("powershell"):
        return _terminal_bell()
    import base64
    env = os.environ.copy()
    env["HERMES_NOTIFY_TITLE"] = title
    env["HERMES_NOTIFY_MSG"] = message
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null\n"
        "$t = [Windows.UI.Notifications.ToastTemplateType]::ToastText02\n"
        "$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($t)\n"
        "$xml.GetElementsByTagName(\'text\')[0].AppendChild($xml.CreateTextNode($env:HERMES_NOTIFY_TITLE)) | Out-Null\n"
        "$xml.GetElementsByTagName(\'text\')[1].AppendChild($xml.CreateTextNode($env:HERMES_NOTIFY_MSG)) | Out-Null\n"
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)\n"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(\'Hermes Agent\').Show($toast)\n"
    )
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    subprocess.run(["powershell", "-EncodedCommand", encoded], timeout=15, env=env)
    return "powershell-toast"


def notify_tool(title, message="", urgency="normal"):
    if not title or not title.strip():
        return json.dumps({"success": False, "error": "Title is required"})
    title = title[:200].strip()
    message = message[:500].strip()
    headless = _is_headless()
    plat = _get_platform()
    if headless:
        method = _terminal_bell()
        return json.dumps({"success": True, "method": method, "note": "SSH/headless fallback"})
    try:
        if plat == "linux":
            method = _notify_linux(title, message, urgency)
        elif plat == "macos":
            method = _notify_macos(title, message)
        elif plat == "windows":
            method = _notify_windows(title, message)
        else:
            method = _terminal_bell()
        return json.dumps({"success": True, "method": method, "platform": plat})
    except Exception as e:
        logger.warning("Notification failed: %s", e)
        _terminal_bell()
        return json.dumps({"success": True, "method": "terminal_bell", "fallback_reason": str(e)})


def notify_sound_tool(sound_type="complete"):
    plat = _get_platform()
    if _is_headless():
        _terminal_bell()
        return json.dumps({"success": True, "method": "terminal_bell", "note": "SSH/headless fallback"})
    try:
        if plat == "linux":
            for cmd in [["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                        ["pw-play", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"]]:
                if shutil.which(cmd[0]) and os.path.exists(cmd[1]):
                    subprocess.run(cmd, timeout=5, capture_output=True)
                    return json.dumps({"success": True, "platform": plat})
        elif plat == "macos":
            sound = "/System/Library/Sounds/Glass.aiff"
            if shutil.which("afplay") and os.path.exists(sound):
                subprocess.run(["afplay", sound], timeout=5)
                return json.dumps({"success": True, "platform": plat})
        elif plat == "windows":
            import base64
            ps = "[System.Media.SystemSounds]::Asterisk.Play()\n"
            encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            subprocess.run(["powershell", "-EncodedCommand", encoded], timeout=10)
            return json.dumps({"success": True, "platform": plat})
        _terminal_bell()
        return json.dumps({"success": True, "method": "terminal_bell"})
    except Exception as e:
        _terminal_bell()
        return json.dumps({"success": True, "method": "terminal_bell", "fallback_reason": str(e)})


def check_notification_requirements():
    return True


from tools.registry import registry

registry.register(
    name="notify",
    toolset="notification",
    schema={
        "name": "notify",
        "description": "Send a desktop/system notification. Falls back to terminal bell over SSH or headless environments.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title."},
                "message": {"type": "string", "description": "Notification body."},
                "urgency": {"type": "string", "enum": ["low", "normal", "critical"], "description": "Urgency level (Linux only)."},
            },
            "required": ["title"]
        }
    },
    handler=lambda args, **kw: notify_tool(
        title=args.get("title", ""),
        message=args.get("message", ""),
        urgency=args.get("urgency", "normal"),
    ),
    check_fn=check_notification_requirements,
)

registry.register(
    name="notify_sound",
    toolset="notification",
    schema={
        "name": "notify_sound",
        "description": "Play a system alert sound.",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_type": {"type": "string", "enum": ["complete"], "description": "Sound type."}
            },
            "required": []
        }
    },
    handler=lambda args, **kw: notify_sound_tool(sound_type=args.get("sound_type", "complete")),
    check_fn=check_notification_requirements,
)
