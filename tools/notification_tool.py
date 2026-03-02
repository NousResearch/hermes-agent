"""
Notification tool for Hermes Agent.

Sends desktop/system notifications without requiring any messaging platform
(Telegram, Discord, etc.). Works on Linux, macOS, and Windows.

Dependencies: none (uses OS built-ins only)
"""

import json
import platform
import subprocess
import shutil
from typing import Optional


# ---------------------------------------------------------------------------
# Registration helper (mirrors the pattern used by other Hermes tools)
# ---------------------------------------------------------------------------

def register(registry):
    """Register notification tools with the tool registry."""

    registry.register(
        name="notify",
        description=(
            "Send a desktop/system notification to the user. "
            "Use this when a long-running task finishes, when something needs "
            "the user's attention, or when explicitly asked to notify. "
            "Works on Linux (libnotify/notify-send), macOS (osascript), and "
            "Windows (PowerShell toast). Falls back gracefully if not supported."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title (e.g. 'Task Complete')",
                },
                "message": {
                    "type": "string",
                    "description": "Notification body text.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "critical"],
                    "description": "Urgency level. Defaults to 'normal'.",
                },
            },
            "required": ["title", "message"],
        },
        handler=handle_notify,
        check_fn=_notify_available,
    )

    registry.register(
        name="notify_sound",
        description=(
            "Play a short system alert sound to get the user's attention. "
            "Useful after notify when you want an audible cue. "
            "Works on Linux (paplay/aplay/beep), macOS (afplay), Windows (PowerShell)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sound": {
                    "type": "string",
                    "enum": ["default", "bell", "complete"],
                    "description": "Which sound to play. Defaults to 'default'.",
                },
            },
            "required": [],
        },
        handler=handle_notify_sound,
        check_fn=_sound_available,
    )


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _notify_available() -> bool:
    """Return True if at least one notification backend is available."""
    system = platform.system()
    if system == "Linux":
        return shutil.which("notify-send") is not None
    if system == "Darwin":
        return shutil.which("osascript") is not None
    if system == "Windows":
        return True  # PowerShell is always available on Windows
    return False


def _sound_available() -> bool:
    """Return True if at least one sound backend is available."""
    system = platform.system()
    if system == "Linux":
        return any(
            shutil.which(cmd) is not None
            for cmd in ("paplay", "aplay", "beep", "pw-play")
        )
    if system == "Darwin":
        return shutil.which("afplay") is not None
    if system == "Windows":
        return True
    return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_notify(title: str, message: str, urgency: str = "normal") -> str:
    """Send a desktop notification. Returns JSON with status."""
    system = platform.system()

    try:
        if system == "Linux":
            result = _notify_linux(title, message, urgency)
        elif system == "Darwin":
            result = _notify_macos(title, message)
        elif system == "Windows":
            result = _notify_windows(title, message)
        else:
            return json.dumps({
                "success": False,
                "error": f"Unsupported platform: {system}",
            })

        return json.dumps(result)

    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def handle_notify_sound(sound: str = "default") -> str:
    """Play a system alert sound. Returns JSON with status."""
    system = platform.system()

    try:
        if system == "Linux":
            result = _sound_linux(sound)
        elif system == "Darwin":
            result = _sound_macos(sound)
        elif system == "Windows":
            result = _sound_windows()
        else:
            return json.dumps({
                "success": False,
                "error": f"Unsupported platform: {system}",
            })

        return json.dumps(result)

    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Linux backends
# ---------------------------------------------------------------------------

def _notify_linux(title: str, message: str, urgency: str) -> dict:
    if not shutil.which("notify-send"):
        return {"success": False, "error": "notify-send not found. Install libnotify-bin."}

    urgency_map = {"low": "low", "normal": "normal", "critical": "critical"}
    level = urgency_map.get(urgency, "normal")

    cmd = ["notify-send", "--urgency", level, title, message]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

    if proc.returncode == 0:
        return {"success": True, "backend": "notify-send", "platform": "linux"}
    return {"success": False, "error": proc.stderr.strip() or "notify-send failed"}


def _sound_linux(sound: str) -> dict:
    # Try common sound players in order of preference
    # paplay (PulseAudio), pw-play (PipeWire), aplay (ALSA), beep (PC speaker)
    sound_files = {
        "default": [
            "/usr/share/sounds/freedesktop/stereo/message.oga",
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/ubuntu/stereo/message.ogg",
        ],
        "bell": [
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/ubuntu/stereo/bell.ogg",
        ],
        "complete": [
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
            "/usr/share/sounds/ubuntu/stereo/dialog-information.ogg",
        ],
    }

    files = sound_files.get(sound, sound_files["default"])

    for player in ("paplay", "pw-play", "aplay"):
        if not shutil.which(player):
            continue
        for f in files:
            try:
                proc = subprocess.run(
                    [player, f], capture_output=True, timeout=5
                )
                if proc.returncode == 0:
                    return {"success": True, "backend": player, "platform": "linux"}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    # Last resort: terminal bell via echo
    try:
        subprocess.run(["bash", "-c", "echo -ne '\\a'"], timeout=2)
        return {"success": True, "backend": "terminal-bell", "platform": "linux"}
    except Exception:
        pass

    return {"success": False, "error": "No sound backend found (tried paplay, pw-play, aplay)"}


# ---------------------------------------------------------------------------
# macOS backends
# ---------------------------------------------------------------------------

def _notify_macos(title: str, message: str) -> dict:
    # osascript display notification
    script = (
        f'display notification "{_esc(message)}" '
        f'with title "{_esc(title)}"'
    )
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=5,
    )
    if proc.returncode == 0:
        return {"success": True, "backend": "osascript", "platform": "macos"}
    return {"success": False, "error": proc.stderr.strip() or "osascript failed"}


def _sound_macos(sound: str) -> dict:
    sound_map = {
        "default": "/System/Library/Sounds/Funk.aiff",
        "bell": "/System/Library/Sounds/Ping.aiff",
        "complete": "/System/Library/Sounds/Glass.aiff",
    }
    sound_file = sound_map.get(sound, sound_map["default"])
    proc = subprocess.run(
        ["afplay", sound_file],
        capture_output=True, timeout=5,
    )
    if proc.returncode == 0:
        return {"success": True, "backend": "afplay", "platform": "macos"}
    return {"success": False, "error": proc.stderr.strip() or "afplay failed"}


# ---------------------------------------------------------------------------
# Windows backends
# ---------------------------------------------------------------------------

def _notify_windows(title: str, message: str) -> dict:
    # PowerShell toast notification (Windows 10+)
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null;"
        "$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        f"$template.GetElementsByTagName('text')[0].InnerText = '{_esc(title)}';"
        f"$template.GetElementsByTagName('text')[1].InnerText = '{_esc(message)}';"
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($template);"
        "$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Hermes Agent');"
        "$notifier.Show($toast);"
    )
    proc = subprocess.run(
        ["powershell", "-Command", script],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode == 0:
        return {"success": True, "backend": "powershell-toast", "platform": "windows"}
    return {"success": False, "error": proc.stderr.strip() or "PowerShell toast failed"}


def _sound_windows() -> dict:
    script = "[System.Media.SystemSounds]::Beep.Play()"
    proc = subprocess.run(
        ["powershell", "-Command", script],
        capture_output=True, text=True, timeout=5,
    )
    if proc.returncode == 0:
        return {"success": True, "backend": "powershell-beep", "platform": "windows"}
    return {"success": False, "error": proc.stderr.strip() or "PowerShell beep failed"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """Escape single quotes for shell/AppleScript strings."""
    return s.replace("'", "\\'").replace('"', '\\"')
