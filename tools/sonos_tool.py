"""
Sonos speaker control tool for Hermes.

Controls Sonos speakers on the local network via SoCo library.
Speaker IPs and coordinator are loaded from config file.
"""

import json
import logging
from pathlib import Path

import yaml

from tools.registry import registry

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".hermes" / "smart_home"
_CONFIG_FILE = _CONFIG_DIR / "sonos.yaml"


def _load_config() -> dict:
    """Load tool config from YAML file."""
    if not _CONFIG_FILE.exists():
        logger.warning("Sonos config not found: %s", _CONFIG_FILE)
        return {}
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _check_sonos_reqs() -> bool:
    try:
        import soco
        config = _load_config()
        if not config.get("coordinator_ip"):
            return False
        return True
    except ImportError:
        return False


def _get_coordinator_ip() -> str:
    """Get Sonos coordinator IP from global cache, fallback to config."""
    config = _load_config()
    default_ip = config.get("coordinator_ip", "")
    try:
        from tools.device_scanner import load_cache
        cache = load_cache()
        sonos = cache.get("sonos", {})
        # Find the Playbar (coordinator)
        for name, ip in sonos.items():
            if "playbar" in name.lower() or "coordinator" in name.lower():
                return ip
        # Fallback to first non-sub device
        for name, ip in sonos.items():
            if "sub" not in name.lower():
                return ip
    except Exception:
        pass
    return default_ip


def sonos_control(action: str, param: str = None) -> str:
    try:
        import soco

        coordinator_ip = _get_coordinator_ip()
        try:
            speaker = soco.SoCo(coordinator_ip)
            _ = speaker.player_name  # Test connection
        except Exception:
            # IP may have changed, try rescan
            try:
                from tools.device_scanner import rescan
                rescan(force=True)
                coordinator_ip = _get_coordinator_ip()
            except Exception:
                pass
            speaker = soco.SoCo(coordinator_ip)

        if action == "status":
            info = speaker.get_speaker_info()
            track = speaker.get_current_track_info()
            transport = speaker.get_current_transport_info()
            return json.dumps({
                "success": True,
                "name": speaker.player_name,
                "model": info.get("model_name", ""),
                "volume": speaker.volume,
                "muted": speaker.mute,
                "playback_state": transport.get("current_transport_state", ""),
                "current_track": {
                    "title": track.get("title", ""),
                    "artist": track.get("artist", ""),
                    "album": track.get("album", ""),
                    "duration": track.get("duration", ""),
                    "position": track.get("position", ""),
                },
                "group_members": [
                    {"name": m.player_name, "ip": m.ip_address, "model": m.get_speaker_info().get("model_name", "")}
                    for m in speaker.group.members
                ] if speaker.group else [],
            }, ensure_ascii=False)

        elif action == "play":
            if param:
                # Play a URI or favorite
                favs = speaker.music_library.get_sonos_favorites()
                for fav in favs:
                    if param.lower() in fav.title.lower():
                        speaker.play_uri(fav.reference)
                        return json.dumps({"success": True, "message": f"Playing favorite: {fav.title}"})
                # Try as direct URI
                speaker.play_uri(param)
                return json.dumps({"success": True, "message": f"Playing: {param}"})
            else:
                speaker.play()
                return json.dumps({"success": True, "message": "Playback resumed"})

        elif action == "pause":
            speaker.pause()
            return json.dumps({"success": True, "message": "Paused"})

        elif action == "stop":
            speaker.stop()
            return json.dumps({"success": True, "message": "Stopped"})

        elif action == "next":
            speaker.next()
            return json.dumps({"success": True, "message": "Next track"})

        elif action == "previous":
            speaker.previous()
            return json.dumps({"success": True, "message": "Previous track"})

        elif action == "volume":
            if not param:
                return json.dumps({"success": True, "volume": speaker.volume})
            vol = int(param)
            speaker.volume = vol
            return json.dumps({"success": True, "message": f"Volume set to {vol}"})

        elif action == "volume_up":
            amount = int(param) if param else 5
            speaker.volume = min(100, speaker.volume + amount)
            return json.dumps({"success": True, "message": f"Volume: {speaker.volume}"})

        elif action == "volume_down":
            amount = int(param) if param else 5
            speaker.volume = max(0, speaker.volume - amount)
            return json.dumps({"success": True, "message": f"Volume: {speaker.volume}"})

        elif action == "mute":
            speaker.mute = True
            return json.dumps({"success": True, "message": "Muted"})

        elif action == "unmute":
            speaker.mute = False
            return json.dumps({"success": True, "message": "Unmuted"})

        elif action == "favorites":
            favs = speaker.music_library.get_sonos_favorites()
            fav_list = [{"title": f.title} for f in favs]
            return json.dumps({"success": True, "favorites": fav_list}, ensure_ascii=False)

        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    except Exception as e:
        logger.error("Sonos tool error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


SONOS_SCHEMA = {
    "name": "sonos",
    "description": (
        "Control Sonos speakers on the local network. "
        "Actions: status, play (param=favorite name or URI, omit to resume), pause, stop, "
        "next, previous, volume (param=0-100), volume_up (param=amount, default 5), "
        "volume_down (param=amount, default 5), mute, unmute, favorites (list saved favorites)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "play", "pause", "stop", "next", "previous",
                    "volume", "volume_up", "volume_down", "mute", "unmute", "favorites",
                ],
                "description": "Action to perform",
            },
            "param": {
                "type": "string",
                "description": "Optional parameter — volume level (0-100), favorite name, or URI",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="sonos",
    toolset="sonos",
    schema=SONOS_SCHEMA,
    handler=lambda args, **kw: sonos_control(
        action=args["action"],
        param=args.get("param"),
    ),
    check_fn=_check_sonos_reqs,
    emoji="🔊",
)
