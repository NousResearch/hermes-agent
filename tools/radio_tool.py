"""Radio tool -- Hermes Radio playback controls.

Registers the radio toolset with tools for playing, tuning, skipping,
and triggering mic breaks.  Gated on mpv being available.

Uses a dedicated persistent event loop in a daemon thread to avoid
creating/destroying event loops (which spews errors on Python 3.14).
"""

import asyncio
import json
import logging
import shutil
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# -- Persistent radio event loop -------------------------------------------
# Single long-lived loop in a daemon thread.  All radio async operations
# are scheduled onto this loop via run_coroutine_threadsafe().

_radio_loop: asyncio.AbstractEventLoop = None
_radio_thread: threading.Thread = None
_radio_lock = threading.Lock()


def _ensure_radio_loop() -> asyncio.AbstractEventLoop:
    """Start the radio event loop thread if it isn't running yet."""
    global _radio_loop, _radio_thread
    with _radio_lock:
        if _radio_loop is not None and _radio_loop.is_running():
            return _radio_loop
        _radio_loop = asyncio.new_event_loop()
        _radio_thread = threading.Thread(
            target=_radio_loop.run_forever,
            daemon=True,
            name="radio-loop",
        )
        _radio_thread.start()
        return _radio_loop


def _run_radio_async(coro):
    """Run an async coroutine on the persistent radio event loop."""
    loop = _ensure_radio_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


def _get_radio():
    from radio.player import HermesRadio
    return HermesRadio.get()


# -- Tool handlers ---------------------------------------------------------

def radio_play_tool(args: Dict[str, Any], **kwargs) -> str:
    """Start playing music from a source."""
    source = args.get("source", "crate").lower()
    query = args.get("query", "")

    radio = _get_radio()

    if source == "crate":
        # Parse crate options from query
        decades = None
        moods = None
        country = None
        if query:
            parts = query.split()
            for part in parts:
                if part.isdigit() and 1900 <= int(part) <= 2020:
                    decades = decades or []
                    decades.append(int(part))
                elif part.upper() in ("SLOW", "FAST", "WEIRD"):
                    moods = moods or []
                    moods.append(part.lower())
                elif len(part) == 3 and part.isalpha():
                    country = part.upper()

        result = _run_radio_async(radio.play_crate(
            decades=decades, moods=moods, country=country,
        ))
        return json.dumps({"success": True, "message": result})

    elif source == "stream":
        if not query:
            return json.dumps({"success": False, "error": "Provide a stream URL or station name"})

        if query.startswith("http://") or query.startswith("https://"):
            result = _run_radio_async(radio.play_stream(query))
            return json.dumps({"success": True, "message": result})

        # Search for the station
        result = _run_radio_async(_search_and_play(radio, query))
        return json.dumps(result)

    elif source == "somafm":
        result = _run_radio_async(_play_somafm(radio, query))
        return json.dumps(result)

    elif source == "local":
        if not query:
            return json.dumps({"success": False, "error": "Provide a file or directory path"})
        result = _run_radio_async(radio.play_local(query))
        return json.dumps({"success": True, "message": result})

    else:
        return json.dumps({"success": False, "error": f"Unknown source: {source}"})


async def _search_and_play(radio, query: str) -> dict:
    """Search Radio Browser and play the top result."""
    from radio.radio_browser import RadioBrowserClient
    client = RadioBrowserClient()
    try:
        stations = await client.search(name=query, limit=1)
        if not stations:
            stations = await client.search(tag=query, limit=1)
        if not stations:
            return {"success": False, "error": f"No stations found for: {query}"}
        station = stations[0]
        msg = await radio.play_stream(station.stream_url, station_name=station.name)
        return {"success": True, "message": msg, "station": station.display}
    finally:
        await client.close()


async def _play_somafm(radio, channel_id: str) -> dict:
    """Play a SomaFM channel."""
    from radio.somafm import get_channel, get_featured
    if not channel_id:
        channels = await get_featured()
        return {
            "success": True,
            "message": "SomaFM channels available:",
            "channels": [{"id": ch.id, "title": ch.title, "genre": ch.genre} for ch in channels],
        }
    ch = await get_channel(channel_id)
    if not ch:
        return {"success": False, "error": f"SomaFM channel not found: {channel_id}"}
    if not ch.stream_url:
        return {"success": False, "error": f"No stream URL for channel: {channel_id}"}
    msg = await radio.play_stream(ch.stream_url, station_name=f"SomaFM {ch.title}")
    return {"success": True, "message": msg}


def radio_pause_tool(args: Dict[str, Any], **kwargs) -> str:
    """Pause or resume radio playback."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"success": False, "error": "Radio is not playing"})
    result = _run_radio_async(_get_radio().toggle_pause())
    return json.dumps({"success": True, "message": result})


def radio_stop_tool(args: Dict[str, Any], **kwargs) -> str:
    """Stop radio playback."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"success": True, "message": "Radio is not playing"})
    radio = _get_radio()
    _run_radio_async(radio.stop())
    return json.dumps({"success": True, "message": "Radio stopped"})


def radio_skip_tool(args: Dict[str, Any], **kwargs) -> str:
    """Skip the current track."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"success": False, "error": "Radio is not playing"})
    result = _run_radio_async(_get_radio().skip())
    return json.dumps({"success": True, "message": result})


def radio_status_tool(args: Dict[str, Any], **kwargs) -> str:
    """Get current radio playback status."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"active": False})
    status = _run_radio_async(_get_radio().status())
    return json.dumps(status)


def radio_volume_tool(args: Dict[str, Any], **kwargs) -> str:
    """Set radio volume."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"success": False, "error": "Radio is not playing"})
    level = args.get("level", 80)
    result = _run_radio_async(_get_radio().set_volume(float(level)))
    return json.dumps({"success": True, "message": result})


def radio_mic_break_tool(args: Dict[str, Any], **kwargs) -> str:
    """Trigger a mic break."""
    from radio.player import HermesRadio
    if not HermesRadio.active():
        return json.dumps({"success": False, "error": "Radio is not playing"})
    text = args.get("text")
    result = _run_radio_async(_get_radio().mic_break(text=text))
    return json.dumps({"success": True, "message": result})


def radio_search_tool(args: Dict[str, Any], **kwargs) -> str:
    """Search for radio stations."""
    query = args.get("query", "")
    source = args.get("source", "radio_browser").lower()

    if not query:
        return json.dumps({"success": False, "error": "Provide a search query"})

    result = _run_radio_async(_do_search(query, source))
    return json.dumps(result)


async def _do_search(query: str, source: str) -> dict:
    if source in ("radio_browser", "rb"):
        from radio.radio_browser import RadioBrowserClient
        client = RadioBrowserClient()
        try:
            stations = await client.search(name=query, limit=10)
            if not stations:
                stations = await client.search(tag=query, limit=10)
            return {
                "success": True,
                "results": [
                    {"name": s.name, "country": s.country, "tags": s.tags,
                     "bitrate": s.bitrate, "url": s.stream_url}
                    for s in stations
                ],
            }
        finally:
            await client.close()

    elif source == "somafm":
        from radio.somafm import get_channels
        channels = await get_channels()
        q = query.lower()
        matches = [
            ch for ch in channels
            if q in ch.title.lower() or q in ch.genre.lower() or q in ch.description.lower()
        ]
        return {
            "success": True,
            "results": [
                {"id": ch.id, "title": ch.title, "genre": ch.genre, "description": ch.description}
                for ch in matches[:10]
            ],
        }

    elif source in ("radio_garden", "rg"):
        from radio.radio_garden import RadioGardenClient
        client = RadioGardenClient()
        try:
            stations = await client.explore(query, limit=10)
            return {
                "success": True,
                "results": [
                    {"name": s.title, "place": s.place, "country": s.country,
                     "url": s.stream_url}
                    for s in stations
                ],
            }
        finally:
            await client.close()

    else:
        return {"success": False, "error": f"Unknown search source: {source}"}


def check_radio_available() -> bool:
    """Check if mpv is installed."""
    return shutil.which("mpv") is not None


# -- Registry --------------------------------------------------------------

from tools.registry import registry

RADIO_SCHEMAS = [
    {
        "name": "radio_play",
        "description": "Start the Hermes Radio player. The radio plays music through the local mpv audio engine. Sources: 'crate' (Radiooooo global archive -- random tracks by decade/country/mood), 'stream' (live internet radio by station name or URL), 'somafm' (curated underground channels like defcon, dronezone, vaporwaves), 'local' (local audio files). For crate mode, query can include decades (e.g. '1970'), moods ('slow'/'fast'/'weird'), and country codes ('JPN'). Use this when the user asks to play music, listen to radio, or start crate digging.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["crate", "stream", "somafm", "local"],
                    "description": "Music source type",
                },
                "query": {
                    "type": "string",
                    "description": "Search query, stream URL, channel ID, file path, or crate params (e.g. '1970 JPN slow')",
                },
            },
            "required": [],
        },
    },
    {
        "name": "radio_pause",
        "description": "Pause or resume the Hermes Radio player. Toggles between paused and playing. Use this when the user says 'pause', 'unpause', 'resume', or 'toggle' the radio/music.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "radio_stop",
        "description": "Stop the Hermes Radio player completely and shut down mpv. Use when the user says 'stop the radio', 'turn off the music', or 'kill the radio'.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "radio_skip",
        "description": "Skip to the next track on the Hermes Radio player. Use when the user says 'skip', 'next', or 'next track'.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "radio_status",
        "description": "Get the current Hermes Radio playback status: what's playing, volume, position, source mode. Call this first if you need to know whether the radio is on or what's currently playing.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "radio_volume",
        "description": "Set the Hermes Radio volume (0-100). Use when the user says 'turn up/down', 'volume', 'louder', 'quieter'.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "number", "description": "Volume level (0-100)"},
            },
            "required": ["level"],
        },
    },
    {
        "name": "radio_mic_break",
        "description": "Trigger a DJ mic break. If text is provided, the DJ says that. Otherwise, auto-generates contextual commentary about the current/upcoming track.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Optional custom mic break text"},
            },
        },
    },
    {
        "name": "radio_search",
        "description": "Search for radio stations. Returns a list of matching stations with stream URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (station name, genre, tag)"},
                "source": {
                    "type": "string",
                    "enum": ["radio_browser", "somafm", "radio_garden"],
                    "description": "Directory to search. radio_garden searches by city name.",
                },
            },
            "required": ["query"],
        },
    },
]

TOOL_HANDLERS = {
    "radio_play": radio_play_tool,
    "radio_pause": radio_pause_tool,
    "radio_stop": radio_stop_tool,
    "radio_skip": radio_skip_tool,
    "radio_status": radio_status_tool,
    "radio_volume": radio_volume_tool,
    "radio_mic_break": radio_mic_break_tool,
    "radio_search": radio_search_tool,
}

for schema in RADIO_SCHEMAS:
    name = schema["name"]
    registry.register(
        name=name,
        toolset="radio",
        schema=schema,
        handler=lambda args, _n=name, **kw: TOOL_HANDLERS[_n](args, **kw),
        check_fn=check_radio_available,
    )
