"""Config, schemas, and session.update payload for xAI realtime voice.

Imported by the session (tools.voice_realtime) and by surfaces that only
need to mint a payload (tui_gateway token RPC). Keep this module cheap:
no websockets/sounddevice at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

REALTIME_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_REALTIME_MODEL = "grok-voice-latest"
DEFAULT_REALTIME_KEYTERMS: Tuple[str, ...] = (
    "Hermes",
    "repository",
    "repo",
    "branch",
    "worktree",
    "pull request",
    "subagent",
)
# xAI caps: at most 100 transcription keyterms, each at most 50 characters. An over-long term is
# rejected with an ``error`` event, which would spend the session's one-shot minimal-config retry.
MAX_KEYTERMS = 100
MAX_KEYTERM_CHARS = 50

# 16 kHz mono int16 — matches the classic recorder; a documented PCM rate.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000  # supervisor speech playback (xAI default)
FRAME_MS = 100  # xAI best practice: ~100 ms per append
FRAME_SAMPLES = INPUT_SAMPLE_RATE * FRAME_MS // 1000

# After the last delay fails, the session goes "dead" (CLI falls back).
RECONNECT_DELAYS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

_SILENT_RELAY_INSTRUCTIONS = (
    "You are a silent transcription relay. Never respond, never speak. "
    "Any response you produce is discarded unheard."
)

CONSULT_TOOL_NAME = "consult_hermes"
STEER_TOOL_NAME = "steer_hermes"
DEFAULT_SUPERVISOR_VOICE = "eve"

_CONSULT_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "name": CONSULT_TOOL_NAME,
    "description": (
        "Delegate a task or question to Hermes, the full agent on this "
        "machine (terminal, files, code, web, memory). Hermes' complete "
        "answer is the tool result to summarize aloud (and lives in "
        "session history). If Hermes is already working and the "
        "user corrects, redirects, or extends that task, call steer_hermes "
        "instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The user's request, restated completely with all "
                    "context needed to act on it."
                ),
            },
        },
        "required": ["task"],
    },
}

_STEER_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "name": STEER_TOOL_NAME,
    "description": (
        "Redirect, adjust, add to, or cancel the Hermes task that is "
        "currently running. Use whenever the user wants to influence the "
        "work in progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": (
                    "The user's steering instruction, restated completely "
                    "(e.g. 'also check the logs', 'skip the tests', "
                    "'stop working on that')."
                ),
            },
        },
        "required": ["instruction"],
    },
}

_SUPERVISOR_INSTRUCTIONS = """\
You are Hermes' voice — the spoken side of Hermes, a powerful agent running \
on this machine. Keep spoken replies short and natural — one or two sentences.

Rules:
- For ANY request needing facts, files, code, commands, the web, or actions, \
you MUST speak one short acknowledgment in your own words FIRST ("On it — \
give me a moment.", "Sure, let me check.") and then call consult_hermes with \
the full task. Never call the tool silently. Never attempt such work \
yourself and never invent technical results.
- You may answer directly only for greetings, small talk, and questions \
about what was already said in this conversation.
- When consult_hermes returns, summarize the outcome aloud in a sentence or \
two. The full answer is in the tool result and session history — do not \
read long output verbatim.
- While Hermes works the user may keep chatting; reply briefly. If asked \
about progress, say Hermes is still working. Do not call consult_hermes \
again for the same task.
- If the user wants to change, extend, or cancel the running task, call \
steer_hermes with their instruction — do not start a new consult for it.
- A consult_hermes busy response means the new request was NOT queued. Never \
claim Hermes is handling it. For a correction or redirect, call steer_hermes; \
for a separate request, wait for the active task to finish and then consult.
"""

# Spoken instantly (force_message — no model turn) when the model calls
# consult_hermes without its mandated filler. Rotated to avoid sounding
# canned; the model's own filler is still preferred when it complies.
_ACK_PHRASES: Tuple[str, ...] = (
    "On it — give me a moment.",
    "Sure, let me check that.",
    "Okay, working on it.",
    "Alright, one moment.",
    "Got it — digging in now.",
    "Let me have Hermes look at that.",
    "On it. This might take a bit.",
    "Sure thing — checking now.",
    "Okay, let me find out.",
    "Alright, Hermes is on it.",
)


class RealtimeVoiceError(RuntimeError):
    """Raised when the realtime voice session cannot be started."""


@dataclass
class RealtimeConfig:
    """Validated ``voice.realtime`` settings."""

    model: str = DEFAULT_REALTIME_MODEL
    # Default: the realtime model converses and consults Hermes (OpenClaw-style). "ears" is the
    # CLI/Discord opt-in where every utterance is a Hermes turn; browser surfaces are supervisor-only.
    brain: str = "supervisor"  # "supervisor" | "ears"
    voice: str = DEFAULT_SUPERVISOR_VOICE
    instructions_extra: str = ""  # appended to the supervisor prompt
    narrate_progress: bool = False  # speak short tool-progress lines while a consult runs
    # Supervisor duplex: False (default) mutes the mic while speech plays so
    # open speakers can't feed the assistant its own voice; True keeps the
    # mic hot for voice barge-in (wear headphones).
    full_duplex: bool = False
    # Loud-barge trigger while half-duplex speech plays: user RMS must exceed
    # the tracked speaker-bleed floor by this factor (voice.barge_in_threshold_multiplier).
    barge_multiplier: float = 3.0
    vad_threshold: Optional[float] = None       # None → server default (0.85)
    vad_silence_ms: Optional[int] = None        # None → server default
    vad_prefix_padding_ms: Optional[int] = None  # None → server default (333)
    language_hint: str = ""
    keyterms: List[str] = field(default_factory=list)
    # Auto-pause after this many silent seconds — a hot mic streams billable
    # audio. 0 disables.
    idle_pause_seconds: float = 120.0
    url: str = REALTIME_URL
    # Discord crowded-room gate. auto = only when 2+ humans are in the VC.
    require_wake_name: str = "auto"  # auto | true | false
    wake_names: List[str] = field(default_factory=list)
    # Opt-in: also post supervisor consult replies to the bound Discord
    # text channel. Default is voice-only (session history still keeps
    # the full answer).
    discord_text_mirror: bool = False

    @property
    def supervisor(self) -> bool:
        return self.brain == "supervisor"


def realtime_client_secrets_url(ws_url: str) -> str:
    """HTTP token-mint endpoint derived from the realtime WS URL.

    Keeps proxy/self-hosted deployments coherent: a configured
    ``voice.realtime.url`` of ``wss://proxy.example/v1/realtime`` mints
    ephemeral tokens against
    ``https://proxy.example/v1/realtime/client_secrets`` instead of a
    hardcoded api.x.ai endpoint (the xAI default still resolves there).
    """
    base = str(ws_url or "").strip() or REALTIME_URL
    base = base.split("?", 1)[0].rstrip("/")
    if base.startswith("wss://"):
        base = "https://" + base[len("wss://"):]
    elif base.startswith("ws://"):
        base = "http://" + base[len("ws://"):]
    return f"{base}/client_secrets"


def _coerce_optional_number(value: Any) -> Optional[float]:
    """YAML-shape-safe numeric coercion (bool is not a number here)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def load_realtime_config(voice_cfg: Any) -> RealtimeConfig:
    """Build a :class:`RealtimeConfig` from the ``voice`` config section."""
    section = {}
    if isinstance(voice_cfg, dict):
        raw = voice_cfg.get("realtime")
        if isinstance(raw, dict):
            section = raw

    cfg = RealtimeConfig()
    model = str(section.get("model") or "").strip()
    if model:
        cfg.model = model
    url = str(section.get("url") or "").strip()
    if url:
        cfg.url = url
    brain = str(section.get("brain") or "").strip().lower()
    if brain in ("ears", "supervisor"):
        cfg.brain = brain
    voice = str(section.get("voice") or "").strip()
    if voice:
        cfg.voice = voice
    extra = section.get("instructions")
    if isinstance(extra, str):
        cfg.instructions_extra = extra.strip()
    cfg.narrate_progress = bool(section.get("narrate_progress", False))
    cfg.full_duplex = bool(section.get("full_duplex", False))
    mult = _coerce_optional_number(
        voice_cfg.get("barge_in_threshold_multiplier") if isinstance(voice_cfg, dict) else None
    )
    if mult is not None and mult > 1.0:
        cfg.barge_multiplier = mult

    threshold = _coerce_optional_number(section.get("vad_threshold"))
    if threshold is not None and 0.0 < threshold <= 1.0:
        cfg.vad_threshold = threshold
    silence_ms = _coerce_optional_number(section.get("vad_silence_ms"))
    if silence_ms is not None and silence_ms >= 0:
        cfg.vad_silence_ms = int(silence_ms)
    prefix_ms = _coerce_optional_number(section.get("vad_prefix_padding_ms"))
    if prefix_ms is not None and prefix_ms >= 0:
        cfg.vad_prefix_padding_ms = int(prefix_ms)

    hint = section.get("language_hint")
    if isinstance(hint, str):
        cfg.language_hint = hint.strip()
    keyterms = section.get("keyterms")
    if isinstance(keyterms, (list, tuple)):
        cfg.keyterms = [str(t).strip()[:MAX_KEYTERM_CHARS] for t in keyterms if str(t).strip()][:MAX_KEYTERMS]

    idle = _coerce_optional_number(section.get("idle_pause_seconds"))
    if idle is not None and idle >= 0:
        cfg.idle_pause_seconds = idle

    raw_policy = section.get("require_wake_name", "auto")
    if isinstance(raw_policy, bool):
        cfg.require_wake_name = "true" if raw_policy else "false"
    else:
        policy = str(raw_policy or "auto").strip().lower()
        cfg.require_wake_name = {
            "always": "true",
            "never": "false",
            "on": "true",
            "off": "false",
        }.get(policy, policy if policy in ("auto", "true", "false") else "auto")
    raw_names = section.get("wake_names")
    if isinstance(raw_names, (list, tuple)):
        cfg.wake_names = [str(n).strip() for n in raw_names if str(n).strip()][:16]
    cfg.discord_text_mirror = bool(section.get("discord_text_mirror", False))
    return cfg


def realtime_voice_enabled(voice_cfg: Any) -> bool:
    """True when ``voice.realtime.enabled`` is truthy in config."""
    if not isinstance(voice_cfg, dict):
        return False
    section = voice_cfg.get("realtime")
    return isinstance(section, dict) and bool(section.get("enabled"))


def merge_realtime_keyterms(*groups: Any) -> List[str]:
    """Stable, case-insensitive keyterm merge within xAI's term-count and term-length caps."""
    merged: List[str] = []
    seen = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for raw in group:
            term = str(raw).strip()
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            merged.append(term[:MAX_KEYTERM_CHARS])
            if len(merged) >= MAX_KEYTERMS:
                return merged
    return merged


def check_realtime_requirements(*, require_local_audio: bool = True) -> Tuple[bool, str]:
    """Return (ok, detail): deps + credentials needed for the realtime backend.

    ``require_local_audio=False`` skips the sounddevice check — surfaces that
    inject their own audio transport (Discord voice channels) need numpy for
    resampling but no local mic/speaker device.
    """
    try:
        import websockets.sync.client  # noqa: F401  (lazy: keep module import cheap)
    except Exception:
        return False, "websockets package not available"
    try:
        if require_local_audio:
            import sounddevice  # noqa: F401  (lazy: optional voice extra)
        import numpy  # noqa: F401
    except Exception:
        return False, "sounddevice/numpy not installed (pip install 'hermes-agent[voice]')"
    try:
        from tools.xai_http import resolve_xai_http_credentials  # lazy: heavy

        creds = resolve_xai_http_credentials()
        if not str(creds.get("api_key") or "").strip():
            return False, "no xAI credentials (set XAI_API_KEY or `hermes auth add xai`)"
    except Exception as exc:
        return False, f"xAI credential resolution failed: {exc}"
    return True, ""


def build_session_update(cfg: RealtimeConfig, *, minimal: bool = False) -> Dict[str, Any]:
    """Build the ``session.update`` payload for this brain mode.
    ``minimal=True`` drops the OpenAI-compat extras (``create_response``,
    ``reasoning``) — the retry payload if the full config draws an error."""
    turn_detection: Dict[str, Any] = {"type": "server_vad"}
    if cfg.vad_threshold is not None:
        turn_detection["threshold"] = cfg.vad_threshold
    if cfg.vad_silence_ms is not None:
        turn_detection["silence_duration_ms"] = cfg.vad_silence_ms
    if cfg.vad_prefix_padding_ms is not None:
        turn_detection["prefix_padding_ms"] = cfg.vad_prefix_padding_ms

    transcription: Dict[str, Any] = {
        "model": "grok-transcribe",
        "keyterms": merge_realtime_keyterms(DEFAULT_REALTIME_KEYTERMS, cfg.keyterms),
    }
    if cfg.language_hint:
        transcription["language_hint"] = cfg.language_hint
    audio: Dict[str, Any] = {
        "input": {
            "format": {"type": "audio/pcm", "rate": INPUT_SAMPLE_RATE},
            "transcription": transcription,
        },
    }

    if cfg.supervisor:
        instructions = _SUPERVISOR_INSTRUCTIONS
        if cfg.instructions_extra:
            instructions += "\n" + cfg.instructions_extra
        audio["output"] = {
            "format": {"type": "audio/pcm", "rate": OUTPUT_SAMPLE_RATE},
        }
        session: Dict[str, Any] = {
            "voice": cfg.voice,
            "instructions": instructions,
            "turn_detection": turn_detection,
            "audio": audio,
            "tools": [dict(_CONSULT_TOOL_SCHEMA), dict(_STEER_TOOL_SCHEMA)],
        }
    else:
        if not minimal:
            # OpenAI-compat param xAI doesn't document: honored → no
            # auto-responses; ignored → the response.cancel path covers it.
            turn_detection["create_response"] = False
        session = {
            "instructions": _SILENT_RELAY_INSTRUCTIONS,
            "turn_detection": turn_detection,
            "audio": audio,
        }
    if not minimal:
        session["reasoning"] = {"effort": "none"}
    return {"type": "session.update", "session": session}
