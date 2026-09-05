"""Cloud TTS backends for ``tools.tts_tool``: Edge, ElevenLabs, xAI, MiniMax, Mistral, Gemini.

Each ``_generate_<provider>(text, output_path, tts_config) -> path`` writes one final-encoded
file. Shared here: bounded upstream response reading (16 MiB cap so a hostile endpoint can't
feed unbounded audio) and the auxiliary-model speech-tag rewrites. OpenAI/DeepInfra live in
``tts_tool_openai``. Origin seams (``get_env_value``, ``_resolve_provider_key``, ``_import_*``)
are resolved through :func:`_origin` at call time.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from tools.tts_tool_delivery import _origin, _section, _wrap_pcm_as_wav, _write_wav_bytes_as
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger("tools.tts_tool")

DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_STREAMING_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_MINIMAX_MODEL = "speech-02-hd"
DEFAULT_MINIMAX_VOICE_ID = "English_expressive_narrator"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1/t2a_v2"
DEFAULT_MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1/t2a_v2"
DEFAULT_MISTRAL_TTS_MODEL = "voxtral-mini-tts-2603"
DEFAULT_MISTRAL_TTS_VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral
DEFAULT_XAI_VOICE_ID = "eve"
DEFAULT_XAI_LANGUAGE = "en"
DEFAULT_XAI_SAMPLE_RATE = 24000
DEFAULT_XAI_BIT_RATE = 128000
DEFAULT_XAI_AUTO_SPEECH_TAGS = False
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
# xAI `speed` accepts 0.7..1.5 (1.0 = API default, omitted from the payload);
# `optimize_streaming_latency` is 0/1/2 (>0 trades quality for time-to-first-audio);
# `text_normalization` speaks numbers/abbreviations/symbols in written form.
DEFAULT_XAI_SPEED_MIN = 0.7
DEFAULT_XAI_SPEED_MAX = 1.5
DEFAULT_XAI_SPEED_DEFAULT = 1.0
DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT = 0
DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT = False
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_GEMINI_TTS_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_AUDIO_TAGS = False
GEMINI_AUDIO_TAG_REWRITE_TASK = "tts_audio_tags"
TTS_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024
TTS_RESPONSE_BODY_CHUNK_BYTES = 64 * 1024

_TRUE_WORDS = {"1", "true", "yes", "on", "enabled"}
_FALSE_WORDS = {"0", "false", "no", "off", "disabled"}


def _config_bool(value: Any, default: bool = False) -> bool:
    """Coerce common YAML/env bool spellings without treating random strings as true."""
    if isinstance(value, (bool, int, float)):
        return bool(value)
    normalized = value.strip().lower() if isinstance(value, str) else None
    return normalized in _TRUE_WORDS if normalized in _TRUE_WORDS | _FALSE_WORDS else default


def _tts_response_format_from_path(output_path: str) -> str:
    """Pick an OpenAI-style response format (opus/wav/flac/mp3) from the output extension."""
    formats = ((".ogg", "opus"), (".wav", "wav"), (".flac", "flac"))
    return next((fmt for ext, fmt in formats if output_path.endswith(ext)), "mp3")


def _require_key(env_var: str, provider_id: str, hint: str) -> str:
    """Resolve *env_var* via the origin key resolver; ValueError ``"<ENV> not set. <hint>"`` when absent."""
    api_key = _origin()._resolve_provider_key(env_var, provider_id) or ""
    if not api_key:
        raise ValueError(f"{env_var} not set. {hint}")
    return api_key


# --- Bounded upstream response reading ---
def _response_has_explicit_stream(response: Any) -> bool:
    """True for real ``requests`` responses (or doubles defining ``iter_content`` themselves)."""
    if not callable(getattr(response, "iter_content", None)):
        return False
    response_type = type(response)
    return response_type.__module__.startswith("requests.") or "iter_content" in vars(response_type)


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _read_tts_response_bytes(response: Any, *, label: str, limit: Optional[int] = None) -> bytes:
    """Read an upstream TTS response with a hard byte cap."""
    limit = TTS_RESPONSE_BODY_LIMIT_BYTES if limit is None else limit
    chunks: list[bytes] = []
    total = 0
    try:
        if _response_has_explicit_stream(response):
            iterator = response.iter_content(chunk_size=TTS_RESPONSE_BODY_CHUNK_BYTES)
        else:
            content = vars(response).get("content", getattr(type(response), "content", b""))
            iterator = (content,) if isinstance(content, (str, bytes, bytearray)) else ()
        for chunk in iterator:
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            chunk = bytes(chunk)
            total += len(chunk)
            if total > limit:
                _close_response(response)
                raise RuntimeError(f"{label} response exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        _close_response(response)


def _parse_json_body(response: Any, raw: bytes) -> Dict[str, Any]:
    """JSON from the already-read *raw* body. Unit-test doubles often only provide ``.json()``;
    real ``requests`` responses took the streaming path, so production never buffers eagerly."""
    if raw:
        return json.loads(raw.decode("utf-8"))
    if not _response_has_explicit_stream(response):
        json_reader = getattr(response, "json", None)
        if callable(json_reader):
            parsed = json_reader()
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_tts_response_json(response: Any, *, label: str, limit: Optional[int] = None) -> Dict[str, Any]:
    return _parse_json_body(response, _read_tts_response_bytes(response, label=label, limit=limit))


def _write_bytes(output_path: str, audio_bytes: bytes) -> str:
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], **extra: Any):
    """Streaming ``requests.post`` with the shared 60s timeout (body read via the bounded readers)."""
    import requests
    return requests.post(url, headers=headers, json=payload, timeout=60, stream=True, **extra)


# --- Auxiliary-model speech-tag rewrites ---
def _auxiliary_reply_text(response: Any) -> str:
    """The first choice's message content with any ```fence``` unwrapped ("" when unreadable)."""
    try:
        message = getattr(response.choices[0], "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    except Exception:
        return ""
    clean = str(content or "").strip()
    fence = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", clean, flags=re.DOTALL)
    return fence.group(1).strip() if fence else clean


_TAG_REWRITE_RULES = (
    "Rules:\n"
    "- Preserve the spoken words, order, and meaning.\n"
    "- Do not add new spoken sentences or remove existing spoken words.\n"
)
_TAG_REWRITE_TAIL = "- Do not explain or comment.\n- Return only the tagged TTS script."


def _rewrite_with_auxiliary_model(
    system_prompt: str, user_prompt: str, fallback: str, *, label: str, fallback_label: str, level: int,
) -> str:
    """Ask the auxiliary model (task ``tts_audio_tags``) to rewrite a script; *fallback* on any failure/empty reply."""
    try:
        from agent.auxiliary_client import call_llm
        response = call_llm(
            task=GEMINI_AUDIO_TAG_REWRITE_TASK, temperature=0.7,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}])
        return _auxiliary_reply_text(response) or fallback
    except Exception as exc:
        logger.log(level, "%s audio tag rewrite failed; using %s: %s", label, fallback_label, exc)
        return fallback


# --- Edge TTS (free default) ---
async def _generate_edge_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    edge_tts = _origin()._import_edge_tts()
    edge_config = tts_config.get("edge") or {}
    speed = float(edge_config.get("speed", tts_config.get("speed", 1.0)))
    kwargs = {"voice": edge_config.get("voice", DEFAULT_EDGE_VOICE)}
    if speed != 1.0:
        kwargs["rate"] = f"{round((speed - 1.0) * 100):+d}%"
    await edge_tts.Communicate(text, **kwargs).save(output_path)
    return output_path


# --- ElevenLabs ---
def _elevenlabs_environment_kwargs(el_config: Dict[str, Any]) -> Dict[str, Any]:
    """SDK client kwargs for ``tts.elevenlabs.base_url``/``wss_url``; empty (SDK default) without a
    base_url. ``wss_url`` defaults to the base_url host with a ``ws(s)://`` scheme."""
    base_url = (el_config.get("base_url") or "").rstrip("/")
    if not base_url:
        return {}
    from elevenlabs.environment import ElevenLabsEnvironment
    wss_url = (el_config.get("wss_url") or "").rstrip("/") or re.sub(r"^http", "ws", base_url)
    return {"environment": ElevenLabsEnvironment(base=base_url, wss=wss_url)}


def _generate_elevenlabs(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    api_key = _require_key("ELEVENLABS_API_KEY", "elevenlabs", "Get one at https://elevenlabs.io/")
    el_config = tts_config.get("elevenlabs") or {}
    client = _origin()._import_elevenlabs()(api_key=api_key, **_elevenlabs_environment_kwargs(el_config))
    audio_generator = client.text_to_speech.convert(
        text=text, voice_id=el_config.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID),
        model_id=el_config.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID),
        output_format="opus_48000_64" if output_path.endswith(".ogg") else "mp3_44100_128")
    with open(output_path, "wb") as f:
        f.writelines(audio_generator)
    return output_path


# --- xAI TTS (dedicated /v1/tts endpoint, not the OpenAI audio shape) ---
_XAI_INLINE_SPEECH_TAGS = (
    "pause", "long-pause", "hum-tune", "laugh", "chuckle", "giggle", "cry", "tsk",
    "tongue-click", "lip-smack", "breath", "inhale", "exhale", "sigh")
_XAI_WRAPPING_SPEECH_TAGS = (
    "soft", "whisper", "loud", "build-intensity", "decrease-intensity", "higher-pitch",
    "lower-pitch", "slow", "fast", "sing-song", "singing", "laugh-speak", "emphasis")
_XAI_SPEECH_TAG_RE = re.compile(
    rf"(\[(?:{'|'.join(_XAI_INLINE_SPEECH_TAGS)})\]|</?(?:{'|'.join(_XAI_WRAPPING_SPEECH_TAGS)})>)",
    flags=re.IGNORECASE)
_XAI_FIRST_SENTENCE_RE = re.compile(r"^(.{12,120}?[.!?…])\s+(?=\S)", flags=re.DOTALL)


def _apply_xai_auto_speech_tags(text: str) -> str:
    """Add xAI speech tags: a conservative local pass ([pause] between paragraphs / after the first
    sentence), then — only when the text carried no explicit tags — an auxiliary-model rewrite
    with the richer xAI tag set, falling back to the locally tagged text on any failure."""
    clean = text.strip()
    if not clean:
        return text
    local = re.sub(r"\s*\n\s*", " ", re.sub(r"\n\s*\n+", " [pause] ", clean))
    if not _XAI_SPEECH_TAG_RE.search(local):
        local = _XAI_FIRST_SENTENCE_RE.sub(r"\1 [pause] ", local, count=1)
    local = re.sub(r"\s{2,}", " ", local).strip()
    if _XAI_SPEECH_TAG_RE.search(clean):  # explicit user/model tags are trusted as-is
        return local
    system_prompt = (
        "You rewrite transcripts for the xAI /v1/tts endpoint by inserting "
        "expressive speech tags.\n\n"
        "Valid inline tags (use as `[tag]`): " + ", ".join(_XAI_INLINE_SPEECH_TAGS) + ".\n"
        "Valid wrapping tags (use as `[tag]...[/tag]`): " + ", ".join(_XAI_WRAPPING_SPEECH_TAGS) + ".\n\n"
        + _TAG_REWRITE_RULES +
        "- Use inline `[tag]` for short modifiers (laughs, sighs, pause, etc.).\n"
        "- Use wrapping `[tag]...[/tag]` for sustained effects (whisper, soft, slow, fast, loud, etc.).\n"
        "- Do not use angle-bracket tags like `<tag>...</tag>` — xAI uses BBCode-style closing tags with `[/tag]`.\n"
        "- Do not use SSML.\n"
        + _TAG_REWRITE_TAIL)
    return _rewrite_with_auxiliary_model(
        system_prompt, f"TRANSCRIPT TO TAG:\n{local}", local, label="xAI TTS", fallback_label="locally-tagged text", level=logging.DEBUG,
    )


def _clamped_number(raw: Any, cast, lo, hi):
    """Parse an optional numeric knob and clamp into [lo, hi]; ``None``/unparseable -> None. An empty
    string is deliberately clamped unconverted (its TypeError surfaces as a generic TTS failure)."""
    if raw is None:
        return None
    if raw != "":
        try:
            raw = cast(raw)
        except (TypeError, ValueError):
            return None
    return max(lo, min(hi, raw))


def _generate_xai_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    from tools.xai_http import resolve_xai_http_credentials

    # TTS is API-billed: a subscription OAuth bearer can authorize chat while
    # returning 403 for /v1/tts, so prefer an explicit XAI_API_KEY over OAuth.
    # See #87045, #88040.
    creds = resolve_xai_http_credentials(prefer_api_key=True)
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.")
    xai_config = tts_config.get("xai") or {}
    voice_id = str(xai_config.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
    language = str(xai_config.get("language", DEFAULT_XAI_LANGUAGE)).strip() or DEFAULT_XAI_LANGUAGE
    sample_rate, bit_rate = (int(xai_config.get("sample_rate", DEFAULT_XAI_SAMPLE_RATE)),
                             int(xai_config.get("bit_rate", DEFAULT_XAI_BIT_RATE)))
    auto_speech_tags = xai_config.get("auto_speech_tags", xai_config.get("speech_tags"))
    if _config_bool(auto_speech_tags, DEFAULT_XAI_AUTO_SPEECH_TAGS):
        text = _apply_xai_auto_speech_tags(text)
    # ``tts.xai.<knob>`` overrides global ``tts.<knob>``; out-of-range values are clamped into the
    # API's band rather than 400ing the request.
    speed = _clamped_number(xai_config.get("speed", tts_config.get("speed")), float,
                            DEFAULT_XAI_SPEED_MIN, DEFAULT_XAI_SPEED_MAX)
    optimize_streaming_latency = _clamped_number(
        xai_config.get("optimize_streaming_latency", tts_config.get("optimize_streaming_latency")),
        int, 0, 2)
    text_normalization = _config_bool(
        xai_config.get("text_normalization"), DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT)
    if creds.get("provider") == "xai-oauth":
        base_url = creds.get("base_url")
    else:
        base_url = xai_config.get("base_url") or creds.get("base_url") or _origin().get_env_value("XAI_BASE_URL")
    base_url = str(base_url or DEFAULT_XAI_BASE_URL).strip().rstrip("/")

    # Documented minimal POST /v1/tts shape; optional fields only when they differ from defaults.
    codec = "wav" if output_path.endswith(".wav") else "mp3"
    payload: Dict[str, Any] = {"text": text, "voice_id": voice_id, "language": language}
    if codec != "mp3" or sample_rate != DEFAULT_XAI_SAMPLE_RATE or bit_rate != DEFAULT_XAI_BIT_RATE:
        output_format: Dict[str, Any] = {"codec": codec}
        if sample_rate:
            output_format["sample_rate"] = sample_rate
        if codec == "mp3" and bit_rate:
            output_format["bit_rate"] = bit_rate
        payload["output_format"] = output_format
    if speed is not None and speed != DEFAULT_XAI_SPEED_DEFAULT:
        payload["speed"] = speed
    if optimize_streaming_latency not in (None, DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT):
        payload["optimize_streaming_latency"] = optimize_streaming_latency
    if text_normalization:
        payload["text_normalization"] = True
    response = _post_json(f"{base_url}/tts", payload, {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent()})
    response.raise_for_status()
    return _write_bytes(output_path, _read_tts_response_bytes(response, label="xAI TTS"))


# --- MiniMax TTS ---
@dataclass(frozen=True)
class _MiniMaxTTSRuntime:
    """A region-bound MiniMax endpoint and credential (key excluded from ``repr``)."""

    region: str
    endpoint: str
    credential_source: str
    api_key: str = field(repr=False)


_MINIMAX_ENDPOINTS = {"global": DEFAULT_MINIMAX_BASE_URL, "cn": DEFAULT_MINIMAX_CN_BASE_URL}
_MINIMAX_OFFICIAL_HOSTS = {
    "global": frozenset({"api.minimax.io", "api.minimax.chat"}),
    "cn": frozenset({"api.minimaxi.com"})}


def _resolve_minimax_tts_runtime(tts_config: Dict[str, Any]) -> _MiniMaxTTSRuntime:
    """Select MiniMax region, endpoint and credential atomically: explicit ``tts.minimax.region`` wins,
    else the legacy global credential; ``cn`` only when it is the sole configured credential."""
    mm_config = _section(tts_config, "minimax")
    resolve_key = _origin()._resolve_provider_key
    credentials = {
        region: (env_var, str(resolve_key(env_var, "minimax") or "").strip())
        for region, env_var in (("global", "MINIMAX_API_KEY"), ("cn", "MINIMAX_CN_API_KEY"))}
    region = str(mm_config.get("region") or "").strip().lower()
    if region and region not in _MINIMAX_ENDPOINTS:
        raise ValueError("tts.minimax.region must be 'global' or 'cn'")
    if not region:
        region = "cn" if credentials["cn"][1] and not credentials["global"][1] else "global"
    credential_source, api_key = credentials[region]
    if not api_key:
        raise ValueError(f"{credential_source} not set for MiniMax TTS region {region!r}")
    endpoint = str(mm_config.get("base_url") or _MINIMAX_ENDPOINTS[region]).strip()
    other_region = "cn" if region == "global" else "global"
    if (urlparse(endpoint).hostname or "").lower() in _MINIMAX_OFFICIAL_HOSTS[other_region]:
        raise ValueError(
            f"tts.minimax.base_url points to the {other_region!r} MiniMax endpoint but region is {region!r}")
    return _MiniMaxTTSRuntime(region=region, endpoint=endpoint, credential_source=credential_source, api_key=api_key)


def _raise_minimax_api_error(result: Dict[str, Any]) -> None:
    base_resp = result.get("base_resp", {})
    status_code = base_resp.get("status_code", -1)
    if status_code != 0:
        raise RuntimeError(
            f"MiniMax TTS API error (code {status_code}): {base_resp.get('status_msg', 'unknown error')}")


def _generate_minimax_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio via MiniMax: ``t2a_v2`` (nested payload, JSON reply with hex audio) or the legacy
    ``text_to_speech`` endpoint (flat payload, raw ``audio/*`` body), detected from the URL."""
    runtime = _resolve_minimax_tts_runtime(tts_config)
    mm_config = _section(tts_config, "minimax")
    model = mm_config.get("model", DEFAULT_MINIMAX_MODEL)
    voice_id = mm_config.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)
    base_url = runtime.endpoint
    # MiniMax scopes TTS requests by GroupId (``?GroupId=<id>`` on the t2a_v2 URL): config or
    # MINIMAX_GROUP_ID, attached only when absent from the URL.
    group_id = (str(mm_config.get("group_id") or "").strip()
                or (_origin().get_env_value("MINIMAX_GROUP_ID") or "").strip())
    if group_id and "GroupId=" not in base_url:
        base_url = f"{base_url}{'&' if '?' in base_url else '?'}GroupId={group_id}"
    is_t2a_v2 = "t2a_v2" in base_url
    if is_t2a_v2:
        payload = {
            "model": model, "text": text,
            "voice_setting": {
                "voice_id": voice_id, "speed": mm_config.get("speed", 1.0), "vol": mm_config.get("vol", 1.0),
                "pitch": mm_config.get("pitch", 0), "emotion": mm_config.get("emotion", "neutral"),
            },
            "audio_setting": {
                "sample_rate": mm_config.get("sample_rate", 32000), "bitrate": mm_config.get("bitrate", 128000),
                "format": "mp3", "channel": 1,
            },
        }
    else:
        payload = {"model": model, "text": text, "voice_id": voice_id}
    response = _post_json(base_url, payload, {
        "Content-Type": "application/json", "Authorization": f"Bearer {runtime.api_key}"})
    if is_t2a_v2:
        response.raise_for_status()
        result = _read_tts_response_json(response, label="MiniMax TTS")
        _raise_minimax_api_error(result)
        hex_audio = result.get("data", {}).get("audio", "")
        if not hex_audio:
            raise RuntimeError("MiniMax TTS returned empty audio data")
        return _write_bytes(output_path, bytes.fromhex(hex_audio))
    content_type = response.headers.get("Content-Type", "")
    if "audio/" in content_type:
        return _write_bytes(output_path, _read_tts_response_bytes(response, label="MiniMax TTS"))
    # Non-audio reply: surface the API error if the body is JSON.
    raw_body = b""
    try:
        raw_body = _read_tts_response_bytes(response, label="MiniMax TTS")
        _raise_minimax_api_error(json.loads(raw_body.decode("utf-8")) if raw_body else {})
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        response.raise_for_status()
        raise RuntimeError(
            f"MiniMax TTS returned unexpected Content-Type '{content_type}' ({len(raw_body)} bytes)")
    raise RuntimeError("MiniMax TTS returned no audio data")


# --- Mistral (Voxtral TTS) — base64 audio, native Opus for voice bubbles ---
def _generate_mistral_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    api_key = _require_key("MISTRAL_API_KEY", "mistral", "Get one at https://console.mistral.ai/")
    mi_config = tts_config.get("mistral") or {}
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if mi_config.get("base_url"):
        client_kwargs["server_url"] = mi_config["base_url"]  # the Mistral SDK calls it server_url
    Mistral = _origin()._import_mistral_client()  # ImportError must escape the RuntimeError wrap
    try:
        with Mistral(**client_kwargs) as client:
            response = client.audio.speech.complete(
                model=mi_config.get("model", DEFAULT_MISTRAL_TTS_MODEL), input=text,
                voice_id=mi_config.get("voice_id") or DEFAULT_MISTRAL_TTS_VOICE_ID,
                response_format=_tts_response_format_from_path(output_path))
            audio_bytes = base64.b64decode(response.audio_data)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Mistral TTS failed: %s", e, exc_info=True)
        raise RuntimeError(f"Mistral TTS failed: {type(e).__name__}") from e
    return _write_bytes(output_path, audio_bytes)


# --- Google Gemini TTS ---
def _read_gemini_persona_prompt(gemini_config: Dict[str, Any]) -> str:
    """Read ``tts.gemini.persona_prompt_file`` (relative -> under HERMES_HOME), failing soft."""
    raw = gemini_config.get("persona_prompt_file")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    path = Path(os.path.expandvars(raw.strip())).expanduser()
    if not path.is_absolute():
        try:
            from hermes_constants import get_hermes_home
            path = get_hermes_home() / path
        except Exception:
            path = Path.cwd() / path
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Gemini TTS persona prompt file unavailable at %s: %s", path, exc)
        return ""


def _gemini_audio_tags_enabled(gemini_config: Dict[str, Any], model: str) -> bool:
    """Audio tags are opt-in and only Gemini 3.1 TTS models are known to honor them."""
    raw = gemini_config.get("audio_tags")
    if isinstance(raw, dict):
        raw = raw.get("enabled")
    if not _config_bool(raw, default=DEFAULT_GEMINI_AUDIO_TAGS):
        return False
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    if "gemini-3.1" in normalized and "tts" in normalized:
        return True
    logger.warning("Gemini TTS audio_tags enabled, but model %s is not known to support "
                   "Gemini audio tags; skipping hidden tag rewrite", model)
    return False


def _rewrite_gemini_tts_audio_tags(text: str, persona_prompt: str = "") -> str:
    """Use the configured auxiliary model to insert Gemini audio tags (falls back to *text*)."""
    transcript = text.strip()
    if not transcript:
        return text
    system_prompt = (
        "You rewrite transcripts for Gemini 3.1 Flash TTS by inserting expressive "
        "audio tags.\n\n"
        "Audio tags are inline square-bracket modifiers such as [whispers], "
        "[excitedly], [very slow], [sarcastically], [laughs], [sighs], or [gasp]. "
        "There is no fixed allowlist. Use creative freeform tags generously but "
        "naturally to control tone, pace, emotional vibe, emphasis, section-level "
        "delivery, and non-verbal sounds. Use English audio tags even when the "
        "spoken transcript is not English.\n\n"
        + _TAG_REWRITE_RULES +
        "- Use square brackets for every audio tag.\n"
        "- Do not use SSML or XML tags.\n"
        + _TAG_REWRITE_TAIL)
    user_prompt = (f"PERSONA AND DIRECTOR CONTEXT:\n{persona_prompt.strip() or '(none)'}\n\n"
                   f"TRANSCRIPT TO TAG:\n{transcript}")
    return _rewrite_with_auxiliary_model(system_prompt, user_prompt, text, label="Gemini TTS",
                                         fallback_label="untagged text", level=logging.WARNING)


def _compose_gemini_tts_prompt(text: str, gemini_config: Dict[str, Any], persona_prompt: Optional[str] = None) -> str:
    """Gemini prompt = persona direction + transcript; a ``{transcript}`` / ``{{transcript}}``
    placeholder is substituted in place, otherwise the transcript is appended under a heading."""
    transcript = text.strip()
    if persona_prompt is None:
        persona_prompt = _read_gemini_persona_prompt(gemini_config)
    if not persona_prompt:
        return transcript
    preamble = (
        "Synthesize speech from the TRANSCRIPT only. Treat AUDIO PROFILE, "
        "SCENE, DIRECTOR'S NOTES, and SAMPLE CONTEXT as performance direction; "
        "do not speak those sections aloud.")
    for pattern in (r"\{\{\s*transcript\s*\}\}", r"\{\s*transcript\s*\}"):
        compiled = re.compile(pattern, flags=re.IGNORECASE)
        if compiled.search(persona_prompt):
            return f"{preamble}\n\n{compiled.sub(transcript, persona_prompt)}".strip()
    return f"{preamble}\n\n{persona_prompt}\n\n#### TRANSCRIPT\n{transcript}".strip()


def _gemini_error_detail(response: Any) -> str:
    """Best-effort ``error.message`` from a non-200 Gemini reply, else the first 300 body chars."""
    raw_body = _read_tts_response_bytes(response, label="Gemini TTS")
    try:
        message = _parse_json_body(response, raw_body).get("error", {}).get("message")
    except Exception:
        message = None
    return message or raw_body.decode("utf-8", errors="replace")[:300]


def _generate_gemini_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Generate audio via Gemini ``generateContent`` (``responseModalities=["AUDIO"]``). The reply is
    base64 24kHz mono 16-bit PCM, wrapped as WAV and ffmpeg-converted to the requested container."""
    origin = _origin()
    api_key = origin._resolve_provider_key("GEMINI_API_KEY", "gemini") or origin._resolve_provider_key(
        "GOOGLE_API_KEY", "gemini")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey")
    gemini_config = _section(tts_config, "gemini")
    model = str(gemini_config.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
    voice = str(gemini_config.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
    base_url = str(gemini_config.get("base_url") or origin.get_env_value("GEMINI_BASE_URL")
                   or DEFAULT_GEMINI_TTS_BASE_URL).strip().rstrip("/")
    persona_prompt = _read_gemini_persona_prompt(gemini_config)
    tts_script = text
    if _gemini_audio_tags_enabled(gemini_config, model):
        tts_script = _rewrite_gemini_tts_audio_tags(text, persona_prompt=persona_prompt)
    prompt_text = _compose_gemini_tts_prompt(
        tts_script, gemini_config, persona_prompt=persona_prompt)
    max_len = origin._resolve_max_text_length("gemini", tts_config)
    if len(prompt_text) > max_len:
        raise ValueError(
            "Gemini TTS composed prompt exceeds the provider request limit "
            f"({len(prompt_text)} > {max_len} chars). Reduce the persona/audio-tag "
            "prompt or lower tts.gemini.max_text_length so long-form text is "
            "split with enough prompt headroom.")
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    headers = {"Content-Type": "application/json"}
    if urlparse(base_url).hostname == "generativelanguage.googleapis.com":
        try:
            import hermes_cli
            version = str(hermes_cli.__version__)
        except Exception:
            version = "0.0.0"
        headers["X-Goog-Api-Client"] = f"hermes-agent/{version}"  # partner-integration guidance
    response = _post_json(f"{base_url}/models/{model}:generateContent", payload, headers, params={"key": api_key})
    if response.status_code != 200:
        raise RuntimeError(f"Gemini TTS API error (HTTP {response.status_code}): {_gemini_error_detail(response)}")
    try:
        data = _read_tts_response_json(response, label="Gemini TTS")
        parts = data["candidates"][0]["content"]["parts"]
        audio_part = next((p for p in parts if "inlineData" in p or "inline_data" in p), None)
        if audio_part is None:
            raise RuntimeError("Gemini TTS response contained no audio data")
        audio_b64 = (audio_part.get("inlineData") or audio_part.get("inline_data") or {}).get("data", "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Gemini TTS response was malformed: {e}") from e
    if not audio_b64:
        raise RuntimeError("Gemini TTS returned empty audio data")
    return _write_wav_bytes_as(_wrap_pcm_as_wav(base64.b64decode(audio_b64)), output_path)
