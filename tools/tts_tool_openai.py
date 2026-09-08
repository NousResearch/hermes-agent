"""OpenAI-compatible TTS backends for ``tools.tts_tool``: OpenAI and DeepInfra.

Also owns the managed-gateway (Nous portal ``openai-audio`` proxy) route selection that
decides where the OpenAI client points. Seams defined on the origin module (``_load_tts_config``,
``_import_openai_client``, ``_resolve_provider_key``, ``_generate_openai_tts``) are resolved
through :func:`_origin` at call time.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    NOUS_MANAGED_PROVIDER, managed_nous_tools_enabled, nous_tool_gateway_unavailable_message,
    read_selection, resolve_openai_audio_api_key, selection_error)
from tools.tts_tool_delivery import _origin, _section
from tools.tts_tool_providers import _tts_response_format_from_path

logger = logging.getLogger("tools.tts_tool")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
# The managed OpenAI audio gateway only proxies these; anything else is 400 "Unsupported".
MANAGED_OPENAI_TTS_MODELS = frozenset({"gpt-4o-mini-tts"})
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# DeepInfra base URL is resolved via hermes_cli.models.deepinfra_base_url (shared).
DEFAULT_DEEPINFRA_TTS_VOICE = "default"


def _managed_openai_audio_route() -> Optional[tuple]:
    gateway = resolve_managed_tool_gateway("openai-audio")
    if gateway is None:
        return None
    return gateway.nous_user_token, urljoin(f"{gateway.gateway_origin.rstrip('/')}/", "v1"), True


def _resolve_openai_audio_client_config() -> tuple[str, str, bool]:
    """``(api_key, base_url, is_managed)`` for the OpenAI audio client (``is_managed`` = the restricted
    Nous proxy, so callers coerce the request). Strict on the stored ``tts`` selection: ``"nous"``
    → managed ONLY (error if unavailable); any other → direct credentials ONLY (``tts.openai.api_key``
    then ``VOICE_TOOLS_OPENAI_KEY``/``OPENAI_API_KEY``); unset → config key → env key → managed."""
    origin = _origin()
    openai_cfg = _section(origin._load_tts_config(), "openai")
    selected = read_selection("tts")
    if selected == NOUS_MANAGED_PROVIDER:
        route = _managed_openai_audio_route()
        if route is None:
            raise ValueError(selection_error(
                "tts", NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or unreachable)"))
        return route
    direct_api_key = openai_cfg.get("api_key") or resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, openai_cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL, False
    if selected is not None:
        raise ValueError(selection_error(
            "tts", selected,
            "neither tts.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set",
        ))
    route = _managed_openai_audio_route()
    if route is None:
        message = "Neither tts.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += ". " + nous_tool_gateway_unavailable_message("managed OpenAI audio for TTS")
        raise ValueError(message)
    return route


def _has_openai_audio_backend() -> bool:
    """Return True when the selected OpenAI audio route is usable."""
    try:
        _resolve_openai_audio_client_config()
        return True
    except ValueError:
        return False


def _generate_openai_tts(
    text: str, output_path: str, tts_config: Dict[str, Any], *, api_key: Optional[str] = None,
    base_url: Optional[str] = None, model: Optional[str] = None, voice: Optional[str] = None,
    speed: Optional[float] = None, instructions: Optional[str] = None) -> str:
    """Generate audio via the OpenAI ``audio.speech.create`` SDK shape.

    Explicit kwargs let OpenAI-compatible backends (DeepInfra) supply credentials/model/voice
    and skip the managed-gateway resolution; otherwise the OpenAI auth chain and ``tts.openai``
    (speed falling back to ``tts.speed``) apply. ``instructions`` is forwarded only when truthy
    so ``tts-1`` and strict OpenAI-compatible servers that reject unknown kwargs are unaffected."""
    fallback_base: Optional[str] = None
    is_managed = False
    explicit_base_url = base_url is not None
    if api_key is None:
        api_key, fallback_base, is_managed = _resolve_openai_audio_client_config()
    oai_config = _section(tts_config, "openai")
    if model is None:
        model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    if voice is None:
        voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)
    config_base_url = oai_config.get("base_url")
    if base_url is None:  # config override beats the auth-chain fallback; explicit arg wins
        base_url = config_base_url or fallback_base or DEFAULT_OPENAI_BASE_URL
    if speed is None:
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        speed = float(oai_config.get("speed", speed_default))
    # The managed gateway only proxies MANAGED_OPENAI_TTS_MODELS; coerce a direct-OpenAI
    # model (e.g. "tts-1-hd") unless the user redirected base_url to their own endpoint.
    if is_managed and not explicit_base_url and not config_base_url and model not in MANAGED_OPENAI_TTS_MODELS:
        logger.warning(
            "TTS: managed OpenAI audio gateway does not support model %r; "
            "falling back to %s. Set VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY "
            "to use %r directly.",
            model, DEFAULT_OPENAI_MODEL, model)
        model = DEFAULT_OPENAI_MODEL
    create_kwargs: Dict[str, Any] = {
        "model": model, "voice": voice, "input": text,
        "response_format": _tts_response_format_from_path(output_path),
        "extra_headers": {"x-idempotency-key": str(uuid.uuid4())}}
    if speed != 1.0:
        create_kwargs["speed"] = max(0.25, min(4.0, speed))
    if instructions:
        create_kwargs["instructions"] = instructions
    if oai_config.get("language"):
        create_kwargs["extra_body"] = {"lang_code": oai_config["language"]}
    client = _origin()._import_openai_client()(api_key=api_key, base_url=base_url)
    try:
        client.audio.speech.create(**create_kwargs).stream_to_file(output_path)
        return output_path
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _generate_deepinfra_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Resolve DeepInfra credentials/model (live ``hermes_cli.models`` catalog, no hardcoded ids), then
    delegate to the OpenAI-compatible handler."""
    api_key = _origin()._resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY not set. Run `hermes setup` to configure, or set the env var directly.")
    di_config = _section(tts_config, "deepinfra")
    from hermes_cli.models import deepinfra_base_url, deepinfra_model_ids
    model = di_config.get("model")
    if not isinstance(model, str) or not model.strip():
        candidates = deepinfra_model_ids("tts")
        if not candidates:
            raise ValueError(
                "No DeepInfra TTS model available. Pin one in config.yaml "
                "under tts.deepinfra.model, or check connectivity to "
                "api.deepinfra.com so the live catalog can be fetched.")
        model = candidates[0]
    return _origin()._generate_openai_tts(
        text, output_path, tts_config, api_key=api_key, base_url=deepinfra_base_url(di_config),
        model=model, voice=di_config.get("voice", DEFAULT_DEEPINFRA_TTS_VOICE),
        speed=float(di_config.get("speed", tts_config.get("speed", 1.0))))
