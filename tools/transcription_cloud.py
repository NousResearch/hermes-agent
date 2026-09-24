"""Cloud STT providers.

OpenAI-SDK-shaped backends (groq, openai, deepinfra), Mistral Voxtral, REST multipart
backends (xAI, ElevenLabs), and OpenAI audio credential resolution (config > keyless
local server > env > managed Nous gateway). Facade-owned state and helpers
(``_HAS_OPENAI``, ``_resolve_provider_key``, ``_resolve_stt_language``, ``_load_stt_config``,
``get_env_value``) are read lazily from ``tools.transcription_tools``.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

from utils import is_truthy_value
from tools.transcription_audio import _transcode_audio_for_stt
from tools.transcription_common import (
    DEFAULT_GROQ_STT_MODEL, DEFAULT_STT_MODEL, ELEVENLABS_STT_BASE_URL, GROQ_BASE_URL, GROQ_MODELS,
    OPENAI_BASE_URL, OPENAI_MODELS, XAI_STT_BASE_URL, _error_result, _get_stt_section,
    _lazy_ensure_quietly, _log_prompt_unsupported, _ok_result)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")
# Voxtral-style ``language xx <asr_text> ...`` prefix some SDK responses carry.
_ASR_TEXT_RE = re.compile(
    r"\s*language\s+[\w.-]+(?:\s*<audio_language>[^<]*</audio_language>)?\s*<asr_text>\s*(?P<text>.*)",
    flags=re.IGNORECASE | re.DOTALL)


def _has_xai_stt_credentials() -> bool:
    from tools.xai_http import resolve_xai_http_credentials
    return bool(resolve_xai_http_credentials().get("api_key"))


def _with_openai_client(api_key: str, base_url: Optional[str], file_path: str, log_label: str, body):
    """Run ``body(client)`` on a fresh OpenAI SDK client (30s timeout, no retries); always closed.
    Errors map to the shared envelope. APIConnectionError is checked before APITimeoutError (its
    subclass) so timeouts report as connection errors, as they always have."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)
        try:
            return body(client)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        try:
            from openai import APIError, APIConnectionError, APITimeoutError
        except ImportError:  # pragma: no cover — callers gate on _HAS_OPENAI
            APIError = APIConnectionError = APITimeoutError = ()
        if isinstance(exc, PermissionError):
            return _error_result(f"Permission denied: {file_path}")
        for cls, label in ((APIConnectionError, "Connection error"), (APITimeoutError, "Request timeout"),
                           (APIError, "API error")):
            if isinstance(exc, cls):
                return _error_result(f"{label}: {exc}")
        logger.error("%s transcription failed: %s", log_label, exc, exc_info=True)
        return _error_result(f"Transcription failed: {exc}")


def _cloud_failure(exc: BaseException, file_path: str, label: str, detail: Optional[str] = None) -> Dict[str, Any]:
    """Map a REST/SDK provider exception to the shared envelope (``label`` e.g. ``"xAI STT transcription"``)."""
    if isinstance(exc, PermissionError):
        return _error_result(f"Permission denied: {file_path}")
    logger.error("%s failed: %s", label, exc, exc_info=True)
    return _error_result(f"{label} failed: {exc if detail is None else detail}")


def _sdk_prompt_kwargs(language: Optional[str], prompt: Optional[str]) -> Dict[str, Any]:
    """``language``/``prompt`` create-kwargs, each only when set so the bare request stays byte-identical."""
    return {key: value for key, value in (("language", language), ("prompt", prompt)) if value}


def _transcribe_groq(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe via the Groq Whisper API; language: hook > ``stt.groq.language`` > ``stt.language`` > env > auto."""
    from tools.transcription_tools import _HAS_OPENAI, _resolve_provider_key, _resolve_stt_language
    api_key = _resolve_provider_key("GROQ_API_KEY", "groq")
    if not api_key:
        return _error_result("GROQ_API_KEY not set")
    if not _HAS_OPENAI:
        return _error_result("openai package not installed")
    if model_name in OPENAI_MODELS:  # auto-correct an OpenAI-only model
        logger.info("Model %s not available on Groq, using %s", model_name, DEFAULT_GROQ_STT_MODEL)
        model_name = DEFAULT_GROQ_STT_MODEL
    language = language or _resolve_stt_language("groq")

    def _run(client):
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(file=audio_file, model=model_name,
                                                               response_format="text",
                                                               **_sdk_prompt_kwargs(language, prompt))
        transcript_text = str(transcription).strip()
        logger.info("Transcribed %s via Groq API (%s, lang=%s, %d chars)",
                     Path(file_path).name, model_name, language or "auto", len(transcript_text))
        return _ok_result(transcript_text, "groq")
    return _with_openai_client(api_key, GROQ_BASE_URL, file_path, "Groq", _run)


def _transcribe_openai(
    file_path: str, model_name: str, *, api_key: Optional[str] = None,
    base_url: Optional[str] = None, provider_label: str = "openai", language: Optional[str] = None,
    prompt: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe via the OpenAI ``audio.transcriptions.create`` SDK shape, shared by every
    OpenAI-compatible endpoint (DeepInfra etc.): explicit ``api_key``/``base_url`` skip the
    OpenAI-only auth chain; ``provider_label`` names the response's provider."""
    from tools.transcription_tools import _HAS_OPENAI, _resolve_stt_language
    if api_key is None:
        try:
            api_key, fallback_base = _resolve_openai_audio_client_config()
        except ValueError as exc:
            return _error_result(str(exc))
        base_url = base_url or fallback_base
    # Language: hook override > stt.<provider>.language > stt.language > env > auto.
    language = language or _resolve_stt_language(provider_label)
    if not _HAS_OPENAI:
        return _error_result("openai package not installed")
    # Auto-correct a Groq-only model on the native OpenAI path only (third-party endpoints may serve it).
    if provider_label == "openai" and model_name in GROQ_MODELS:
        logger.info("Model %s not available on OpenAI, using %s", model_name, DEFAULT_STT_MODEL)
        model_name = DEFAULT_STT_MODEL

    def _run(client):
        from openai import BadRequestError

        def _create_transcription(path: str):
            create_kwargs: Dict[str, Any] = {
                "model": model_name, "response_format": "text" if model_name == "whisper-1" else "json",
            }
            if language:
                # gpt-transcribe takes a ``languages`` list and rejects the legacy field.
                if model_name == "gpt-transcribe":
                    create_kwargs["extra_body"] = {"languages": [language]}
                else:
                    create_kwargs["language"] = language
                logger.debug("Using language hint '%s' for OpenAI STT", language)
            if prompt:  # only when set so the bare request stays byte-identical
                create_kwargs["prompt"] = prompt
            with open(path, "rb") as audio_file:
                return client.audio.transcriptions.create(file=audio_file, **create_kwargs)
        with tempfile.TemporaryDirectory(prefix="hermes-stt-") as work_dir:
            try:
                transcription = _create_transcription(file_path)
            except BadRequestError as exc:
                if not any(k in str(exc).lower() for k in ("unsupported", "corrupted", "invalid file")):
                    raise
                # Newer models reject containers whisper-1 accepted (Ogg/Opus voice notes): transcode, retry once.
                converted_path, transcode_error = _transcode_audio_for_stt(file_path, work_dir)
                if transcode_error:
                    return _error_result(transcode_error)
                logger.info("Retrying %s STT after transcoding %s to m4a (API rejected the original container)",
                            provider_label, Path(file_path).name)
                transcription = _create_transcription(converted_path)
        transcript_text = _extract_transcript_text(transcription)
        logger.info("Transcribed %s via %s (%s, %d chars)",
                    Path(file_path).name, provider_label, model_name, len(transcript_text))
        return _ok_result(transcript_text, provider_label)
    return _with_openai_client(api_key, base_url, file_path, provider_label, _run)


def _transcribe_mistral(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe with the ``mistralai`` SDK (``/v1/audio/transcriptions``); requires ``MISTRAL_API_KEY``."""
    from tools.transcription_tools import _resolve_provider_key, _resolve_stt_language
    api_key = _resolve_provider_key("MISTRAL_API_KEY", "mistral")
    if not api_key:
        return _error_result("MISTRAL_API_KEY not set")
    try:
        _lazy_ensure_quietly("stt.mistral")
        from mistralai.client import Mistral
        with Mistral(api_key=api_key) as client, open(file_path, "rb") as audio_file:
            # Language: hook override > stt.mistral.language > stt.language > env > auto.
            language = language or _resolve_stt_language("mistral")
            result = client.audio.transcriptions.complete(
                model=model_name, file={"content": audio_file, "file_name": Path(file_path).name},
                **_sdk_prompt_kwargs(language, prompt))
        transcript_text = _extract_transcript_text(result)
        logger.info("Transcribed %s via Mistral API (%s, %d chars)",
                    Path(file_path).name, model_name, len(transcript_text))
        return _ok_result(transcript_text, "mistral")
    except Exception as e:
        return _cloud_failure(e, file_path, "Mistral transcription", type(e).__name__)


# ---- REST multipart backends (xAI, ElevenLabs) ----------------------------
def _post_audio_multipart(url: str, headers: Dict[str, str], file_path: str, data: Dict[str, str]):
    import requests
    with open(file_path, "rb") as audio_file:
        return requests.post(url, headers=headers, files={"file": (Path(file_path).name, audio_file)},
                             data=data, timeout=120)


def _rest_provider(
    file_path: str, provider: str, label: str, post: Callable[[], Any], extract_detail,
    extract_text, log: Callable[[str, Dict[str, Any]], None]) -> Dict[str, Any]:
    """Shared multipart REST flow: ``post()`` -> ``log(text, body)`` -> ok envelope. Non-200 ->
    ``"<label> API error (HTTP n): detail"`` (JSON detail via *extract_detail*, else the first 300
    body chars); empty text -> the ``no_speech`` envelope (silence is non-fatal); exceptions ->
    ``_cloud_failure``."""
    try:
        response = post()
        if response.status_code != 200:
            try:
                detail = extract_detail(response.json()) or response.text[:300]
            except Exception:
                detail = response.text[:300]
            return _error_result(f"{label} API error (HTTP {response.status_code}): {detail}")
        body = response.json()
        transcript_text = extract_text(body)
        if not transcript_text:
            return _error_result(f"{label} returned empty transcript", no_speech=True)
        log(transcript_text, body)
        return _ok_result(transcript_text, provider)
    except Exception as e:
        return _cloud_failure(e, file_path, f"{label} transcription")


def _transcribe_xai(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe via xAI ``POST /v1/stt`` (multipart). Supports ITN, diarization, word timestamps."""
    from tools.transcription_tools import _load_stt_config, _resolve_stt_language, get_env_value
    from tools.xai_http import resolve_xai_http_credentials
    if prompt:
        _log_prompt_unsupported("STT provider 'xai'")
    # STT is API-billed: prefer the explicit XAI_API_KEY over the xAI OAuth/Grok-subscription
    # credential, which may be valid for Grok yet hit spending-limit errors on /v1/stt.
    direct_api_key = str(get_env_value("XAI_API_KEY") or "").strip()
    creds = {"provider": "xai", "api_key": direct_api_key,
             "base_url": str(get_env_value("XAI_BASE_URL") or "https://api.x.ai/v1").strip().rstrip("/")
             } if direct_api_key else resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        return _error_result("No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY")
    stt_config = _load_stt_config()
    xai_config = stt_config.get("xai") or {}

    def _resolve_base_url(resolved_creds: Dict[str, str]) -> str:
        # OAuth bearers are pinned to the resolver-validated origin; overrides apply to API keys only.
        url = resolved_creds.get("base_url")
        if resolved_creds.get("provider") != "xai-oauth":
            url = xai_config.get("base_url") or get_env_value("XAI_STT_BASE_URL") or url
        return str(url or XAI_STT_BASE_URL).strip().rstrip("/")

    # Language: hook override > stt.xai.language > stt.language > env.
    language = language or _resolve_stt_language("xai", stt_config) or ""

    def _post() -> Any:
        from tools.xai_http import hermes_xai_user_agent
        data: Dict[str, str] = {"language": language} if language else {}
        data.update({flag: "true" for flag, default in (("format", True), ("diarize", False))
                     if is_truthy_value(xai_config.get(flag, default))})

        def _post_transcription(bearer: str, endpoint_base_url: str):
            headers = {"Authorization": f"Bearer {bearer}", "User-Agent": hermes_xai_user_agent()}
            return _post_audio_multipart(f"{endpoint_base_url}/stt", headers, file_path, data)

        response = _post_transcription(api_key, _resolve_base_url(creds))
        if response.status_code in {401, 403} and creds.get("provider") == "xai-oauth":
            logger.info("xAI STT got HTTP %d; refreshing OAuth credentials and retrying once", response.status_code)
            try:
                refreshed_creds = resolve_xai_http_credentials(force_refresh=True, api_key_hint=api_key)
                refreshed_key = str(refreshed_creds.get("api_key") or "").strip()
                if refreshed_key and refreshed_key != api_key:
                    response = _post_transcription(refreshed_key, _resolve_base_url(refreshed_creds))
            except Exception as retry_exc:
                logger.warning("xAI STT OAuth refresh-and-retry after HTTP %d failed: %s",
                               response.status_code, retry_exc)
        return response

    def _log(transcript_text: str, result: Dict[str, Any]) -> None:
        logger.info("Transcribed %s via xAI Grok STT (lang=%s, %.1fs audio, %d chars)", Path(file_path).name,
                    result.get("language", language), result.get("duration", 0), len(transcript_text))

    return _rest_provider(file_path, "xai", "xAI STT", _post, lambda body: body.get("error", {}).get("message", ""),
                          lambda body: body.get("text", "").strip(), _log)


def _elevenlabs_error_detail(err_body: Dict[str, Any]) -> str:
    error_value = err_body.get("detail") or err_body.get("error")
    if isinstance(error_value, dict):
        return str(error_value.get("message") or error_value)
    return str(error_value) if error_value else ""


def _transcribe_elevenlabs(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe using ElevenLabs Scribe STT API."""
    from tools.transcription_tools import _load_stt_config, _resolve_provider_key, _resolve_stt_language, get_env_value
    if prompt:
        _log_prompt_unsupported("STT provider 'elevenlabs'")
    api_key = _resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
    if not api_key:
        return _error_result("ELEVENLABS_API_KEY not set")
    stt_config = _load_stt_config()
    elevenlabs_config = stt_config.get("elevenlabs") or {}
    base_url = str(
        elevenlabs_config.get("base_url") or get_env_value("ELEVENLABS_STT_BASE_URL") or ELEVENLABS_STT_BASE_URL
    ).strip().rstrip("/")
    # Language: hook override > stt.elevenlabs.language(_code) > stt.language.
    language_code = language or _resolve_stt_language("elevenlabs", stt_config, extra_keys=("language_code",)) or ""

    def _post() -> Any:
        data: Dict[str, str] = {
            "model_id": model_name,
            "tag_audio_events": str(is_truthy_value(elevenlabs_config.get("tag_audio_events", False))).lower(),
            "diarize": str(is_truthy_value(elevenlabs_config.get("diarize", False))).lower(),
            **({"language_code": language_code} if language_code else {})}
        return _post_audio_multipart(f"{base_url}/speech-to-text", {"xi-api-key": api_key}, file_path, data)

    def _log(transcript_text: str, _body: Dict[str, Any]) -> None:
        logger.info("Transcribed %s via ElevenLabs Scribe (%s, %d chars)",
                    Path(file_path).name, model_name, len(transcript_text))

    return _rest_provider(file_path, "elevenlabs", "ElevenLabs STT", _post, _elevenlabs_error_detail,
                          _extract_transcript_text, _log)


def _transcribe_deepinfra(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve DeepInfra credentials/model (shared ``hermes_cli.models`` helpers), then delegate to :func:`_transcribe_openai`."""
    from tools.transcription_tools import _load_stt_config, _resolve_provider_key
    api_key = _resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
    if not api_key:
        return _error_result("DEEPINFRA_API_KEY not set")
    from hermes_cli.models import deepinfra_base_url, deepinfra_model_ids
    # ``stt.deepinfra: null`` in YAML yields None, not {} — coalesce.
    base_url = deepinfra_base_url(_get_stt_section(_load_stt_config(), "deepinfra"))
    model_name = model_name or next(iter(deepinfra_model_ids("stt")), None)
    if not model_name:
        return _error_result(
            "No DeepInfra STT model available. Pin one in config.yaml under stt.deepinfra.model, "
            "or check connectivity to api.deepinfra.com so the live catalog can be fetched.")
    return _transcribe_openai(file_path, model_name, api_key=api_key, base_url=base_url,
                              provider_label="deepinfra", language=language, prompt=prompt)


# ---- OpenAI audio credential resolution -----------------------------------
def _is_local_or_private_url(url: str) -> bool:
    """True for loopback/RFC-1918/LAN-internal hosts, where an empty ``stt.openai.api_key`` is acceptable
    (local OpenAI-compatible servers ignore the auth header — no sham ``api_key: not-needed`` needed)."""
    try:
        from urllib.parse import urlparse
        import ipaddress
        host = (urlparse(url).hostname or "").lower()
        if host == "localhost" or host.endswith((".local", ".lan", ".internal")):
            return True
        addr = ipaddress.ip_address(host)  # raises for "" and non-IP hostnames
        return addr.is_private or addr.is_loopback
    except Exception:  # unparsable URL or non-IP hostname
        return False


def _direct_openai_credentials(cfg_api_key: str, cfg_base_url: str) -> Optional[tuple[str, str]]:
    """Direct-credential ladder: config key > keyless local base_url (placeholder key so the SDK
    constructs a client) > env key; None if none apply."""
    from tools.tool_backend_helpers import resolve_openai_audio_api_key
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or OPENAI_BASE_URL)
    # A local OpenAI-compatible server needs no key — send a placeholder so the SDK doesn't refuse to
    # construct a client (#25193, credit @nnnet).
    if cfg_base_url and _is_local_or_private_url(cfg_base_url):
        return "not-needed", cfg_base_url
    direct_api_key = resolve_openai_audio_api_key()
    return (direct_api_key, OPENAI_BASE_URL) if direct_api_key else None


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """``(api_key, base_url)`` for the OpenAI STT client, strict on the stored ``stt`` selection:
    ``"nous"`` -> managed gateway ONLY (a direct OPENAI_API_KEY must NOT override it); any other
    stored provider -> direct credentials ONLY (no silent managed fallback); never-configured ->
    legacy ladder: direct credentials, then the managed gateway. Failures raise ValueError."""
    from tools.transcription_tools import _load_stt_config
    from tools.managed_tool_gateway import resolve_managed_tool_gateway
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER, managed_nous_tools_enabled, nous_tool_gateway_unavailable_message,
        read_selection, selection_error)
    openai_cfg = _load_stt_config().get("openai") or {}
    selected = read_selection("stt")

    def _managed() -> Optional[tuple[str, str]]:
        gateway = resolve_managed_tool_gateway("openai-audio")
        if gateway is None:
            return None
        return gateway.nous_user_token, urljoin(f"{gateway.gateway_origin.rstrip('/')}/", "v1")

    if selected == NOUS_MANAGED_PROVIDER:
        managed = _managed()
        if managed is None:
            raise ValueError(selection_error("stt", NOUS_MANAGED_PROVIDER,
                                             "the Nous Tool Gateway is not available (not entitled or unreachable)"))
        return managed
    direct = _direct_openai_credentials(openai_cfg.get("api_key", ""), openai_cfg.get("base_url", ""))
    if direct is not None:
        return direct
    if selected is not None:
        raise ValueError(selection_error(
            "stt", selected,
            "neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"))
    managed = _managed()
    if managed is None:
        message = "Neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += ". " + nous_tool_gateway_unavailable_message("managed OpenAI audio for transcription")
        raise ValueError(message)
    return managed


def _extract_transcript_text(transcription: Any) -> str:
    """Normalize text / object / dict transcription responses to a plain string."""
    value = transcription if isinstance(transcription, str) else getattr(transcription, "text", None)
    if not isinstance(value, str) and isinstance(transcription, dict):
        value = transcription.get("text")
    text = (value if isinstance(value, str) else str(transcription)).strip()
    match = _ASR_TEXT_RE.match(text)
    return match.group("text").strip() if match else text
