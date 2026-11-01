"""Ollama speech backend for ``tools.tts_tool``.

Ollama ≥ 0.40.9 serves text-to-speech models (Phonon today) over the OpenAI-compatible
``POST /v1/audio/speech`` endpoint, streaming a WAV or raw PCM. This module only speaks HTTP with
:mod:`urllib` — no SDK, no key — so it costs nothing to import and works against a local daemon or
a remote ``OLLAMA_HOST``.

Two properties of that endpoint shape the code here:

* **Options are not ignored, they are refused.** An unsupported ``response_format``, ``speed`` or
  ``instructions`` is a ``400``; an unknown voice is a ``400`` whose body lists the model's voices.
  So the request carries only fields Ollama documents and the error text is surfaced verbatim.
* **A model has to be speech-capable.** Chat/embedding models answer a speech request with a
  ``400``; :func:`ollama_speech_models` filters ``/api/tags`` through ``/api/show`` capabilities so
  the settings page can offer only real speech models.

The WAV Ollama streams has its RIFF size fields set to the maximum value (the header is written
before the length is known), which is why the result goes through ``_finalize_wav_output`` rather
than being trusted as a well-formed container.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from tools.tts_tool_delivery import _finalize_wav_output, _section, _wav_sidecar_path

logger = logging.getLogger("tools.tts_tool")

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "phonon"  # `phonon` == `phonon:31m-en`; 40m-* adds the other languages
DEFAULT_OLLAMA_VOICE = "default"  # the only voice the current Phonon packages ship
DEFAULT_OLLAMA_TIMEOUT = 120.0
# Documented per-request input cap for Ollama speech models.
OLLAMA_MAX_TEXT_LENGTH = 4096


def _ollama_base_url(raw: Any) -> str:
    """Base URL ending in ``/v1`` (the OpenAI-compatible root), from either spelling of the host.

    ``http://localhost:11434``, ``localhost:11434`` and ``http://localhost:11434/v1`` all resolve to
    the same root; a trailing slash is dropped so ``/v1/audio/speech`` joins cleanly.
    """
    value = str(raw or "").strip().rstrip("/")
    if not value:
        return DEFAULT_OLLAMA_BASE_URL
    if "://" not in value:
        value = f"http://{value}"
    return value if value.endswith("/v1") else f"{value}/v1"


def _ollama_root(url: str) -> str:
    """The native (``/api/...``) root for a ``/v1`` base URL."""
    return _ollama_base_url(url).removesuffix("/v1")


def _ollama_timeout(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_OLLAMA_TIMEOUT
    return value if value > 0 else DEFAULT_OLLAMA_TIMEOUT


def _request_json(url: str, payload: Optional[dict[str, Any]], timeout: float) -> Any:
    """GET (``payload=None``) or POST ``payload`` as JSON, returning the decoded body.

    ``HTTPError`` bodies carry Ollama's own message (``{"error": "..."}``); they are raised as
    ``RuntimeError`` with that text so the tool result explains what to fix instead of showing a
    bare status code.
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if data is None else "POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8", "replace").strip()
            detail = (json.loads(body).get("error") or body) if body else ""
        except (ValueError, OSError):  # non-JSON body, or a stream that failed mid-read
            detail = ""
        raise RuntimeError(f"Ollama {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {_ollama_root(url)} ({exc.reason}). Is `ollama serve` running, "
            "and is tts.ollama.base_url right?") from exc


def ollama_voices(model: str, base_url: str = DEFAULT_OLLAMA_BASE_URL,
                  timeout: float = DEFAULT_OLLAMA_TIMEOUT) -> list[str]:
    """Voice names ``model`` exposes, from ``/api/show`` (``[]`` when the daemon is unreachable).

    Ollama lists them under ``voices``; the key is absent on builds and models that carry none, in
    which case the caller should fall back to :data:`DEFAULT_OLLAMA_VOICE`.
    """
    try:
        shown = _request_json(f"{_ollama_root(base_url)}/api/show", {"model": model}, timeout)
    except RuntimeError as exc:
        logger.debug("[Ollama] voice lookup failed for %s: %s", model, exc)
        return []
    raw = shown.get("voices") if isinstance(shown, dict) else None
    return [str(v) for v in raw] if isinstance(raw, list) else []


def ollama_speech_models(base_url: str = DEFAULT_OLLAMA_BASE_URL,
                         timeout: float = DEFAULT_OLLAMA_TIMEOUT) -> list[str]:
    """Installed models whose capabilities include ``speech`` (``[]`` when offline).

    One ``/api/show`` per installed model; a daemon with dozens of models pays dozens of tiny
    local calls, which is why this is only called from the settings/CLI surfaces, never per turn.
    """
    try:
        tags = _request_json(f"{_ollama_root(base_url)}/api/tags", None, timeout)
    except RuntimeError as exc:
        logger.debug("[Ollama] model listing failed: %s", exc)
        return []
    models = [str(m.get("name")) for m in (tags.get("models") or []) if isinstance(m, dict) and m.get("name")]
    speech: list[str] = []
    for name in models:
        try:
            shown = _request_json(f"{_ollama_root(base_url)}/api/show", {"model": name}, timeout)
        except RuntimeError:
            continue
        capabilities = shown.get("capabilities") if isinstance(shown, dict) else None
        if isinstance(capabilities, list) and "speech" in capabilities:
            speech.append(name)
    return speech


def _generate_ollama_tts(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
    """Synthesize *text* through Ollama's ``/v1/audio/speech`` and write it to *output_path*."""
    ol_config = _section(tts_config, "ollama")
    base_url = _ollama_base_url(ol_config.get("base_url"))
    model = str(ol_config.get("model") or DEFAULT_OLLAMA_MODEL).strip()
    voice = str(ol_config.get("voice") or DEFAULT_OLLAMA_VOICE).strip()
    timeout = _ollama_timeout(ol_config.get("timeout"))
    payload: dict[str, Any] = {
        "model": model, "input": text, "voice": voice, "response_format": "wav"}
    # Only fields Ollama documents: anything else (speed, instructions) is a 400, not a no-op.
    request = urllib.request.Request(
        f"{base_url}/audio/speech", data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    logger.info("[Ollama] Synthesizing with %s (voice=%s) at %s", model, voice, base_url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8", "replace").strip()
            detail = (json.loads(body).get("error") or body) if body else ""
        except (ValueError, OSError):  # non-JSON body, or a stream that failed mid-read
            detail = ""
        # Order matters: Ollama's unknown-voice message also contains "not found", so the voice
        # hint has to be checked before the missing-model one.
        hint = ""
        if "voice" in detail.lower():
            hint = " Set tts.ollama.voice to one of the voices listed above."
        elif exc.code == 404 or "not found" in detail.lower():
            # A 404 is either a missing model or a daemon too old to serve speech at all:
            # /v1/audio/speech landed in Ollama 0.40.9.
            hint = (f" Run `ollama pull {model}`, and check `ollama --version` — speech models"
                    " need Ollama 0.40.9 or newer.")
        raise RuntimeError(f"Ollama speech request failed ({exc.code}): {detail or exc.reason}.{hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url} ({exc.reason}). Is `ollama serve` running?") from exc
    if not audio:
        raise RuntimeError(f"Ollama returned no audio for model '{model}'.")
    wav_path = _wav_sidecar_path(output_path)
    with open(wav_path, "wb") as handle:
        handle.write(audio)
    logger.info("[Ollama] Wrote %d bytes of audio", len(audio))
    return _finalize_wav_output(wav_path, output_path)
