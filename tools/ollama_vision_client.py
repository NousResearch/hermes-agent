#!/usr/bin/env python3
"""Unified model-invocation adapter for the local vision Orchestrator
(V0.1 inactive foundation).

Stage B1 reuses the existing auxiliary-client call chain
(``agent.auxiliary_client.async_call_llm`` with ``task="vision"``) and the
existing image-preparation pipeline in :mod:`tools.vision_tools` /
:mod:`tools.image_source`:

- ``resolve_image_source`` — download/read + website-policy checks;
- ``_normalize_to_supported_image`` — SVG/BMP/TIFF → supported format;
- ``_image_to_base64_data_url`` — base64 + data-URL construction.

This adapter is *not* a heavy SDK: it is one thin, uniform entry point that
accepts an exact model identity per call (per-call model override already
exists in ``async_call_llm`` / ``vision_analyze_tool``).

No live inference is performed by Stage B1 — tests are fully mocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.vision_policy import ExecutionStatus

# Lazily bound auxiliary-client entry (mirrors tools/vision_tools.py pattern
# so tests can patch the same names).
_async_call_llm: Any = None
_extract_content_or_reasoning: Any = None


def _noop_extractor(response: Any) -> str:
    """Sentinel used only when the canonical extractor failed to load."""
    return ""


def _load_auxiliary_client() -> None:
    global _async_call_llm, _extract_content_or_reasoning
    if _async_call_llm is None or _extract_content_or_reasoning is None:
        from agent.auxiliary_client import (
            async_call_llm as _acl,
            extract_content_or_reasoning as _ecr,
        )

        if _async_call_llm is None:
            _async_call_llm = _acl
        if _extract_content_or_reasoning is None:
            _extract_content_or_reasoning = _ecr


async def _resolve_image_bytes_async(
    image_source: str, task_id: Optional[str] = None
) -> bytes:
    from tools.image_source import ResolveContext, resolve_image_source

    resolved = await resolve_image_source(image_source, ResolveContext(task_id=task_id))
    return resolved.data


def _ensure_ollama_transport_compatible_image(
    normalized_path: Path, mime: Optional[str]
) -> Tuple[bytes, str, bool]:
    """Deterministic endpoint-transport compatibility conversion.

    The canonical normalization pipeline (``_normalize_to_supported_image``)
    preserves WebP bytes; the Ollama OpenAI-compatible endpoint rejects
    ``data:image/webp;base64,...`` with HTTP 400. This adapter-local step
    converts WebP to PNG *for the model request only* — the canonical
    calibration identity (``normalized_image_sha256``) is untouched.

    Returns ``(transport_bytes, transport_mime, transcoded)``:
    - PNG/JPEG inputs pass through unchanged (``transcoded=False``);
    - WebP inputs are deterministically re-encoded to PNG
      (``transcoded=True``), alpha preserved, dimensions unchanged;
    - MIME is detected from bytes, never from the filename suffix.

    Never modifies the source file. The caller owns cleanup of any temporary
    output created here.
    """
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    actual_mime = mime or ""
    if not actual_mime or actual_mime == "image/webp":
        try:
            actual_mime = _detect_image_mime_type_from_bytes(normalized_path.read_bytes()) or actual_mime
        except Exception:  # noqa: BLE001 — best-effort detection
            pass

    if actual_mime != "image/webp":
        # PNG/JPEG/GIF and other accepted formats pass through unchanged.
        data = normalized_path.read_bytes()
        return data, actual_mime or "application/octet-stream", False

    # WebP → deterministic PNG conversion (alpha preserved, dimensions kept).
    from PIL import Image

    with Image.open(normalized_path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "A" in im.mode:
            png_im = im.convert("RGBA")
        else:
            png_im = im.convert("RGB")
        buf = io.BytesIO()
        png_im.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    return png_bytes, "image/png", True


async def prepare_image(
    image_source: str,
    task_id: Optional[str] = None,
) -> Tuple[str, Optional[int], Optional[int], Optional[str], Optional[str], Dict[str, Any]]:
    """Reuse the existing image pipeline to produce a base64 data URL.

    Returns
    ``(data_url, width_px, height_px, mime, normalized_sha256, transport_meta)``
    where:

    - ``width/height`` are populated only when the resolver provides them
      (may be ``None`` for opaque sources);
    - ``normalized_sha256`` is the SHA-256 of the EXACT canonical normalized
      asset bytes — this is the calibration identity and always matches the
      committed manifest;
    - ``transport_meta`` is ``{"transport_image_sha256": str,
      "transport_mime_type": str, "transport_transcoded": bool}`` describing
      the exact bytes encoded in the model-request data URL.

    Two-layer hash contract: canonical bytes define calibration identity;
    transport bytes are what the endpoint actually receives. WebP is
    converted to PNG only for transport (Ollama rejects WebP data URLs).
    No new source-resolution or canonical-normalization pipeline is
    introduced.
    """
    from tools.vision_tools import (
        _image_to_base64_data_url,
        _normalize_to_supported_image,
    )

    data = await _resolve_image_bytes_async(image_source, task_id=task_id)

    mime: Optional[str] = None
    try:
        from tools.vision_tools import _detect_image_mime_type_from_bytes

        mime = _detect_image_mime_type_from_bytes(data)
    except Exception:  # noqa: BLE001 — mime detection is best-effort
        mime = None

    temp_dir = Path("/tmp") / "vision_orchestrator"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = temp_dir / f"img_{hashlib.sha256(data).hexdigest()[:16]}.img"
    if not tmp.exists():
        tmp.write_bytes(data)

    normalized_path, mime, _err = await asyncio.to_thread(
        _normalize_to_supported_image, tmp, mime or ""
    )
    if _err or normalized_path is None:
        raise ValueError(_err or "Image normalization failed.")
    if normalized_path != tmp and tmp.exists():
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass

    # Canonical identity: SHA-256 over the exact canonical normalized bytes.
    normalized_bytes = normalized_path.read_bytes()
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()

    # Endpoint transport compatibility (WebP → PNG only for the request).
    transport_bytes, transport_mime, transcoded = await asyncio.to_thread(
        _ensure_ollama_transport_compatible_image, normalized_path, mime
    )
    transport_sha256 = hashlib.sha256(transport_bytes).hexdigest()
    transport_meta = {
        "transport_image_sha256": transport_sha256,
        "transport_mime_type": transport_mime,
        "transport_transcoded": transcoded,
    }

    data_url = await asyncio.to_thread(
        _image_to_base64_data_url, normalized_path, mime_type=transport_mime
    )
    # If transport bytes differ from canonical bytes, the data URL must be
    # built from the transport bytes, not the canonical file.
    if transcoded:
        transport_path = temp_dir / f"transport_{transport_sha256[:16]}.png"
        if not transport_path.exists():
            transport_path.write_bytes(transport_bytes)
        data_url = await asyncio.to_thread(
            _image_to_base64_data_url, transport_path, mime_type="image/png"
        )

    width = height = None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(transport_bytes)) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 — dimensions are best-effort
        pass
    return data_url, width, height, mime, normalized_sha256, transport_meta


async def invoke_vision_model(
    *,
    model: str,
    prompt: str,
    image_data_url: str,
    timeout_seconds: float = 120.0,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Invoke exactly one vision-model call through the unified path.

    Uses the existing ``async_call_llm(task="vision", model=...)`` chain —
    the per-call model override already exists and does not alter any other
    caller. Structured (JSON) responses are requested via the prompt; the
    raw text is returned for the quality evaluator to parse.

    Returns ``{"execution_status": ..., "raw_text": ..., "error": ...}``.
    """
    _load_auxiliary_client()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    call_kwargs: Dict[str, Any] = {
        "task": "vision",
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
    }
    if base_url:
        call_kwargs["base_url"] = base_url
    if api_key:
        call_kwargs["api_key"] = api_key

    try:
        response = await _async_call_llm(**call_kwargs)
    except TimeoutError:
        return {
            "execution_status": ExecutionStatus.TIMEOUT.value,
            "raw_text": "",
            "error": "vision_model_timeout",
        }
    except Exception as exc:  # noqa: BLE001 — classified below
        status = _classify_error(exc)
        return {
            "execution_status": status,
            "raw_text": "",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }

    text = _extract_text(response)
    if text is None:
        return {
            "execution_status": ExecutionStatus.INVALID_RESPONSE.value,
            "raw_text": "",
            "error": "unparseable_model_response",
        }
    return {
        "execution_status": ExecutionStatus.SUCCESS.value,
        "raw_text": text,
        "error": None,
    }


def _extract_text(response: Any) -> Optional[str]:
    """Extract assistant text from the auxiliary-client response object.

    Resolution order (no second parser is introduced):

    1. Canonical Hermes extractor — ``agent.auxiliary_client.
       extract_content_or_reasoning`` — handles the real ``async_call_llm``
       contract: ``response.choices[0].message.content`` with reasoning-field
       fallback. This is the same helper the existing ``vision_analyze_tool``
       path uses, so the two paths cannot diverge.
    2. Legacy string/dict forms (``content``/``text``/list-of-text) are kept
       for the verified mock-test contract and older provider adapters that
       return plain dicts.

    Returns ``None`` when no usable text exists (empty content, malformed
    choices/message, or unsupported shape) so the caller can classify the
    invocation as INVALID_RESPONSE. Never calls the model again.
    """
    if response is None:
        return None
    if isinstance(response, str):
        return response or None
    if isinstance(response, dict):
        content = response.get("content") or response.get("text")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
            return "".join(parts) if parts else None
        return None
    # Object shape: prefer the canonical Hermes extractor, which understands
    # response.choices[0].message.content (+ reasoning fallback).
    if _extract_content_or_reasoning is None:
        _load_auxiliary_client()
    extractor = _extract_content_or_reasoning or _noop_extractor
    try:
        canonical = extractor(response)
    except Exception:
        canonical = ""
    if canonical and canonical.strip():
        return canonical
    # Fallback for object shapes the canonical helper cannot parse but that
    # still expose a top-level content/text attribute (e.g. some test fakes).
    content = getattr(response, "content", None)
    if content is not None:
        return str(content) or None
    text = getattr(response, "text", None)
    if text is not None:
        return str(text) or None
    return None


def _classify_error(exc: Exception) -> str:
    """Classify an invocation exception into an ExecutionStatus value."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return ExecutionStatus.TIMEOUT.value
    if "model" in msg and ("not found" in msg or "does not exist" in msg):
        return ExecutionStatus.MODEL_NOT_FOUND.value
    if "connect" in msg or "connection" in msg or "refused" in msg:
        return ExecutionStatus.ENDPOINT_UNAVAILABLE.value
    if "schema" in msg or "json" in msg:
        return ExecutionStatus.SCHEMA_INVALID.value
    return ExecutionStatus.INVALID_RESPONSE.value
