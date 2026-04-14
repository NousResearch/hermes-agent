"""MiniMax image-01 backend for the image_generate tool.

This module isolates all MiniMax-specific image-generation logic so
the main ``tools/image_generation_tool.py`` file can add backend
dispatch with a minimal diff — it imports from here instead of
growing inline MiniMax code.

Design mirrors the TTS file's per-provider helpers (e.g.
``_generate_minimax_tts`` in ``tools/tts_tool.py``): a single public
function that takes the existing tool-level kwargs and returns the
tool's standard result dict.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


#: API host; overridable via MINIMAX_API_HOST for CN users.
_DEFAULT_HOST = "https://api.minimax.io"

#: MiniMax image-01 supported aspect ratios.  Mirrors what the CLI
#: ``mmx image generate --aspect-ratio`` accepts.
_VALID_ASPECT_RATIOS = {
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9",
}

#: Map the tool's generic aspect-ratio names (``landscape``/``portrait``/
#: ``square``) to MiniMax-specific ratios.  Keeps the tool's public
#: schema unchanged while routing cleanly to MiniMax.
_GENERIC_TO_MINIMAX = {
    "landscape": "16:9",
    "portrait": "9:16",
    "square": "1:1",
}


def _host() -> str:
    raw = (os.environ.get("MINIMAX_API_HOST") or _DEFAULT_HOST).strip().rstrip("/")
    return raw or _DEFAULT_HOST


def _api_key() -> Optional[str]:
    """Return the MiniMax API key, preferring the international env var."""
    for var in ("MINIMAX_API_KEY", "MINIMAX_CN_API_KEY"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    return None


def check_minimax_image_requirements() -> bool:
    """Tool registry ``check_fn`` for the MiniMax image backend."""
    return _api_key() is not None


def _normalise_aspect(value: str) -> str:
    v = (value or "").strip().lower()
    if v in _VALID_ASPECT_RATIOS:
        return v
    if v in _GENERIC_TO_MINIMAX:
        return _GENERIC_TO_MINIMAX[v]
    # Fall back to the most common; don't error out — the tool's outer
    # schema already validates user-facing names.
    return "16:9"


def _post_json(path: str, payload: Dict[str, Any], *, timeout: int = 120) -> Dict[str, Any]:
    key = _api_key()
    if not key:
        raise RuntimeError(
            "MINIMAX_API_KEY not configured — set it in ~/.hermes/.env "
            "or switch image_gen.provider away from 'minimax'."
        )
    url = f"{_host()}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = _safe_read(exc)
        raise RuntimeError(
            f"MiniMax image API error ({exc.code}): {_extract_error(body)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"MiniMax image API returned non-JSON: {raw[:200]}"
        ) from exc


def _download(url: str, output_path: str, *, timeout: int = 120) -> str:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
        with open(path, "wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
    return str(path)


def _extract_urls(resp: Dict[str, Any]) -> List[str]:
    """Pull image URLs out of a MiniMax image_generation response."""
    data = resp.get("data") or {}
    if isinstance(data, dict):
        for key in ("image_urls", "urls"):
            v = data.get(key)
            if isinstance(v, list):
                return [u for u in v if isinstance(u, str) and u.startswith("http")]
        for key in ("image_url", "url"):
            v = data.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return [v]
    for key in ("image_urls", "urls"):
        v = resp.get(key)
        if isinstance(v, list):
            return [u for u in v if isinstance(u, str) and u.startswith("http")]
    return []


def _safe_read(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_error(body: str) -> str:
    if not body:
        return "(empty response body)"
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:400]
    if isinstance(data, dict):
        base = data.get("base_resp") or {}
        if isinstance(base, dict) and base.get("status_msg"):
            return f"{base['status_msg']} (status_code={base.get('status_code')})"
        err = data.get("error") or {}
        if isinstance(err, dict) and err.get("message"):
            return err["message"]
    return body[:400]


def _output_dir() -> str:
    """Default output directory — matches the existing image_generate tool."""
    from hermes_constants import get_hermes_dir
    try:
        return str(get_hermes_dir("cache/images", "image_cache"))
    except Exception:
        return str(Path.home() / "Desktop")


def generate_minimax_image(
    *,
    prompt: str,
    aspect_ratio: str = "landscape",
    num_images: int = 1,
    output_format: str = "png",
) -> Dict[str, Any]:
    """Generate one or more images via MiniMax image-01.

    Signature matches the subset of ``image_generate_tool`` arguments
    that MiniMax supports.  Returns the existing tool's result shape so
    the dispatching layer in ``image_generation_tool.py`` doesn't need
    format adapters.

    Parameters
    ----------
    prompt:
        Text description of the desired image.
    aspect_ratio:
        Accepts MiniMax-native ratios (``16:9``, ``1:1`` …) or the
        image_generate tool's generic names (``landscape`` / ``portrait``
        / ``square``).  Falls back to ``16:9`` on unknown input.
    num_images:
        1-4 images per request.  Values outside that range are clamped.
    output_format:
        ``png`` or ``jpeg``.  Stored as the file extension;
        image-01 returns PNGs at the URL, so we keep the byte stream
        as-is regardless of user preference for now.
    """
    if not prompt or not prompt.strip():
        return {"success": False, "error": "prompt is required"}

    n = max(1, min(4, int(num_images or 1)))
    payload = {
        "model": "image-01",
        "prompt": prompt.strip(),
        "aspect_ratio": _normalise_aspect(aspect_ratio),
        "n": n,
        "response_format": "url",
    }

    try:
        resp = _post_json("/v1/image_generation", payload)
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    # MiniMax wraps errors in base_resp even on HTTP 200.  status_code 0
    # means success; everything else is a MiniMax-side failure we should
    # surface verbatim (e.g. 1008 "insufficient balance", 1004 "auth
    # failed") rather than masking as "no URL".  Mirrors what
    # vision_minimax does for /v1/coding_plan/vlm.
    base = resp.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        return {
            "success": False,
            "error": f"MiniMax image API error {base.get('status_code')}: "
                     f"{base.get('status_msg', '')}",
            "status": base.get("status_code"),
        }

    urls = _extract_urls(resp)
    if not urls:
        return {
            "success": False,
            "error": "MiniMax image API returned no URL(s)",
            "raw_keys": list(resp.keys()),
        }

    ext = "jpeg" if (output_format or "").lower() in ("jpeg", "jpg") else "png"
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _output_dir()
    saved: List[str] = []
    for i, url in enumerate(urls):
        if i == 0:
            filename = f"minimax_image_{ts}.{ext}"
        else:
            filename = f"minimax_image_{ts}_{i + 1}.{ext}"
        path = str(Path(out_dir) / filename)
        try:
            saved_path = _download(url, path)
            saved.append(saved_path)
        except Exception as exc:
            logger.warning("MiniMax image download failed for %s: %s", url, exc)

    if not saved:
        return {"success": False, "error": "all image downloads failed", "urls": urls}

    return {
        "success": True,
        "provider": "minimax",
        "model": "image-01",
        "prompt": prompt,
        "aspect_ratio": payload["aspect_ratio"],
        "paths": saved,
        "urls": urls,
        "count": len(saved),
    }
