"""MiniMax VLM backend for the vision_analyze tool.

Dedicated helper for the image-understanding endpoint at
``/v1/coding_plan/vlm``, which is the path MiniMax's own ``mmx vision``
CLI uses and the same one their Coding Plan MCP wraps.  Unlike the
chat models (M2.7 et al.) on ``/v1/chat/completions`` or
``/anthropic/v1/messages``, the VLM endpoint is purpose-built for
vision — it takes a prompt + image and returns a text description,
matching the "image → caption → text" flow Hermes expects from a
vision backend.

Request::

    POST /v1/coding_plan/vlm
    Authorization: Bearer $MINIMAX_API_KEY
    {"prompt": "Describe...", "image_url": "data:image/png;base64,…"}

Response::

    {"content": "A rain-slicked street in a futuristic neon city…",
     "base_resp": {"status_code": 0, "status_msg": "success"}}

The ``image_url`` field accepts:

- A ``data:image/<mime>;base64,…`` URI (we convert local files this way).
- A plain ``http(s)://`` URL to an image MiniMax can fetch.
- A pre-uploaded ``file_id`` (passed as the ``file_id`` body field instead).

This module exposes a single public function, :func:`describe_image`,
plus an ``is_available()`` helper for the tool registry's ``check_fn``.
Zero new dependencies — pure stdlib.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_HOST = "https://api.minimax.io"
VLM_PATH = "/v1/coding_plan/vlm"
DEFAULT_TIMEOUT = 60

# Same MIME whitelist as mmx-cli's vision describe command.
_SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp"}


def _api_key() -> Optional[str]:
    for var in ("MINIMAX_API_KEY", "MINIMAX_CN_API_KEY"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    return None


def _host() -> str:
    raw = (os.environ.get("MINIMAX_API_HOST") or DEFAULT_HOST).strip().rstrip("/")
    return raw or DEFAULT_HOST


def is_available() -> bool:
    """Tool registry ``check_fn``: True when a MiniMax key is configured."""
    return _api_key() is not None


def _read_local_as_data_uri(path: str) -> str:
    """Read a local image file and return a ``data:`` URI with inferred MIME."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        # Sniff common magic bytes to avoid sending an unknown MIME.
        with p.open("rb") as fh:
            head = fh.read(12)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif head.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "image/jpeg"
    if mime not in _SUPPORTED_MIME:
        raise ValueError(
            f"Unsupported image MIME {mime!r}. MiniMax VLM accepts "
            f"JPEG / PNG / WebP only."
        )
    data = p.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _to_image_url(image_source: str) -> str:
    """Normalise ``image_source`` to something the VLM API accepts.

    Accepts:
      - Existing ``data:image/…;base64,…`` URIs (passed through).
      - Absolute ``http(s)://`` URLs (passed through; MiniMax fetches).
      - ``file://`` URLs (stripped, treated as local paths).
      - Local file paths (converted to a base64 data URI).
    """
    s = image_source.strip()
    if s.startswith("data:"):
        return s
    if s.startswith(("http://", "https://")):
        return s
    if s.startswith("file://"):
        s = s[len("file://"):]
    return _read_local_as_data_uri(s)


def describe_image(
    *,
    prompt: str,
    image_source: Optional[str] = None,
    file_id: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Call the MiniMax VLM endpoint and return the result.

    Pass either ``image_source`` (URL / local path / data URI) or
    ``file_id`` (pre-uploaded file reference via the MiniMax File API),
    never both.  Returns a dict in the shape the ``vision_analyze_tool``
    already uses:

        {"success": True, "analysis": "…", "provider": "minimax",
         "model": "MiniMax-VL-01"}

    Or on failure::

        {"success": False, "error": "…"}
    """
    if (not image_source) and (not file_id):
        return {"success": False, "error": "Either image_source or file_id is required"}
    if image_source and file_id:
        return {"success": False, "error": "Pass image_source OR file_id, not both"}

    key = _api_key()
    if not key:
        return {
            "success": False,
            "error": (
                "MINIMAX_API_KEY not configured.  Set it in "
                "~/.hermes/.env to enable MiniMax vision."
            ),
        }

    body: Dict[str, Any] = {"prompt": prompt}
    if file_id:
        body["file_id"] = file_id
    else:
        try:
            body["image_url"] = _to_image_url(image_source or "")
        except (FileNotFoundError, ValueError) as exc:
            return {"success": False, "error": str(exc)}

    url = f"{_host()}{VLM_PATH}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
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
        return {
            "success": False,
            "error": f"MiniMax VLM HTTP {exc.code}: {_safe_error_body(exc)}",
            "status": exc.code,
        }
    except urllib.error.URLError as exc:
        return {"success": False, "error": f"Network error: {exc.reason}"}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": f"Non-JSON response: {raw[:200]}"}

    # Check base_resp (MiniMax's uniform status wrapper).
    base = payload.get("base_resp") or {}
    if isinstance(base, dict) and base.get("status_code") not in (0, None):
        return {
            "success": False,
            "error": f"MiniMax VLM error {base.get('status_code')}: "
                     f"{base.get('status_msg', '')}",
            "status": base.get("status_code"),
        }

    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        return {
            "success": False,
            "error": "VLM response contained no content",
            "raw_keys": list(payload.keys()),
        }

    return {
        "success": True,
        "analysis": content,
        "provider": "minimax",
        "model": "MiniMax-VL-01",
        "endpoint": VLM_PATH,
    }


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "(no response body)"
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            err = parsed.get("error") or {}
            if isinstance(err, dict) and err.get("message"):
                return err["message"]
            base = parsed.get("base_resp") or {}
            if isinstance(base, dict) and base.get("status_msg"):
                return base["status_msg"]
    except Exception:
        pass
    return body[:400]
