"""Agent-facing tool: ``nopecha_recognition``.

Provider-independent interface over ``tools/nopecha_client.py`` (the
NopeCHA-specific HTTP layer). The agent calls this when
``browser_navigate``/``browser_snapshot``'s CAPTCHA detection
(``tools/browser_captcha_detection.py``) has flagged a page as
``"captcha"`` and the workflow is authorized for automated verification.

This tool ONLY recognizes a challenge (returns what to click / what text is
shown). It never submits anything back to a website and never requests a
solved token for a site's widget (NopeCHA's separate Token API, which is
unsupported here by design -- see ``tools/nopecha_client.py`` module
docstring). After this tool returns, the agent is expected to act on the
result itself via the normal browser tools (``browser_click`` etc.) if, and
only if, that is consistent with the target site's own permitted
verification workflow.

IMPLEMENTATION STATUS (Phase 2, 2026-11-13): ``challenge_type`` is still
restricted to ``"recaptcha"`` at the schema level -- the other 6 NopeCHA
Recognition-API types are spec'd in ``tools/nopecha_client.py`` but not yet
exercised against a real job each (blocked on free-tier access -- see that
module's docstring). Expanding this enum is follow-up work once that's
resolved.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools.nopecha_client import (
    MAX_IMAGE_BYTES,
    NopeCHAError,
    poll_recognition,
    record_call_result,
    submit_recognition,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
}


NOPECHA_RECOGNITION_SCHEMA: Dict[str, Any] = {
    "name": "nopecha_recognition",
    "description": (
        "Recognize a CAPTCHA challenge using NopeCHA's Recognition API -- returns "
        "what to click or what text is shown, it does NOT submit anything to the "
        "website and does NOT return a ready-to-inject solved token. Use only after "
        "browser_navigate/browser_snapshot has flagged the current page as a CAPTCHA "
        "and the site/workflow is authorized for automated verification. Currently "
        "only challenge_type='recaptcha' (image-grid challenges, e.g. 'Select all "
        "squares with X') is supported.\n\n"
        "Provide the challenge image via image_path (a local screenshot/crop path, "
        "e.g. from browser_vision) OR image_data (already base64-encoded) -- exactly "
        "one of the two. Free mode (no API key configured) works from some IPs but "
        "is unreliable in practice -- if you see error reason 'ip_ineligible', a paid "
        "NOPECHA_API_KEY is required; do not retry the same request repeatedly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "challenge_type": {
                "type": "string",
                "enum": ["recaptcha"],
                "description": "NopeCHA Recognition API challenge type. Only 'recaptcha' is currently supported.",
            },
            "task": {
                "type": "string",
                "description": (
                    "The CAPTCHA's own instruction text, in English, verbatim from the "
                    "challenge (e.g. 'Select all squares with vehicles')."
                ),
            },
            "grid": {
                "type": "string",
                "enum": ["1x1", "3x3", "4x4"],
                "description": (
                    "Size of the puzzle grid: 3x3 for a 9-tile challenge, 4x4 for 16 "
                    "tiles, 1x1 when images are shown one at a time. Omit if unknown."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Local file path to a screenshot/crop of the challenge image. Mutually exclusive with image_data.",
            },
            "image_data": {
                "type": "string",
                "description": "Already base64-encoded (optionally data-URI-prefixed) challenge image. Mutually exclusive with image_path.",
            },
            "task_id": {
                "type": "string",
                "description": "Browser session/task identifier, for logging correlation only (not sent to NopeCHA).",
            },
        },
        "required": ["challenge_type", "task"],
    },
}


def _load_image_data_uri(image_path: Optional[str], image_data: Optional[str]) -> str:
    """Resolve exactly one of image_path/image_data into a data-URI string."""
    if image_path and image_data:
        raise ValueError("Pass only one of image_path or image_data, not both.")
    if not image_path and not image_data:
        raise ValueError("One of image_path or image_data is required.")

    if image_data:
        return image_data if image_data.startswith("data:") else f"data:image/png;base64,{image_data}"

    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")
    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(raw)} bytes (max {MAX_IMAGE_BYTES} bytes). "
            f"Crop to just the challenge widget, not a full-page screenshot."
        )
    mime = _MIME_BY_EXT.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def nopecha_recognition(
    challenge_type: str,
    task: str,
    grid: Optional[str] = None,
    image_path: Optional[str] = None,
    image_data: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Submit + poll a NopeCHA Recognition API job. Returns a normalized JSON envelope."""
    started = time.monotonic()

    def _envelope(status: str, result: Any = None, request_id: Optional[str] = None,
                  error: Optional[Dict[str, Any]] = None) -> str:
        return json.dumps({
            "provider": "nopecha",
            "status": status,
            "challenge_type": challenge_type,
            "result": result,
            "request_id": request_id,
            "error": error,
        }, ensure_ascii=False)

    try:
        data_uri = _load_image_data_uri(image_path, image_data)
    except ValueError as exc:
        logger.info("nopecha_recognition: input validation failed (task_id=%s): %s", task_id, exc)
        # Not a provider call at all -- don't count this against call stats.
        return _envelope("error", error={"reason": "invalid_request", "message": str(exc), "code": None})

    job_id = None
    try:
        job_id = submit_recognition(
            challenge_type, task=task, grid=grid, image_data=[data_uri],
        )
        result = poll_recognition(challenge_type, job_id)
        duration = time.monotonic() - started
        logger.info(
            "nopecha_recognition: success challenge_type=%s task_id=%s request_id=%s duration=%.2fs",
            challenge_type, task_id, job_id, duration,
        )
        record_call_result(challenge_type, success=True, request_id=job_id)
        return _envelope("success", result=result, request_id=job_id)
    except NotImplementedError as exc:
        logger.info("nopecha_recognition: unsupported type (task_id=%s): %s", task_id, exc)
        record_call_result(challenge_type, success=False, reason="unsupported_challenge_type")
        return _envelope("error", error={"reason": "unsupported_challenge_type", "message": str(exc), "code": None})
    except NopeCHAError as exc:
        duration = time.monotonic() - started
        logger.info(
            "nopecha_recognition: failed challenge_type=%s task_id=%s reason=%s code=%s duration=%.2fs",
            challenge_type, task_id, exc.reason, exc.code, duration,
        )
        record_call_result(challenge_type, success=False, request_id=job_id, code=exc.code, reason=exc.reason)
        return _envelope(
            "error", request_id=job_id,
            error={"reason": exc.reason, "message": exc.message, "code": exc.code},
        )


def _nopecha_check() -> bool:
    """Gate: enabled unless NOPECHA_ENABLED is explicitly falsy.

    Empty/unset NOPECHA_API_KEY does NOT disable the tool -- that is free
    mode, which is a supported, documented NopeCHA tier, not a
    misconfiguration.
    """
    raw = None
    try:
        import os
        raw = os.getenv("NOPECHA_ENABLED")
        if raw is None:
            from hermes_cli.config import get_env_value
            raw = get_env_value("NOPECHA_ENABLED")
    except Exception:
        raw = None
    if raw is None or not str(raw).strip():
        return True
    from utils import is_truthy_value
    return is_truthy_value(raw, default=True)


registry.register(
    name="nopecha_recognition",
    toolset="browser-captcha",
    schema=NOPECHA_RECOGNITION_SCHEMA,
    handler=lambda args, **kw: nopecha_recognition(
        challenge_type=args.get("challenge_type", ""),
        task=args.get("task", ""),
        grid=args.get("grid"),
        image_path=args.get("image_path"),
        image_data=args.get("image_data"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_nopecha_check,
    emoji="🧩",
)
