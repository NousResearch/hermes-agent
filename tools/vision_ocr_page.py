"""vision_ocr_page — bounded OCR full-text retrieval (Stage-3 limited use).

HERMES_VISION_ROUTER_STAGE3_LIMITED_USE_OPERATING_MODE_DESIGN_V0_1:

- reads one deterministic page (<=``ocr_page_chars``, default 65536) of a
  session-registered OCR result by private handle;
- never auto-injects full text; never puts full text in repository-safe
  trace;
- handle is session-bound and revoked when the Router session ends;
- zero model calls: pure file read from the private cache.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

DEFAULT_OCR_PAGE_CHARS = 65536

VISION_OCR_PAGE_SCHEMA: Dict[str, Any] = {
    "name": "vision_ocr_page",
    "description": (
        "Retrieve one bounded page of a previously truncated OCR result. "
        "Requires the private_handle from the OCR envelope and an explicit "
        "page number. Returns page text, page bounds, remaining characters "
        "and a sha256. Never returns more than the configured page limit; "
        "full text is never auto-injected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "Opaque OCR result handle (private_handle from "
                               "the initial OCR envelope).",
            },
            "page": {
                "type": "integer",
                "description": "1-based page number (each page <= configured "
                               "limit).",
                "minimum": 1,
            },
        },
        "required": ["handle", "page"],
    },
}


def _resolve_ocr_path(handle: str) -> Optional[str]:
    """Session-bound private OCR result lookup. Unknown/expired -> None."""
    from tools.vision_session_state import vision_session_state

    h = handle.strip()
    if h.startswith("ocr://"):
        h = h[len("ocr://"):]
    return vision_session_state.resolve_ocr_result(f"ocr://{h}")


def _page_chars() -> int:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        from tools.vision_policy import resolve_vision_router_value
        return int(resolve_vision_router_value(cfg, "ocr_page_chars",
                                               DEFAULT_OCR_PAGE_CHARS))
    except Exception:  # noqa: BLE001
        return DEFAULT_OCR_PAGE_CHARS


def _handle_vision_ocr_page(args: Dict[str, Any], **kw: Any) -> str:
    """Registry handler (synchronous; zero model calls)."""
    from tools.vision_session_state import vision_session_state

    if not vision_session_state.enabled:
        return json.dumps({
            "execution_status": "POLICY_BLOCKED",
            "quality_decision": "NOT_EVALUATED",
            "logical_model_calls": 0,
            "error": "SESSION_DISABLED",
        }, ensure_ascii=False)

    handle = str(args.get("handle") or "").strip()
    try:
        page = int(args.get("page") or 1)
    except (TypeError, ValueError):
        page = 1
    if page < 1:
        page = 1
    if not handle:
        return json.dumps({
            "execution_status": "POLICY_BLOCKED",
            "quality_decision": "NOT_EVALUATED",
            "logical_model_calls": 0,
            "error": "SOURCE_DENIED: missing OCR handle",
        }, ensure_ascii=False)

    path = _resolve_ocr_path(handle)
    if not path:
        return json.dumps({
            "execution_status": "POLICY_BLOCKED",
            "quality_decision": "NOT_EVALUATED",
            "logical_model_calls": 0,
            "error": "SOURCE_DENIED: OCR handle not authorized or expired",
        }, ensure_ascii=False)

    try:
        with open(path, "r", encoding="utf-8") as f:
            full = f.read()
    except OSError:
        return json.dumps({
            "execution_status": "INVALID_RESPONSE",
            "quality_decision": "NOT_EVALUATED",
            "logical_model_calls": 0,
            "error": "OCR_RESULT_UNREADABLE",
        }, ensure_ascii=False)

    import hashlib

    limit = _page_chars()
    start = (page - 1) * limit
    end = start + limit
    page_text = full[start:end]
    total = len(full)
    return json.dumps({
        "execution_status": "SUCCESS",
        "quality_decision": "PASS",
        "logical_model_calls": 0,
        "handle": handle,
        "page": page,
        "page_start": start,
        "page_end": min(end, total),
        "page_chars": len(page_text),
        "total_chars": total,
        "remaining_chars": max(0, total - end),
        "sha256": hashlib.sha256(full.encode("utf-8")).hexdigest()[:16],
        "truncated": total > end,
        "page_text": page_text,
    }, ensure_ascii=False)


from tools.registry import registry  # noqa: E402

registry.register(
    name=VISION_OCR_PAGE_SCHEMA["name"],
    toolset="vision",
    schema=VISION_OCR_PAGE_SCHEMA,
    handler=_handle_vision_ocr_page,
    is_async=False,
    emoji="📄",
)
