#!/usr/bin/env python3
"""Local document extraction helper backed by the doc-tools sidecar."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:9478"
_DEFAULT_STACK_DIR = Path.home() / "docker" / "doc-tools"
_DEFAULT_TIMEOUT = 120.0
_HEALTH_CACHE_TTL_SECONDS = 5.0
_health_cache: dict[str, tuple[float, bool]] = {}


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_document_tools_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw = load_config().get("document_tools", {})
        if isinstance(raw, dict):
            cfg = raw
    except Exception:
        cfg = {}

    base_url = str(
        cfg.get("base_url")
        or os.getenv("HERMES_DOC_TOOLS_BASE_URL")
        or _DEFAULT_BASE_URL
    ).strip().rstrip("/")
    stack_dir_raw = str(
        cfg.get("stack_dir")
        or os.getenv("HERMES_DOC_TOOLS_STACK_DIR")
        or _DEFAULT_STACK_DIR
    ).strip()
    intake_dir_raw = str(
        cfg.get("intake_dir") or os.getenv("HERMES_DOC_TOOLS_INTAKE_DIR") or ""
    ).strip()
    timeout = _coerce_positive_float(
        cfg.get("timeout") or os.getenv("HERMES_DOC_TOOLS_TIMEOUT"),
        _DEFAULT_TIMEOUT,
    )
    cleanup_after_extract = bool(cfg.get("cleanup_after_extract", True))

    stack_dir = _expand_path(stack_dir_raw)
    intake_dir = _expand_path(intake_dir_raw) if intake_dir_raw else (stack_dir / "intake").resolve()

    return {
        "base_url": base_url,
        "stack_dir": stack_dir,
        "intake_dir": intake_dir,
        "timeout": timeout,
        "cleanup_after_extract": cleanup_after_extract,
    }


def _is_probable_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _resolve_source_path(source: str) -> Path:
    candidate = Path(os.path.expanduser(source))
    if not candidate.is_absolute():
        base = Path(os.environ.get("TERMINAL_CWD", os.getcwd()))
        candidate = base / candidate
    return candidate.resolve()


def _container_intake_path(staged_path: Path, intake_dir: Path) -> str:
    relative = staged_path.resolve().relative_to(intake_dir.resolve())
    return f"/data/intake/{relative.as_posix()}"


def _stage_source_path(source_path: Path, intake_dir: Path) -> tuple[Path, bool]:
    resolved_source = source_path.resolve()
    resolved_intake = intake_dir.resolve()
    intake_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_source.relative_to(resolved_intake)
        return resolved_source, False
    except ValueError:
        pass

    suffix = resolved_source.suffix
    staged_name = f"{resolved_source.stem[:40]}-{uuid.uuid4().hex}{suffix}"
    staged_path = resolved_intake / staged_name
    shutil.copy2(resolved_source, staged_path)
    return staged_path, True


def _check_document_tools_health(base_url: str) -> bool:
    now = time.monotonic()
    cached = _health_cache.get(base_url)
    if cached and cached[0] > now:
        return cached[1]

    ok = False
    try:
        response = httpx.get(f"{base_url}/health", timeout=1.0)
        response.raise_for_status()
        payload = response.json()
        ok = bool(payload.get("ok"))
    except Exception:
        ok = False

    _health_cache[base_url] = (now + _HEALTH_CACHE_TTL_SECONDS, ok)
    return ok


def check_document_tools_requirements() -> bool:
    cfg = _load_document_tools_config()
    intake_dir = cfg["intake_dir"]
    if not intake_dir.exists() or not intake_dir.is_dir():
        return False
    return _check_document_tools_health(cfg["base_url"])


def _delegate_url_extract(source: str, max_chars: int | None = None) -> str:
    from tools.web_tools import web_extract_tool

    raw = web_extract_tool([source], "markdown")
    parsed = json.loads(raw)
    results = parsed.get("results") if isinstance(parsed, dict) else None
    first = results[0] if isinstance(results, list) and results else {}
    markdown = first.get("content") if isinstance(first, dict) else None
    if markdown is None:
        markdown = ""

    truncated = False
    if isinstance(max_chars, int) and max_chars > 0 and len(markdown) > max_chars:
        markdown = markdown[:max_chars]
        truncated = True

    error = first.get("error") if isinstance(first, dict) else None
    return tool_result(
        {
            "ok": not bool(error),
            "backend_used": "web_extract",
            "source": source,
            "source_kind": "url",
            "markdown": markdown,
            "structured_data": None,
            "metadata": {
                "title": first.get("title") if isinstance(first, dict) else None,
                "truncated": truncated,
            },
            "warnings": [
                "URL sources are handled by Hermes web_extract instead of the local doc-tools sidecar"
            ],
            "error": error,
        }
    )


def document_extract_tool(
    source: str,
    source_kind: str = "auto",
    backend: str = "auto",
    mode: str = "markdown",
    ocr: bool = False,
    structured: bool = False,
    extract_tables: bool = False,
    metadata_only: bool = False,
    max_chars: int | None = 200_000,
) -> str:
    if not source or not str(source).strip():
        return tool_error("source is required")

    normalized_source = str(source).strip()
    normalized_source_kind = (source_kind or "auto").strip().lower()
    if normalized_source_kind not in {"auto", "local_path", "url"}:
        return tool_error(f"Unsupported source_kind: {source_kind}")

    if normalized_source_kind == "url" or (
        normalized_source_kind == "auto" and _is_probable_url(normalized_source)
    ):
        return _delegate_url_extract(normalized_source, max_chars=max_chars)

    cfg = _load_document_tools_config()
    source_path = _resolve_source_path(normalized_source)
    if not source_path.exists():
        return tool_error(f"Source file not found: {source_path}")
    if not source_path.is_file():
        return tool_error(f"Source is not a file: {source_path}")

    staged_path: Path | None = None
    copied_to_intake = False

    try:
        staged_path, copied_to_intake = _stage_source_path(source_path, cfg["intake_dir"])
        container_source = _container_intake_path(staged_path, cfg["intake_dir"])
        payload = {
            "source": container_source,
            "source_kind": "local_path",
            "backend": backend,
            "mode": mode,
            "ocr": bool(ocr),
            "structured": bool(structured),
            "extract_tables": bool(extract_tables),
            "metadata_only": bool(metadata_only),
            "max_chars": max_chars,
        }
        response = httpx.post(
            f"{cfg['base_url']}/extract",
            json=payload,
            timeout=cfg["timeout"],
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            return tool_error("doc-tools returned a non-object JSON response")

        result.setdefault("metadata", {})
        if not isinstance(result["metadata"], dict):
            result["metadata"] = {"raw_metadata": result["metadata"]}
        result["original_source"] = str(source_path)
        result["staged_source"] = str(staged_path)
        result["cleanup_performed"] = bool(copied_to_intake and cfg["cleanup_after_extract"])
        return tool_result(result)
    except httpx.HTTPStatusError as exc:
        message = exc.response.text.strip() or str(exc)
        return tool_error(f"doc-tools request failed: {message}", status_code=exc.response.status_code)
    except Exception as exc:
        logger.exception("document_extract failed for %s", normalized_source)
        return tool_error(str(exc))
    finally:
        if (
            copied_to_intake
            and staged_path is not None
            and cfg["cleanup_after_extract"]
            and staged_path.exists()
        ):
            try:
                staged_path.unlink()
            except OSError:
                logger.warning("Failed to remove staged document: %s", staged_path)


_DEFAULT_DOCUMENT_AI_BASE_URL = "http://192.168.50.10:8098"
_DOCUMENT_AI_TOKEN_ENV_KEYS = (
    "HERMES_DOCUMENT_AI_TOKEN",
    "DOCUMENT_AI_TOKEN",
    "SPARK_DOCUMENT_AI_TOKEN",
)


def _load_document_ai_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw_cfg = load_config()
        raw = raw_cfg.get("document_ai", {})
        if not isinstance(raw, dict):
            raw = {}
        # Backward/nearby shape for profile-local configs that keep all document
        # knobs under document_tools.
        spark_raw = (raw_cfg.get("document_tools", {}) or {}).get("spark", {})
        if isinstance(spark_raw, dict):
            cfg.update(spark_raw)
        cfg.update(raw)
    except Exception:
        cfg = {}

    base_url = str(
        cfg.get("base_url")
        or os.getenv("HERMES_DOCUMENT_AI_BASE_URL")
        or os.getenv("DOCUMENT_AI_BASE_URL")
        or _DEFAULT_DOCUMENT_AI_BASE_URL
    ).strip().rstrip("/")
    timeout = _coerce_positive_float(
        cfg.get("timeout") or os.getenv("HERMES_DOCUMENT_AI_TIMEOUT"),
        _DEFAULT_TIMEOUT,
    )
    token = str(cfg.get("token") or "").strip()
    if not token:
        for key in _DOCUMENT_AI_TOKEN_ENV_KEYS:
            token = str(os.getenv(key) or "").strip()
            if token:
                break
    return {"base_url": base_url, "timeout": timeout, "token": token}


def _document_ai_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"X-Document-AI-Token": token}


def check_document_ai_requirements() -> bool:
    cfg = _load_document_ai_config()
    if not cfg.get("token"):
        return False
    return _check_document_tools_health(cfg["base_url"])


def _truncate_document_ai_result(result: Any, max_chars: int | None) -> Any:
    if not isinstance(max_chars, int) or max_chars <= 0:
        return result
    if isinstance(result, str):
        return result[:max_chars]
    if isinstance(result, list):
        return [_truncate_document_ai_result(item, max_chars) for item in result]
    if not isinstance(result, dict):
        return result

    remaining = max_chars
    out: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, str) and key.lower() in {
            "text", "markdown", "content", "body", "html", "document", "raw_text"
        }:
            out[key] = value[:remaining]
            remaining = max(0, remaining - len(out[key]))
            if len(value) > len(out[key]):
                out[f"{key}_truncated"] = True
        else:
            out[key] = value
    return out


def _document_ai_file_type(source_path: Path) -> int:
    """Return OpenWebUI/PaddleOCR fileType: 0 for PDF, 1 for image-like files."""
    return 0 if source_path.suffix.lower() == ".pdf" else 1


def _document_ai_payload(source_path: Path) -> dict[str, Any]:
    """Build the JSON contract expected by the Spark document-ai gateway.

    The gateway intentionally mimics OpenWebUI's PaddleOCR-VL loader: it
    accepts JSON with a base64 ``file`` field, not multipart/form-data.
    """
    encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
    return {
        "file": encoded,
        "fileType": _document_ai_file_type(source_path),
        "filename": source_path.name,
    }


def _extract_document_ai_markdown(payload: dict[str, Any]) -> str:
    result = payload.get("result") if isinstance(payload, dict) else None
    layout_results = None
    if isinstance(result, dict):
        layout_results = result.get("layoutParsingResults")
    if layout_results is None:
        layout_results = payload.get("layoutParsingResults")
    if not isinstance(layout_results, list):
        return ""

    chunks: list[str] = []
    for item in layout_results:
        if not isinstance(item, dict):
            continue
        markdown = item.get("markdown")
        text = markdown.get("text") if isinstance(markdown, dict) else markdown
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n\n".join(chunks)


def document_ai_extract_tool(
    source: str,
    redacted: bool = True,
    max_chars: int | None = 200_000,
) -> str:
    """Extract a local document through the authenticated Spark document-ai lane."""
    if not source or not str(source).strip():
        return tool_error("source is required")
    source_path = _resolve_source_path(str(source).strip())
    if not source_path.exists():
        return tool_error(f"Source file not found: {source_path}")
    if not source_path.is_file():
        return tool_error(f"Source is not a file: {source_path}")

    cfg = _load_document_ai_config()
    token = cfg.get("token") or ""
    if not token:
        return tool_error(
            "Spark document-ai token is not configured. Set HERMES_DOCUMENT_AI_TOKEN, "
            "DOCUMENT_AI_TOKEN, or document_ai.token in the active Hermes profile."
        )

    endpoint = "/layout-parsing/redacted" if redacted else "/layout-parsing"
    params = {} if redacted else {"redact": "false"}
    try:
        payload = _document_ai_payload(source_path)
        response = httpx.post(
            f"{cfg['base_url']}{endpoint}",
            headers=_document_ai_headers(token),
            json=payload,
            params=params,
            timeout=cfg["timeout"],
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            result = {"result": result}
        markdown = _extract_document_ai_markdown(result)
        if markdown:
            result.setdefault("markdown", markdown)
        result = _truncate_document_ai_result(result, max_chars)
        result.setdefault("metadata", {})
        if isinstance(result["metadata"], dict):
            result["metadata"].update(
                {
                    "backend_used": "spark_document_ai",
                    "base_url": cfg["base_url"],
                    "endpoint": endpoint,
                    "redacted": bool(redacted),
                    "source_filename": source_path.name,
                }
            )
        result["ok"] = bool(result.get("ok", True))
        return tool_result(result)
    except httpx.HTTPStatusError as exc:
        message = exc.response.text.strip() or str(exc)
        return tool_error(f"Spark document-ai request failed: {message}", status_code=exc.response.status_code)
    except Exception as exc:
        logger.exception("document_ai_extract failed for %s", source_path)
        return tool_error(str(exc))


DOCUMENT_EXTRACT_SCHEMA = {
    "name": "document_extract",
    "description": (
        "Extract text or structure from a local document using the localhost doc-tools sidecar. "
        "Hermes stages local files into the sidecar intake directory automatically. "
        "For URLs, this tool delegates to web_extract."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or URL to extract",
            },
            "source_kind": {
                "type": "string",
                "enum": ["auto", "local_path", "url"],
                "description": "Source type. auto detects http/https URLs and treats everything else as a local file path.",
                "default": "auto",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "markitdown", "docling"],
                "description": "Preferred extraction backend for local files",
                "default": "auto",
            },
            "mode": {
                "type": "string",
                "enum": ["markdown", "structured"],
                "description": "Return markdown text or structured Docling output",
                "default": "markdown",
            },
            "ocr": {
                "type": "boolean",
                "description": "Force OCR-oriented extraction for local files",
                "default": False,
            },
            "structured": {
                "type": "boolean",
                "description": "Request structured data from Docling for local files",
                "default": False,
            },
            "extract_tables": {
                "type": "boolean",
                "description": "Hint that table extraction matters for local files",
                "default": False,
            },
            "metadata_only": {
                "type": "boolean",
                "description": "Return metadata without markdown text when supported",
                "default": False,
            },
            "max_chars": {
                "type": ["integer", "null"],
                "description": "Maximum markdown characters to return",
                "minimum": 1,
                "default": 200000,
            },
        },
        "required": ["source"],
    },
}


DOCUMENT_AI_EXTRACT_SCHEMA = {
    "name": "document_ai_extract",
    "description": (
        "Extract a local document through the authenticated Spark document-ai / Docling gateway. "
        "Use this heavy lane for large, scanned, OCR-heavy, table/layout-heavy, or sensitive documents. "
        "Defaults to the redacted endpoint. Requires HERMES_DOCUMENT_AI_TOKEN or DOCUMENT_AI_TOKEN."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path to send to Spark document-ai.",
            },
            "redacted": {
                "type": "boolean",
                "description": "Use the redacted layout-parsing endpoint. Keep true unless the user explicitly needs unredacted extraction.",
                "default": True,
            },
            "max_chars": {
                "type": ["integer", "null"],
                "description": "Maximum characters to retain from large text fields in the returned JSON.",
                "minimum": 1,
                "default": 200000,
            },
        },
        "required": ["source"],
    },
}


registry.register(
    name="document_extract",
    toolset="document",
    schema=DOCUMENT_EXTRACT_SCHEMA,
    handler=lambda args, **kw: document_extract_tool(
        source=args.get("source", ""),
        source_kind=args.get("source_kind", "auto"),
        backend=args.get("backend", "auto"),
        mode=args.get("mode", "markdown"),
        ocr=args.get("ocr", False),
        structured=args.get("structured", False),
        extract_tables=args.get("extract_tables", False),
        metadata_only=args.get("metadata_only", False),
        max_chars=args.get("max_chars", 200_000),
    ),
    check_fn=check_document_tools_requirements,
    emoji="📑",
)

registry.register(
    name="document_ai_extract",
    toolset="document",
    schema=DOCUMENT_AI_EXTRACT_SCHEMA,
    handler=lambda args, **kw: document_ai_extract_tool(
        source=args.get("source", ""),
        redacted=bool(args.get("redacted", True)),
        max_chars=args.get("max_chars", 200_000),
    ),
    check_fn=check_document_ai_requirements,
    emoji="📄",
    max_result_size_chars=12000,
)

