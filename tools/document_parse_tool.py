#!/usr/bin/env python3
"""Document parsing tool backed by Hermes' parser abstraction."""

import json
from typing import Any, Dict

from agent.document_parsing import (
    create_document_screenshots,
    DocumentParserUnavailable,
    DocumentParsingError,
    parse_document,
)
from tools.registry import registry


DEFAULT_MAX_CHARS = 20_000


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 0 or len(text) <= max_chars:
        return text, False
    if max_chars == 0:
        return "", bool(text)
    return text[:max_chars] + "\n...[truncated]...", True


def document_parse_tool(
    path: str,
    backend: str = "auto",
    max_chars: int = DEFAULT_MAX_CHARS,
    include_pages: bool = False,
    include_text_items: bool = False,
    include_bounding_boxes: bool = False,
    ocr_enabled: bool | None = None,
    ocr_server_url: str | None = None,
    ocr_language: str | None = None,
    target_pages: str | None = None,
    dpi: int | None = None,
    precise_bounding_box: bool | None = None,
    preserve_small_text: bool | None = None,
    generate_screenshots: bool = False,
    screenshot_output_dir: str | None = None,
    screenshot_image_format: str | None = None,
) -> str:
    """Parse a local document and return normalized extracted text."""
    if not path or not str(path).strip():
        return json.dumps({"success": False, "error": "Path is required."}, ensure_ascii=False)

    max_chars = max(0, min(int(max_chars), 200_000))
    parse_options = {
        "ocr_enabled": ocr_enabled,
        "ocr_server_url": ocr_server_url,
        "ocr_language": ocr_language,
        "target_pages": target_pages,
        "dpi": dpi,
        "no_precise_bbox": None if precise_bounding_box is None else not bool(precise_bounding_box),
        "preserve_small_text": preserve_small_text,
    }
    screenshot_options = {
        "target_pages": target_pages,
        "dpi": dpi,
        "image_format": screenshot_image_format,
        "screenshot_output_dir": screenshot_output_dir,
    }

    try:
        parsed = parse_document(path=path, backend=backend, parse_options=parse_options)
    except (DocumentParsingError, DocumentParserUnavailable) as exc:
        return json.dumps(
            {
                "success": False,
                "error": str(exc),
                "path": path,
                "backend": backend,
            },
            ensure_ascii=False,
        )

    text, truncated = _truncate_text(parsed.text, max_chars)
    result: Dict[str, Any] = {
        "success": True,
        "path": parsed.source_path,
        "parser_backend": parsed.parser_backend,
        "text": text,
        "truncated": truncated,
        "metadata": parsed.metadata,
        "page_count": len(parsed.pages),
    }

    if include_pages:
        pages = []
        for page in parsed.pages:
            page_text, page_truncated = _truncate_text(page.text, max_chars)
            pages.append(
                {
                    "page_number": page.page_number,
                    "text": page_text,
                    "truncated": page_truncated,
                    "item_count": len(page.items),
                    "bounding_box_count": len(page.metadata.get("bounding_boxes", [])),
                    "width": page.width,
                    "height": page.height,
                }
            )
            if include_text_items:
                pages[-1]["text_items"] = [
                    {
                        "text": item.text,
                        "bbox": item.bbox,
                        "confidence": item.confidence,
                    }
                    for item in page.items
                ]
            if include_bounding_boxes:
                pages[-1]["bounding_boxes"] = page.metadata.get("bounding_boxes", [])
        result["pages"] = pages

    if generate_screenshots:
        try:
            screenshots = create_document_screenshots(
                path=path,
                backend=backend,
                screenshot_options=screenshot_options,
            )
            result["screenshots"] = [shot.to_dict() for shot in screenshots]
        except (DocumentParsingError, DocumentParserUnavailable) as exc:
            result["screenshot_error"] = str(exc)

    return json.dumps(result, ensure_ascii=False)


def check_document_parse_requirements() -> bool:
    """The tool is always available; unsupported files fail at runtime with a clear error."""
    return True


DOCUMENT_PARSE_SCHEMA = {
    "name": "document_parse",
    "description": (
        "Extract text from a local document path. Supports plain text-like files directly and "
        "can use LiteParse for PDFs, Office files, and images when installed. Use this when "
        "you need the contents of a local document that read_file cannot handle well, such as "
        "PDF, DOCX, PPTX, XLSX, or image-based documents. Returns normalized extracted text "
        "with optional per-page output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the local document file to parse.",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "liteparse", "basic"],
                "description": "Parsing backend. Use auto unless you need to force LiteParse or basic text parsing.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of extracted text to return. Default 20000.",
                "minimum": 0,
            },
            "include_pages": {
                "type": "boolean",
                "description": "When true, include per-page extracted text and metadata.",
            },
            "include_text_items": {
                "type": "boolean",
                "description": "When true and include_pages is enabled, include per-page text items with bbox/confidence metadata.",
            },
            "include_bounding_boxes": {
                "type": "boolean",
                "description": "When true and include_pages is enabled, include per-page bounding box arrays when available from LiteParse.",
            },
            "ocr_enabled": {
                "type": "boolean",
                "description": "Override OCR on/off for this parse call.",
            },
            "ocr_server_url": {
                "type": "string",
                "description": "Optional OCR server URL to use instead of the default LiteParse OCR flow.",
            },
            "ocr_language": {
                "type": "string",
                "description": "OCR language code such as en, fra, or deu.",
            },
            "target_pages": {
                "type": "string",
                "description": "Specific pages to parse or screenshot, for example '1-5,10'.",
            },
            "dpi": {
                "type": "integer",
                "description": "Rendering DPI for OCR and screenshots. Higher values can improve OCR quality but are slower.",
            },
            "precise_bounding_box": {
                "type": "boolean",
                "description": "Enable or disable precise bounding boxes for LiteParse-backed parsing.",
            },
            "preserve_small_text": {
                "type": "boolean",
                "description": "Preserve very small text when supported by LiteParse.",
            },
            "generate_screenshots": {
                "type": "boolean",
                "description": "When true, generate page screenshots for the parsed document using LiteParse.",
            },
            "screenshot_output_dir": {
                "type": "string",
                "description": "Optional directory where screenshots should be written.",
            },
            "screenshot_image_format": {
                "type": "string",
                "enum": ["png", "jpg"],
                "description": "Image format for generated screenshots.",
            },
        },
        "required": ["path"],
    },
}


registry.register(
    name="document_parse",
    toolset="documents",
    schema=DOCUMENT_PARSE_SCHEMA,
    handler=lambda args, **kw: document_parse_tool(
        path=args.get("path", ""),
        backend=args.get("backend", "auto"),
        max_chars=args.get("max_chars", DEFAULT_MAX_CHARS),
        include_pages=bool(args.get("include_pages", False)),
        include_text_items=bool(args.get("include_text_items", False)),
        include_bounding_boxes=bool(args.get("include_bounding_boxes", False)),
        ocr_enabled=args.get("ocr_enabled"),
        ocr_server_url=args.get("ocr_server_url"),
        ocr_language=args.get("ocr_language"),
        target_pages=args.get("target_pages"),
        dpi=args.get("dpi"),
        precise_bounding_box=args.get("precise_bounding_box"),
        preserve_small_text=args.get("preserve_small_text"),
        generate_screenshots=bool(args.get("generate_screenshots", False)),
        screenshot_output_dir=args.get("screenshot_output_dir"),
        screenshot_image_format=args.get("screenshot_image_format"),
    ),
    check_fn=check_document_parse_requirements,
    emoji="📄",
)
