"""First-class document reading for the agent.

The ``read_document`` tool extracts text from PDFs (and other pymupdf-supported
formats) in-process, so the agent reaches for a real tool instead of
hand-rolling ``python3 -c`` / ``pip install pymupdf`` in code-exec — the
"make the right path the path of least resistance" lever from decision
2026-06-04-opinionated-hsm-base-package.

pymupdf (``fitz``) ships eagerly in the image (Dockerfile ``--extra
documents``) and is also registered as a lazy feature (``documents.pymupdf``)
so older images self-heal on first use.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# Cap extracted text so a giant PDF can't blow up the context window. The
# registry also enforces ``max_result_size_chars``; this is a friendly,
# in-tool truncation with a clear marker.
_MAX_TEXT_CHARS = 200_000


def _import_fitz():
    """Return the ``fitz`` (pymupdf) module, lazy-installing if needed.

    Raises the original ImportError / FeatureUnavailable if it can't be made
    importable so the caller can surface an actionable message.
    """
    try:
        import fitz  # type: ignore
        return fitz
    except ImportError:
        from tools.lazy_deps import ensure  # local import: avoid hard dep at module load
        ensure("documents.pymupdf", prompt=False)
        import fitz  # type: ignore
        return fitz


def _resolve_local_path(path: str) -> str:
    """Strip a ``file://`` scheme and expand ``~`` so paths resolve locally."""
    resolved = path.strip()
    if resolved.startswith("file://"):
        resolved = resolved[len("file://"):]
    return os.path.expanduser(resolved)


def _parse_pages(spec: Optional[str], page_count: int) -> List[int]:
    """Parse a 1-indexed page spec into sorted 0-indexed page numbers.

    ``"1"`` -> [0]; ``"1-3"`` -> [0,1,2]; ``"1,3,5"`` -> [0,2,4];
    ``"2-3,5"`` -> [1,2,4]. Out-of-range pages are dropped. ``None`` or an
    unparseable spec returns every page.
    """
    if not spec or not spec.strip():
        return list(range(page_count))

    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                for p in range(lo, hi + 1):
                    wanted.add(p - 1)
            else:
                wanted.add(int(part) - 1)
        except ValueError:
            continue

    pages = sorted(p for p in wanted if 0 <= p < page_count)
    return pages or list(range(page_count))


def _read_document_sync(path: str, pages: Optional[str]) -> str:
    """Blocking extraction — run via :func:`asyncio.to_thread`."""
    local = _resolve_local_path(path)
    if not os.path.isfile(local):
        return f"❌ Document not found: {path}"

    try:
        fitz = _import_fitz()
    except Exception as exc:  # ImportError / FeatureUnavailable
        logger.warning("read_document: pymupdf unavailable: %s", exc)
        return (
            "❌ Document reading is unavailable: the pymupdf library could not "
            f"be loaded ({exc})."
        )

    try:
        doc = fitz.open(local)
    except Exception as exc:
        return f"❌ Could not open document {path!r}: {exc}"

    try:
        page_count = doc.page_count
        selected = _parse_pages(pages, page_count)
        chunks: List[str] = []
        for idx in selected:
            text = doc.load_page(idx).get_text().strip()
            if text:
                chunks.append(f"--- page {idx + 1} ---\n{text}")
        body = "\n\n".join(chunks).strip()
    finally:
        doc.close()

    if not body:
        # A document with no extractable text layer (scanned page, Canva-style
        # deck where text is drawn as vectors/images). Redirect to the OCR path
        # rather than returning a misleading empty success.
        return (
            f"⚠️ No extractable text found in {path} ({page_count} page(s)). "
            "This looks like an image-only / scanned document — its text lives "
            "in the page images, not a text layer. Use the vision tool "
            "(vision_analyze) on the rendered page image(s) to OCR it."
        )

    if len(body) > _MAX_TEXT_CHARS:
        body = body[:_MAX_TEXT_CHARS] + "\n\n…[truncated]"
    return body


async def read_document_tool(path: str, pages: Optional[str] = None) -> str:
    """Extract text from a local document (PDF, EPUB, XPS, …) via pymupdf.

    ``path``: local file path or ``file://`` URI.
    ``pages``: optional 1-indexed page spec (``"1"``, ``"1-3"``, ``"1,3,5"``).
    Returns the extracted text, or an actionable message on error / image-only
    documents. Never raises for the common failure cases.
    """
    return await asyncio.to_thread(_read_document_sync, path, pages)


def check_document_requirements() -> bool:
    """True when pymupdf is importable now, or lazy-installable on first use."""
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from tools.lazy_deps import LAZY_DEPS, _allow_lazy_installs
        return "documents.pymupdf" in LAZY_DEPS and _allow_lazy_installs()
    except Exception:
        return False


READ_DOCUMENT_SCHEMA = {
    "name": "read_document",
    "description": (
        "Extract the text of a document (PDF, EPUB, XPS, and similar) from a "
        "local file path. Use this whenever the user sends or references a "
        "document file (a PDF attachment, a .pdf path in their message, a file "
        "saved by another tool). Prefer this over writing your own code to "
        "parse the file. For image-only / scanned documents with no text "
        "layer, this tells you to fall back to vision_analyze for OCR."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local file path or file:// URI of the document to read.",
            },
            "pages": {
                "type": "string",
                "description": (
                    "Optional 1-indexed page selection, e.g. '1', '1-3', or "
                    "'1,3,5'. Omit to read the whole document."
                ),
            },
        },
        "required": ["path"],
    },
}


async def _handle_read_document(args: Dict[str, Any], **kw: Any) -> str:
    path = args.get("path") or args.get("document_url") or ""
    pages = args.get("pages")
    return await read_document_tool(path, pages)


registry.register(
    name="read_document",
    toolset="documents",
    schema=READ_DOCUMENT_SCHEMA,
    handler=_handle_read_document,
    check_fn=check_document_requirements,
    is_async=True,
    emoji="📄",
    max_result_size_chars=_MAX_TEXT_CHARS,
)
