"""Stdlib document-to-text extraction for ``read_file``.

Supports Jupyter notebooks, DOCX, and XLSX without adding hard dependencies.
When the optional ``firecrawl-anydoc`` package is installed (``pip install
firecrawl-anydoc``, imports as ``anydoc``), coverage widens to legacy Office
(.doc/.ppt/.xls), OpenDocument, RTF, EPUB, and PDF — converted to Markdown by
its Rust core. The stdlib extractors remain authoritative for their three
formats so behavior is identical whether or not anydoc is present.
Malformed documents raise :class:`ExtractionError`; callers can then fall back to
normal text/binary handling.
"""

from __future__ import annotations

import importlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

__all__ = [
    "EXTRACTABLE_EXTENSIONS",
    "ExtractionError",
    "extract_document_bytes",
    "extract_document_text",
    "is_extractable_document",
]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx"})
# Formats handled only when the optional anydoc converter is installed.
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".pdf",
})
MAX_XLSX_BYTES = 50 * 1024 * 1024
# Refuse to convert huge documents. anydoc loads the whole file through its
# Rust core with no streaming, and the read_file char budget only applies
# after conversion, so an unbounded input can pin a tool turn and spike RAM.
MAX_ANYDOC_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class ExtractionError(Exception):
    """Raised when a supported-looking document cannot be rendered as text."""


def _extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in EXTRACTABLE_EXTENSIONS:
        return ext
    if ext in ANYDOC_EXTENSIONS and _anydoc() is not None:
        return ext
    return ""


_ANYDOC_UNSET = object()
_anydoc_module: Any = _ANYDOC_UNSET
_anydoc_lock = threading.Lock()
# After a failed first load, wait this long before trying again. The attempt
# can shell out to pip, so retrying on every call would hammer the network
# in environments where the install can never succeed.
ANYDOC_RETRY_SECONDS = 300.0
_anydoc_failed_at: Optional[float] = None


def _anydoc() -> Optional[Any]:
    """Lazily import the optional anydoc converter; None when unavailable.

    A failed load is retried after :data:`ANYDOC_RETRY_SECONDS` rather than
    disabling extraction for the rest of the process, so one transient
    failure (network blip, pip race) does not stick in long-lived workers.
    """
    global _anydoc_module, _anydoc_failed_at
    if _anydoc_module is not _ANYDOC_UNSET:
        return _anydoc_module
    with _anydoc_lock:
        if _anydoc_module is not _ANYDOC_UNSET:
            return _anydoc_module
        if (
            _anydoc_failed_at is not None
            and time.monotonic() - _anydoc_failed_at < ANYDOC_RETRY_SECONDS
        ):
            return None
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            # prompt=False: read_file must never block on an install prompt.
            _lazy_ensure("tool.doc_extract", prompt=False)
        except Exception:
            _anydoc_failed_at = time.monotonic()
            return None
        try:
            _anydoc_module = importlib.import_module("anydoc")
        except Exception:  # ImportError or a broken native binding
            _anydoc_failed_at = time.monotonic()
            return None
        _anydoc_failed_at = None
    return _anydoc_module  # type: ignore[return-value]


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


def extract_document_text(path: str) -> str:
    ext = _extension(path)
    if ext == ".ipynb":
        return _extract_notebook(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".xlsx":
        return _extract_xlsx(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract a document already fetched across a file backend boundary."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, limit is {MAX_DOCUMENT_BYTES:,})"
        )
    ext = _extension(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc_bytes(data, path)
    if ext not in EXTRACTABLE_EXTENSIONS:
        raise ExtractionError(f"Unsupported document type: {path!r}")

    # The stdlib extractors are path-oriented. Materialize backend bytes in a
    # private host temp file, then remove it even when parsing fails.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
            fh.write(data)
            temp_path = fh.name
        return extract_document_text(temp_path)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _anydoc_missing_error(path: str) -> str:
    """Teaching error for anydoc-gated formats when the converter is absent.

    Response-time hint (#95681 pattern): the schema no longer lists the
    anydoc-gated formats or the availability caveat — a session that never
    touches a .doc/.odt/.epub never pays for the explanation, and one that
    does gets the full story here, with the fix.
    """
    return (
        f"Cannot convert {path!r}: this format needs the optional anydoc "
        "converter, which is not installed (install blocked or first "
        "attempt failed; retried every 5 minutes). Fix: `pip install "
        "firecrawl-anydoc` in Hermes's environment, or convert the file "
        "yourself via terminal (e.g. libreoffice --headless --convert-to "
        "txt)."
    )


def _coerce_hosted_ocr(value: Any) -> Optional[bool]:
    """Tri-state coercion for ``file_tools.hosted_ocr``: True/False/None.

    Hand-edited YAML regularly carries quoted strings ("true", "false")
    where the strict ``is True``/``is False`` branches would silently read
    them as auto: a quoted 'false' would not disable the outbound call and
    a quoted 'true' would not opt in. Uses the same token table as
    ``hermes config set`` (true/1/yes/on, false/0/no/off). Anything else
    stays None (auto).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    return None


def _hosted_ocr_config() -> tuple:
    """Resolve hosted-OCR settings: (enabled, api_key, api_url).

    Hosted OCR sends scanned PDFs (pages with no text layer) to
    Firecrawl's Parse endpoint through the bundled anydoc converter.
    Resolution of ``file_tools.hosted_ocr``:
      false  - fully disabled, even with a key.
      true   - enabled. With ``FIRECRAWL_API_KEY`` when set, keyless
               (free tier, no account) when not. anydoc itself falls
               back from the key to keyless.
      unset/null (default) - enabled only when ``FIRECRAWL_API_KEY`` is
               set. The key stays the consent signal for document
               egress, as decided in #97195.
    The Nous managed gateway is not used: its Parse proxy was
    live-probed broken (uniform HTTP 500, 2026-08-28) while
    scrape/search worked. Revisit when the gateway grows Parse support.
    Never raises.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY") or None
    enabled = api_key is not None
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        section = cfg.get("file_tools") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            setting = _coerce_hosted_ocr(section.get("hosted_ocr"))
            if setting is False:
                enabled = False
            elif setting is True:
                enabled = True
    except Exception:  # noqa: BLE001
        pass
    return enabled, api_key, None


def hosted_ocr_available() -> bool:
    """Public probe for read_file's schema line: is hosted OCR unlocked?

    Same resolution as ``_hosted_ocr_config``: a direct
    ``FIRECRAWL_API_KEY`` unlocks (the consent signal for document
    egress), ``file_tools.hosted_ocr: true`` unlocks without a key
    (keyless free tier), ``false`` disables both. Env and config probe
    only, no network at schema-build time. A route that fails at
    conversion time lands in the NEEDS-OCR warning.
    """
    try:
        enabled, _api_key, _api_url = _hosted_ocr_config()
        return enabled
    except Exception:  # noqa: BLE001
        return False


def _needs_ocr_warning(path: str, pages, hosted_error: str = "") -> str:
    """Typed replacement for the heuristic coverage note on full-OCR PDFs.

    Fired when anydoc raises NeedsOcrError and hosted OCR is disabled,
    unavailable, or failed. Maintainer-directed shape: hint at CHECKING
    for an OCR skill (never name one — none is guaranteed to exist), and
    never advertise the hosted_ocr config knob — when hosted fails or is
    absent, a skill or ignoring the gap are the paths that exist.
    """
    page_list = ", ".join(str(p) for p in pages) if pages else "unknown"
    msg = (
        f"[NEEDS OCR: pages {page_list} of this PDF are scanned images "
        "with no text layer — their content is MISSING below. "
    )
    if hosted_error:
        msg += f"Hosted OCR was attempted and failed ({hosted_error}). "
    msg += (
        "If the missing pages matter: render just those pages with "
        f"`pdftoppm -jpeg -r 150 -f <first> -l <last> '{path}' /tmp/page` "
        "and inspect via vision_analyze, or check whether an OCR skill is "
        "available (skills_list)."
    )
    return msg + "]\n"


def _extract_anydoc(path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(_anydoc_missing_error(path))
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if size > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({size:,} bytes, limit is {MAX_ANYDOC_BYTES:,})"
        )
    needs_ocr = getattr(mod, "NeedsOcrError", None)
    try:
        text = mod.to_markdown(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        if needs_ocr is not None and isinstance(exc, needs_ocr):
            # Typed scanned-pages signal (anydoc >= 0.2). Try hosted OCR
            # when a Firecrawl route exists; otherwise teach recovery.
            pages = list(getattr(exc, "pages", []) or [])
            enabled, api_key, api_url = _hosted_ocr_config()
            hosted_error = ""
            if enabled:
                try:
                    kwargs = {"ocr": "hosted"}
                    if api_key:
                        kwargs["api_key"] = api_key
                    if api_url:
                        kwargs["api_url"] = api_url
                    text = mod.to_markdown(path, **kwargs)
                    return text.rstrip("\n") + "\n"
                except Exception as hosted_exc:  # noqa: BLE001
                    hosted_error = f"{type(hosted_exc).__name__}: {hosted_exc}"
            # No route / disabled / hosted failed: whole doc is scans —
            # nothing to extract, so the warning IS the result.
            return _needs_ocr_warning(path, pages, hosted_error)
        # anydoc raises one ConvertError subclass per failure mode
        # (Unsupported, Malformed, Encrypted, ResourceLimit, MissingPart).
        # Any of them means "no meaningful text": fall back to the normal
        # path/binary handling rather than crash read_file.
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = _pdf_coverage_note(path)
        if note:
            # Prepend: read_file paginates the extraction, so a footer on a
            # long document would sit on a page the model may never fetch.
            # This heuristic note survives for PARTIAL coverage gaps —
            # documents with a text layer plus some scanned pages, which
            # convert without raising NeedsOcrError.
            text = note + text
    return text


# ── Scanned-PDF coverage detection ──────────────────────────────────
#
# anydoc (like every text-layer extractor) returns nothing for scanned
# image pages and emits no image placeholders or page markers, so a
# mostly-scanned PDF converts "successfully" into a few headers with
# empty bodies — silent data loss the model cannot detect. Count per-page
# text via poppler's pdftotext (form-feed page separators) and append a
# loud footer when a meaningful share of pages yielded no text.

# A page with fewer extracted characters than this is considered empty.
PDF_EMPTY_PAGE_CHARS = 20
# Warn when at least this many pages are empty AND they exceed the ratio,
# or when the absolute count alone is overwhelming.
PDF_COVERAGE_MIN_EMPTY = 2
PDF_COVERAGE_MIN_RATIO = 0.2
PDF_COVERAGE_ABSOLUTE_EMPTY = 10
PDF_PAGE_SCAN_TIMEOUT = 20.0


def _pdf_page_texts(path: str) -> Optional[list[str]]:
    """Per-page extracted text, or None when undeterminable."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        proc = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True,
            timeout=PDF_PAGE_SCAN_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    pages = proc.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # trailing form-feed artifact
    return pages or None


def _pdf_page_char_counts(path: str) -> Optional[list[int]]:
    """Per-page extracted-text char counts, or None when undeterminable."""
    pages = _pdf_page_texts(path)
    if pages is None:
        return None
    return [len(page.strip()) for page in pages]


def _page_ranges(pages: list[int]) -> str:
    """Compact 1-based range list, e.g. '2-29, 33-35, 42'."""
    parts = [f"{a}-{b}" if a != b else str(a) for a, b in _group_ranges(pages)]
    if len(parts) > 12:
        parts = parts[:12] + ["…"]
    return ", ".join(parts)


def _group_ranges(pages: list[int]) -> list[list[int]]:
    """Group sorted 1-based page numbers into [start, end] runs."""
    ranges: list[list[int]] = []
    for p in pages:
        if ranges and p == ranges[-1][1] + 1:
            ranges[-1][1] = p
        else:
            ranges.append([p, p])
    return ranges


# Cap the per-gap breakdown so a pathological PDF (hundreds of alternating
# text/scan pages) cannot balloon the warning. Ranges beyond the cap are
# summarized in one line.
PDF_GAP_MAP_MAX_ENTRIES = 20
_GAP_CONTEXT_CHARS = 60


def _gap_map(counts: list[int], texts: list[str], empty: list[int]) -> str:
    """Per-gap breakdown: each empty range labeled with the last text seen
    before it (usually a section divider/header page), so the agent can
    decide WHICH gaps it actually needs to read instead of OCRing all of
    them."""
    ranges = _group_ranges(empty)
    lines: list[str] = []
    for a, b in ranges[:PDF_GAP_MAP_MAX_ENTRIES]:
        label = ""
        # Walk back to the nearest preceding page with text.
        for prev in range(a - 2, -1, -1):
            if counts[prev] >= PDF_EMPTY_PAGE_CHARS:
                snippet = " ".join(texts[prev].split())[:_GAP_CONTEXT_CHARS]
                label = f' — after "{snippet}" (p{prev + 1})'
                break
        span = f"page {a}" if a == b else f"pages {a}-{b}"
        n = b - a + 1
        lines.append(f"  {span} ({n} page{'s' if n != 1 else ''}){label}")
    if len(ranges) > PDF_GAP_MAP_MAX_ENTRIES:
        rest = ranges[PDF_GAP_MAP_MAX_ENTRIES:]
        rest_pages = sum(b - a + 1 for a, b in rest)
        lines.append(f"  … {len(rest)} more gaps ({rest_pages} pages)")
    return "\n".join(lines)


def _pdf_coverage_note(path: str, display_path: Optional[str] = None) -> str:
    """A warning header when many PDF pages produced no text, else ''.

    ``path`` is the file scanned with pdftotext (may be a host temp file
    for backend-transferred bytes); ``display_path`` is the path shown in
    the recovery command — the one the agent's terminal can actually see.
    """
    texts = _pdf_page_texts(path)
    if not texts or len(texts) < 2:
        return ""
    counts = [len(page.strip()) for page in texts]
    empty = [i + 1 for i, n in enumerate(counts) if n < PDF_EMPTY_PAGE_CHARS]
    total = len(counts)
    if len(empty) < PDF_COVERAGE_MIN_EMPTY:
        return ""
    if (
        len(empty) / total < PDF_COVERAGE_MIN_RATIO
        and len(empty) < PDF_COVERAGE_ABSOLUTE_EMPTY
    ):
        return ""
    shown = display_path or path
    return (
        "[EXTRACTION COVERAGE WARNING: "
        f"{len(empty)} of {total} pages in this PDF yielded no text. "
        "Those pages are likely scanned images (or blank) — their content "
        "is MISSING from the extracted text below, even where section "
        "headers appear with empty bodies. Unreadable gaps, each labeled "
        "with the last text extracted before it:\n"
        f"{_gap_map(counts, texts, empty)}\n"
        "Decide which gaps you actually need — do NOT OCR or render "
        "everything. For the gaps that matter, render just that range with "
        f"`pdftoppm -jpeg -r 150 -f <first> -l <last> '{shown}' /tmp/page` "
        "and inspect each image with the vision_analyze tool, or use the "
        "ocr-and-documents skill (marker-pdf) for bulk OCR of large "
        "ranges.]\n"
    )


def _extract_anydoc_bytes(data: bytes, path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(_anydoc_missing_error(path))
    if len(data) > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        text = mod.to_markdown_bytes(data)
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = _pdf_coverage_note_from_bytes(data, path)
        if note:
            # Prepend: read_file paginates the extraction, so a footer on a
            # long document would sit on a page the model may never fetch.
            text = note + text
    return text


def _pdf_coverage_note_from_bytes(data: bytes, display_path: str) -> str:
    """Coverage note for backend-transferred PDF bytes.

    pdftotext is path-oriented, so materialize the bytes in a private host
    temp file for the scan; the recovery command still names
    ``display_path`` — the path the agent's terminal backend can see.
    """
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(data)
            temp_path = fh.name
        return _pdf_coverage_note(temp_path, display_path=display_path)
    except OSError:
        return ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def _human_size(n_bytes: int) -> str:
    return f"{round(n_bytes / 1024)} KB" if n_bytes >= 1024 else f"{n_bytes} B"


def _base64_bytes(payload: str) -> int:
    """Approximate decoded size of a base64 payload (whitespace ignored)."""
    clean = re.sub(r"[^0-9+/=A-Za-z]", "", payload)
    padding = min(2, len(clean) - len(clean.rstrip("=")))
    return max(0, (len(clean) * 3) // 4 - padding)


def _clean_stream_text(text: str) -> str:
    """Strip ANSI escapes and collapse ``\\r`` progress-bar rewrites.

    tqdm and friends redraw the same line via carriage returns; Jupyter
    renders only the final frame, so keeping the text after the last ``\\r``
    of each line reproduces what the notebook displays without the invisible
    intermediate frames.
    """
    from tools.ansi_strip import strip_ansi

    cleaned = strip_ansi(text).replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        frames = [frame for frame in line.split("\r") if frame]
        lines.append(frames[-1] if frames else "")
    return "\n".join(lines)


# Notebook outputs longer than this are tail-truncated per output block so a
# single runaway training log cannot flood the extracted text.
_MAX_OUTPUT_CHARS = 20_000


def _notebook_output_text(output: Any) -> str:
    """Render one notebook output as compact text.

    Keeps stream text, error tracebacks, and textual results; replaces
    token-heavy payloads (base64 images, HTML, widget state) with short
    sized placeholders. Handles both nbformat v4 output shapes and the
    legacy v3 ones (``pyout``/``pyerr``; data flat on the output dict).
    """
    if not isinstance(output, dict):
        return ""
    otype = output.get("output_type")

    if otype == "stream":
        body = _clean_stream_text(_source_text(output.get("text", "")))
        return body if body.strip() else ""

    if otype in {"error", "pyerr"}:
        traceback = output.get("traceback")
        tb_text = ""
        if isinstance(traceback, list):
            tb_text = _clean_stream_text(
                "\n".join(line for line in traceback if isinstance(line, str))
            )
        header = f"Error: {output.get('ename', '')}: {output.get('evalue', '')}".rstrip(": ")
        return f"{header}\n{tb_text}".rstrip()

    if otype in {"execute_result", "display_data", "pyout"}:
        data = output.get("data")
        if not isinstance(data, dict):
            # nbformat v3 stores mime data flat on the output dict.
            data = {}
            if isinstance(output.get("text"), (str, list)):
                data["text/plain"] = output["text"]
            for v3_key, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                                 ("svg", "image/svg+xml"), ("html", "text/html")):
                if v3_key in output:
                    data[mime] = output[v3_key]

        if "application/vnd.jupyter.widget-view+json" in data:
            return "[interactive widget — omitted]"

        # Prefer readable text: models consume text/plain (e.g. the pandas
        # twin of an HTML table) far better than markup.
        for mime in ("text/plain", "text/markdown"):
            if mime in data:
                body = _clean_stream_text(_source_text(data[mime]))
                if body.strip():
                    return body

        for mime, value in data.items():
            if isinstance(mime, str) and mime.startswith("image/"):
                size = _base64_bytes(_source_text(value))
                return f"[{mime} output — {_human_size(size)}, omitted]"

        if "text/html" in data:
            html = _source_text(data["text/html"])
            return f"[text/html output — {len(html):,} chars, omitted]"

        mimes = ", ".join(str(m) for m in data) or "unknown"
        return f"[{mimes} output — omitted]"

    return ""


def _notebook_outputs(cell: dict, jq_pointer: str = "", filename: str = "") -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    blocks = [text for text in (_notebook_output_text(o) for o in outputs) if text]
    if not blocks:
        return ""
    joined = "\n".join(blocks)
    if len(joined) > _MAX_OUTPUT_CHARS:
        omitted = len(joined) - _MAX_OUTPUT_CHARS
        hint = ""
        if jq_pointer and filename:
            hint = f" — full output: jq -r '{jq_pointer}' {filename}"
        joined = joined[:_MAX_OUTPUT_CHARS] + f"\n… [{omitted:,} output chars truncated{hint}]"
    return joined


def _extract_notebook(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            nb = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(nb, dict):
        raise ExtractionError("Notebook root is not an object")

    raw_cells = nb.get("cells")
    if isinstance(raw_cells, list):
        cells = [(f".cells[{i}].outputs", cell) for i, cell in enumerate(raw_cells)]
    else:
        cells = [
            (f".worksheets[{wi}].cells[{ci}].outputs", cell)
            for wi, ws in enumerate(nb.get("worksheets", []))
            if isinstance(ws, dict)
            for ci, cell in enumerate(ws.get("cells", []))
        ]
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    nb_name = os.path.basename(path)
    counts = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    out: list[str] = []
    for jq_pointer, cell in cells:
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        out.extend((f"# ── {labels[typ]} cell{suffix} ──", _source_text(cell.get("source", "")).rstrip("\n"), ""))
        if typ == "code":
            rendered = _notebook_outputs(cell, jq_pointer, nb_name)
            if rendered:
                out.extend((f"# ── Output (cell {counts[typ]}) ──", rendered.rstrip("\n"), ""))
    if not out:
        raise ExtractionError("Notebook contains no readable cells")
    return "\n".join(out).rstrip("\n") + "\n"


def _zip_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError as exc:
        raise ExtractionError(f"Missing {name}") from exc
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {name}: {exc}") from exc


def _extract_docx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            root = _zip_xml(zf, "word/document.xml")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid DOCX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    w = f"{{{_NS_W}}}"
    lines: list[str] = []
    for para in root.iter(f"{w}p"):
        buf: list[str] = []
        for node in para.iter():
            if node.tag == f"{w}t":
                buf.append(node.text or "")
            elif node.tag == f"{w}tab":
                buf.append("\t")
            elif node.tag in {f"{w}br", f"{w}cr"}:
                buf.append("\n")
        lines.extend("".join(buf).split("\n"))
    if not any(line.strip() for line in lines):
        raise ExtractionError("DOCX contains no extractable text")
    return "\n".join(lines).rstrip("\n") + "\n"


def _extract_xlsx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            shared = _shared_strings(zf, names)
            sheets = _workbook_sheets(zf)
            rels = _workbook_rels(zf, names)
            out: list[str] = []
            for name, state, rid in sheets:
                if state in {"hidden", "veryHidden"}:
                    continue
                part = _sheet_part(rels.get(rid, ""))
                if part not in names:
                    continue
                try:
                    rows = _sheet_rows(zf.read(part), shared)
                except ET.ParseError:
                    continue
                out.append(f"# ── Sheet: {name} ──")
                out.extend("\t".join(row) for row in rows)
                if not rows:
                    out.append("(empty)")
                out.append("")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid XLSX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    if not out:
        raise ExtractionError("XLSX has no visible sheets with content")
    return "\n".join(out).rstrip("\n") + "\n"


def _shared_strings(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except ET.ParseError:
        return []
    s = f"{{{_NS_S}}}"
    return ["".join(t.text or "" for t in item.iter(f"{s}t")) for item in root.iter(f"{s}si")]


def _workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    root = _zip_xml(zf, "xl/workbook.xml")
    s, r = f"{{{_NS_S}}}", f"{{{_NS_REL}}}"
    return [
        (sheet.get("name", "Sheet"), sheet.get("state", "visible"), sheet.get(f"{r}id", ""))
        for sheet in root.iter(f"{s}sheet")
    ]


def _workbook_rels(zf: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    rels_path = "xl/_rels/workbook.xml.rels"
    if rels_path not in names:
        return {}
    try:
        root = ET.fromstring(zf.read(rels_path))
    except ET.ParseError:
        return {}
    rel_tag = f"{{{_NS_PKG_REL}}}Relationship"
    return {rel.get("Id", ""): rel.get("Target", "") for rel in root.iter(rel_tag) if rel.get("Id")}


def _sheet_part(target: str) -> str:
    target = target.lstrip("/")
    return posixpath.normpath(target if target.startswith("xl/") else f"xl/{target}")


def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return max(idx - 1, 0)


def _sheet_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(xml_bytes)
    s = f"{{{_NS_S}}}"
    rows: list[list[str]] = []
    for row in root.iter(f"{s}row"):
        if len(rows) >= _MAX_XLSX_ROWS_PER_SHEET:
            break
        cells: dict[int, str] = {}
        max_col = -1
        for cell in row.iter(f"{s}c"):
            col = _col_index(cell.get("r", "")) if cell.get("r") else max_col + 1
            if col >= _MAX_XLSX_COLS:
                continue
            cells[col] = _cell_value(cell, shared, s)
            max_col = max(max_col, col)
        rows.append([cells.get(i, "") for i in range(max_col + 1)] if max_col >= 0 else [])
    while rows and not any(value.strip() for value in rows[-1]):
        rows.pop()
    return rows


def _cell_value(cell: ET.Element, shared: list[str], s: str) -> str:
    value = cell.findtext(f"{s}v") or ""
    typ = cell.get("t", "")
    if typ == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if typ == "inlineStr":
        inline = cell.find(f"{s}is")
        return "" if inline is None else "".join(t.text or "" for t in inline.iter(f"{s}t"))
    if typ == "b":
        return "TRUE" if value.strip() in {"1", "true", "TRUE"} else "FALSE"
    if typ == "e":
        return value or "#ERROR"
    return value
