"""Document-to-text extraction for ``read_file``: stdlib Jupyter/DOCX/XLSX (always
authoritative for those three), plus legacy Office/OpenDocument/RTF/EPUB/PDF when the
optional ``firecrawl-anydoc`` package (imports as ``anydoc``) is installed. Malformed
documents raise :class:`ExtractionError`; callers fall back to text/binary handling."""

from __future__ import annotations

import contextlib
import functools
import importlib
import itertools
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
from typing import Any, Callable, Iterator, Optional
from xml.etree import ElementTree as ET

__all__ = ["EXTRACTABLE_EXTENSIONS", "ExtractionError", "extract_document_bytes",
           "extract_document_text", "is_extractable_document"]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx"})
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docm", ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsm", ".xlsb", ".odt", ".ods", ".odp", ".rtf", ".epub", ".pdf"})
# anydoc loads whole files (no streaming); read_file's char budget applies only post-conversion.
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
    known = ext in EXTRACTABLE_EXTENSIONS or (ext in ANYDOC_EXTENSIONS and _anydoc() is not None)
    return ext if known else ""


_ANYDOC_UNSET = object()
_anydoc_module: Any = _ANYDOC_UNSET
_anydoc_lock = threading.Lock()
# Cooldown after a failed load: the attempt can shell out to pip, so retrying every call would
# hammer the network where install can't succeed.
ANYDOC_RETRY_SECONDS = 300.0
_anydoc_failed_at: Optional[float] = None


def _anydoc() -> Optional[Any]:
    """Lazily import the optional anydoc converter (None when unavailable; failures retried after
    ANYDOC_RETRY_SECONDS so one transient pip/network blip does not stick)."""
    global _anydoc_module, _anydoc_failed_at
    if _anydoc_module is not _ANYDOC_UNSET:
        return _anydoc_module
    with _anydoc_lock:
        if _anydoc_module is not _ANYDOC_UNSET:
            return _anydoc_module
        if (_anydoc_failed_at is not None
                and time.monotonic() - _anydoc_failed_at < ANYDOC_RETRY_SECONDS):
            return None
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("tool.doc_extract", prompt=False)  # read_file must never block on a prompt
            _anydoc_module = importlib.import_module("anydoc")
        except Exception:  # install failure, ImportError or a broken native binding
            _anydoc_failed_at = time.monotonic()
            return None
        _anydoc_failed_at = None
    return _anydoc_module  # type: ignore[return-value]


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


def _check_size(size: int, limit: int) -> None:
    if size > limit:
        raise ExtractionError(f"Document too large to convert ({size:,} bytes, limit is {limit:,})")


@contextlib.contextmanager
def _temp_copy(data: bytes, suffix: str) -> Iterator[str]:
    """Materialize backend bytes in a private host temp file; removed even when parsing fails."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(data)
    try:
        yield fh.name
    finally:
        with contextlib.suppress(OSError):
            os.unlink(fh.name)


def extract_document_text(path: str) -> str:
    ext = _extension(path)
    if ext in _STDLIB_EXTRACTORS:
        return _STDLIB_EXTRACTORS[ext](path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract a document already fetched across a file backend boundary."""
    _check_size(len(data), MAX_DOCUMENT_BYTES)
    ext = _extension(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc_bytes(data, path)
    if ext not in EXTRACTABLE_EXTENSIONS:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    with _temp_copy(data, ext) as temp_path:  # the stdlib extractors are path-oriented
        return _STDLIB_EXTRACTORS[ext](temp_path)


def _anydoc_missing_error(path: str) -> str:
    """Teaching text for anydoc-gated formats (not in the schema: only sessions hitting one pay).

    Response-time hint (#95681 pattern): the schema no longer lists the anydoc-gated formats or the
    availability caveat — a session that never touches a .doc/.odt/.epub never pays for the explanation, and
    one that does gets the full story here, with the fix.
    """
    return (
        f"Cannot convert {path!r}: this format needs the optional anydoc "
        "converter, which is not installed (install blocked or first "
        "attempt failed; retried every 5 minutes). Fix: `pip install "
        "firecrawl-anydoc` in Hermes's environment, or convert the file "
        "yourself via terminal (e.g. libreoffice --headless --convert-to "
        "txt).")


def _hosted_ocr_config() -> tuple:
    """(enabled, api_key, api_url); never raises, no network. Maintainer decision: the ONLY route
    is a direct ``FIRECRAWL_API_KEY`` (anydoc defaults api_url); the Nous gateway's Parse proxy
    live-probed broken, so it is NOT used. ``file_tools.hosted_ocr: false`` disables even with a
    key."""
    api_key = os.environ.get("FIRECRAWL_API_KEY") or None
    enabled = api_key is not None
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("file_tools")
        if isinstance(section, dict) and section.get("hosted_ocr") is False:
            enabled = False
    return enabled, api_key, None


def hosted_ocr_available() -> bool:
    """Probe for read_file's schema line; a key failing at conversion time surfaces in NEEDS-OCR."""
    return _hosted_ocr_config()[0]


def _needs_ocr_warning(path: str, pages, hosted_error: str = "") -> str:
    """NeedsOcrError result when hosted OCR is off/failed; hints at CHECKING for an OCR skill
    (never names one) and never advertises the hosted_ocr knob."""
    page_list = ", ".join(str(p) for p in pages) if pages else "unknown"
    hosted = f"Hosted OCR was attempted and failed ({hosted_error}). " if hosted_error else ""
    return (
        f"[NEEDS OCR: pages {page_list} of this PDF are scanned images "
        f"with no text layer — their content is MISSING below. {hosted}"
        "If the missing pages matter: render just those pages with "
        f"`pdftoppm -jpeg -r 150 -f <first> -l <last> '{path}' /tmp/page` "
        "and inspect via vision_analyze, or check whether an OCR skill is "
        "available (skills_list).]\n")


def _finalize_anydoc_text(text: Any, path: str, pdf_note: Callable[[], str]) -> str:
    """Normalize converter output; PDFs get the coverage note PREPENDED (read_file paginates, so a
    footer may never be fetched) — this covers PARTIAL scan gaps that raise no NeedsOcrError."""
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    return (pdf_note() if Path(path).suffix.lower() == ".pdf" else "") + text.rstrip("\n") + "\n"


def _ocr_scanned_pdf(mod: Any, path: str, exc: BaseException) -> str:
    """anydoc >= 0.2 scanned-pages signal: hosted OCR when a route exists, else teach recovery."""
    pages = list(getattr(exc, "pages", []) or [])
    enabled, api_key, api_url = _hosted_ocr_config()
    hosted_error = ""
    if enabled:
        try:
            extra = {k: v for k, v in (("api_key", api_key), ("api_url", api_url)) if v}
            return mod.to_markdown(path, ocr="hosted", **extra).rstrip("\n") + "\n"
        except Exception as hosted_exc:  # noqa: BLE001
            hosted_error = f"{type(hosted_exc).__name__}: {hosted_exc}"
    return _needs_ocr_warning(path, pages, hosted_error)  # whole doc is scans: the warning IS it


def _extract_anydoc(path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(_anydoc_missing_error(path))
    try:
        _check_size(os.path.getsize(path), MAX_ANYDOC_BYTES)
        text = mod.to_markdown(path)
    except ExtractionError:
        raise
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        needs_ocr = getattr(mod, "NeedsOcrError", None)
        if needs_ocr is not None and isinstance(exc, needs_ocr):
            return _ocr_scanned_pdf(mod, path, exc)
        # Any ConvertError subclass (Unsupported/Malformed/Encrypted/...) = "no meaningful text".
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    return _finalize_anydoc_text(text, path, lambda: _pdf_coverage_note(path))


def _extract_anydoc_bytes(data: bytes, path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(_anydoc_missing_error(path))
    _check_size(len(data), MAX_ANYDOC_BYTES)
    try:
        text = mod.to_markdown_bytes(data)
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    return _finalize_anydoc_text(text, path, lambda: _pdf_coverage_note_from_bytes(data, path))


# ── Scanned-PDF coverage: text-layer extractors return nothing for scanned pages, so a mostly
# scanned PDF converts "successfully" into silent data loss. Count per-page text via pdftotext.
PDF_EMPTY_PAGE_CHARS = 20  # fewer extracted chars than this = empty page
# Warn when empty pages reach both MIN_EMPTY and MIN_RATIO, or ABSOLUTE_EMPTY alone.
PDF_COVERAGE_MIN_EMPTY, PDF_COVERAGE_MIN_RATIO, PDF_COVERAGE_ABSOLUTE_EMPTY = 2, 0.2, 10
PDF_PAGE_SCAN_TIMEOUT = 20.0
PDF_GAP_MAP_MAX_ENTRIES = 20  # cap so alternating text/scan pages can't balloon the warning
_GAP_CONTEXT_CHARS = 60


def _pdf_page_texts(path: str) -> Optional[list[str]]:
    """Per-page extracted text, or None when undeterminable."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        proc = subprocess.run(
            ["pdftotext", path, "-"], capture_output=True, timeout=PDF_PAGE_SCAN_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.decode("utf-8", errors="replace") if proc.returncode == 0 else ""
    pages = out.split("\f") if out else []
    if pages and not pages[-1].strip():
        pages.pop()  # trailing form-feed artifact
    return pages or None


def _gap_map(counts: list[int], texts: list[str], empty: list[int]) -> str:
    """Per-gap breakdown labeled with the text before each gap, so the agent picks which to OCR."""
    # Sorted 1-based page numbers -> (start, end) runs; consecutive pages share ``page - index``.
    runs = [list(g) for _k, g in itertools.groupby(enumerate(empty), lambda e: e[1] - e[0])]
    ranges = [(run[0][1], run[-1][1]) for run in runs]
    lines: list[str] = []
    for a, b in ranges[:PDF_GAP_MAP_MAX_ENTRIES]:
        label = ""
        for prev in range(a - 2, -1, -1):  # nearest preceding page with text
            if counts[prev] >= PDF_EMPTY_PAGE_CHARS:
                snippet = " ".join(texts[prev].split())[:_GAP_CONTEXT_CHARS]
                label = f' — after "{snippet}" (p{prev + 1})'
                break
        span = f"page {a}" if a == b else f"pages {a}-{b}"
        n = b - a + 1
        lines.append(f"  {span} ({n} page{'s' if n != 1 else ''}){label}")
    if len(ranges) > PDF_GAP_MAP_MAX_ENTRIES:
        rest = ranges[PDF_GAP_MAP_MAX_ENTRIES:]
        lines.append(f"  … {len(rest)} more gaps ({sum(b - a + 1 for a, b in rest)} pages)")
    return "\n".join(lines)


def _pdf_coverage_note(path: str, display_path: Optional[str] = None) -> str:
    """Warning header when many pages yielded no text, else ''. ``display_path`` (default ``path``,
    which may be a host temp file) is what the recovery command shows."""
    texts = _pdf_page_texts(path)
    if not texts or len(texts) < 2:
        return ""
    counts = [len(page.strip()) for page in texts]
    empty = [i + 1 for i, n in enumerate(counts) if n < PDF_EMPTY_PAGE_CHARS]
    total = len(counts)
    n_empty = len(empty)
    enough = n_empty / total >= PDF_COVERAGE_MIN_RATIO or n_empty >= PDF_COVERAGE_ABSOLUTE_EMPTY
    if n_empty < PDF_COVERAGE_MIN_EMPTY or not enough:
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
        "ranges.]\n")


def _pdf_coverage_note_from_bytes(data: bytes, display_path: str) -> str:
    """Coverage note for backend PDF bytes via a host temp copy (pdftotext needs a path)."""
    with contextlib.suppress(OSError), _temp_copy(data, ".pdf") as temp_path:
        return _pdf_coverage_note(temp_path, display_path=display_path)
    return ""


def _joined(lines: list[str], empty_error: str) -> str:
    """Join extracted lines with a single trailing newline; raise when nothing non-blank."""
    if not any(line.strip() for line in lines):
        raise ExtractionError(empty_error)
    return "\n".join(lines).rstrip("\n") + "\n"


def _source_text(source) -> str:
    """Notebook source/text fields are a str or a list of str fragments."""
    if isinstance(source, list):
        source = "".join(item for item in source if isinstance(item, str))
    return source if isinstance(source, str) else ""


def _human_size(n_bytes: int) -> str:
    return f"{round(n_bytes / 1024)} KB" if n_bytes >= 1024 else f"{n_bytes} B"


def _base64_bytes(payload: str) -> int:
    """Approximate decoded size of a base64 payload (whitespace ignored)."""
    clean = re.sub(r"[^0-9+/=A-Za-z]", "", payload)
    return max(0, (len(clean) * 3) // 4 - min(2, len(clean) - len(clean.rstrip("="))))


def _clean_stream_text(text: str) -> str:
    """Strip ANSI escapes; keep only the final ``\\r`` frame of each line (tqdm redraws)."""
    from tools.ansi_strip import strip_ansi
    return "\n".join(([f for f in line.split("\r") if f] or [""])[-1]
                     for line in strip_ansi(text).replace("\r\n", "\n").split("\n"))


_MAX_OUTPUT_CHARS = 20_000  # per code cell, so one runaway training log cannot flood the extraction
# nbformat v3 stores mime data flat on the output dict under these keys.
_V3_MIME_KEYS = (("png", "image/png"), ("jpeg", "image/jpeg"), ("svg", "image/svg+xml"), ("html", "text/html"))


def _notebook_output_text(output: Any) -> str:
    """One notebook output as compact text: stream/traceback/textual results kept; token-heavy
    payloads (images, HTML, widgets) become sized placeholders. Handles v4 and legacy v3 shapes."""
    if not isinstance(output, dict):
        return ""
    otype = output.get("output_type")
    if otype == "stream":
        body = _clean_stream_text(_source_text(output.get("text", "")))
        return body if body.strip() else ""
    if otype in {"error", "pyerr"}:
        tb = output.get("traceback")
        tb_text = _clean_stream_text("\n".join(filter(lambda l: isinstance(l, str), tb))
                                     if isinstance(tb, list) else "")
        header = f"Error: {output.get('ename', '')}: {output.get('evalue', '')}".rstrip(": ")
        return f"{header}\n{tb_text}".rstrip()
    if otype not in {"execute_result", "display_data", "pyout"}:
        return ""
    data = output.get("data")
    if not isinstance(data, dict):  # legacy v3: mime payloads sit flat on the output dict
        data = {"text/plain": output["text"]} if isinstance(output.get("text"), (str, list)) else {}
        data.update((mime, output[k]) for k, mime in _V3_MIME_KEYS if k in output)
    if "application/vnd.jupyter.widget-view+json" in data:
        return "[interactive widget — omitted]"
    for mime in ("text/plain", "text/markdown"):  # models consume text far better than markup
        body = _clean_stream_text(_source_text(data[mime])) if mime in data else ""
        if body.strip():
            return body
    for mime, value in data.items():
        if isinstance(mime, str) and mime.startswith("image/"):
            return f"[{mime} output — {_human_size(_base64_bytes(_source_text(value)))}, omitted]"
    if "text/html" in data:
        return f"[text/html output — {len(_source_text(data['text/html'])):,} chars, omitted]"
    return f"[{', '.join(str(m) for m in data) or 'unknown'} output — omitted]"


def _notebook_outputs(cell: dict, jq_pointer: str = "", filename: str = "") -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    joined = "\n".join(filter(None, map(_notebook_output_text, outputs)))
    if len(joined) <= _MAX_OUTPUT_CHARS:
        return joined
    hint = f" — full output: jq -r '{jq_pointer}' {filename}" if jq_pointer and filename else ""
    omitted = len(joined) - _MAX_OUTPUT_CHARS
    return joined[:_MAX_OUTPUT_CHARS] + f"\n… [{omitted:,} output chars truncated{hint}]"


_CELL_LABELS = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}


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
    else:  # nbformat v3: cells live under worksheets
        cells = [
            (f".worksheets[{wi}].cells[{ci}].outputs", cell)
            for wi, ws in enumerate(nb.get("worksheets", [])) if isinstance(ws, dict)
            for ci, cell in enumerate(ws.get("cells", []))]
    if not cells:
        raise ExtractionError("Notebook contains no cells")
    nb_name = os.path.basename(path)
    counts = dict.fromkeys(_CELL_LABELS, 0)
    out: list[str] = []
    for jq_pointer, cell in cells:
        typ = cell.get("cell_type") if isinstance(cell, dict) else None
        if typ not in _CELL_LABELS:
            continue
        counts[typ] += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        source = _source_text(cell.get("source", "")).rstrip("\n")
        out += [f"# ── {_CELL_LABELS[typ]} cell{suffix} ──", source, ""]
        rendered = _notebook_outputs(cell, jq_pointer, nb_name) if typ == "code" else ""
        if rendered:
            out += [f"# ── Output (cell {counts[typ]}) ──", rendered.rstrip("\n"), ""]
    return _joined(out, "Notebook contains no readable cells")


@contextlib.contextmanager
def _open_zip(path: str, kind: str) -> Iterator[zipfile.ZipFile]:
    """Open an OOXML package; bad-zip/OS failures (body included) become ExtractionError."""
    try:
        with zipfile.ZipFile(path) as zf:
            yield zf
    except (zipfile.BadZipFile, OSError) as exc:
        bad_zip = isinstance(exc, zipfile.BadZipFile)
        raise ExtractionError(f"Not a valid {kind}: {exc}" if bad_zip else str(exc)) from exc


def _zip_xml(zf: zipfile.ZipFile, name: str, optional: bool = False) -> Any:
    """Parse a package part; ``optional`` parts yield an empty element when absent or malformed."""
    try:
        return ET.fromstring(zf.read(name))
    except (KeyError, ET.ParseError) as exc:
        if optional:
            return ET.Element("missing")
        raise ExtractionError(
            f"Missing {name}" if isinstance(exc, KeyError) else f"Malformed XML in {name}: {exc}"
        ) from exc


def _extract_docx(path: str) -> str:
    with _open_zip(path, "DOCX") as zf:
        root = _zip_xml(zf, "word/document.xml")
    w = f"{{{_NS_W}}}"
    breaks = {f"{w}tab": "\t", f"{w}br": "\n", f"{w}cr": "\n"}
    lines: list[str] = []
    for para in root.iter(f"{w}p"):
        text = "".join(
            (n.text or "") if n.tag == f"{w}t" else breaks.get(n.tag, "") for n in para.iter())
        lines.extend(text.split("\n"))
    return _joined(lines, "DOCX contains no extractable text")


def _extract_xlsx(path: str) -> str:
    s, r, pr = f"{{{_NS_S}}}", f"{{{_NS_REL}}}", f"{{{_NS_PKG_REL}}}"
    with _open_zip(path, "XLSX") as zf:
        names = set(zf.namelist())
        sst = _zip_xml(zf, "xl/sharedStrings.xml", optional=True)
        shared = ["".join(t.text or "" for t in item.iter(f"{s}t")) for item in sst.iter(f"{s}si")]
        rels_root = _zip_xml(zf, "xl/_rels/workbook.xml.rels", optional=True)
        rels = {rel.get("Id", ""): rel.get("Target", "")
                for rel in rels_root.iter(f"{pr}Relationship") if rel.get("Id")}
        out: list[str] = []
        for sheet in _zip_xml(zf, "xl/workbook.xml").iter(f"{s}sheet"):
            target = rels.get(sheet.get(f"{r}id", ""), "").lstrip("/")
            part = posixpath.normpath(target if target.startswith("xl/") else f"xl/{target}")
            if sheet.get("state", "visible") in {"hidden", "veryHidden"} or part not in names:
                continue
            with contextlib.suppress(ET.ParseError):
                rows = _sheet_rows(zf.read(part), shared)
                out += [f"# ── Sheet: {sheet.get('name', 'Sheet')} ──",
                        *(["\t".join(row) for row in rows] or ["(empty)"]), ""]
    return _joined(out, "XLSX has no visible sheets with content")


def _col_index(ref: str) -> int:
    """0-based column of a cell ref: ``A1`` -> 0, ``AB7`` -> 27 (bijective base-26 letters)."""
    idx = functools.reduce(lambda acc, ch: acc * 26 + ord(ch.upper()) - ord("A") + 1,
                           itertools.takewhile(str.isalpha, ref), 0)
    return max(idx - 1, 0)


def _sheet_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(xml_bytes)
    s = f"{{{_NS_S}}}"
    rows: list[list[str]] = []
    for row in itertools.islice(root.iter(f"{s}row"), _MAX_XLSX_ROWS_PER_SHEET):
        cells: dict[int, str] = {}
        max_col = -1
        for cell in row.iter(f"{s}c"):
            col = _col_index(cell.get("r", "")) if cell.get("r") else max_col + 1
            if col < _MAX_XLSX_COLS:
                cells[col] = _cell_value(cell, shared, s)
                max_col = max(max_col, col)
        rows.append([cells.get(i, "") for i in range(max_col + 1)])
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
    return (value or "#ERROR") if typ == "e" else value


# Extension -> stdlib extractor; anydoc formats fall through in extract_document_text.
_STDLIB_EXTRACTORS: dict[str, Callable[[str], str]] = {
    ".ipynb": _extract_notebook, ".docx": _extract_docx, ".xlsx": _extract_xlsx}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

MAX_XLSX_BYTES = 50 * 1024 * 1024
# ---- END PLUGIN-COMPAT ----
