"""Stdlib document-to-text extraction for ``read_file``.

Supports Jupyter notebooks, DOCX, XLSX, and PPTX without adding hard
dependencies.
When the optional ``firecrawl-anydoc`` package is installed (``pip install
firecrawl-anydoc``, imports as ``anydoc``), coverage widens to legacy Office
(.doc/.ppt/.xls), OpenDocument, RTF, EPUB, and PDF — converted to Markdown by
its Rust core. The stdlib extractors remain authoritative for their four
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
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

__all__ = ["EXTRACTABLE_EXTENSIONS", "ExtractionError", "extract_document_text", "is_extractable_document"]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx", ".pptx"})
# Formats handled only when the optional anydoc converter is installed.
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docm",
    ".ppt", ".pps", ".pot", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".pdf",
})
MAX_XLSX_BYTES = 50 * 1024 * 1024
# Refuse to convert huge documents. anydoc loads the whole file through its
# Rust core with no streaming, and the read_file char budget only applies
# after conversion, so an unbounded input can pin a tool turn and spike RAM.
MAX_ANYDOC_BYTES = 50 * 1024 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_PPTX_SLIDE_REL_TYPES = frozenset({
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
    "https://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
})
_SUPPORTED_MCE_NAMESPACES = frozenset({_NS_P, _NS_A})


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
            pass  # lazy install unavailable — fall through to a plain import
        try:
            _anydoc_module = importlib.import_module("anydoc")
        except Exception:  # ImportError or a broken native binding
            _anydoc_failed_at = time.monotonic()
            return None
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
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def _extract_anydoc(path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if size > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({size:,} bytes, limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        text = mod.to_markdown(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        # anydoc raises one ConvertError subclass per failure mode
        # (Unsupported, Malformed, Encrypted, ResourceLimit, MissingPart).
        # Any of them means "no meaningful text": fall back to the normal
        # path/binary handling rather than crash read_file.
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    return text.rstrip("\n") + "\n"


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def _extract_notebook(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            nb = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(nb, dict):
        raise ExtractionError("Notebook root is not an object")

    cells = nb.get("cells")
    if not isinstance(cells, list):
        cells = [
            cell
            for ws in nb.get("worksheets", [])
            if isinstance(ws, dict)
            for cell in ws.get("cells", [])
        ]
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    counts = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    out: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        out.extend((f"# ── {labels[typ]} cell{suffix} ──", _source_text(cell.get("source", "")).rstrip("\n"), ""))
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


# ---------------------------------------------------------------------------
# PowerPoint (.pptx)
# ---------------------------------------------------------------------------

def _extract_pptx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            parts = _pptx_slide_parts(zf, names)
            out: list[str] = []
            for index, part in enumerate(parts, 1):
                try:
                    lines = _slide_text(zf.read(part))
                except ET.ParseError:
                    continue
                out.append(f"# ── Slide {index} ──")
                out.extend(lines)
                if not any(line.strip() for line in lines):
                    out.append("(no text)")
                out.append("")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid PPTX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    if not out:
        raise ExtractionError("PPTX has no slides with content")
    return "\n".join(out).rstrip("\n") + "\n"


def _pptx_slide_parts(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    """Ordered slide part names.

    Preferred order comes from ``ppt/presentation.xml`` (``<p:sldId r:id=...>``)
    resolved through its ``.rels`` — this respects slide reordering, which the
    ``slideN.xml`` filenames do not. Falls back to numeric filename order when
    the presentation part or its rels are missing/malformed.
    """
    p, r = f"{{{_NS_P}}}", f"{{{_NS_REL}}}"
    try:
        root = ET.fromstring(zf.read("ppt/presentation.xml"))
        rids = [sld.get(f"{r}id") for sld in root.iter(f"{p}sldId")]
    except (KeyError, ET.ParseError):
        rids = []

    rels = _pptx_rels(zf, names)
    if rids:
        ordered: list[str] = []
        for rid in rids:
            target = rels.get(rid or "")
            if not target:
                break
            part = _pptx_part(target)
            if part not in names:
                break
            ordered.append(part)
        else:
            return ordered

    # Fallback: every ppt/slides/slideN.xml, in numeric (not lexical) order so
    # slide10 sorts after slide2.
    slides = [n for n in names if _SLIDE_RE.match(n)]
    return sorted(slides, key=lambda n: int(_SLIDE_RE.match(n).group(1)))


def _pptx_rels(zf: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    rels_path = "ppt/_rels/presentation.xml.rels"
    if rels_path not in names:
        return {}
    try:
        root = ET.fromstring(zf.read(rels_path))
    except ET.ParseError:
        return {}
    rel_tag = f"{{{_NS_PKG_REL}}}Relationship"
    rels: dict[str, str] = {}
    for rel in root.iter(rel_tag):
        rid = rel.get("Id", "")
        is_slide = rel.get("Type") in _PPTX_SLIDE_REL_TYPES
        is_internal = rel.get("TargetMode", "Internal") == "Internal"
        if not rid or not is_slide or not is_internal:
            continue
        rels[rid] = rel.get("Target", "")
    return rels


def _pptx_part(target: str) -> str:
    target = target.lstrip("/")
    return posixpath.normpath(target if target.startswith("ppt/") else f"ppt/{target}")


def _xml_with_choice_scopes(
    xml_bytes: bytes,
) -> tuple[ET.Element, dict[ET.Element, dict[str, str]]]:
    """Parse XML while retaining namespace scopes for MCE Choice elements."""
    pending: dict[str, str] = {}
    stack: list[dict[str, str]] = []
    scopes: dict[ET.Element, dict[str, str]] = {}
    choice_tag = f"{{{_NS_MC}}}Choice"
    parser = ET.iterparse(BytesIO(xml_bytes), events=("start-ns", "start", "end"))
    for event, value in parser:
        if event == "start-ns":
            prefix, uri = value
            pending[prefix or ""] = uri
        elif event == "start":
            scope = stack[-1] if stack else {}
            if pending:
                scope = scope.copy()
                scope.update(pending)
                pending.clear()
            stack.append(scope)
            if value.tag == choice_tag:
                scopes[value] = scope
        else:
            stack.pop()
    return parser.root, scopes


def _ignored_mce_elements(
    root: ET.Element,
    scopes: dict[ET.Element, dict[str, str]],
) -> set[ET.Element]:
    """Elements in unselected mc:AlternateContent branches."""
    mc = f"{{{_NS_MC}}}"
    choice_tag, fallback_tag = f"{mc}Choice", f"{mc}Fallback"
    ignored: set[ET.Element] = set()
    for alternate in root.iter(f"{mc}AlternateContent"):
        branches = [child for child in alternate if child.tag in {choice_tag, fallback_tag}]
        selected: ET.Element | None = None
        fallback: ET.Element | None = None
        for branch in branches:
            if branch.tag == fallback_tag:
                if fallback is None:
                    fallback = branch
                continue
            required = branch.get("Requires", "").split()
            scope = scopes.get(branch, {})
            requirements_met = required and all(
                scope.get(prefix) in _SUPPORTED_MCE_NAMESPACES
                for prefix in required
            )
            if selected is None and requirements_met:
                selected = branch
        if selected is None:
            selected = fallback
        for child in alternate:
            if child is not selected:
                ignored.update(child.iter())
    return ignored


def _slide_text(xml_bytes: bytes) -> list[str]:
    root, scopes = _xml_with_choice_scopes(xml_bytes)
    a = f"{{{_NS_A}}}"
    lines: list[str] = []
    # MCE AlternateContent is a choice, not a container whose branches should
    # all be traversed. Select the first Choice whose required namespaces this
    # extractor understands, or its Fallback, and ignore every node in the
    # other branches. This covers both whole-paragraph and inline alternatives.
    # Element identity keeps genuinely repeated text elsewhere.
    ignored_elements = _ignored_mce_elements(root, scopes)
    # Each DrawingML paragraph (<a:p>) — including those inside text boxes,
    # tables and placeholders — is one logical line. <a:t> holds the runs;
    # <a:br> is a soft line break within a paragraph.
    for para in root.iter(f"{a}p"):
        if para in ignored_elements:
            continue
        buf: list[str] = []
        for node in para.iter():
            if node in ignored_elements:
                continue
            if node.tag == f"{a}t":
                buf.append(node.text or "")
            elif node.tag == f"{a}br":
                buf.append("\n")
        lines.extend("".join(buf).split("\n"))
    return lines
