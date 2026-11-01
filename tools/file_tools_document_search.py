"""Content search inside PDFs and Office documents for ``search_files``.

ripgrep skips binary files, so a content search over a folder of paperwork came
back empty while ``read_file`` could extract the very text being searched for
(#101427). This extracts the documents in scope with read_file's own extractor
and runs ripgrep itself over the extracted text, so the pattern means exactly
what it means for the ordinary search (same regex engine, linear time, killable
on the time budget), and reports line numbers that match ``read_file`` on the
same path: ``read_file(path, offset=<line>)`` opens the passage.

The shape follows Perplexity Computer's Projects workspace (Oct 2026 engineering
post "Building shared workspaces for people and agents"): parsed text is kept
outside the files, keyed by the source document's content hash, and agents search
it with their ordinary file tools instead of a semantic index; documents whose
text cannot be produced are named, never silently dropped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from tools import interrupt as tool_interrupt
from tools.file_operations_search import _pattern_has_regex_newline
from tools.read_extract import (
    ANYDOC_EXTENSIONS, EXTRACTABLE_EXTENSIONS, ExtractionError, extract_document_bytes,
    is_extractable_document)

logger = logging.getLogger(__name__)

# .ipynb is JSON text rg already reads; SQLite renders a schema dump, not prose.
SEARCHABLE_DOCUMENT_EXTENSIONS = frozenset(
    (EXTRACTABLE_EXTENSIONS | ANYDOC_EXTENSIONS) - {".ipynb", ".db", ".sqlite", ".sqlite3"})
MAX_SEARCH_DOCUMENTS = 50
MAX_SEARCH_DOCUMENT_BYTES = 20 * 1024 * 1024
SEARCH_BUDGET_SECONDS = 20.0
MAX_DOCUMENT_MATCHES = 50
_SNIPPET_RADIUS = 120
_HINT_MAX_FILES = 10
_CACHE_MAX_CHARS = 32_000_000
_CACHE_MAX_ENTRIES = 1024

# sha256(bytes) + path -> (text, error). The path is part of the key because the
# converter is chosen by extension and PDF coverage notes name the path; content in
# the key means an edited file is a new entry. Failures are cached too (a scanned PDF
# is not re-parsed on every search). In-memory only: extracted text is never persisted.
_cache: OrderedDict[str, tuple[str, str]] = OrderedDict()
_cache_chars = 0
_cache_lock = threading.Lock()


def _cached_extract(data: bytes, path: str) -> tuple[str, str]:
    global _cache_chars
    key = f"{hashlib.sha256(data).hexdigest()}\0{path}"
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    try:
        entry = (extract_document_bytes(data, path), "")
    except ExtractionError as exc:
        entry = ("", str(exc).split(". ")[0][:160] or "extraction failed")
    size = len(entry[0]) + len(entry[1])
    with _cache_lock:
        if key not in _cache and size <= _CACHE_MAX_CHARS:
            _cache[key] = entry
            _cache_chars += size
            while _cache_chars > _CACHE_MAX_CHARS or len(_cache) > _CACHE_MAX_ENTRIES:
                _, (old_text, old_error) = _cache.popitem(last=False)
                _cache_chars -= len(old_text) + len(old_error)
    return entry


def _host_rg() -> Optional[str]:
    """The ripgrep this process can run: pm's pinned store copy first (nothing puts it
    on PATH), then PATH."""
    try:
        from pm import installed_package

        installed = installed_package("ripgrep")
        if installed is not None and installed.binary:
            return str(installed.binary)
    except Exception:
        logger.debug("pm ripgrep lookup failed", exc_info=True)
    return shutil.which("rg")


def _glob_can_select_documents(file_glob: Optional[str]) -> bool:
    """False only when ``file_glob`` pins a literal extension no document has (``*.py``),
    so a narrowed code search skips the document walk; the glob itself is applied by
    the backend's own walk."""
    tail = (file_glob or "").replace("\\", "/").rsplit("/", 1)[-1]
    ext = re.search(r"\.([A-Za-z0-9]+)$", tail)
    return ext is None or f".{ext.group(1).lower()}" in SEARCHABLE_DOCUMENT_EXTENSIONS


def _extract_all(file_ops: Any, docs: list[str], deadline: float) -> tuple[dict[str, str], dict[str, str]]:
    """``(texts, skipped)``: extracted text per document, reason per unsearchable one."""
    texts: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for doc in docs:
        if tool_interrupt.is_interrupted() or time.monotonic() > deadline:
            skipped[doc] = "time budget reached"
        elif not is_extractable_document(doc):
            skipped[doc] = "converter unavailable (run `hermes pm repair`)"
        else:
            raw = file_ops.read_file_bytes(doc, max_bytes=MAX_SEARCH_DOCUMENT_BYTES)
            if raw.error or raw.base64_content is None:
                skipped[doc] = (raw.error or "unreadable").split(" (")[0]
                continue
            text, error = _cached_extract(base64.b64decode(raw.base64_content), doc)
            if error:
                skipped[doc] = error
            else:
                texts[doc] = text
    return texts, skipped


def _rg_args(rg: str, pattern: str, output_mode: str) -> list[str]:
    mode = {"files_only": ["--files-with-matches"], "count": ["--count-matches", "--with-filename"]}
    args = [rg, "--no-config", "--hidden", "--no-ignore",
            *mode.get(output_mode, ["--json", "--max-count", str(MAX_DOCUMENT_MATCHES)])]
    if _pattern_has_regex_newline(pattern):
        args.append("--multiline")
    return [*args, "-e", pattern, "--"]


def _snippet(data: dict) -> str:
    text = data["lines"].get("text", "")
    sub = (data.get("submatches") or [{}])[0]
    raw = text.encode("utf-8")
    start, end = sub.get("start", 0), sub.get("end", 0)  # byte offsets
    lo, hi = max(0, start - _SNIPPET_RADIUS), min(len(raw), end + _SNIPPET_RADIUS)
    body = raw[lo:hi].decode("utf-8", errors="ignore").replace("\n", " ").strip()
    return ("…" if lo else "") + body + ("…" if hi < len(raw) else "")


def _rg_over_texts(rg: str, texts: dict[str, str], pattern: str, output_mode: str,
                   timeout: float) -> dict[str, Any]:
    """Run rg over the extracted texts; ``{doc: [(line, snippet)] | count | True}``."""
    with tempfile.TemporaryDirectory(prefix="hermes-docsearch-") as tmp:
        by_name: dict[str, str] = {}
        for i, (doc, text) in enumerate(texts.items()):
            name = f"{i:04d}.txt"
            by_name[name] = doc
            # splitlines() is read_file's numbering; rg counts \n, so normalise to it.
            with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as fh:  # windows-footgun: ok — a write
                fh.write("\n".join(text.splitlines()) + "\n")
        proc = subprocess.run(
            [*_rg_args(rg, pattern, output_mode), tmp], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError((proc.stderr or "ripgrep failed").strip().splitlines()[-1][:160])
    found: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        if output_mode == "files_only":
            found[by_name.get(os.path.basename(line), line)] = True
        elif output_mode == "count":
            path, _sep, count = line.rpartition(":")
            if count.isdigit():
                found[by_name.get(os.path.basename(path), path)] = int(count)
        else:
            event = json.loads(line)
            if event.get("type") == "match":
                data = event["data"]
                doc = by_name.get(os.path.basename(data["path"].get("text", "")))
                if doc:
                    found.setdefault(doc, []).append((data["line_number"], _snippet(data)))
    return found


def _format_hits(hits: dict[str, Any], output_mode: str,
                 redact: Callable[[str, str], str]) -> dict[str, Any]:
    if output_mode == "files_only":
        out: dict[str, Any] = {"document_files": list(hits)}
        total = len(hits)
    elif output_mode == "count":
        out = {"document_counts": hits}
        total = sum(hits.values())
    else:
        rows: list[str] = []
        total, shown = sum(len(v) for v in hits.values()), 0
        for doc, found in hits.items():
            room = MAX_DOCUMENT_MATCHES - shown
            if room <= 0:
                break
            rows.append(doc)
            rows.extend(f"  {line_no}: {redact(doc, snippet)}" for line_no, snippet in found[:room])
            shown += len(found[:room])
        out = {"document_matches": "\n".join(rows)}
    out["document_matches_note"] = (
        f"{total} match(es) in {len(hits)} PDF/Office document(s), searched in their "
        "extracted text (ripgrep cannot read these files). Format: path, then "
        "'<line>: <snippet>' rows; line numbers match read_file on that path, so "
        "read_file(path, offset=<line>) opens the passage."
        + (f" Showing at most {MAX_DOCUMENT_MATCHES}." if output_mode == "content"
           and total > MAX_DOCUMENT_MATCHES else ""))
    return out


def _not_searched(skipped: dict[str, str], more: bool) -> dict[str, str]:
    if not skipped and not more:
        return {}
    parts = []
    if skipped:
        names = list(skipped)[:_HINT_MAX_FILES]
        reasons = sorted(set(skipped.values()))[:3]
        parts.append(
            f"{len(skipped)} document(s) in scope could NOT be searched "
            f"({'; '.join(reasons)}): {', '.join(names)}"
            + (" …" if len(skipped) > len(names) else "") + ".")
    if more:
        parts.append(
            f"More than {MAX_SEARCH_DOCUMENTS} documents are in scope; only the first "
            f"{MAX_SEARCH_DOCUMENTS} were searched. Narrow path or file_glob to reach the rest.")
    parts.append("Do not conclude the text is absent from documents that were not searched.")
    return {"_documents_not_searched": " ".join(parts)}


def search_documents(file_ops: Any, path: str, pattern: str, file_glob: Optional[str],
                     output_mode: str, *, admit: Callable[[str], bool],
                     redact: Callable[[str, str], str]) -> dict:
    """Search the extracted text of documents under ``path``; return keys for the
    search_files result (empty dict when no document is in scope).

    ``admit`` is the read-block policy (credential/secret paths stay unsearched) and
    ``redact`` the same secret scrubber content matches get.
    """
    if not _glob_can_select_documents(file_glob):
        return {}
    try:
        candidates, more = file_ops.list_documents(
            path, SEARCHABLE_DOCUMENT_EXTENSIONS, MAX_SEARCH_DOCUMENTS, file_glob)
    except Exception:
        logger.debug("document listing failed for %s", path, exc_info=True)
        return {}
    docs = [d for d in candidates if admit(d)]
    if not docs:
        return _not_searched({}, more)
    rg = _host_rg()
    if rg is None:
        return _not_searched({d: "ripgrep is not installed on the Hermes host" for d in docs}, more)

    deadline = time.monotonic() + SEARCH_BUDGET_SECONDS
    texts, skipped = _extract_all(file_ops, docs, deadline)
    out: dict[str, Any] = {}
    if texts:
        try:
            hits = _rg_over_texts(rg, texts, pattern, output_mode,
                                  timeout=max(1.0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, RuntimeError, OSError, ValueError) as exc:
            reason = "time budget reached" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
            skipped.update(dict.fromkeys(texts, reason))
            hits = {}
        if hits:
            out.update(_format_hits(hits, output_mode, redact))
    out.update(_not_searched(skipped, more))
    return out
