"""Local PDF reading for ``web_extract``.

Paid extract vendors (Firecrawl in particular) bill PDF scrapes PER PAGE: a
single 200-page hardware manual costs ~200 credits. Downloading the file and
reading its text layer locally costs nothing, so ``web_extract`` classifies
each URL first and routes PDFs here instead of to any vendor.

Classification: the URL path ends in ``.pdf`` (case-insensitive, query string
ignored), or a bounded HEAD request reports ``Content-Type: application/pdf``.

Extraction: pymupdf when importable (same library the ``ocr-and-documents``
skill uses), else poppler's ``pdftotext``. A PDF with no text layer returns
what the extractor gave plus ``metadata.warning`` — never a silent fallback to
a paid vendor.

Config (``web:`` in config.yaml):
    local_pdf: true          # false restores vendor dispatch for PDFs
    pdf_max_bytes: 52428800  # download cap (50 MB)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

DEFAULT_PDF_MAX_BYTES = 50 * 1024 * 1024
HEAD_TIMEOUT_SECONDS = 5.0
HEAD_MAX_REDIRECTS = 3
GET_MAX_REDIRECTS = 5
GET_TIMEOUT_SECONDS = 60.0
# httpx timeouts bound each network operation, not the whole transfer: a
# server trickling a byte just under the read timeout would otherwise hold
# the batch forever. These cap the WHOLE probe / download, redirects included.
HEAD_DEADLINE_SECONDS = 10.0
GET_DEADLINE_SECONDS = 120.0
PDFTOTEXT_TIMEOUT_SECONDS = 60.0
# A whole document with fewer extracted non-whitespace characters than this
# is treated as having no text layer (a scan).
MIN_TEXT_LAYER_CHARS = 20
NO_TEXT_WARNING = "no text layer — run OCR"
SERVED_BY = "local-pdf"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class NotAPdf(Exception):
    """The URL looked like a PDF but the server returned something else."""


def local_pdf_settings(web_config: Optional[dict]) -> Tuple[bool, int]:
    """Return ``(enabled, max_bytes)`` from the ``web:`` config section."""
    cfg = web_config if isinstance(web_config, dict) else {}
    raw_enabled = cfg.get("local_pdf", True)
    if isinstance(raw_enabled, str):
        enabled = raw_enabled.strip().lower() not in {"false", "0", "no", "off"}
    else:
        enabled = raw_enabled is None or bool(raw_enabled)
    try:
        max_bytes = int(cfg.get("pdf_max_bytes") or DEFAULT_PDF_MAX_BYTES)
        if max_bytes <= 0:
            max_bytes = DEFAULT_PDF_MAX_BYTES
    except (TypeError, ValueError):
        max_bytes = DEFAULT_PDF_MAX_BYTES
    return enabled, max_bytes


def has_pdf_extension(url: str) -> bool:
    """True when the URL path (query/fragment ignored) ends in ``.pdf``."""
    try:
        path = urlsplit(url).path
    except ValueError:
        return False
    return path.lower().endswith(".pdf")


def _is_pdf_content_type(value: Optional[str]) -> bool:
    return bool(value) and value.split(";", 1)[0].strip().lower() == "application/pdf"


def _client(timeout: float):
    from tools.url_safety import create_ssrf_safe_async_client

    return create_ssrf_safe_async_client(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": BROWSER_UA, "Accept": "application/pdf,*/*;q=0.8"},
    )


async def _check_hop(url: str) -> Optional[Dict[str, Any]]:
    """SSRF + website-policy gate for one redirect hop. Returns an error result or None."""
    from tools.url_safety import async_is_safe_url
    from tools.website_policy import check_website_access

    if not await async_is_safe_url(url):
        return {
            "url": url, "title": "", "content": "",
            "error": "Blocked: URL targets a private or internal network address",
        }
    try:
        blocked = check_website_access(url)
    except Exception:  # noqa: BLE001 — policy errors fail open, like dispatch
        blocked = None
    if blocked:
        return {
            "url": url, "title": "", "content": "",
            "error": blocked["message"],
            "blocked_by_policy": {
                "host": blocked["host"],
                "rule": blocked["rule"],
                "source": blocked["source"],
            },
        }
    return None


async def head_is_pdf(url: str) -> bool:
    """Bounded HEAD probe: 5 s per op, 10 s overall, <=3 redirects. Any failure => False."""
    try:
        return await asyncio.wait_for(_head_is_pdf(url), HEAD_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        logger.debug("web_extract: HEAD probe for %s exceeded %ss", url, HEAD_DEADLINE_SECONDS)
    except Exception as exc:  # noqa: BLE001 — a failed probe means "not known PDF"
        logger.debug("web_extract: HEAD probe failed for %s: %s", url, exc)
    return False


async def _head_is_pdf(url: str) -> bool:
    async with _client(HEAD_TIMEOUT_SECONDS) as client:
        current = url
        for _ in range(HEAD_MAX_REDIRECTS + 1):
            if await _check_hop(current) is not None:
                return False
            resp = await client.head(current)
            if resp.is_redirect and resp.headers.get("location"):
                current = urljoin(str(resp.url), resp.headers["location"])
                continue
            return resp.status_code < 400 and _is_pdf_content_type(
                resp.headers.get("content-type")
            )
    return False


async def classify_pdf_urls(urls: List[str]) -> List[bool]:
    """Classify each URL; HEAD is skipped when the extension already says pdf."""

    async def _one(u: str) -> bool:
        if has_pdf_extension(u):
            return True
        return await head_is_pdf(u)

    return list(await asyncio.gather(*(_one(u) for u in urls)))


async def _download(url: str, max_bytes: int) -> Tuple[str, bytes, Optional[Dict[str, Any]]]:
    """Stream the PDF. Returns ``(final_url, data, error_result)``."""
    async with _client(GET_TIMEOUT_SECONDS) as client:
        current = url
        for _ in range(GET_MAX_REDIRECTS + 1):
            hop_error = await _check_hop(current)
            if hop_error is not None:
                # Keep the result attributed to the REQUESTED url so batch
                # callers can match it; name the redirect target in the error.
                if current != url:
                    hop_error["error"] = f"{hop_error['error']} (redirected to {current})"
                hop_error["url"] = url
                return current, b"", hop_error
            async with client.stream("GET", current) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    current = urljoin(str(resp.url), resp.headers["location"])
                    continue
                if resp.status_code >= 400:
                    return current, b"", {
                        "url": url, "title": "", "content": "",
                        "error": f"HTTP {resp.status_code} fetching PDF {current}",
                    }
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    return current, b"", _too_large(url, int(declared), max_bytes)
                chunks: List[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return current, b"", _too_large(url, None, max_bytes)
                    chunks.append(chunk)
                data = b"".join(chunks)
                if not data.lstrip()[:5].startswith(b"%PDF") and not _is_pdf_content_type(
                    resp.headers.get("content-type")
                ):
                    raise NotAPdf(resp.headers.get("content-type") or "unknown type")
                return current, data, None
        return current, b"", {
            "url": url, "title": "", "content": "",
            "error": f"Too many redirects fetching PDF {url}",
        }


def _too_large(url: str, size: Optional[int], max_bytes: int) -> Dict[str, Any]:
    size_txt = f"{size} bytes" if size is not None else f"more than {max_bytes} bytes"
    return {
        "url": url, "title": "", "content": "",
        "error": (
            f"PDF is too large to read locally ({size_txt}; limit web.pdf_max_bytes="
            f"{max_bytes}). Raise web.pdf_max_bytes or download it with the terminal."
        ),
    }


def _extract_pymupdf(data: bytes) -> Optional[Tuple[List[str], str]]:
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        pages = [page.get_text() for page in doc]
        title = (doc.metadata or {}).get("title") or ""
    finally:
        doc.close()
    return pages, title


def _extract_pdftotext(data: bytes) -> Optional[Tuple[List[str], str]]:
    if shutil.which("pdftotext") is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        proc = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=PDFTOTEXT_TIMEOUT_SECONDS,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            "pdftotext failed: " + proc.stderr.decode("utf-8", "replace").strip()[:200]
        )
    pages = proc.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # trailing form-feed artifact
    return pages, ""


def local_extractor_available() -> bool:
    """True when pymupdf is importable or poppler's ``pdftotext`` is on PATH."""
    import importlib.util

    for name in ("pymupdf", "fitz"):
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return shutil.which("pdftotext") is not None


_warned_no_extractor = False


def extract_pdf_text(data: bytes) -> Tuple[List[str], str, str]:
    """Return ``(page_texts, title, extractor)``; raises when no extractor exists."""
    got = _extract_pymupdf(data)
    if got is not None:
        return got[0], got[1], "pymupdf"
    got = _extract_pdftotext(data)
    if got is not None:
        return got[0], got[1], "pdftotext"
    raise RuntimeError(
        "No local PDF extractor available: install pymupdf (pip install pymupdf) "
        "or poppler (pdftotext)."
    )


async def read_pdf_locally(url: str, max_bytes: int) -> Optional[Dict[str, Any]]:
    """Fetch and read a PDF URL locally.

    Returns a web_extract result dict (``url``/``title``/``content``/``error``
    plus ``metadata``), or ``None`` when the server did not actually return a
    PDF (e.g. a ``.pdf`` link that lands on an HTML page) — the caller then
    dispatches that URL normally, since HTML is not billed per page.
    """
    try:
        final_url, data, error = await asyncio.wait_for(
            _download(url, max_bytes), GET_DEADLINE_SECONDS
        )
    except asyncio.TimeoutError:
        return {"url": url, "title": "", "content": "",
                "error": (f"Timed out fetching PDF locally: download exceeded "
                          f"{GET_DEADLINE_SECONDS:g}s")}
    except NotAPdf as exc:
        logger.info("web_extract: %s is not a PDF after fetch (%s); using backend", url, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — report the local fetch error, not a vendor one
        return {"url": url, "title": "", "content": "",
                "error": f"Error fetching PDF locally: {exc}"}
    if error is not None:
        return error
    try:
        pages, title, extractor = await asyncio.to_thread(extract_pdf_text, data)
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "title": "", "content": "",
                "error": f"Error reading PDF locally: {exc}"}

    page_count = len(pages)
    metadata: Dict[str, Any] = {
        "served_by": SERVED_BY,
        "pages": page_count,
        "bytes": len(data),
        "extractor": extractor,
    }
    if final_url != url:
        metadata["final_url"] = final_url
    parts = [
        f"<!-- page {i} -->\n{text.strip()}" for i, text in enumerate(pages, 1)
    ]
    content = "\n\n".join(parts).strip()
    if sum(len("".join(p.split())) for p in pages) < MIN_TEXT_LAYER_CHARS:
        metadata["warning"] = NO_TEXT_WARNING
        content = (content + "\n\n" if content else "") + (
            f"[{NO_TEXT_WARNING}: this PDF ({page_count} pages) has no extractable "
            "text layer; OCR it locally (see the ocr-and-documents skill).]"
        )
    logger.info(
        "web_extract: %s is a PDF (%d pages); read locally, no vendor credits",
        url, page_count,
    )
    return {
        "url": url,
        "title": title or os.path.basename(urlsplit(final_url).path) or url,
        "content": content,
        "raw_content": content,
        "error": None,
        "metadata": metadata,
    }


async def split_local_pdfs(
    urls: List[str], indices: List[int], web_config: Optional[dict]
) -> Tuple[Dict[int, Dict[str, Any]], List[str], List[int]]:
    """Read every PDF in *urls* locally; return ``(done_by_index, vendor_urls, vendor_indices)``."""
    global _warned_no_extractor
    enabled, max_bytes = local_pdf_settings(web_config)
    if not enabled or not urls:
        return {}, list(urls), list(indices)
    if not local_extractor_available():
        # No way to read a PDF here: keep today's backend dispatch rather than
        # turning every PDF into an error. Warn once so the cost is visible.
        if not _warned_no_extractor:
            _warned_no_extractor = True
            logger.warning(
                "web_extract: web.local_pdf is on but neither pymupdf nor pdftotext "
                "is installed; PDFs go to the extract backend (billed per page). "
                "Install pymupdf (pip install pymupdf) or poppler to read them locally."
            )
        return {}, list(urls), list(indices)
    flags = await classify_pdf_urls(urls)
    reads = iter(await asyncio.gather(
        *(read_pdf_locally(u, max_bytes) for u, f in zip(urls, flags) if f)
    ))
    done: Dict[int, Dict[str, Any]] = {}
    vendor_urls: List[str] = []
    vendor_indices: List[int] = []
    for url, index, flag in zip(urls, indices, flags):
        local = next(reads) if flag else None
        if local is not None:
            done[index] = local
        else:
            vendor_urls.append(url)
            vendor_indices.append(index)
    return done, vendor_urls, vendor_indices
