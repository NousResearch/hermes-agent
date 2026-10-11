"""web_extract reads PDFs locally instead of sending them to a per-page vendor.

Real-imports E2E: a local http.server serves real PDF bytes, the conftest
sandbox provides a temp Hermes home, and a recording fake extract provider
proves which URLs would have been billed by a vendor.
"""

import asyncio
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools import url_safety
from tools import web_pdf_local
from tools import web_tools


def _make_pdf(page_texts):
    """Build a minimal valid PDF with one Helvetica text line per page."""
    objs = []
    n = len(page_texts)
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_num = 3 + n * 2
    for i, text in enumerate(page_texts):
        content_num = 4 + i * 2
        if text:
            stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
        else:
            stream = b""
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>".encode()
        )
        objs.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


THREE_PAGE = _make_pdf(["Alpha page one", "Bravo page two", "Charlie page three"])
SCANNED = _make_pdf(["", ""])
HTML = b"<html><head><title>Hi</title></head><body>hello html</body></html>"

ROUTES = {
    "/manual.pdf": (200, "application/pdf", THREE_PAGE),
    "/files/Manual.PDF": (200, "application/octet-stream", THREE_PAGE),
    "/getfile": (200, "application/pdf", THREE_PAGE),
    "/page.html": (200, "text/html; charset=utf-8", HTML),
    "/scan.pdf": (200, "application/pdf", SCANNED),
    "/landing.pdf": (200, "text/html", HTML),
}


# A server that never stalls long enough to trip httpx's per-operation
# timeouts but never finishes either: one byte per TRICKLE_INTERVAL.
TRICKLE_INTERVAL = 0.1
TRICKLE_SECONDS = 15.0


def _trickle(wfile, data):
    try:
        for i in range(len(data)):
            wfile.write(data[i:i + 1])
            wfile.flush()
            time.sleep(TRICKLE_INTERVAL)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the client gave up — the point of the test


class _Handler(BaseHTTPRequestHandler):
    def _route(self):
        path = self.path.split("?", 1)[0]
        if path in ("/redirect-to-pdf", "/redirect-to-metadata.pdf"):
            return 302, None, b""
        return ROUTES.get(path, (404, "text/html", b"not found"))

    def _send(self, with_body):
        status, ctype, body = self._route()
        self.send_response(status)
        if status == 302 and self.path.startswith("/redirect-to-metadata"):
            self.send_header("Location", "http://169.254.169.254/latest/doc.pdf")
        elif status == 302:
            self.send_header("Location", "/manual.pdf")
        else:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/trickle.pdf"):
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            _trickle(self.wfile, b"%PDF-1.4\n" + b"x" * int(TRICKLE_SECONDS / TRICKLE_INTERVAL))
            return
        self._send(True)

    def do_HEAD(self):  # noqa: N802
        if self.path.startswith("/trickle-head"):
            # Status line + a header that never ends.
            _trickle(
                self.wfile,
                b"HTTP/1.1 200 OK\r\nX-Slow: " + b"y" * int(TRICKLE_SECONDS / TRICKLE_INTERVAL),
            )
            return
        self._send(False)

    def log_message(self, *args):
        pass


class _RecordingProvider:
    name = "fakevendor"
    display_name = "Fake Vendor"

    def __init__(self):
        self.calls = []

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def extract(self, urls, format=None):
        self.calls.append(list(urls))
        return [
            {"url": u, "title": "vendor", "content": f"VENDOR:{u}", "error": None}
            for u in urls
        ]


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _write_config(web_section):
    home = Path(os.environ["HERMES_HOME"])
    lines = ["security:", "  allow_private_urls: true", "web:"]
    for key, value in web_section.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def vendor(monkeypatch):
    provider = _RecordingProvider()
    import agent.web_search_registry as registry

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fakevendor")
    monkeypatch.setattr(
        registry, "get_provider", lambda name: provider if name == "fakevendor" else None
    )
    monkeypatch.setattr("tools.web_tools_extract._rescue_eligible", lambda p: False)
    # Routing tests (size cap, 404, blocked redirect) never reach extraction;
    # pin the extractor gate so they hold on hosts without pymupdf/pdftotext.
    monkeypatch.setattr(web_pdf_local, "local_extractor_available", lambda: True)
    url_safety._reset_allow_private_cache()
    yield provider
    url_safety._reset_allow_private_cache()


def _needs_extractor():
    try:
        import pymupdf  # noqa: F401
        return
    except ImportError:
        pass
    if shutil.which("pdftotext") is None:
        pytest.skip("no local PDF extractor (pymupdf / pdftotext)")


def _run(urls):
    return json.loads(asyncio.run(web_tools.web_extract_tool(urls)))["results"]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://x.test/a.pdf", True),
        ("https://x.test/a.PDF", True),
        ("https://x.test/a.pdf?download=1", True),
        ("https://x.test/a.pdf#page=3", True),
        ("https://x.test/a.pdf.html", False),
        ("https://x.test/?file=a.pdf", False),
        ("https://x.test/page", False),
    ],
)
def test_has_pdf_extension(url, expected):
    assert web_pdf_local.has_pdf_extension(url) is expected


def test_settings_defaults_and_overrides():
    assert web_pdf_local.local_pdf_settings({}) == (True, 50 * 1024 * 1024)
    assert web_pdf_local.local_pdf_settings({"local_pdf": False})[0] is False
    assert web_pdf_local.local_pdf_settings({"local_pdf": "false"})[0] is False
    assert web_pdf_local.local_pdf_settings({"pdf_max_bytes": 1234})[1] == 1234


def test_pdf_routes_local_html_goes_to_vendor(server, vendor):
    _needs_extractor()
    _write_config({})
    urls = [
        f"{server}/manual.pdf",
        f"{server}/files/Manual.PDF?download=1",
        f"{server}/getfile",
        f"{server}/redirect-to-pdf",
        f"{server}/page.html",
    ]
    results = _run(urls)
    assert [r["url"] for r in results] == urls

    # Only the HTML URL was sent to the (billing) vendor.
    assert vendor.calls == [[f"{server}/page.html"]]
    assert results[4]["content"] == f"VENDOR:{server}/page.html"
    assert "metadata" not in results[4]

    for r in results[:4]:
        assert r["error"] is None, r
        meta = r["metadata"]
        assert meta["served_by"] == "local-pdf"
        assert meta["pages"] == 3
        assert meta["bytes"] == len(THREE_PAGE)
        assert "warning" not in meta
        for word in ("Alpha", "Bravo", "Charlie"):
            assert word in r["content"]


def test_mixed_invalid_blocked_local_and_vendor_preserve_order(server, vendor):
    _needs_extractor()
    _write_config({})
    urls = [None, f"{server}/manual.pdf", "http://169.254.169.254/doc.pdf",
            f"{server}/page.html", f"{server}/manual.pdf"]
    results = _run(urls)
    assert len(results) == len(urls)
    assert results[0]["error"].startswith("Invalid URL item")
    assert results[2]["error"].startswith("Blocked")
    assert [r["url"] for r in results[1:]] == urls[1:]
    assert results[1]["metadata"]["served_by"] == "local-pdf"
    assert results[4]["metadata"]["served_by"] == "local-pdf"
    assert vendor.calls == [[urls[3]]]


def test_local_pdf_needs_no_extract_provider(server, vendor, monkeypatch):
    _needs_extractor()
    _write_config({})

    def no_provider_resolution(*args):
        pytest.fail("local-only batch must not resolve an extract provider")

    monkeypatch.setattr(web_tools, "_resolve_extract_provider", no_provider_resolution)
    results = _run([f"{server}/manual.pdf"])
    assert results[0]["metadata"]["served_by"] == "local-pdf"
    assert vendor.calls == []


def test_local_pdf_false_restores_vendor(server, vendor):
    _write_config({"local_pdf": False})
    url = f"{server}/manual.pdf"
    results = _run([url])
    assert vendor.calls == [[url]]
    assert results[0]["content"] == f"VENDOR:{url}"


def test_size_cap_refuses_with_clear_error(server, vendor):
    _write_config({"pdf_max_bytes": 100})
    results = _run([f"{server}/manual.pdf"])
    assert vendor.calls == []
    assert "too large" in results[0]["error"]
    assert "web.pdf_max_bytes=100" in results[0]["error"]


def test_404_pdf_reports_http_error_not_vendor(server, vendor):
    _write_config({})
    results = _run([f"{server}/missing.pdf"])
    assert vendor.calls == []
    assert results[0]["error"].startswith("HTTP 404 fetching PDF")


def test_scanned_pdf_warns_and_never_falls_back(server, vendor):
    _needs_extractor()
    _write_config({})
    results = _run([f"{server}/scan.pdf"])
    assert vendor.calls == []
    meta = results[0]["metadata"]
    assert meta["served_by"] == "local-pdf"
    assert meta["pages"] == 2
    assert meta["warning"] == "no text layer — run OCR"
    assert "run OCR" in results[0]["content"]


def test_pdf_link_landing_on_html_uses_vendor(server, vendor):
    _write_config({})
    url = f"{server}/landing.pdf"
    results = _run([url])
    assert vendor.calls == [[url]]
    assert results[0]["content"] == f"VENDOR:{url}"

def test_blocked_redirect_keeps_requested_url(server, vendor):
    _write_config({})
    url = f"{server}/redirect-to-metadata.pdf"
    results = _run([url])
    assert vendor.calls == []
    assert results[0]["url"] == url
    assert results[0]["error"].startswith("Blocked")
    assert "redirected to http://169.254.169.254/" in results[0]["error"]


def test_no_local_extractor_keeps_backend_dispatch(server, vendor, monkeypatch):
    _write_config({})
    monkeypatch.setattr(web_pdf_local, "local_extractor_available", lambda: False)
    url = f"{server}/manual.pdf"
    results = _run([url])
    assert vendor.calls == [[url]]
    assert results[0]["content"] == f"VENDOR:{url}"


def _tight_deadlines(monkeypatch):
    # raising=False so this runs red (not AttributeError) on a base without them.
    monkeypatch.setattr(web_pdf_local, "GET_DEADLINE_SECONDS", 0.5, raising=False)
    monkeypatch.setattr(web_pdf_local, "HEAD_DEADLINE_SECONDS", 0.5, raising=False)


def test_trickling_pdf_download_hits_overall_deadline(server, vendor, monkeypatch):
    """httpx's timeout is per read; a byte-every-100ms server must still be cut off."""
    _write_config({})
    _tight_deadlines(monkeypatch)
    url = f"{server}/trickle.pdf"
    started = time.monotonic()
    results = _run([url])
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"download ran {elapsed:.1f}s past its deadline"
    assert vendor.calls == []
    assert results[0]["url"] == url
    assert results[0]["error"].startswith("Timed out fetching PDF locally"), results[0]


def test_trickling_head_probe_hits_overall_deadline(server, vendor, monkeypatch):
    """A HEAD probe that never finishes means "not known PDF" -> vendor, promptly."""
    _write_config({})
    _tight_deadlines(monkeypatch)
    url = f"{server}/trickle-head"
    started = time.monotonic()
    flags = asyncio.run(web_pdf_local.classify_pdf_urls([url]))
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"HEAD probe ran {elapsed:.1f}s past its deadline"
    assert flags == [False]
