"""PDF attach JSON-RPC handler: renders a PDF's pages with ``pdftoppm`` and queues them as images.

Split from ``methods_prompt`` so the prompt/attachment handlers carry no subprocess of their
own; this module is the one place the gateway shells out for attachments. Bodies are rebound
onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG (``pdftoppm``; 5028 if missing) and
    queuing the pages as images.  Host ``path`` or base64 ``content_base64``."""
    import shutil
    import subprocess
    import tempfile
    session, err = _sess_building(params, rid)
    if err:
        return err
    if shutil.which("pdftoppm") is None:
        return _err(rid, 5028, "pdftoppm not installed (poppler-utils package required)")
    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")
    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        pdf_path, display_name, err = _pdf_attach_source(rid, params, td_path, raw_path, raw_b64)
        if err is not None:
            return err
        first_page, last_page, err = _pdf_page_range(rid, params)
        if err is not None:
            return err
        argv = [
            "pdftoppm", "-png", "-r", "150", "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(td_path / "page")]
        from hermes_cli._subprocess_compat import windows_hide_flags
        try:
            # UTF-8 + lossy decode: non-UTF-8 child output must not crash the gateway
            # thread on locale-mismatched Windows.
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                encoding="utf-8", errors="replace", creationflags=windows_hide_flags())
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))
        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")
        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(
                session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}")
            attached_pages.append({"path": str(dst), "page": page_int, **_image_meta(dst)})
        return _ok(rid, {
            "attached": True, "filename": display_name, "pages_attached": len(attached_pages),
            "pages": attached_pages, "count": len(session["attached_images"]),
            "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]"})


def _pdf_attach_source(rid, params, td_path, raw_path, raw_b64):
    """Materialize the PDF to render: ``(pdf_path, display_name, err)``."""
    if raw_b64:
        pdf_bytes, err = _decode_attach_payload(
            rid, raw_b64, mime_prefix="application/pdf", max_bytes=_PDF_ATTACH_MAX_BYTES,
            label="PDF", empty_msg="decoded PDF is empty")
        if err is not None:
            return None, None, err
        if pdf_bytes[:5] != b"%PDF-":
            return None, None, _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
        pdf_path = td_path / "input.pdf"
        pdf_path.write_bytes(pdf_bytes)
        return pdf_path, str(params.get("filename", "") or "uploaded.pdf"), None
    try:
        from cli import _resolve_attachment_path
        resolved = _resolve_attachment_path(raw_path)
    except Exception:
        resolved = None
    if resolved is None or not (pdf := Path(resolved)).is_file():
        return None, None, _err(rid, 4016, f"PDF not found: {raw_path}")
    if pdf.suffix.lower() != ".pdf":
        return None, None, _err(rid, 4016, f"not a PDF: {pdf.name}")
    if pdf.stat().st_size > _PDF_ATTACH_MAX_BYTES:
        mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
        return None, None, _err(rid, 4018, f"PDF too large; cap is {mb} MB")
    return pdf, pdf.name, None


def _pdf_page_range(rid, params):
    """Validate first/last page against the per-call cap: ``(first, last, err)``."""
    try:
        first_page = int(params.get("first_page") or 1)
        last_page = None if params.get("last_page") is None else int(params.get("last_page"))
    except (TypeError, ValueError):
        return None, None, _err(rid, 4015, "first_page/last_page must be integers")
    if first_page < 1:
        return None, None, _err(rid, 4015, "first_page must be >= 1")
    if last_page is None:
        last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
    if last_page < first_page:
        return None, None, _err(rid, 4015, "last_page must be >= first_page")
    if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
        return None, None, _err(
            rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")
    return first_page, last_page, None


def register(server) -> None:
    """Publish this module's helpers + handler onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
