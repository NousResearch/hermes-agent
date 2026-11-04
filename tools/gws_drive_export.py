"""HTML -> Google Doc -> DOCX/PDF export, for DRA Content's content_export tool.

Deliberately NOT part of tools/gws_ops.py's ``_OP_FUNCS`` registry. Every
function registered there is auto-exposed as a generic, directly-callable
tool to any session with the "oauth" toolset enabled -- appropriate for
CRUD primitives like drive_create_file, wrong for this: an arbitrary-file-id
export capability with no connection to DRA Content's own ACL should not be
a raw end-user-invokable action. These functions are imported and called
only from tools/dra_content_tool.py's content_export handler, which is
itself gated behind the (separately toolset-gated, not-yet-enabled)
"dra_content" toolset.

The conversion mechanism was validated live against a real account before
this module was written (see the Stage 10 pilot commit): Drive's
files.create converts source content to a Google Workspace format when the
request body's mimeType (the TARGET format) differs from the uploaded
media's mimetype (the SOURCE format). tools/gws_ops.py's existing
drive_create_file uses the SAME value for both, so it cannot do this
conversion -- that is a real gap in the existing function, not something
this module works around by accident.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MIME = "application/pdf"

MAX_HTML_BYTES = 25 * 1024 * 1024  # matches DRA Content's own per-version cap


class DriveExportError(RuntimeError):
    """Raised for user-facing export failures (safe to show the model)."""


def _build(creds):
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def create_google_doc_from_html(
    creds, name: str, html: str, parent_id: Optional[str] = None
) -> dict:
    """Upload HTML, converted by Drive itself into a native Google Doc.

    Returns {id, name, mimeType}. The resulting file is a real Google Doc --
    editable, commentable, shareable exactly like one a person created by
    hand in Drive.
    """
    html_bytes = html.encode("utf-8")
    if len(html_bytes) > MAX_HTML_BYTES:
        raise DriveExportError(
            f"document is {len(html_bytes)} bytes, over the {MAX_HTML_BYTES} byte export limit"
        )

    service = _build(creds)
    body = {"name": name, "mimeType": GOOGLE_DOC_MIME}
    if parent_id:
        body["parents"] = [parent_id]
    media = MediaIoBaseUpload(BytesIO(html_bytes), mimetype="text/html", resumable=False)
    try:
        f = service.files().create(body=body, media_body=media, fields="id,name,mimeType").execute()
    except Exception as exc:
        raise DriveExportError(f"Drive rejected the document: {exc}") from exc
    return {"id": f["id"], "name": f["name"], "mime_type": f["mimeType"]}


def export_bytes(creds, file_id: str, export_mime_type: str) -> bytes:
    """Export a Google Workspace file (a Doc, typically) to a plain format.

    Uses MediaIoBaseDownload rather than a single .execute() call: Drive's
    export endpoint can chunk large documents, and the download helper
    handles that correctly where a bare .execute() would only return the
    first chunk for anything non-trivial.
    """
    service = _build(creds)
    request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
    buf = BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        try:
            _, done = downloader.next_chunk()
        except Exception as exc:
            raise DriveExportError(f"Drive export failed: {exc}") from exc
    return buf.getvalue()


def upload_bytes(
    creds, name: str, content: bytes, mime_type: str, parent_id: Optional[str] = None
) -> dict:
    """Upload raw bytes (e.g. exported DOCX/PDF) as a new Drive file."""
    service = _build(creds)
    body = {"name": name, "mimeType": mime_type}
    if parent_id:
        body["parents"] = [parent_id]
    media = MediaIoBaseUpload(BytesIO(content), mimetype=mime_type, resumable=False)
    try:
        f = service.files().create(body=body, media_body=media, fields="id,name,mimeType,webViewLink").execute()
    except Exception as exc:
        raise DriveExportError(f"Drive rejected the upload: {exc}") from exc
    return {"id": f["id"], "name": f["name"], "mime_type": f["mimeType"], "url": f.get("webViewLink")}


def delete_file(creds, file_id: str) -> None:
    _build(creds).files().delete(fileId=file_id).execute()


def export_html_as(
    creds, name: str, html: str, target_format: str, parent_id: Optional[str] = None
) -> dict:
    """The full Phase 29 pipeline for one artifact export.

    target_format: "gdoc" | "docx" | "pdf"

    For "gdoc" the Google Doc created by the conversion step IS the
    deliverable -- returned as-is. For "docx"/"pdf" that Google Doc is only
    an intermediate: it is exported to the requested bytes, those bytes are
    uploaded as a second, native-format file, and the intermediate Doc is
    deleted so Drive is not left with a confusing duplicate. If the upload
    or export step fails after the intermediate was created, the
    intermediate is still deleted before the error propagates -- a failed
    export should not leave debris in the user's Drive.
    """
    doc = create_google_doc_from_html(creds, name, html, parent_id)

    if target_format == "gdoc":
        return {"id": doc["id"], "name": doc["name"], "mime_type": doc["mime_type"],
                "url": f"https://docs.google.com/document/d/{doc['id']}/edit"}

    export_mime = {"docx": DOCX_MIME, "pdf": PDF_MIME}.get(target_format)
    if export_mime is None:
        delete_file(creds, doc["id"])
        raise DriveExportError(f"unsupported target_format: {target_format!r} (expected gdoc, docx, or pdf)")

    try:
        content = export_bytes(creds, doc["id"], export_mime)
        ext = {"docx": ".docx", "pdf": ".pdf"}[target_format]
        final_name = name if name.endswith(ext) else name + ext
        final = upload_bytes(creds, final_name, content, export_mime, parent_id)
    finally:
        try:
            delete_file(creds, doc["id"])
        except Exception:
            pass  # best-effort cleanup; the primary result still returns/raises correctly

    return final
