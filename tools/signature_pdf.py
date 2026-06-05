"""PDF stamping helpers for Signature Core completed documents.

This module intentionally has no runtime dependency on the signing database. It takes
an already-approved PDF plus approval evidence and writes a visible signed PDF with
an audit page. The caller remains responsible for persisting the completed artifact
in Signature Core, usually as `signature.attachments.kind = 'completed_pdf'`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_signed_at(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return value.replace("T", " ").replace("+00:00", " UTC")


def stamp_signed_pdf(
    *,
    input_pdf: str | Path,
    output_pdf: str | Path,
    request_id: str,
    source_id: str = "",
    signer: str,
    signed_at: str | None,
    approval_hash: str,
    document_hash: str,
    event_id: str = "",
    disclaimer: str | None = None,
) -> dict[str, Any]:
    """Write a visibly signed PDF and return output metadata.

    The original `document_hash` should be the hash of the exact document that was
    approved by the signer. The signed output receives its own SHA-256 for the
    completed artifact record.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on deployment extras
        raise RuntimeError("PyMuPDF (`fitz`) is required to stamp signed PDFs") from exc

    input_path = Path(input_pdf)
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_sha = sha256_file(input_path)
    doc = fitz.open(str(input_path))
    if doc.page_count < 1:
        raise ValueError("PDF has no pages")

    signed_at_text = _format_signed_at(signed_at)
    signer_text = signer.strip() or "Firmante"

    page = doc[0]
    rect = page.rect
    stamp = fitz.Rect(rect.x1 - 270, rect.y1 - 155, rect.x1 - 36, rect.y1 - 42)
    page.draw_rect(stamp, color=(0.02, 0.38, 0.18), fill=(0.93, 1.0, 0.95), width=1.4)
    page.insert_text((stamp.x0 + 12, stamp.y0 + 22), "FIRMADO DIGITALMENTE", fontsize=12, fontname="helv", color=(0.02, 0.32, 0.14))
    page.insert_text((stamp.x0 + 12, stamp.y0 + 45), f"Por: {signer_text[:42]}", fontsize=9.5, fontname="helv", color=(0, 0, 0))
    page.insert_text((stamp.x0 + 12, stamp.y0 + 62), f"Fecha: {signed_at_text[:34]}", fontsize=8.5, fontname="helv", color=(0, 0, 0))
    page.insert_text((stamp.x0 + 12, stamp.y0 + 82), f"Hash aprobación: {approval_hash[:18]}…", fontsize=8, fontname="cour", color=(0, 0, 0))
    page.insert_text((stamp.x0 + 12, stamp.y0 + 98), f"Hash documento: {document_hash[:18]}…", fontsize=8, fontname="cour", color=(0, 0, 0))

    audit = doc.new_page(width=612, height=792)
    audit.draw_rect(fitz.Rect(36, 36, 576, 756), color=(0.02, 0.28, 0.16), width=1.2)
    audit.insert_text((56, 78), "Certificado de aprobación y firma digital", fontsize=18, fontname="helv", color=(0.02, 0.28, 0.16))
    audit.insert_text((56, 110), "SitioUno / Zeus Signature Core", fontsize=11, fontname="helv", color=(0.25, 0.25, 0.25))

    details = [
        ("Documento", source_id or request_id),
        ("Solicitud de firma", request_id),
        ("Firmante", signer_text),
        ("Fecha de firma", signed_at_text),
        ("Hash SHA-256 del documento aprobado", document_hash),
        ("Hash SHA-256 de aprobación", approval_hash),
        ("Evento sandbox", event_id or "N/A"),
    ]
    y = 150
    for label, value in details:
        audit.insert_text((56, y), label, fontsize=9, fontname="helv", color=(0.25, 0.25, 0.25))
        audit.insert_textbox(fitz.Rect(210, y - 12, 548, y + 30), str(value), fontsize=9, fontname="helv", color=(0, 0, 0))
        y += 44 if len(str(value)) < 60 else 58

    audit_text = disclaimer or (
        "Este PDF fue actualizado automáticamente después de recibir la aprobación "
        "del firmante en el workspace público. La evidencia de aprobación queda "
        "persistida en Signature Core con hash de aprobación y eventos encadenados. "
        "Este sello visual no sustituye todavía una firma cualificada PAdES/TSA; "
        "representa evidencia interna de aprobación digital."
    )
    audit.insert_textbox(fitz.Rect(56, 565, 548, 650), audit_text, fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))

    if output_path.resolve() == input_path.resolve():
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        doc.save(str(tmp), garbage=4, deflate=True)
        doc.close()
        os.replace(tmp, output_path)
    else:
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "original_sha256": original_sha,
        "signed_sha256": sha256_file(output_path),
        "byte_size": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp a Signature Core PDF as signed")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-id", default="")
    parser.add_argument("--signer", required=True)
    parser.add_argument("--signed-at", default="")
    parser.add_argument("--approval-hash", required=True)
    parser.add_argument("--document-hash", required=True)
    parser.add_argument("--event-id", default="")
    args = parser.parse_args()
    result = stamp_signed_pdf(
        input_pdf=args.input,
        output_pdf=args.output,
        request_id=args.request_id,
        source_id=args.source_id,
        signer=args.signer,
        signed_at=args.signed_at,
        approval_hash=args.approval_hash,
        document_hash=args.document_hash,
        event_id=args.event_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
