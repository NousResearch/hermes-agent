"""Contract & security tests for read_file native PDF fast-path, tools.media_io, and OOXML zip-bomb bounds."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
from tools import file_tools as _file_tools  # registers read_file on registry
from tools.media_io import is_valid_audio_bytes, is_valid_pdf_bytes, normalize_audio_mime, read_file_bounded_nofollow
from tools.file_tools_read_tracking import _programmatic_read
from tools.read_extract import ExtractionError, _extract_docx, _extract_xlsx
from tools.registry import registry
from tools.vision_tools_history_budget import _repeat_counts

_TINY_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"


def test_read_file_registry_dispatch_execute_code_and_subagent_cap(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_TINY_PDF)

    _repeat_counts.clear()
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        # 1. Interactive model dispatch via registry -> native _multimodal envelope
        res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
        assert isinstance(res, dict) and res.get("_multimodal") is True
        assert res["content"][1]["type"] == "document"
        assert res["content"][1]["mime_type"] == "application/pdf"

        # 2. execute_code (_programmatic_read=True) -> NEVER returns _multimodal dict
        tok = _programmatic_read.set(True)
        try:
            prog_res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
            assert isinstance(prog_res, str)
        finally:
            _programmatic_read.reset(tok)

        # 3. Explicit pagination (offset > 1) -> stays on text extraction
        paged_res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 2, "limit": 100})
        assert isinstance(paged_res, str)

        # 4. Subagent reserve_embed cap (3): 4th read is refused with JSON error, not _multimodal
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_SESSION_ID", f"subagent-{tmp_path.name}")
        for _ in range(3):
            ok = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
            assert isinstance(ok, dict) and ok.get("_multimodal") is True
        refused = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
        assert isinstance(refused, str)
        parsed_refusal = json.loads(refused)
        assert parsed_refusal.get("success") is False
        assert "3 time(s)" in parsed_refusal.get("error", "")


def test_read_file_rejects_spoofed_pdf_magic_and_media_io_security(tmp_path: Path):
    fake_pdf = tmp_path / "spoofed.pdf"
    fake_pdf.write_text("<html>not a pdf</html>", encoding="utf-8")
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        res = registry.dispatch("read_file", {"path": str(fake_pdf), "offset": 1, "limit": 100})
        assert isinstance(res, str)

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(_TINY_PDF)
    sym_pdf = tmp_path / "link.pdf"
    sym_pdf.symlink_to(real_pdf)
    assert read_file_bounded_nofollow(str(sym_pdf)) is None
    assert is_valid_pdf_bytes(_TINY_PDF) is True
    assert is_valid_pdf_bytes(b"<html>fake</html>") is False

    # Audio magic validation: reject HTML and UTF-16LE/BE BOMs; accept WAV & AIFF; normalize aliases
    assert is_valid_audio_bytes(b"<html><body>not audio</body></html>") is False
    assert is_valid_audio_bytes(b"\xff\xfeH\x00e\x00l\x00l\x00o\x00") is False
    assert is_valid_audio_bytes(b"\xfe\xff\x00H\x00e\x00l\x00l\x00o") is False
    assert is_valid_audio_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ") is True
    assert is_valid_audio_bytes(b"FORM\x00\x00\x00\x24AIFFCOMM") is True
    assert normalize_audio_mime("audio/x-wav") == "audio/wav"


def test_read_extract_docx_and_multisheet_xlsx_zip_bombs_rejected(tmp_path: Path):
    # Single-member DOCX zip bomb (> 25 MB member cap)
    docx_bomb = tmp_path / "bomb.docx"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"0" * (26 * 1024 * 1024))
    docx_bomb.write_bytes(buf.getvalue())
    with pytest.raises(ExtractionError, match="exceeds safe decompressed size limit"):
        _extract_docx(str(docx_bomb))

    # Multi-sheet cumulative XLSX zip bomb: each sheet is < 25 MB, total > 100 MB
    xlsx_bomb = tmp_path / "sheets_bomb.xlsx"
    xbuf = io.BytesIO()
    with zipfile.ZipFile(xbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        sheets_xml = "".join(
            f'<sheet name="S{i}" sheetId="{i}" r:id="rId{i}"/>' for i in range(1, 6)
        )
        rels_xml = "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, 6)
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheets_xml}</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{rels_xml}</Relationships>",
        )
        pad = b" " * (22 * 1024 * 1024)
        sheet_body = (
            b'<?xml version="1.0"?>'
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<sheetData><row><c><v>1</v></c></row></sheetData><!--"
            + pad
            + b"--></worksheet>"
        )
        for i in range(1, 6):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_body)
    xlsx_bomb.write_bytes(xbuf.getvalue())
    with pytest.raises(ExtractionError, match="exceeds safe decompressed size limit"):
        _extract_xlsx(str(xlsx_bomb))


def test_read_file_native_pdf_never_bypasses_secret_redaction(tmp_path: Path, monkeypatch):
    import zlib

    secret = "ghp_1234567890abcdefghij1234567890abcd"
    plain_secret_pdf = tmp_path / "plain_secret.pdf"
    plain_secret_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n"
        + f"% Token: {secret}\n".encode("ascii")
        + b"%%EOF\n"
    )
    flate_stream = zlib.compress(f"BT ({secret}) Tj ET".encode("ascii"))
    flate_secret_pdf = tmp_path / "flate_secret.pdf"
    flate_secret_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj << /Filter /FlateDecode >>\nstream\n"
        + flate_stream
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    tj_split_pdf = tmp_path / "tj_secret.pdf"
    tj_split_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj\nstream\n"
        + zlib.compress(f"BT [({secret[:14]}) -10 ({secret[14:]})] TJ ET".encode("ascii"))
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    clean_pdf = tmp_path / "clean.pdf"
    clean_pdf.write_bytes(_TINY_PDF)

    _repeat_counts.clear()
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        for secret_pdf in (plain_secret_pdf, flate_secret_pdf, tj_split_pdf):
            res = registry.dispatch("read_file", {"path": str(secret_pdf), "offset": 1, "limit": 100})
            assert isinstance(res, str), f"Secret-bearing PDF {secret_pdf.name} must NOT attach natively"
            assert secret not in res

        clean_res = registry.dispatch("read_file", {"path": str(clean_pdf), "offset": 1, "limit": 100})
        assert isinstance(clean_res, dict) and clean_res.get("_multimodal") is True


def test_pdf_has_redacted_text_closes_encoding_and_filter_bypasses():
    import base64
    import zlib

    from agent.redact import (
        clear_vault_redaction_values,
        redact_sensitive_text,
        register_vault_redaction_value,
    )
    from tools.media_io import pdf_has_redacted_text

    redactor = lambda t: redact_sensitive_text(t, file_read=True, secret_file=False)
    tok = "ghp_A1b2C3d4E5f6G7h8I9j0"
    raw_def_comp = zlib.compressobj(wbits=-15)
    raw_deflate = raw_def_comp.compress(f"({tok}) Tj".encode()) + raw_def_comp.flush()

    bypasses = {
        "hex": b"%PDF-1.4\n1 0 obj\n<" + tok.encode().hex().encode() + b"> Tj\nendobj\n",
        "utf16be_hex": b"%PDF-1.4\n1 0 obj\n<" + (b"\xfe\xff" + tok.encode("utf-16-be")).hex().encode() + b"> Tj\nendobj\n",
        "line_cont": b"%PDF-1.4\n1 0 obj\n(ghp_\\\nA1b2C3d4E5f6G7h8I9j0) Tj\nendobj\n",
        "octal": b"%PDF-1.4\n1 0 obj\n(ghp\\137A1b2C3d4E5f6G7h8I9j0) Tj\nendobj\n",
        "ignored_bs": b"%PDF-1.4\n1 0 obj\n(g\\hp_A1b2C3d4E5f6G7h8I9j0) Tj\nendobj\n",
        "tj_no_num": b"%PDF-1.4\n1 0 obj\n[(ghp_)(A1b2C3d4E5f6G7h8I9j0)] TJ\nendobj\n",
        "consec_tj": b"%PDF-1.4\n1 0 obj\n(ghp_A1b2) Tj (C3d4E5f6G7h8I9j0) Tj\nendobj\n",
        "flate_utf16be": (
            b"%PDF-1.4\n1 0 obj\n<</Filter/FlateDecode>>\nstream\n"
            + zlib.compress(b"(\xfe\xff" + tok.encode("utf-16-be") + b") Tj")
            + b"\nendstream\nendobj\n"
        ),
        "utf16be_escaped_paren": (
            b"%PDF-1.4\n(\xfe\xff"
            + ("(note) " + tok).encode("utf-16-be").replace(b")", b"\\)")
            + b") Tj\n"
        ),
        "ascii85_flate": (
            b"%PDF-1.4\n1 0 obj\n<</Filter[/ASCII85Decode/FlateDecode]>>\nstream\n"
            + base64.a85encode(zlib.compress(f"({tok}) Tj".encode()), adobe=True)
            + b"\nendstream\nendobj\n"
        ),
        "double_flate": (
            b"%PDF-1.4\n1 0 obj\n<</Filter[/FlateDecode/FlateDecode]>>\nstream\n"
            + zlib.compress(zlib.compress(f"({tok}) Tj".encode()))
            + b"\nendstream\nendobj\n"
        ),
        "raw_deflate": (
            b"%PDF-1.4\n1 0 obj\n<</Filter/FlateDecode>>\nstream\n" + raw_deflate + b"\nendstream\nendobj\n"
        ),
        "asciihex_stream": (
            b"%PDF-1.4\n1 0 obj\n<</Filter/ASCIIHexDecode>>\nstream\n"
            + f"({tok}) Tj".encode().hex().encode()
            + b">\nendstream\nendobj\n"
        ),
        "lzw_fail_closed": b"%PDF-1.4\n1 0 obj\n<</Filter/LZWDecode>>\nstream\n\x80\x01\x02\nendstream\nendobj\n",
        "stream_desync": (
            b"%PDF-1.4\n% comment ending in stream\n1 0 obj\n<</Filter/FlateDecode>>\nstream\n"
            + zlib.compress(f"({tok}) Tj".encode())
            + b"\nendstream\nendobj\n"
        ),
        "encrypt_hex_name": b"%PDF-1.4\ntrailer <</#45ncrypt 2 0 R>>\n",
    }
    for label, payload in bypasses.items():
        assert pdf_has_redacted_text(payload, redactor) is True, f"PDF bypass not caught: {label}"

    register_vault_redaction_value("VaultPassSecret99")
    try:
        vault_pdf = (
            b"%PDF-1.4\n1 0 obj\n<</Filter/FlateDecode>>\nstream\n"
            + zlib.compress(b"(\xfe\xff" + "VaultPassSecret99".encode("utf-16-be") + b") Tj")
            + b"\nendstream\nendobj\n"
        )
        assert pdf_has_redacted_text(vault_pdf, redactor) is True
    finally:
        clear_vault_redaction_values()

    clean_jpeg = (
        b"%PDF-1.4\n1 0 obj\n<</Type/XObject/Subtype/Image/Filter/DCTDecode>>\n"
        b"stream\n\xff\xd8\xff\xe0JFIF\nendstream\nendobj\n"
    )
    clean_flate = (
        b"%PDF-1.4\n1 0 obj\n<</Filter/FlateDecode>>\nstream\n"
        + zlib.compress(b"BT (Hello world 2026) Tj ET")
        + b"\nendstream\nendobj\n"
    )
    assert pdf_has_redacted_text(clean_jpeg, redactor) is False
    assert pdf_has_redacted_text(clean_flate, redactor) is False


def test_media_io_fifo_nonblocking_and_ooxml_amplification_caps(tmp_path: Path):
    import os
    import sys

    from tools.media_io import peek_file_header_nofollow
    from tools.read_extract import extract_document_bytes

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.pdf"
        os.mkfifo(fifo)
        assert peek_file_header_nofollow(str(fifo)) is None
        assert read_file_bounded_nofollow(str(fifo)) is None

    # XLSX sharedStrings expansion bomb
    buf = io.BytesIO()
    ns_s = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_pr = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns_s}"><si><t>{"A" * (1024 * 1024)}</t></si></sst>')
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns_s}" xmlns:r="{ns_r}"><sheets><sheet name="S1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{ns_pr}"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        cells = '<c t="s"><v>0</v></c>' * 50
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns_s}"><sheetData><row>{cells}</row></sheetData></worksheet>',
        )
    with pytest.raises(ExtractionError, match="exceeds safe output size limit"):
        extract_document_bytes(buf.getvalue(), "bomb.xlsx")

    # DOCX nested <w:p> stays O(N) linear without quadratic text duplication
    buf2 = io.BytesIO()
    ns_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{ns_w}"><w:body>{"<w:p>" * 200}<w:r><w:t>{"X" * 10000}</w:t></w:r>{"</w:p>" * 200}</w:body></w:document>',
        )
    out = extract_document_bytes(buf2.getvalue(), "nested.docx")
    assert out.count("X") == 10000 and len(out) <= 10500
    assert sys.version_info >= (3, 10)


def test_try_read_native_pdf_slot_release_deleted_fallback_and_main_loop_guard(tmp_path: Path):
    from tools.file_tools import _try_read_native_pdf, read_file_tool
    from tools.vision_tools_history_budget import _count_key

    pdf_p = tmp_path / "slot.pdf"
    pdf_p.write_bytes(_TINY_PDF)
    resolved = pdf_p.resolve()
    key = _count_key(str(resolved))
    _repeat_counts.pop(key, None)

    with (
        patch("tools.file_tools._get_file_ops"),
        patch("tools.file_tools._file_ops_uses_host_paths", return_value=True),
        patch("agent.image_routing._runtime_main", return_value="gemini-3-flash"),
        patch("agent.image_routing.supports_pdf_in_tool_results", return_value=True),
        patch("agent.image_routing.decide_pdf_input_mode", return_value="native"),
        patch("tools.vision_tools_history_budget.resolve_repeat_cap", return_value=1),
        patch("tools.media_io.pdf_has_redacted_text", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            _try_read_native_pdf(str(pdf_p), resolved, 1, "slot-exc")
    assert _repeat_counts.get(key, 0) == 0

    _repeat_counts[key] = 1
    pdf_p.unlink()
    with (
        patch("tools.file_tools._get_file_ops"),
        patch("tools.file_tools._file_ops_uses_host_paths", return_value=True),
        patch("agent.image_routing._runtime_main", return_value="gemini-3-flash"),
        patch("agent.image_routing.supports_pdf_in_tool_results", return_value=True),
        patch("agent.image_routing.decide_pdf_input_mode", return_value="native"),
        patch("tools.vision_tools_history_budget.resolve_repeat_cap", return_value=1),
    ):
        assert _try_read_native_pdf(str(pdf_p), resolved, 1, "slot-gone") is None
    _repeat_counts.pop(key, None)

    pdf_p.write_bytes(_TINY_PDF)
    with (
        patch("tools.file_tools._get_file_ops"),
        patch("tools.file_tools._file_ops_uses_host_paths", return_value=True),
        patch("agent.image_routing._runtime_main", return_value="gemini-3-flash"),
        patch("agent.image_routing.supports_pdf_in_tool_results", return_value=True),
        patch("agent.image_routing.decide_pdf_input_mode", return_value="native"),
        patch("tools.vision_tools_history_budget.resolve_repeat_cap", return_value=0),
    ):
        results = [read_file_tool(str(pdf_p), task_id=f"main-loop-{tmp_path.name}") for _ in range(5)]
    assert all(isinstance(r, dict) and r.get("_multimodal") is True for r in results[:3])
    assert isinstance(results[3], str) and "BLOCKED" in results[3]
