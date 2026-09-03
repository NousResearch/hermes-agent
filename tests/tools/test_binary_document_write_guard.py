"""Tests for the binary-document write guard (port of nearai/ironclaw#7109).

A plain-text write can never produce a valid OOXML/OLE/ODF container, so
write_file/patch must refuse to write text into .docx/.xlsx/.pptx (and
friends), and must refuse to OVERWRITE an existing .pdf — while still
allowing new-.pdf creation (raw PDF syntax is text-authorable).
"""

import json
import zipfile
from pathlib import Path

from tools.binary_extensions import (
    has_opaque_document_extension,
    is_pdf_path,
)
from tools.file_tools import (
    _check_binary_document_write,
    patch_tool,
    write_file_tool,
)


def _make_minimal_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="xml" '
            'ContentType="application/xml"/></Types>',
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.'
            'openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r>'
            "<w:t>Quarterly numbers look good.</w:t></w:r></w:p></w:body>"
            "</w:document>",
        )


class TestExtensionHelpers:
    def test_opaque_document_extensions(self):
        for p in ("a.docx", "b.XLSX", "c.pptx", "d.doc", "e.odt", "f.ods", "g.odp",
                  "h.docm", "i.xlsm", "j.xlsb", "k.pptm", "l.ppsx", "m.ppsm",
                  "n.pps", "o.pot", "p.rtf", "q.epub"):
            assert has_opaque_document_extension(p) is True, f"{p} should be opaque"

    def test_non_opaque_paths(self):
        for p in ("a.txt", "b.py", "c.pdf", "d.md", "noext", "e.csv"):
            assert has_opaque_document_extension(p) is False

    def test_is_pdf_path(self):
        assert is_pdf_path("report.pdf") is True
        assert is_pdf_path("report.PDF") is True
        assert is_pdf_path("report.txt") is False


class TestCheckBinaryDocumentWrite:
    def test_docx_always_rejected(self, tmp_path: Path):
        # Even a NON-existing docx is rejected — text can't be a valid container.
        err = _check_binary_document_write(str(tmp_path / "new.docx"))
        assert err is not None
        assert ".docx" in err

    def test_existing_pdf_rejected(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        err = _check_binary_document_write(str(pdf))
        assert err is not None
        assert "overwrite" in err.lower()

    def test_new_pdf_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "fresh.pdf")) is None

    def test_plain_text_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "notes.txt")) is None


class TestWriteFileToolGuard:
    def test_write_file_rejects_existing_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        result = json.loads(write_file_tool(str(docx), "edited text"))

        assert result.get("error"), "text write into .docx must be refused"
        assert docx.read_bytes() == original, "document bytes must be untouched"
        assert zipfile.is_zipfile(docx), "document must remain a valid container"

    def test_write_file_rejects_docm(self, tmp_path: Path):
        """Regression: .docm is extractable by read_file (anydoc) but was
        missing from OPAQUE_DOCUMENT_EXTENSIONS in the original PR #82818.
        Flagged by @egilewski — proven live: text write corrupted the zip."""
        docm = tmp_path / "macro.docm"
        _make_minimal_docx(docm)  # same OOXML zip structure
        original = docm.read_bytes()

        result = json.loads(write_file_tool(str(docm), "edited text"))

        assert result.get("error"), "text write into .docm must be refused"
        assert docm.read_bytes() == original, "document bytes must be untouched"
        assert zipfile.is_zipfile(docm), "document must remain a valid container"

    def test_write_file_rejects_new_docx(self, tmp_path: Path):
        result = json.loads(write_file_tool(str(tmp_path / "new.docx"), "hello"))
        assert result.get("error")
        assert not (tmp_path / "new.docx").exists()

    def test_write_file_rejects_existing_pdf_overwrite(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n1 0 obj\nendobj\n%%EOF\n")
        original = pdf.read_bytes()

        result = json.loads(write_file_tool(str(pdf), "replacement text"))

        assert result.get("error")
        assert pdf.read_bytes() == original

    def test_write_file_allows_new_pdf_creation(self, tmp_path: Path):
        pdf = tmp_path / "generated.pdf"
        result = json.loads(write_file_tool(str(pdf), "%PDF-1.4\n%%EOF\n"))
        assert not result.get("error")
        assert pdf.exists()

    def test_write_file_plain_text_unaffected(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        result = json.loads(write_file_tool(str(target), "hello world"))
        assert not result.get("error")
        assert target.read_text() == "hello world"


class TestPatchToolGuard:
    def test_patch_replace_rejects_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        result = json.loads(
            patch_tool(mode="replace", path=str(docx),
                       old_string="good", new_string="great")
        )

        assert result.get("error")
        assert docx.read_bytes() == original

    def test_patch_v4a_update_rejects_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {docx}\n"
            "@@\n"
            "-good\n"
            "+great\n"
            "*** End Patch"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))

        assert result.get("error")
        assert docx.read_bytes() == original

    def test_patch_v4a_delete_of_docx_not_blocked_by_guard(self, tmp_path: Path):
        # Delete doesn't write text content — the binary-document guard must
        # not fire for it (delete may still fail/succeed for other reasons).
        docx = tmp_path / "old.docx"
        _make_minimal_docx(docx)

        v4a = (
            "*** Begin Patch\n"
            f"*** Delete File: {docx}\n"
            "*** End Patch"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))
        err = result.get("error") or ""
        assert "binary document" not in err.lower()

    def test_patch_replace_plain_text_unaffected(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("hello world")
        result = json.loads(
            patch_tool(mode="replace", path=str(target),
                       old_string="world", new_string="there")
        )
        assert not result.get("error")
        assert target.read_text() == "hello there"
