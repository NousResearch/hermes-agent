"""DOCX non-breaking hyphens are visible text, not disposable run metadata."""

import io
import zipfile

from tools.read_extract import extract_document_bytes, extract_document_text


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _document_bytes():
    document = (
        f'<w:document xmlns:w="{_NS_W}"><w:body>'
        '<w:p><w:r><w:t>Use a non</w:t><w:noBreakHyphen/>'
        '<w:t>disclosure agreement.</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>part</w:t>'
        '<w:noBreakHyphen/><w:t>time</w:t><w:tab/><w:t>role</w:t>'
        '<w:br/><w:t>next line</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        '</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="word/document.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>'
            '</Relationships>',
        )
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


_EXPECTED = "Use a non-disclosure agreement.\npart-time\trole\nnext line\n"


def test_docx_path_preserves_nonbreaking_hyphens(tmp_path):
    path = tmp_path / "agreement.docx"
    path.write_bytes(_document_bytes())
    assert extract_document_text(str(path)) == _EXPECTED


def test_docx_backend_bytes_preserve_nonbreaking_hyphens():
    assert extract_document_bytes(_document_bytes(), "remote.docx") == _EXPECTED
