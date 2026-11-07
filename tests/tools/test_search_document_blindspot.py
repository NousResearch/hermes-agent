"""search_files must not be blind to the text inside PDFs and Office documents.

ripgrep skips binary files while read_file extracts them, so a content search
over a folder of paperwork returned a confident zero while the answer sat in the
.docx next to the searched .txt (#101427).
"""

import json
import zipfile

from tools.file_tools import read_file_tool, search_tool


def _docx(path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", (
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/></Types>'))
        z.writestr("word/document.xml", (
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
            f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'))


def test_content_search_finds_document_text_at_read_file_line_numbers(tmp_path):
    (tmp_path / "cover_letter.txt").write_text("nothing relevant here\n")
    doc = tmp_path / "violations.docx"
    _docx(doc, ["Case file 2026-114", "Background.", "Notice of violation issued by OATH and the ECB."])

    out = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path)))
    assert f"{doc}\n  3: Notice of violation issued by OATH" in out["document_matches"]
    # The reported line is the line read_file shows for that path.
    assert "3|Notice of violation issued by OATH" in read_file_tool(path=str(doc), offset=3, limit=1)

    # The document itself as the search root is searched the same way.
    single = json.loads(search_tool(pattern="OATH", target="content", path=str(doc)))
    assert "3: Notice of violation" in single["document_matches"]

    # A search narrowed to source files never walks or reports documents, while a
    # name-pinned document glob still reaches it.
    scoped = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path),
                                    file_glob="*.py"))
    assert not any(key.startswith(("document", "_documents")) for key in scoped)
    pinned = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path),
                                    file_glob="viol*.docx"))
    assert "3: Notice of violation" in pinned["document_matches"]


def test_documents_that_cannot_be_extracted_are_named_not_dropped(tmp_path):
    (tmp_path / "notes.txt").write_text("OATH appears in plain text too\n")
    (tmp_path / "tickets.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / "sheet.xlsx").write_bytes(b"PK\x03\x04")

    # Named even when the text files matched: the unread PDF may hold the better answer.
    out = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path)))
    hint = out["_documents_not_searched"]
    assert "tickets.pdf" in hint and "sheet.xlsx" in hint
    assert "document_matches" not in out
