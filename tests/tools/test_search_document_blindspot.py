"""A content search that finds nothing must not imply the text is absent.

search_files is ripgrep-backed and cannot see inside PDFs or Office documents,
while read_file extracts them. On a folder of paperwork that mismatch produces
a confident false negative: the search returns empty and the answer is in the
PDF sitting next to it.
"""

import json

from tools.file_tools import _unsearched_documents, search_tool


def test_lists_documents_a_content_search_cannot_read(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing relevant here")
    (tmp_path / "tickets.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / "sheet.xlsx").write_bytes(b"PK\x03\x04")

    out = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path)))
    hint = out.get("_documents_not_searched", "")
    assert "tickets.pdf" in hint and "sheet.xlsx" in hint
    assert "read_file" in hint


def test_no_hint_when_there_are_no_documents(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing relevant here")
    out = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path)))
    assert "_documents_not_searched" not in out


def test_no_hint_when_the_search_matched(tmp_path):
    (tmp_path / "notes.txt").write_text("OATH appears here")
    (tmp_path / "tickets.pdf").write_bytes(b"%PDF-1.4 binary")
    out = json.loads(search_tool(pattern="OATH", target="content", path=str(tmp_path)))
    assert "_documents_not_searched" not in out


def test_scan_is_bounded_and_never_raises(tmp_path):
    assert _unsearched_documents(str(tmp_path / "does-not-exist")) == []
    for i in range(15):
        (tmp_path / f"doc{i}.pdf").write_bytes(b"%PDF")
    assert len(_unsearched_documents(str(tmp_path))) == 10  # capped


def test_hint_respects_file_glob(tmp_path):
    """A search narrowed to source files must not point at excluded documents."""
    (tmp_path / "notes.py").write_text("nothing relevant here")
    (tmp_path / "tickets.pdf").write_bytes(b"%PDF-1.4 binary")

    scoped = json.loads(search_tool(pattern="OATH", target="content",
                                    path=str(tmp_path), file_glob="*.py"))
    assert "_documents_not_searched" not in scoped

    # A glob that selects documents is exactly when the blind spot bites.
    on_docs = json.loads(search_tool(pattern="OATH", target="content",
                                     path=str(tmp_path), file_glob="*.pdf"))
    assert "tickets.pdf" in on_docs["_documents_not_searched"]
