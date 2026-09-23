"""Phonetic guides (XLSX ``rPh``, DOCX ``w:rt`` ruby) annotate base text; they are not the value.

See https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.spreadsheet.phoneticrun
(rPh is permitted under both si and is).
"""
import json
import zipfile

import pytest

from tools import file_tools  # registers the public read_file handler
from tools.read_extract import extract_document_text
from tools.registry import registry

S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"


def _workbook(path, storage, text):
    cell = '<c r="A1" t="s"><v>0</v></c>' if storage == "shared" else f'<c r="A1" t="inlineStr"><is>{text}</is></c>'
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>',
        "_rels/.rels": f'<Relationships xmlns="{P}"><Relationship Id="rId1" Type="{R}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Cities" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{P}"><Relationship Id="rId1" Type="{R}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{R}/sharedStrings" Target="sharedStrings.xml"/></Relationships>',
        "xl/sharedStrings.xml": f'<sst xmlns="{S}" count="1" uniqueCount="1"><si>{text}</si></sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{S}"><sheetData><row r="1">{cell}<c r="B1" t="inlineStr"><is><t>sentinel</t></is></c></row></sheetData></worksheet>',
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        for name, xml in parts.items():
            package.writestr(name, xml)
    return path


def _docx(path, paragraph_xml):
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr("word/document.xml",
                         f'<w:document xmlns:w="{W}"><w:body><w:p>{paragraph_xml}</w:p></w:body></w:document>')
    return path


_XLSX_PHONETICS = '<rPh sb="0" eb="1"><t>トウ</t></rPh><rPh sb="1" eb="2"><t>キョウ</t></rPh>'
_DOCX_RUBY = ('<w:r><w:ruby><w:rt><w:r><w:t>トウキョウ</w:t></w:r></w:rt>'
              '<w:rubyBase><w:r><w:t>東京</w:t></w:r></w:rubyBase></w:ruby></w:r><w:r><w:tab/><w:t>sentinel</w:t></w:r>')


def _document(tmp_path, kind):
    if kind == "docx-ruby":
        return _docx(tmp_path / "cities.docx", _DOCX_RUBY)
    storage, rich = kind.split("-")
    base = '<r><rPr><b/></rPr><t>東</t></r><r><t>京</t></r>' if rich == "rich" else '<t>東京</t>'
    return _workbook(tmp_path / "cities.xlsx", storage, base + _XLSX_PHONETICS)


@pytest.mark.parametrize("kind", ["shared-plain", "shared-rich", "inline-plain", "inline-rich", "docx-ruby"])
def test_read_file_excludes_phonetic_guides(tmp_path, monkeypatch, kind):
    path = _document(tmp_path, kind)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    task_id = f"phonetic-{kind}"
    try:
        result = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task_id))
        assert not result.get("error"), result
        assert result["extracted_document"] is True
        row = result["content"].splitlines()[0 if kind == "docx-ruby" else 1]  # XLSX line 1 is the sheet header
        assert row.split("|", 1)[1] == "東京\tsentinel"
    finally:
        file_tools.clear_file_ops_cache(task_id)


@pytest.mark.parametrize("storage", ["shared", "inline"])
@pytest.mark.parametrize("text, expected", [
    ('<t xml:space="preserve"> plain </t>', ' plain '),
    ('<r><t xml:space="preserve"> first </t></r><r><rPr><b/></rPr><t>second</t></r>', ' first second'),
    ('<t/>', ''),
])
def test_string_values_without_phonetics_are_preserved(tmp_path, storage, text, expected):
    path = _workbook(tmp_path / "control.xlsx", storage, text)
    assert extract_document_text(str(path)).splitlines()[1] == expected + "\tsentinel"
