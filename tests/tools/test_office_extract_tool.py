import json

import pytest

from tools.office_extract_tool import office_extract, check_office_extract_requirements
from toolsets import get_toolset, resolve_toolset


def _load(result: str) -> dict:
    return json.loads(result)


def test_office_extract_reads_docx_text_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "brief.docx"
    doc = docx.Document()
    doc.add_heading("Client Brief", level=1)
    doc.add_paragraph("Launch a summer grill campaign for Miratorg.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Audience"
    table.cell(0, 1).text = "Families"
    table.cell(1, 0).text = "Tone"
    table.cell(1, 1).text = "Warm"
    doc.save(path)

    result = _load(office_extract(str(path)))

    assert result["success"] is True
    assert result["format"] == "docx"
    assert result["file_path"] == str(path)
    assert "# Client Brief" in result["markdown"]
    assert "Launch a summer grill campaign" in result["markdown"]
    assert "Audience | Families" in result["markdown"]
    assert result["metadata"]["paragraph_count"] >= 2
    assert result["metadata"]["table_count"] == 1


def test_office_extract_reads_pptx_slide_text(tmp_path):
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "deck.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Big idea"
    slide.placeholders[1].text = "Fire + family + weekend ritual"
    presentation.save(path)

    result = _load(office_extract(str(path)))

    assert result["success"] is True
    assert result["format"] == "pptx"
    assert "## Slide 1" in result["markdown"]
    assert "Big idea" in result["markdown"]
    assert "weekend ritual" in result["markdown"]
    assert result["metadata"]["slide_count"] == 1


def test_office_extract_reads_xlsx_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "budget.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Plan"
    sheet.append(["Channel", "Budget"])
    sheet.append(["Telegram", 100000])
    sheet.append(["Outdoor", 250000])
    workbook.save(path)

    result = _load(office_extract(str(path)))

    assert result["success"] is True
    assert result["format"] == "xlsx"
    assert "## Sheet: Plan" in result["markdown"]
    assert "Channel | Budget" in result["markdown"]
    assert "Telegram | 100000" in result["markdown"]
    assert result["metadata"]["sheet_count"] == 1


def test_office_extract_rejects_legacy_binary_office_formats(tmp_path):
    path = tmp_path / "legacy.doc"
    path.write_bytes(b"not really a doc")

    result = _load(office_extract(str(path)))

    assert result["success"] is False
    assert result["error_code"] == "unsupported_legacy_format"
    assert ".docx" in result["error"]


def test_office_extract_toolset_is_registered_when_requirements_are_available():
    assert check_office_extract_requirements() is True
    assert "office_extract" in get_toolset("office")["tools"]
    assert "office_extract" in resolve_toolset("office")
