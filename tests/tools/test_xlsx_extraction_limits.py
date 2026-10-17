"""The read_file result must disclose cells omitted by XLSX extraction limits."""

import json
import zipfile

import pytest

from tools import file_tools, read_extract  # noqa: F401 -- registers read_file
from tools.registry import registry


def _column(number):
    result = ""
    while number:
        number, digit = divmod(number - 1, 26)
        result = chr(65 + digit) + result
    return result


def _workbook(path, axis, exceeds):
    def cell(reference, text):
        return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'

    if axis == "rows":
        cap = read_extract._MAX_XLSX_ROWS_PER_SHEET
        rows = '<row r="1">' + cell("A1", "first-value") + '</row>'
        rows += ''.join(f'<row r="{i}"/>' for i in range(2, cap))
        rows += f'<row r="{cap}">' + cell(f"A{cap}", "boundary-value") + '</row>'
        if exceeds:
            rows += f'<row r="{cap + 1}">' + cell(f"A{cap + 1}", "omitted-value") + '</row>'
    else:
        cap = read_extract._MAX_XLSX_COLS
        rows = '<row r="1">' + cell("A1", "first-value")
        rows += cell(f"{_column(cap)}1", "boundary-value")
        if exceeds:
            rows += cell(f"{_column(cap + 1)}1", "omitted-value")
        rows += '</row>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{rows}</sheetData></worksheet>')


def _read_tail(path, axis):
    offset = read_extract._MAX_XLSX_ROWS_PER_SHEET if axis == "rows" else 1
    result = json.loads(registry.dispatch("read_file", {"path": str(path), "offset": offset, "limit": 10}))
    assert result.get("extracted_document"), result
    first_page = json.loads(registry.dispatch("read_file", {"path": str(path), "limit": 2}))
    assert first_page.get("extracted_document"), first_page
    return first_page["content"] + "\n" + result["content"]


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_read_file_discloses_omitted_xlsx_cells(tmp_path, axis):
    path = tmp_path / "exceeds.xlsx"
    _workbook(path, axis, exceeds=True)
    text = _read_tail(path, axis)
    assert "boundary-value" in text
    assert "omitted-value" not in text
    assert "XLSX extraction truncated" in text
    assert axis in text and "omitted" in text


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_read_file_keeps_boundary_cells_without_a_false_truncation_warning(tmp_path, axis):
    path = tmp_path / "boundary.xlsx"
    _workbook(path, axis, exceeds=False)
    text = _read_tail(path, axis)
    assert "boundary-value" in text
    assert "XLSX extraction truncated" not in text
