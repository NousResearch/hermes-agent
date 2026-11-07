"""Invalid shared-string references must never borrow another cell's text."""

import json
import zipfile

import pytest

from tools import file_tools
from tools.registry import registry

S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"


def _read_cell(tmp_path, monkeypatch, index):
    path = tmp_path / "indices.xlsx"
    parts = {
        "[Content_Types].xml": "<Types/>",
        "xl/workbook.xml": (
            f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets>'
            '<sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<Relationships xmlns="{P}"><Relationship Id="rId1" '
            f'Type="{R}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/sharedStrings.xml": (
            f'<sst xmlns="{S}"><si><t>first string</t></si>'
            '<si><r><t>東</t></r><r><t>京</t></r></si></sst>'
        ),
        "xl/worksheets/sheet1.xml": (
            f'<worksheet xmlns="{S}"><sheetData><row r="1">'
            f'<c r="A1" t="s"><v>{index}</v></c>'
            '<c r="B1" t="inlineStr"><is><t>sentinel</t></is></c>'
            '</row></sheetData></worksheet>'
        ),
    }
    with zipfile.ZipFile(path, "w") as package:
        for name, xml in parts.items():
            package.writestr(name, xml)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    task_id = "shared-string-indices"
    try:
        result = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task_id))
        assert not result.get("error"), result
        assert result["extracted_document"] is True
        return result["content"].splitlines()[1].split("|", 1)[1]
    finally:
        file_tools.clear_file_ops_cache(task_id)


@pytest.mark.parametrize("index", ["-1", "-2", "-3", "2", "invalid", ""])
def test_invalid_shared_string_references_leave_the_cell_empty(tmp_path, monkeypatch, index):
    assert _read_cell(tmp_path, monkeypatch, index) == "\tsentinel"


@pytest.mark.parametrize("index, expected", [("0", "first string"), ("1", "東京")])
def test_valid_shared_string_boundaries_preserve_plain_and_rich_text(
    tmp_path, monkeypatch, index, expected
):
    assert _read_cell(tmp_path, monkeypatch, index) == expected + "\tsentinel"
