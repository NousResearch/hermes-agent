"""Workbook extraction distinguishes an uncached formula from an empty cell."""

import json
import zipfile

from tools import file_tools, terminal_tool
from tools.registry import registry


def test_read_file_keeps_uncached_formulas_and_cached_results(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    task = "xlsx-formula-read"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    path = tmp_path / "formulas.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'))
        archive.writestr("_rels/.rels", f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Target="xl/workbook.xml" Type="{rel}/officeDocument"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets><sheet name="FormulaOnly" sheetId="1" r:id="rId1"/><sheet name="Cached" sheetId="2" r:id="rId2"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="{rel}/worksheet"/><Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="{rel}/worksheet"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="A1"><f>SUM(1,2)</f><v/></c><c r="B1"><f>2+3</f></c></row></sheetData></worksheet>')
        archive.writestr("xl/worksheets/sheet2.xml", f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="A1"><f>SUM(1,2)</f><v>3</v></c><c r="B1" t="str"><f>""</f><v/></c></row></sheetData></worksheet>')
    terminal_tool.register_task_env_overrides(task, {"env_type": "local", "cwd": str(tmp_path)})
    try:
        result = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task))
        assert "error" not in result, result
        content = result["content"]
        assert "[uncached formula: =SUM(1,2)]" in content, content
        assert "[uncached formula: =2+3]" in content, content
        assert "(empty)" not in content, content
        cached = content.split("Sheet: Cached", 1)[1]
        cached_rows = [line.partition("|")[2] for line in cached.splitlines() if "|" in line]
        assert "3\t" in cached_rows and "uncached" not in cached, cached
    finally:
        file_tools.clear_file_ops_cache(task)
        terminal_tool.clear_task_env_overrides(task)
