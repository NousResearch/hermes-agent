"""SQLite previews keep column boundaries when names or values contain Markdown syntax."""

import json
import re
import sqlite3
from contextlib import closing

import pytest
from markdown_it import MarkdownIt

from tools.file_tools import registry


def _preview_cells(tmp_path, monkeypatch, column, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    path = tmp_path / "preview.db"
    quoted = '"' + column.replace('"', '""') + '"'
    with closing(sqlite3.connect(path)) as db:
        db.execute(f"CREATE TABLE sample ({quoted} TEXT, control TEXT)")
        db.execute("INSERT INTO sample VALUES (?, ?)", (value, "plain control"))
        db.commit()
    original = path.read_bytes()

    result = json.loads(registry.dispatch(
        "read_file", {"path": str(path), "limit": 1000}, task_id=str(tmp_path),
    ))
    assert result.get("extracted_document"), result
    assert path.read_bytes() == original
    lines = [re.sub(r"^\s*\d+\|", "", line) for line in result["content"].splitlines()]
    table = "\n".join(line for line in lines if line.startswith("|"))
    tokens = MarkdownIt("commonmark").enable("table").parse(table)
    return [
        "".join(child.content for child in token.children or [])
        for token in tokens if token.type == "inline"
    ]


@pytest.mark.parametrize("column", [
    "buyer|seller", "first\nlast", "first\r\nlast", "first\rlast", r"back\slash|column",
])
def test_sqlite_column_names_render_as_single_header_cells(tmp_path, monkeypatch, column):
    cells = _preview_cells(tmp_path, monkeypatch, column, "simple value")
    assert cells == [" ".join(column.splitlines()), "control", "simple value", "plain control"]


@pytest.mark.parametrize("value", [
    "alpha|beta", "first\nlast", "first\r\nlast", "first\rlast", r"part\name|tail", r"part\|tail",
])
def test_sqlite_values_render_as_single_data_cells(tmp_path, monkeypatch, value):
    cells = _preview_cells(tmp_path, monkeypatch, "description", value)
    assert cells == ["description", "control", " ".join(value.splitlines()), "plain control"]
