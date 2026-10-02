"""read_file should expose current DOCX text, excluding removed revision content."""

import json
import zipfile

import pytest

from tools import file_tools
from tools.registry import registry

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _read_docx(tmp_path, monkeypatch, body):
    path = tmp_path / "revisions.docx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", "<Types/>")
        package.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W}" xmlns:v="urn:schemas-microsoft-com:vml">'
            f"<w:body>{body}</w:body></w:document>",
        )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    task_id = "docx-revisions"
    try:
        result = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task_id))
        assert not result.get("error"), result
        assert result["extracted_document"] is True
        return [line.split("|", 1)[1] for line in result["content"].splitlines()]
    finally:
        file_tools.clear_file_ops_cache(task_id)


@pytest.mark.parametrize("location", ["body", "table"])
def test_removed_runs_do_not_duplicate_text_or_emit_breaks(tmp_path, monkeypatch, location):
    paragraph = (
        '<w:p><w:r><w:t>A</w:t></w:r>'
        '<w:moveFromRangeStart w:id="0" w:name="move1"/>'
        '<w:moveFrom w:id="1" w:author="Editor"><w:r><w:t> moved</w:t></w:r></w:moveFrom>'
        '<w:moveFromRangeEnd w:id="0"/>'
        '<w:del w:id="2" w:author="Editor"><w:r><w:delText>old</w:delText>'
        '<w:tab/><w:br/><w:cr/></w:r></w:del>'
        '<w:ins w:id="3" w:author="Editor"><w:r><w:t>B</w:t></w:r></w:ins>'
        '<w:moveToRangeStart w:id="4" w:name="move1"/>'
        '<w:moveTo w:id="5" w:author="Editor"><w:r><w:t> moved</w:t></w:r></w:moveTo>'
        '<w:moveToRangeEnd w:id="4"/>'
        '<w:r><w:tab/><w:t>C</w:t><w:br/><w:t>D</w:t></w:r></w:p>'
    )
    body = paragraph if location == "body" else f"<w:tbl><w:tr><w:tc>{paragraph}</w:tc></w:tr></w:tbl>"
    assert _read_docx(tmp_path, monkeypatch, body) == ["AB moved\tC", "D"]


@pytest.mark.parametrize("revision", ["del", "moveFrom"])
def test_paragraphs_inside_removed_text_boxes_are_not_read(tmp_path, monkeypatch, revision):
    removed = (
        f'<w:{revision} w:id="1" w:author="Editor"><w:r><w:pict><v:shape><v:textbox>'
        '<w:txbxContent><w:p><w:r><w:t>removed callout</w:t></w:r></w:p></w:txbxContent>'
        f'</v:textbox></v:shape></w:pict></w:r></w:{revision}>'
    )
    if revision == "moveFrom":
        removed = '<w:moveFromRangeStart w:id="0" w:name="move1"/>' + removed + '<w:moveFromRangeEnd w:id="0"/>'
    body = f'<w:p><w:r><w:t>Visible body</w:t></w:r>{removed}</w:p>'
    if revision == "moveFrom":
        body += (
            '<w:p><w:moveToRangeStart w:id="2" w:name="move1"/>'
            '<w:moveTo w:id="3" w:author="Editor"><w:r><w:t>Current callout</w:t></w:r></w:moveTo>'
            '<w:moveToRangeEnd w:id="2"/></w:p>'
        )
    expected = ["Visible body"] + (["Current callout"] if revision == "moveFrom" else [])
    assert _read_docx(tmp_path, monkeypatch, body) == expected
