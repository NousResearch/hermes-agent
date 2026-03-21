"""Tests for tools/document_parse_tool.py."""

from __future__ import annotations

import json

from model_tools import get_tool_definitions
from tools import document_parse_tool as tool


def test_document_parse_tool_requires_path():
    result = json.loads(tool.document_parse_tool(path=""))

    assert result["success"] is False
    assert "Path is required" in result["error"]


def test_document_parse_tool_returns_truncated_text(monkeypatch):
    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-cli"
        text = "abcdefghij"
        metadata = {"page_count": 1}
        pages = []

    monkeypatch.setattr(tool, "parse_document", lambda path, backend="auto", parse_options=None: FakeParsed())

    result = json.loads(tool.document_parse_tool(path="/tmp/report.pdf", max_chars=5))

    assert result["success"] is True
    assert result["parser_backend"] == "liteparse-cli"
    assert result["truncated"] is True
    assert "...[truncated]..." in result["text"]


def test_document_parse_tool_honors_zero_max_chars(monkeypatch):
    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-cli"
        text = "abcdefghij"
        metadata = {"page_count": 1}
        pages = []

    monkeypatch.setattr(tool, "parse_document", lambda path, backend="auto", parse_options=None: FakeParsed())

    result = json.loads(tool.document_parse_tool(path="/tmp/report.pdf", max_chars=0))

    assert result["success"] is True
    assert result["text"] == ""
    assert result["truncated"] is True


def test_document_parse_tool_includes_pages(monkeypatch):
    class FakePage:
        page_number = 1
        text = "page text"
        items = [object(), object()]
        metadata = {"bounding_boxes": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}]}
        width = 100
        height = 200

    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-python"
        text = "all text"
        metadata = {"page_count": 1}
        pages = [FakePage()]

    monkeypatch.setattr(tool, "parse_document", lambda path, backend="auto", parse_options=None: FakeParsed())

    result = json.loads(tool.document_parse_tool(path="/tmp/report.pdf", include_pages=True))

    assert result["success"] is True
    assert result["page_count"] == 1
    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][0]["item_count"] == 2
    assert result["pages"][0]["bounding_box_count"] == 1


def test_document_parse_tool_passes_parse_options(monkeypatch):
    captured = {}

    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-python"
        text = "all text"
        metadata = {"page_count": 0}
        pages = []

    def fake_parse_document(path, backend="auto", parse_options=None):
        captured["path"] = path
        captured["backend"] = backend
        captured["parse_options"] = parse_options
        return FakeParsed()

    monkeypatch.setattr(tool, "parse_document", fake_parse_document)

    result = json.loads(
        tool.document_parse_tool(
            path="/tmp/report.pdf",
            backend="liteparse",
            ocr_enabled=False,
            ocr_language="fra",
            target_pages="1-3",
            dpi=300,
            precise_bounding_box=False,
            preserve_small_text=True,
        )
    )

    assert result["success"] is True
    assert captured["backend"] == "liteparse"
    assert captured["parse_options"]["ocr_enabled"] is False
    assert captured["parse_options"]["ocr_language"] == "fra"
    assert captured["parse_options"]["target_pages"] == "1-3"
    assert captured["parse_options"]["dpi"] == 300
    assert captured["parse_options"]["no_precise_bbox"] is True
    assert captured["parse_options"]["preserve_small_text"] is True


def test_document_parse_tool_includes_text_items_and_bounding_boxes(monkeypatch):
    class FakeItem:
        text = "hello"
        bbox = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
        confidence = 0.9

    class FakePage:
        page_number = 1
        text = "page text"
        items = [FakeItem()]
        metadata = {"bounding_boxes": [{"x1": 5, "y1": 6, "x2": 7, "y2": 8}]}
        width = 100
        height = 200

    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-python"
        text = "all text"
        metadata = {"page_count": 1}
        pages = [FakePage()]

    monkeypatch.setattr(tool, "parse_document", lambda path, backend="auto", parse_options=None: FakeParsed())

    result = json.loads(
        tool.document_parse_tool(
            path="/tmp/report.pdf",
            include_pages=True,
            include_text_items=True,
            include_bounding_boxes=True,
        )
    )

    assert result["pages"][0]["text_items"][0]["text"] == "hello"
    assert result["pages"][0]["bounding_boxes"][0]["x1"] == 5


def test_document_parse_tool_generates_screenshots(monkeypatch):
    class FakeParsed:
        source_path = "/tmp/report.pdf"
        parser_backend = "liteparse-python"
        text = "all text"
        metadata = {"page_count": 0}
        pages = []

    class FakeShot:
        def to_dict(self):
            return {"page_number": 1, "image_path": "/tmp/page_1.png", "image_format": "png"}

    monkeypatch.setattr(tool, "parse_document", lambda path, backend="auto", parse_options=None: FakeParsed())
    monkeypatch.setattr(tool, "create_document_screenshots", lambda **kwargs: [FakeShot()])

    result = json.loads(
        tool.document_parse_tool(
            path="/tmp/report.pdf",
            generate_screenshots=True,
            screenshot_image_format="png",
        )
    )

    assert result["screenshots"][0]["page_number"] == 1
    assert result["screenshots"][0]["image_format"] == "png"


def test_document_parse_tool_surfaces_parser_errors(monkeypatch):
    monkeypatch.setattr(
        tool,
        "parse_document",
        lambda path, backend="auto", parse_options=None: (_ for _ in ()).throw(tool.DocumentParsingError("bad parse")),
    )

    result = json.loads(tool.document_parse_tool(path="/tmp/report.pdf"))

    assert result["success"] is False
    assert result["error"] == "bad parse"


def test_document_parse_tool_is_exposed_in_tool_definitions():
    tool_names = [entry["function"]["name"] for entry in get_tool_definitions(["documents"], quiet_mode=True)]

    assert "document_parse" in tool_names
