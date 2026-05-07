import json

from tools import feishu_doc_tool as doc


def test_doc_read_requires_token():
    result = json.loads(doc._handle_doc_read({}))
    assert "doc_token is required" in result["error"]


def test_doc_read_calls_raw_content(monkeypatch):
    calls = []

    def fake_request(method, uri, **kwargs):
        calls.append((method, uri, kwargs))
        return {"content": "hello"}

    monkeypatch.setattr(doc, "request_json", fake_request)

    result = json.loads(doc._handle_doc_read({"doc_token": "doc123"}))

    assert result["success"] is True
    assert result["content"] == "hello"
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/raw_content")
    assert calls[0][2]["paths"] == {"document_id": "doc123"}


def test_doc_create_requires_title():
    result = json.loads(doc._handle_doc_create({}))
    assert "title is required" in result["error"]


def test_doc_create_sends_title_and_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(doc, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"document": {"document_id": "d"}})

    result = json.loads(doc._handle_doc_create({"title": "PRD", "folder_token": "fld"}))

    assert result["success"] is True
    assert calls[0][0] == "POST"
    assert calls[0][2]["body"] == {"title": "PRD", "folder_token": "fld"}


def test_doc_append_text_builds_paragraph_blocks(monkeypatch):
    calls = []
    monkeypatch.setattr(doc, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"children": [1, 2]})

    result = json.loads(doc._handle_doc_append_text({"doc_token": "doc", "text": "one\ntwo"}))

    assert result["success"] is True
    body = calls[0][2]["body"]
    assert len(body["children"]) == 2
    assert calls[0][2]["paths"] == {"document_id": "doc", "block_id": "doc"}


def test_doc_append_markdown_converts_to_native_blocks(monkeypatch):
    calls = []

    def fake_request(method, uri, **kwargs):
        calls.append((method, uri, kwargs))
        if uri.endswith("/blocks/convert"):
            return {"blocks": [{"block_id": "b", "block_type": 3, "heading1": {"elements": []}}], "first_level_block_ids": ["b"]}
        return {"children": [{"block_type": 3}]}

    monkeypatch.setattr(doc, "request_json", fake_request)

    result = json.loads(doc._handle_doc_append_markdown({"doc_token": "doc", "markdown": "# Title"}))

    assert result["success"] is True
    assert result["block_count"] == 1
    assert calls[0][1].endswith("/blocks/convert")
    assert calls[1][2]["body"]["children"][0]["block_type"] == 3


def test_convert_markdown_uses_first_level_order(monkeypatch):
    def fake_request(method, uri, **kwargs):
        return {
            "blocks": [
                {"block_id": "second", "block_type": 2},
                {"block_id": "first", "block_type": 3},
            ],
            "first_level_block_ids": ["first", "second"],
        }

    monkeypatch.setattr(doc, "request_json", fake_request)

    blocks = doc._convert_markdown("# Title\n\nBody")

    assert [block["block_id"] for block in blocks] == ["first", "second"]


def test_insert_blocks_chunks_feishu_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(doc, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"children": kwargs["body"]["children"]})

    data = doc._insert_blocks("doc", [{"block_type": 2, "i": i} for i in range(51)])

    assert len(data["children"]) == 51
    assert len(calls) == 2
    assert len(calls[0][2]["body"]["children"]) == 50
    assert len(calls[1][2]["body"]["children"]) == 1


def test_doc_replace_markdown_clears_then_inserts(monkeypatch):
    calls = []

    def fake_request(method, uri, **kwargs):
        calls.append((method, uri, kwargs))
        if uri.endswith("/blocks/convert"):
            return {"blocks": [{"block_id": "b", "block_type": 4, "heading2": {"elements": []}}], "first_level_block_ids": ["b"]}
        if uri.endswith("/blocks"):
            return {"items": [{"block_type": 1}, {"block_type": 2, "parent_id": "doc"}]}
        return {}

    monkeypatch.setattr(doc, "request_json", fake_request)

    result = json.loads(doc._handle_doc_replace_markdown({"doc_token": "doc", "markdown": "## Title"}))

    assert result["success"] is True
    assert result["deleted_blocks"] == 1
    assert [call[0] for call in calls] == ["POST", "GET", "DELETE", "POST"]
    assert calls[2][2]["body"] == {"start_index": 0, "end_index": 1}