import json

from tools import feishu_drive_tool as drive


def _patch(monkeypatch):
    calls = []
    monkeypatch.setattr(drive, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"ok": True})
    return calls


def test_get_meta_requires_file_token():
    assert "file_token is required" in json.loads(drive._get_meta({}))["error"]


def test_get_meta_uses_batch_meta_endpoint(monkeypatch):
    calls = _patch(monkeypatch)
    drive._get_meta({"file_token": "tok", "file_type": "docx"})
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/metas/batch_query")
    assert calls[0][2]["body"] == {"request_docs": [{"doc_token": "tok", "doc_type": "docx"}]}


def test_search_files_requires_query():
    assert "query is required" in json.loads(drive._search_files({}))["error"]


def test_search_files_posts_search_key(monkeypatch):
    calls = _patch(monkeypatch)
    drive._search_files({"query": "prd", "page_size": 20})
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/files/search")
    assert calls[0][2]["body"] == {"search_key": "prd"}
    assert calls[0][2]["queries"]["page_size"] == 20


def test_create_folder_requires_name():
    assert "name is required" in json.loads(drive._create_folder({}))["error"]


def test_comments_and_reply(monkeypatch):
    calls = _patch(monkeypatch)
    drive._list_comments({"file_token": "file", "is_whole": True})
    drive._reply_comment({"file_token": "file", "comment_id": "c", "content": "ok"})
    assert calls[0][0] == "GET"
    assert calls[0][2]["paths"] == {"file_token": "file"}
    assert calls[1][0] == "POST"
    assert calls[1][2]["paths"] == {"file_token": "file", "comment_id": "c"}
    assert "reply" in calls[1][2]["body"]