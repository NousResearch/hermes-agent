import json

from tools import feishu_bitable_tool as bitable


def _patch(monkeypatch):
    calls = []
    monkeypatch.setattr(bitable, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"ok": True})
    return calls


def test_create_app_calls_endpoint(monkeypatch):
    calls = _patch(monkeypatch)
    result = json.loads(bitable._create_app({"name": "PM Base", "folder_token": "fld"}))
    assert result["success"] is True
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/apps")
    assert calls[0][2]["body"] == {"name": "PM Base", "folder_token": "fld"}


def test_list_tables_requires_app_token():
    assert "app_token required" in json.loads(bitable._list_tables({}))["error"]


def test_list_tables_calls_endpoint(monkeypatch):
    calls = _patch(monkeypatch)
    result = json.loads(bitable._list_tables({"app_token": "app", "page_size": 10}))
    assert result["success"] is True
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/tables")
    assert calls[0][2]["paths"] == {"app_token": "app"}
    assert calls[0][2]["queries"]["page_size"] == 10


def test_get_fields_requires_table_id():
    assert "table_id required" in json.loads(bitable._get_fields({"app_token": "app"}))["error"]


def test_update_table_and_field_shapes(monkeypatch):
    calls = _patch(monkeypatch)
    bitable._update_table({"app_token": "app", "table_id": "tbl", "name": "项目任务"})
    bitable._create_field({"app_token": "app", "table_id": "tbl", "field_name": "模块", "type": 3, "property": {"options": [{"name": "A"}]}})
    bitable._update_field({"app_token": "app", "table_id": "tbl", "field_id": "fld", "field_name": "状态", "type": 3})
    assert calls[0][0] == "PATCH"
    assert calls[0][2]["body"] == {"name": "项目任务"}
    assert calls[1][0] == "POST"
    assert calls[1][2]["body"]["field_name"] == "模块"
    assert calls[1][2]["body"]["type"] == 3
    assert calls[2][0] == "PUT"
    assert calls[2][2]["paths"]["field_id"] == "fld"


def test_search_records_posts_search_body(monkeypatch):
    calls = _patch(monkeypatch)
    bitable._search_records({"app_token": "app", "table_id": "tbl", "filter": {"x": 1}})
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/records/search")
    assert calls[0][2]["body"] == {"filter": {"x": 1}}


def test_create_record_requires_fields_object():
    assert "fields must be a non-empty object" in json.loads(bitable._create_record({"app_token": "app", "table_id": "tbl"}))["error"]


def test_create_update_delete_record_shapes(monkeypatch):
    calls = _patch(monkeypatch)
    bitable._create_record({"app_token": "app", "table_id": "tbl", "fields": {"Name": "A"}})
    bitable._update_record({"app_token": "app", "table_id": "tbl", "record_id": "rec", "fields": {"Name": "B"}})
    bitable._delete_record({"app_token": "app", "table_id": "tbl", "record_id": "rec"})

    assert calls[0][0] == "POST"
    assert calls[0][2]["body"] == {"fields": {"Name": "A"}}
    assert calls[1][0] == "PUT"
    assert calls[1][2]["paths"]["record_id"] == "rec"
    assert calls[2][0] == "DELETE"