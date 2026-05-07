import json

from tools import feishu_wiki_tool as wiki


def _patch(monkeypatch):
    calls = []
    monkeypatch.setattr(wiki, "request_json", lambda method, uri, **kwargs: calls.append((method, uri, kwargs)) or {"ok": True})
    return calls


def test_list_spaces_calls_endpoint(monkeypatch):
    calls = _patch(monkeypatch)
    wiki._list_spaces({"page_size": 5})
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/spaces")
    assert calls[0][2]["queries"]["page_size"] == 5


def test_list_nodes_requires_space_id():
    assert "space_id is required" in json.loads(wiki._list_nodes({}))["error"]


def test_list_nodes_calls_space_nodes(monkeypatch):
    calls = _patch(monkeypatch)
    wiki._list_nodes({"space_id": "sp", "parent_node_token": "parent"})
    assert calls[0][0] == "GET"
    assert calls[0][2]["paths"] == {"space_id": "sp"}
    assert calls[0][2]["queries"]["parent_node_token"] == "parent"


def test_get_node_requires_node_token():
    assert "node_token is required" in json.loads(wiki._get_node({"space_id": "sp"}))["error"]


def test_get_node_calls_endpoint(monkeypatch):
    calls = _patch(monkeypatch)
    wiki._get_node({"space_id": "sp", "node_token": "node"})
    assert calls[0][0] == "GET"
    assert calls[0][2]["paths"] == {"space_id": "sp", "node_token": "node"}


def test_create_node_body(monkeypatch):
    calls = _patch(monkeypatch)
    wiki._create_node({"space_id": "sp", "parent_node_token": "parent", "title": "Doc", "obj_token": "doc"})
    assert calls[0][0] == "POST"
    assert calls[0][2]["paths"] == {"space_id": "sp"}
    assert calls[0][2]["body"]["title"] == "Doc"
    assert calls[0][2]["body"]["obj_token"] == "doc"