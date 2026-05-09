import json

from tools.recall_tool import recall


class DummyMemoryManager:
    def __init__(self):
        self.calls = []

    def recall_now_all(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return "graph memory hit"


def test_recall_light_defaults_to_graph_when_provider_available():
    manager = DummyMemoryManager()
    raw = recall("Graphiti auth", depth="light", memory_manager=manager, db=None)
    data = json.loads(raw)

    assert data["success"] is True
    assert data["sources"] == ["graph"]
    assert data["results"]["graph"] == "graph memory hit"
    assert manager.calls[0][1]["mode"] == "manual"
    assert manager.calls[0][1]["depth"] == "light"


def test_recall_standard_routes_graph_and_session_search(monkeypatch):
    manager = DummyMemoryManager()

    def fake_session_search(**kwargs):
        return json.dumps({"success": True, "results": [{"summary": "session hit"}]})

    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)
    raw = recall(
        "recall router",
        depth="standard",
        sources=["graph", "session_fts"],
        budget="small",
        limit=5,
        memory_manager=manager,
        db=object(),
        current_session_id="s1",
    )
    data = json.loads(raw)

    assert data["success"] is True
    assert data["budget"] == "small"
    assert data["sources"] == ["graph", "session_fts"]
    assert data["results"]["graph"] == "graph memory hit"
    assert data["results"]["sessions"]["results"][0]["summary"] == "session hit"
    assert data["recall_key"]
    assert data["auto_recall_key"]
    assert data["auto_recall_key"] != data["recall_key"]


def test_recall_auto_key_matches_manual_query_for_auto_suppression():
    manager = DummyMemoryManager()
    manual = json.loads(recall("Graphiti auth", depth="light", memory_manager=manager))
    auto = json.loads(recall("Graphiti auth", mode="auto", depth="light", memory_manager=manager))

    assert manual["auto_recall_key"] == auto["recall_key"]


def test_recall_reports_missing_sources():
    raw = recall("history", depth="deep", sources=["graph", "session_summary"])
    data = json.loads(raw)

    assert data["success"] is False
    assert "graph source requested" in "\n".join(data["errors"])
    assert "session source requested" in "\n".join(data["errors"])


def test_recall_budget_clamps_session_limit(monkeypatch):
    seen = {}

    def fake_session_search(**kwargs):
        seen.update(kwargs)
        return json.dumps({"success": True, "results": []})

    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)
    raw = recall(
        "history",
        depth="deep",
        sources=["session_summary"],
        budget="tiny",
        limit=5,
        db=object(),
    )
    data = json.loads(raw)

    assert data["success"] is True
    assert seen["limit"] == 1


def test_recall_surfaces_nested_session_failure(monkeypatch):
    def fake_session_search(**kwargs):
        return json.dumps({"success": False, "error": "fts unavailable"})

    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)
    raw = recall("history", depth="standard", sources=["session_fts"], db=object())
    data = json.loads(raw)

    assert data["success"] is False
    assert data["results"]["sessions"]["success"] is False
    assert "session recall returned unsuccessful result" in data["errors"]
