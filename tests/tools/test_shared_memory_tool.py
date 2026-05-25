import json

from tools import shared_memory_tool


def test_check_requirements_false_without_key(monkeypatch):
    monkeypatch.delenv("SAM_API_KEY", raising=False)
    monkeypatch.delenv("SHARED_MEMORY_API_KEY", raising=False)

    assert shared_memory_tool.check_shared_memory_requirements() is False


def test_search_dispatch_redacts_actor_fields(monkeypatch):
    monkeypatch.setenv("SAM_API_KEY", "test-key")

    def fake_request(method, path, body=None):
        assert method == "POST"
        assert path == "/search"
        assert body == {"query": "q", "limit": 2}
        return {
            "results": [
                {
                    "id": "mem-1",
                    "created_by": "test-key",
                    "reviewed_by": "reviewer-key",
                    "metadata": {"actor": "test-key"},
                }
            ]
        }

    monkeypatch.setattr(shared_memory_tool, "_request", fake_request)

    payload = json.loads(shared_memory_tool.shared_memory_search(query="q", limit=2))
    record = payload["results"][0]
    assert record["created_by"] == "[REDACTED]"
    assert record["reviewed_by"] == "[REDACTED]"
    assert record["metadata"]["actor"] == "[REDACTED]"


def test_update_status_rejects_invalid_action():
    payload = json.loads(shared_memory_tool.shared_memory_update_status("mem-1", "publish"))

    assert payload["error"] == "invalid_action"
