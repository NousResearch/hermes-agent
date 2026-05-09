import json

from tools import mem0_shadow_tool
from toolsets import resolve_toolset


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def _clear_env(monkeypatch):
    for key in [
        "MEM0_SHADOW_BASE_URL",
        "MEM0_SHADOW_API_KEY",
        "MEM0_SHADOW_USER_ID",
        "MEM0_SHADOW_AGENT_ID",
        "MEM0_SHADOW_PROJECT",
        "MEM0_SHADOW_CANDIDATE_K",
        "MEM0_SHADOW_LOCAL_RERANK",
        "MEM0_SHADOW_STRICT_SEARCH",
        "MEM0_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_requirements_false_without_shadow_config(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert mem0_shadow_tool._requirements_met() is False


def test_requirements_true_from_shadow_config_file(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mem0_shadow.json").write_text(
        json.dumps({"base_url": "http://127.0.0.1:8888", "api_key": "m0adm_test"})
    )

    assert mem0_shadow_tool._requirements_met() is True


def test_mem0_shadow_search_uses_strict_filters_candidate_expansion_and_preserves_metadata(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_SHADOW_BASE_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("MEM0_SHADOW_API_KEY", "m0adm_test")
    monkeypatch.setenv("MEM0_SHADOW_USER_ID", "joohyun-memory-shadow-r3")
    monkeypatch.setenv("MEM0_SHADOW_AGENT_ID", "cortex-shadow-structured-r3")
    monkeypatch.setenv("MEM0_SHADOW_PROJECT", "mina-operating-system")
    monkeypatch.setenv("MEM0_SHADOW_CANDIDATE_K", "50")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({
            "results": [
                {
                    "memory": "MEMORY_RECORD\ntitle: mem0 strict adapter\ncontent: source metadata survives",
                    "score": 0.3,
                    "metadata": {
                        "source_id": 17408,
                        "project": "mina-operating-system",
                        "record_type": "decision",
                    },
                }
            ]
        })

    monkeypatch.setattr("plugins.memory.mem0.urllib.request.urlopen", fake_urlopen)

    out = json.loads(mem0_shadow_tool._search({"query": "mem0 strict adapter", "top_k": 3}))

    assert captured["url"] == "http://127.0.0.1:8888/search"
    assert captured["headers"].get("X-api-key") == "m0adm_test"
    assert "Authorization" not in captured["headers"]
    assert captured["body"] == {
        "query": "mem0 strict adapter",
        "filters": {
            "user_id": "joohyun-memory-shadow-r3",
            "agent_id": "cortex-shadow-structured-r3",
            "project": "mina-operating-system",
        },
        "top_k": 50,
        "rerank": False,
    }
    assert out["shadow_only"] is True
    assert out["authoritative"] is False
    assert out["results"][0]["source_id"] == 17408
    assert out["results"][0]["metadata"]["record_type"] == "decision"


def test_mem0_shadow_search_can_override_project_and_namespace(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_SHADOW_BASE_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("MEM0_SHADOW_API_KEY", "m0sk_test")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"results": []})

    monkeypatch.setattr("plugins.memory.mem0.urllib.request.urlopen", fake_urlopen)

    out = json.loads(mem0_shadow_tool._search({
        "query": "agentb",
        "project": "agentb",
        "user_id": "u-override",
        "agent_id": "agent-override",
    }))

    assert captured["body"]["filters"] == {
        "user_id": "u-override",
        "agent_id": "agent-override",
        "project": "agentb",
    }
    assert out["filters"] == captured["body"]["filters"]


def test_memory_toolset_contains_mem0_shadow_search():
    assert "mem0_shadow_search" in resolve_toolset("memory")
