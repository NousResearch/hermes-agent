from pathlib import Path

import pytest


def test_candidate_store_uses_default_path_under_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore()

    assert store.path == tmp_path / "auto_learning" / "candidates.jsonl"
    assert store.path.parent.is_dir()


def test_candidate_store_append_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore(max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise answers",
        confidence=0.92,
        evidence={"source": "post_task_review"},
    )

    reloaded = AutoLearningStore(max_entries=10)
    items = reloaded.list_candidates()

    assert len(items) == 1
    assert items[0]["summary"] == "User prefers concise answers"
    assert items[0]["id"] == entry["id"]
    assert items[0]["status"] == "candidate"
    assert items[0]["category"] == "memory"
    assert items[0]["fingerprint"]
    assert items[0]["created_at"]


def test_candidate_store_enforces_max_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore(max_entries=2)
    store.add_candidate(category="memory", summary="one", confidence=0.9, evidence={"n": 1})
    store.add_candidate(category="memory", summary="two", confidence=0.9, evidence={"n": 2})
    store.add_candidate(category="memory", summary="three", confidence=0.9, evidence={"n": 3})

    items = store.list_candidates()
    assert [item["summary"] for item in items] == ["two", "three"]


def test_candidate_store_can_mark_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore(max_entries=10)
    entry = store.add_candidate(
        category="skill",
        summary="OpenVINO setup needs bootstrap first",
        confidence=0.88,
        evidence={"source": "post_task_review"},
    )

    updated = store.mark_status(entry["id"], "promoted", note="saved as skill")

    assert updated["status"] == "promoted"
    assert updated["promotion_note"] == "saved as skill"
    assert store.list_candidates(status="promoted")[0]["id"] == entry["id"]


def test_candidate_store_dedupes_exact_repeat_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore(max_entries=10)
    first = store.add_candidate(
        category="memory",
        summary="User prefers concise answers",
        confidence=0.92,
        evidence={"source": "post_task_review"},
        target="user",
    )
    second = store.add_candidate(
        category="memory",
        summary="  User prefers concise answers  ",
        confidence=0.95,
        evidence={"source": "different_review"},
        target="user",
    )

    assert second["id"] == first["id"]
    assert second["fingerprint"] == first["fingerprint"]
    assert len(store.list_candidates()) == 1


def test_candidate_store_find_by_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.auto_learning_store import AutoLearningStore

    store = AutoLearningStore(max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise answers",
        confidence=0.92,
        evidence={"source": "post_task_review"},
        target="user",
    )

    found = store.find_by_fingerprint(entry["fingerprint"])

    assert found is not None
    assert found["id"] == entry["id"]
