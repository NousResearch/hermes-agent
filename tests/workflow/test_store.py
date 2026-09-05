"""Workflow documents persist under HERMES_HOME, not the renderer cache."""

from workflow.store import (
    get_document,
    load_documents,
    remove_document,
    save_documents,
    upsert_document,
)


def test_round_trip(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    saved = save_documents(
        [{"id": "ship", "name": "Ship", "scenario": {"steps": [], "edges": []}}],
        "ship",
    )
    assert saved["currentId"] == "ship"
    loaded = load_documents()
    assert loaded["docs"][0]["id"] == "ship"
    assert (home / "workflows" / "documents.json").exists()


def test_lookup_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    upsert_document({"id": "figma-to-pr", "name": "Figma → PR", "scenario": {"steps": [], "edges": []}})
    assert get_document("Figma → PR")["id"] == "figma-to-pr"


def test_remove_falls_to_neighbour(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents(
        [
            {"id": "a", "name": "A", "scenario": {"steps": [], "edges": []}},
            {"id": "b", "name": "B", "scenario": {"steps": [], "edges": []}},
        ],
        "a",
    )
    out = remove_document("a")
    assert [d["id"] for d in out["docs"]] == ["b"]
    assert out["currentId"] == "b"
