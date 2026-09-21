"""GBrain tool tests — markdown-native filesystem backend.

``tools/gbrain.py`` was rewritten in dc765fd29 from a CLI-backed
``_run_gbrain`` to a markdown-native filesystem search/get/graph that
operates directly on ``~/brain`` (or ``$GBRAIN_REPO``).  These tests build
a temporary markdown-native repo and exercise the real functions — no
``_run_gbrain`` monkeypatch, no legacy CLI protocol.
"""

import json

from tools import gbrain


def _make_brain(tmp_path, pages):
    """Write a tiny markdown brain repo and return the path."""
    repo = tmp_path / "brain"
    repo.mkdir()
    for slug, content in pages.items():
        target = repo / f"{slug}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo


def test_gbrain_search_limits_results(monkeypatch, tmp_path):
    """keyword search returns structured page results and honours ``limit``."""
    repo = _make_brain(tmp_path, {
        "people/sahil-saghir": "# Sahil\nSahil founded KenseiAgent.",
        "projects/kensei": "# Kensei\nKensei is the agent platform.",
    })
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_search({"query": "sahil", "limit": 1}))

    assert data["query"] == "sahil"
    assert data["limit"] == 1
    assert data["total"] == 1
    slugs = [r["slug"] for r in data["results"]]
    assert "people/sahil-saghir" in slugs


def test_gbrain_get_returns_content(monkeypatch, tmp_path):
    """``gbrain_get`` reads the actual markdown fixture."""
    repo = _make_brain(tmp_path, {"people/sahil-saghir": "# Page\n"})
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_get({"slug": "people/sahil-saghir.md"}))

    assert data == {"slug": "people/sahil-saghir", "content": "# Page\n"}


def test_gbrain_search_error_on_missing_query():
    data = json.loads(gbrain.gbrain_search({"query": ""}))

    assert "error" in data
    assert "query is required" in data["error"]


def test_gbrain_graph_bounds_depth(monkeypatch, tmp_path):
    """graph traversal uses real [[wikilinks]] and clamps depth to 5."""
    repo = _make_brain(tmp_path, {
        "people/sahil-saghir": "# Sahil\nLinks: [[projects/kensei]]",
        "projects/kensei": "# Kensei\nLinks: [[people/sahil-saghir]]",
    })
    monkeypatch.setattr(gbrain, "GBRAIN_REPO", repo)

    data = json.loads(gbrain.gbrain_graph({"slug": "people/sahil-saghir", "depth": 99}))

    # _bounded_int clamps depth to maximum=5
    assert data["depth"] == 5
    # The real wikilink to projects/kensei must appear as a connected node
    connected_slugs = {n["slug"] for n in data["nodes"]}
    assert "projects/kensei" in connected_slugs
