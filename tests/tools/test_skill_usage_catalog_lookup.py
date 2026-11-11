"""A skill load must not re-read the library to label its provenance.

One catalog scan already parsed every SKILL.md. `telemetry_provenance` used to walk the tree and
read every frontmatter again to answer "external or local?" (~2,675 file reads, ~50 ms on every
skill invocation). The label now rides the scanned catalog; the exact filesystem lookups stay for
callers that act on existence, so stale rows can never resurrect a deleted skill.
"""
import time
from pathlib import Path

import pytest


def _rows(local: Path, external: Path):
    return [
        {"name": "local-skill", "path": str(local / "local-skill" / "SKILL.md"), "root": str(local)},
        {"name": "ext-skill", "path": str(external / "ext-skill" / "SKILL.md"), "root": str(external)},
    ]


@pytest.fixture
def scanned(monkeypatch, tmp_path):
    """A warm catalog covering one local and one external skill, plus a hard no-walk guard."""
    import tools.skills_tool as st
    import tools.skill_usage as su
    import agent.skill_utils as sku

    local = tmp_path / "skills"
    external = tmp_path / "ext"
    for root, name in ((local, "local-skill"), (external, "ext-skill")):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")

    rows = _rows(local, external)
    monkeypatch.setattr(st, "_SKILLS_CACHE", {("sig",): (("sig",), time.time(), rows)}, raising=False)
    monkeypatch.setattr(su, "_catalog_dirs_cache", None, raising=False)
    monkeypatch.setattr(sku, "get_all_skills_dirs", lambda *a, **k: [local, external])
    monkeypatch.setattr(su, "_skills_dir", lambda: local)
    monkeypatch.setattr(su, "is_hub_installed", lambda *a, **k: False)
    monkeypatch.setattr(su, "is_bundled", lambda *a, **k: False)

    def no_walk(*args, **kwargs):
        raise AssertionError("walked the library for a label the catalog already knows")

    monkeypatch.setattr(sku, "iter_skill_index_files", no_walk)
    return local, external, rows


def test_provenance_labels_a_local_skill_from_the_catalog(scanned):
    import tools.skill_usage as su

    assert su.telemetry_provenance("local-skill") == "local"
    assert su._catalog_location("local-skill") == "local"


def test_provenance_labels_an_external_skill_from_the_catalog(scanned):
    import tools.skill_usage as su

    assert su.telemetry_provenance("ext-skill") == "external"
    assert su._catalog_location("ext-skill") == "external"


def test_an_unlisted_skill_has_no_catalog_labels(scanned):
    import tools.skill_usage as su

    assert su._catalog_location("never-seen") is None


def test_the_dirs_index_is_rebuilt_only_when_the_scan_is(scanned):
    import tools.skill_usage as su

    first = su._catalog_skill_dirs()
    assert first and su._catalog_skill_dirs() is first


def test_without_a_catalog_provenance_falls_back_to_the_walk(monkeypatch, tmp_path):
    """Cold process, no scan yet: the label still resolves, by looking, exactly as before."""
    import tools.skills_tool as st
    import tools.skill_usage as su

    monkeypatch.setattr(st, "_SKILLS_CACHE", {}, raising=False)
    monkeypatch.setattr(su, "_catalog_dirs_cache", None, raising=False)
    monkeypatch.setattr(su, "is_hub_installed", lambda *a, **k: False)
    monkeypatch.setattr(su, "is_bundled", lambda *a, **k: False)
    monkeypatch.setattr(su, "_find_external_skill_dir", lambda name: tmp_path / "ext" / name if name == "ext-skill" else None)
    monkeypatch.setattr(su, "_find_skill_dir", lambda name: None)

    assert su.telemetry_provenance("ext-skill") == "external"
    assert su.telemetry_provenance("ghost") == "unknown"


def test_stale_rows_cannot_resurrect_a_deleted_skill(monkeypatch, tmp_path):
    """`_find_skill_dir` decides existence, so it keeps looking at disk, not at the catalog."""
    import tools.skills_tool as st
    import tools.skill_usage as su
    import agent.skill_utils as sku

    local = tmp_path / "skills"
    local.mkdir()
    monkeypatch.setattr(st, "_SKILLS_CACHE", {("sig",): (("sig",), time.time(), _rows(local, tmp_path / "ext"))}, raising=False)
    monkeypatch.setattr(su, "_skills_dir", lambda: local)
    monkeypatch.setattr(sku, "is_external_skill_path", lambda *a, **k: False)
    monkeypatch.setattr(sku, "iter_skill_index_files", lambda *a, **k: iter(()))

    assert su._find_skill_dir("local-skill") is None
