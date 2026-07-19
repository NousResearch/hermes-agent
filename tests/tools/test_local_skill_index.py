"""Regression tests for the local, token-efficient skill index."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.local_skill_index import (
    build_skill_index,
    ensure_skill_index,
    search_skill_index,
)
from tools.skills_tool import skills_list


def _write_skill(root: Path, name: str, *, body: str, category: str = "general") -> Path:
    skill_dir = root / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Instructions for {name}.\n"
        "metadata:\n"
        "  hermes:\n"
        "    tags: [oauth, provider]\n"
        "---\n\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_build_search_and_obsidian_export(tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = _write_skill(
        skills_root,
        "provider-ops",
        body="## Setup\nConfigure the provider.\n\n## Quotas\nInspect cached token quota usage.",
        category="operations",
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "oauth.md").write_text(
        "# OAuth \"extra usage\" troubleshooting\n\nCheck subscription quota classification and cached input.",
        encoding="utf-8",
    )
    usage_path = skills_root / ".usage.json"
    usage_path.write_text(
        json.dumps(
            {
                "provider-ops": {
                    "view_count": 7,
                    "use_count": 5,
                    "last_used_at": "2026-07-14T10:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    routing_map = tmp_path / "Hermes-Skill-Map.md"
    routing_map.write_text(
        "`provider-ops` supports [[Operations/Provider Operations]].\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "skill-index.sqlite3"
    dashboard_path = tmp_path / "Hermes-Skill-Index.md"

    result = build_skill_index(
        [skills_root],
        db_path,
        usage_path=usage_path,
        obsidian_map_path=routing_map,
        obsidian_output_path=dashboard_path,
    )

    assert result["skill_count"] == 1
    assert result["document_count"] >= 3
    assert db_path.exists()
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert "# Hermes Skill Index" in dashboard
    assert "provider-ops" in dashboard
    assert "[[Operations/Provider Operations]]" in dashboard
    assert "7" in dashboard

    hits = search_skill_index(db_path, "oauth quota cached", limit=5)
    assert hits
    assert hits[0]["name"] == "provider-ops"
    assert any(hit["file_path"] == "references/oauth.md" for hit in hits)
    oauth_hit = next(hit for hit in hits if hit["file_path"] == "references/oauth.md")
    assert '\\"extra usage\\"' in oauth_hit["recommended_skill_view"]
    assert all("content" not in hit for hit in hits)
    assert all(hit["recommended_skill_view"] for hit in hits)

    # Runtime routing does not pass dashboard-only usage/map inputs. The same
    # content index must remain reusable instead of rebuilding and erasing the
    # richer dashboard metadata.
    reused = ensure_skill_index([skills_root], db_path)
    assert reused["rebuilt"] is False


def test_ensure_index_reuses_unchanged_database_and_rebuilds_after_edit(tmp_path):
    skills_root = tmp_path / "skills"
    skill_dir = _write_skill(skills_root, "cache-ops", body="## Cache\nInspect cache reads.")
    db_path = tmp_path / "skill-index.sqlite3"

    first = ensure_skill_index([skills_root], db_path)
    second = ensure_skill_index([skills_root], db_path)
    assert first["rebuilt"] is True
    assert second["rebuilt"] is False

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "\nNew routing term.\n", encoding="utf-8")
    third = ensure_skill_index([skills_root], db_path)
    assert third["rebuilt"] is True


def test_skills_list_query_uses_local_index_without_returning_full_skill(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "provider-ops",
        body="## Quotas\nDiagnose OAuth cached token quota accounting.",
        category="operations",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with patch("tools.skills_tool.SKILLS_DIR", skills_root):
        raw = skills_list(query="OAuth quota", limit=3)

    result = json.loads(raw)
    assert result["success"] is True
    assert result["query"] == "OAuth quota"
    assert result["count"] == 1
    assert result["skills"][0]["name"] == "provider-ops"
    assert "content" not in result["skills"][0]
    assert result["matches"][0]["recommended_skill_view"]
