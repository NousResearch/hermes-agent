"""skill_manage telemetry must key on the canonical frontmatter name — the same
key skill_view bumps — so a `category/name` address cannot split one skill's
.usage.json record into two (#136130)."""

import json
from pathlib import Path
from unittest.mock import patch

SKILL_CONTENT = """\
---
name: homelab-kb
description: Frontmatter name diverges from the category-qualified directory.
---

# body

original line.
"""


def _seed_category_skill(skills_root: Path, category="homelab", slug="homelab-kb",
                         frontmatter_name=None) -> None:
    skill_dir = skills_root / category / slug
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_CONTENT.replace("name: homelab-kb",
                              f"name: {frontmatter_name or slug}"),
        encoding="utf-8")


def _usage_records(hermes_home: Path) -> dict:
    sidecar = hermes_home / "skills" / ".usage.json"
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _manage(tmp_path, monkeypatch, **kwargs) -> dict:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "skills").mkdir(parents=True, exist_ok=True)
    from tools.skill_manager_tool import skill_manage
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        return json.loads(skill_manage(**kwargs))


def test_patch_by_qualified_name_bumps_the_canonical_record(tmp_path, monkeypatch):
    _seed_category_skill(tmp_path)
    res = _manage(
        tmp_path, monkeypatch, action="patch", name="homelab/homelab-kb",
        old_string="original line.", new_string="patched line.")
    assert res["success"] is True

    records = _usage_records(tmp_path / ".hermes")
    assert "homelab/homelab-kb" not in records, "qualified duplicate split the record"
    assert records["homelab-kb"]["patch_count"] == 1


def test_write_file_by_qualified_name_bumps_the_canonical_record(tmp_path, monkeypatch):
    _seed_category_skill(tmp_path)
    res = _manage(
        tmp_path, monkeypatch, action="write_file", name="homelab/homelab-kb",
        file_path="assets/notes.md", file_content="extra")
    assert res["success"] is True

    records = _usage_records(tmp_path / ".hermes")
    assert "homelab/homelab-kb" not in records
    assert records["homelab-kb"]["patch_count"] == 1


def test_delete_by_qualified_name_forgets_the_canonical_record(tmp_path, monkeypatch):
    _seed_category_skill(tmp_path)
    hermes_home = tmp_path / ".hermes"
    sidecar = hermes_home / "skills" / ".usage.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({"homelab-kb": {"created_by": "learn", "patch_count": 2}}),
        encoding="utf-8")

    res = _manage(tmp_path, monkeypatch, action="delete", name="homelab/homelab-kb")
    assert res["success"] is True

    assert "homelab-kb" not in _usage_records(hermes_home), (
        "qualified forget missed the canonical record and left it orphaned")


def test_bare_name_still_keys_unchanged(tmp_path, monkeypatch):
    _seed_category_skill(tmp_path, category="misc", slug="bare-skill")
    res = _manage(
        tmp_path, monkeypatch, action="patch", name="misc/bare-skill",
        old_string="original line.", new_string="patched line.")
    assert res["success"] is True

    # A skill whose directory equals its frontmatter name keeps a single record
    # keyed by that shared name — canonical resolution must not move it.
    records = _usage_records(tmp_path / ".hermes")
    assert set(records) == {"bare-skill"}
    assert records["bare-skill"]["patch_count"] == 1
