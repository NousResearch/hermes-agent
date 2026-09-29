"""Regression: the Capabilities skills list must honour frontmatter ``category:``.

A flat skill (``skills/<name>/SKILL.md``, no category dir) used to be listed with
``category: None`` — the desktop grouped it under the "General" catch-all even
though its SKILL.md declared a category, splitting it from its category-mates
(two "github" groups: the flat umbrella skill in General, curator-created
skills under Github).
"""

from pathlib import Path

from tools import skills_tool


def _write_skill(root: Path, rel: str | Path, category: str | None) -> None:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True)
    lines = ["---", "name: " + skill_dir.name, "description: test skill."]
    if category is not None:
        lines.insert(2, "category: " + category)
    lines += ["---", "# body"]
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_frontmatter_category_wins_over_path_fallback(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    # Flat skill WITH an explicit frontmatter category...
    _write_skill(root, "github", "github")
    # ...and a category-nested one WITHOUT frontmatter category (path fallback).
    _write_skill(root, Path("mlops") / "axolotl", None)

    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: root)
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    skills = {s["name"]: s["category"] for s in skills_tool._find_all_skills(skip_disabled=True)}

    assert skills["github"] == "github"  # frontmatter, not the General fallback
    assert skills["axolotl"] == "mlops"  # path fallback still works
