"""Tests for the recipes layer (skill frontmatter <-> cron automation bridge).

A recipe is a skill with a metadata.hermes.recipe block. These verify parsing,
the create-job bridge, and the export round-trip without touching the real
cron store.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.recipes import (
    RecipeError,
    RecipeSpec,
    create_recipe_job,
    export_recipe,
    parse_recipe,
    recipe_spec_for_installed,
)


RECIPE_SKILL = """---
name: morning-brief
description: Summarize unread email and calendar every morning.
version: 1.0.0
metadata:
  hermes:
    tags: [recipe, email]
    recipe:
      schedule: "0 8 * * *"
      deliver: telegram
      prompt: "Summarize my unread email and today's calendar."
---

# Morning Brief

Every morning, gather unread email and the day's calendar and send a digest.
"""

PLAIN_SKILL = """---
name: not-a-recipe
description: Just a regular skill.
metadata:
  hermes:
    tags: [misc]
---

# Not a recipe
"""

MALFORMED_RECIPE = """---
name: broken
description: Recipe with no schedule.
metadata:
  hermes:
    recipe:
      deliver: origin
---

# Broken
"""


class TestParseRecipe:
    def test_parses_full_recipe(self):
        spec = parse_recipe(RECIPE_SKILL)
        assert spec is not None
        assert spec.skill_name == "morning-brief"
        assert spec.schedule == "0 8 * * *"
        assert spec.deliver == "telegram"
        assert spec.prompt is not None and spec.prompt.startswith("Summarize")

    def test_plain_skill_is_not_a_recipe(self):
        assert parse_recipe(PLAIN_SKILL) is None

    def test_no_frontmatter_is_not_a_recipe(self):
        assert parse_recipe("just some text, no frontmatter") is None

    def test_missing_schedule_raises(self):
        with pytest.raises(RecipeError):
            parse_recipe(MALFORMED_RECIPE)

    def test_recipe_not_mapping_raises(self):
        bad = "---\nname: x\nmetadata:\n  hermes:\n    recipe: not-a-dict\n---\n\nbody"
        with pytest.raises(RecipeError):
            parse_recipe(bad)

    def test_deliver_defaults_to_origin(self):
        skill = (
            "---\nname: r\ndescription: d\nmetadata:\n  hermes:\n"
            '    recipe:\n      schedule: "every 1h"\n---\n\nbody'
        )
        spec = parse_recipe(skill)
        assert spec is not None
        assert spec.deliver == "origin"


class TestRecipeSpecForInstalled:
    def test_finds_and_parses_installed_recipe(self, tmp_path):
        skills_dir = tmp_path / "skills"
        rec_dir = skills_dir / "productivity" / "morning-brief"
        rec_dir.mkdir(parents=True)
        (rec_dir / "SKILL.md").write_text(RECIPE_SKILL, encoding="utf-8")

        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            spec = recipe_spec_for_installed("morning-brief")
        assert spec is not None
        assert spec.schedule == "0 8 * * *"

    def test_missing_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert recipe_spec_for_installed("nope") is None

    def test_plain_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "misc" / "not-a-recipe"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(PLAIN_SKILL, encoding="utf-8")
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert recipe_spec_for_installed("not-a-recipe") is None


class TestCreateRecipeJob:
    def test_bridges_to_create_job(self):
        spec = parse_recipe(RECIPE_SKILL)
        assert spec is not None
        captured = {}

        def fake_create_job(**kwargs):
            captured.update(kwargs)
            return {"id": "abc123", **kwargs}

        with patch("cron.jobs.create_job", fake_create_job):
            job = create_recipe_job(spec, origin={"platform": "telegram"})

        assert captured["schedule"] == "0 8 * * *"
        assert captured["skills"] == ["morning-brief"]
        assert captured["deliver"] == "telegram"
        assert captured["prompt"].startswith("Summarize")
        assert job["id"] == "abc123"


class TestExportRecipe:
    def test_round_trips_job_to_skill_md(self):
        job = {
            "name": "My Morning Brief",
            "schedule_display": "0 8 * * *",
            "skills": ["morning-brief"],
            "deliver": "telegram",
            "prompt": "Summarize my unread email.",
        }
        md = export_recipe(job, "# Morning Brief\n\nDoes the morning digest.")
        # The exported SKILL.md must itself parse back as a recipe.
        spec = parse_recipe(md)
        assert spec is not None
        assert spec.schedule == "0 8 * * *"
        assert spec.deliver == "telegram"
        # Name is sanitized to a valid skill identifier.
        assert spec.skill_name == "my-morning-brief"

    def test_export_has_recipe_tag(self):
        job = {"name": "x", "schedule_display": "every 2h", "skills": ["x"]}
        md = export_recipe(job, "body")
        assert "recipe" in md
        assert "automation" in md
