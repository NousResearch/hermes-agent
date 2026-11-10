"""Tests for the blueprints layer (skill frontmatter <-> cron automation bridge).

A blueprint is a skill with a metadata.hermes.blueprint block. These verify parsing
and installed-skill lookup without touching the real cron store.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.blueprints import (
    BlueprintError,
    BlueprintSpec,
    parse_blueprint,
    blueprint_spec_for_installed,
)


BLUEPRINT_SKILL = """---
name: morning-brief
description: Summarize unread email and calendar every morning.
version: 1.0.0
metadata:
  hermes:
    tags: [blueprint, email]
    blueprint:
      schedule: "0 8 * * *"
      deliver: telegram
      prompt: "Summarize my unread email and today's calendar."
---

# Morning Brief

Every morning, gather unread email and the day's calendar and send a digest.
"""

PLAIN_SKILL = """---
name: not-a-blueprint
description: Just a regular skill.
metadata:
  hermes:
    tags: [misc]
---

# Not a blueprint
"""

MALFORMED_BLUEPRINT = """---
name: broken
description: Blueprint with no schedule.
metadata:
  hermes:
    blueprint:
      deliver: origin
---

# Broken
"""


class TestParseBlueprint:
    def test_parses_full_blueprint(self):
        spec = parse_blueprint(BLUEPRINT_SKILL)
        assert spec is not None
        assert spec.skill_name == "morning-brief"
        assert spec.schedule == "0 8 * * *"
        assert spec.deliver == "telegram"
        assert spec.prompt is not None and spec.prompt.startswith("Summarize")


    def test_deliver_defaults_to_origin(self):
        skill = (
            "---\nname: r\ndescription: d\nmetadata:\n  hermes:\n"
            '    blueprint:\n      schedule: "every 1h"\n---\n\nbody'
        )
        spec = parse_blueprint(skill)
        assert spec is not None
        assert spec.deliver == "origin"


class TestBlueprintSpecForInstalled:
    def test_finds_and_parses_installed_blueprint(self, tmp_path):
        skills_dir = tmp_path / "skills"
        rec_dir = skills_dir / "productivity" / "morning-brief"
        rec_dir.mkdir(parents=True)
        (rec_dir / "SKILL.md").write_text(BLUEPRINT_SKILL, encoding="utf-8")

        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            spec = blueprint_spec_for_installed("morning-brief")
        assert spec is not None
        assert spec.schedule == "0 8 * * *"


    def test_plain_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "misc" / "not-a-blueprint"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(PLAIN_SKILL, encoding="utf-8")
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert blueprint_spec_for_installed("not-a-blueprint") is None
