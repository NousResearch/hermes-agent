"""The review fork must not write the local copy of a bundled/hub-installed skill.

A bundled skill's local copy IS the upstream copy, matched by an exact directory hash
(``tools/skills_sync._dir_hash``), so a single autonomous write strands the skill as
``user_modified`` and no later update can reconcile it. The write is refused and staged as an
upstream patch under ``<HERMES_HOME>/pending/skill-upstream/<skill>/`` instead, so the lesson the
fork learned is kept rather than dropped.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import skill_manage

VALID_SKILL_CONTENT = """\
---
name: gated-skill
description: A skill used by the bundled-write-gate tests.
---

# Gated Skill

Step 1: do the thing.
"""


@contextmanager
def _skill_dir(tmp_path):
    """Search only *tmp_path* for skills, never the real ``~/.hermes/skills``."""
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


@contextmanager
def _background_review():
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        yield
    finally:
        reset_current_write_origin(token)


def _create(name="gated-skill"):
    return json.loads(skill_manage(action="create", name=name, content=VALID_SKILL_CONTENT))


def test_patch_to_a_bundled_skill_is_refused_and_staged_as_an_upstream_patch(tmp_path):
    from hermes_constants import get_hermes_home
    from tools.skill_manager_guards import mark_background_review_skill_read

    with _skill_dir(tmp_path), patch("tools.skill_usage.is_bundled", return_value=True):
        assert _create()["success"] is True
        skill_md = tmp_path / "gated-skill" / "SKILL.md"
        before = skill_md.read_bytes()
        with _background_review():
            mark_background_review_skill_read(skill_md)
            result = json.loads(skill_manage(
                action="patch", name="gated-skill",
                old_string="Step 1: do the thing.", new_string="Step 1: do it better."))
        after = skill_md.read_bytes()

    assert result["success"] is False, result
    assert "bundled" in result["error"]
    assert before == after, "the refusal must leave the skill untouched"

    patch_path = Path(result["_upstream_patch"])
    assert patch_path.is_file()
    assert patch_path.parent == get_hermes_home() / "pending" / "skill-upstream" / "gated-skill"
    diff = patch_path.read_text(encoding="utf-8")
    assert "--- a/SKILL.md" in diff and "+++ b/SKILL.md" in diff
    assert "-Step 1: do the thing." in diff
    assert "+Step 1: do it better." in diff
    # The refusal has to say the change was kept, not dropped.
    assert str(patch_path) in result["error"]


@pytest.mark.parametrize("ownership", ["is_bundled", "is_hub_installed"])
@pytest.mark.parametrize("action,kwargs", [
    ("edit", {"content": VALID_SKILL_CONTENT.replace("do the thing", "do the other thing")}),
    ("patch", {"old_string": "Step 1: do the thing.", "new_string": "Step 1: do it better."}),
    ("write_file", {"file_path": "references/lesson.md", "file_content": "# Lesson\n"}),
    ("remove_file", {"file_path": "references/lesson.md"}),
])
def test_every_content_action_is_gated(tmp_path, ownership, action, kwargs):
    """Bundled and hub-installed skills are both owned upstream, for all four content actions."""
    with _skill_dir(tmp_path), patch(f"tools.skill_usage.{ownership}", return_value=True):
        assert _create()["success"] is True
        if action == "remove_file":
            assert json.loads(skill_manage(
                action="write_file", name="gated-skill",
                file_path="references/lesson.md", file_content="# Lesson\n"))["success"] is True
        with _background_review():
            result = json.loads(skill_manage(action=action, name="gated-skill", **kwargs))

    assert result["success"] is False, result
    assert result.get("_upstream_patch"), f"{action} was refused without staging the patch"


def test_a_user_owned_skill_is_not_gated(tmp_path):
    """#134289 stays intact where it belongs: the fork still improves the user's own skills."""
    from tools.skill_manager_guards import mark_background_review_skill_read

    with _skill_dir(tmp_path), \
         patch("tools.skill_usage.is_bundled", return_value=False), \
         patch("tools.skill_usage.is_hub_installed", return_value=False):
        assert _create()["success"] is True
        with _background_review():
            mark_background_review_skill_read(tmp_path / "gated-skill" / "SKILL.md")
            result = json.loads(skill_manage(
                action="patch", name="gated-skill",
                old_string="Step 1: do the thing.", new_string="Step 1: do it better."))

    assert result["success"] is True, result
    assert "Step 1: do it better." in (tmp_path / "gated-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert not (tmp_path / ".hermes" / "pending").exists()


def test_a_foreground_write_to_a_bundled_skill_is_untouched(tmp_path):
    """The gate is on the autonomous fork only — a user-directed write still lands."""
    with _skill_dir(tmp_path), patch("tools.skill_usage.is_bundled", return_value=True):
        assert _create()["success"] is True
        result = json.loads(skill_manage(
            action="patch", name="gated-skill",
            old_string="Step 1: do the thing.", new_string="Step 1: do it better."))

    assert result["success"] is True, result


def test_the_allow_switch_opens_the_gate(tmp_path):
    """`auxiliary.background_review.bundled_writes: allow` restores the ungated behaviour,
    read through the real config loader."""
    from hermes_constants import get_hermes_home
    from tools.skill_manager_guards import mark_background_review_skill_read

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "auxiliary:\n  background_review:\n    bundled_writes: allow\n", encoding="utf-8")

    with _skill_dir(tmp_path), patch("tools.skill_usage.is_bundled", return_value=True):
        assert _create()["success"] is True
        with _background_review():
            mark_background_review_skill_read(tmp_path / "gated-skill" / "SKILL.md")
            result = json.loads(skill_manage(
                action="patch", name="gated-skill",
                old_string="Step 1: do the thing.", new_string="Step 1: do it better."))

    assert result["success"] is True, result
    assert "Step 1: do it better." in (tmp_path / "gated-skill" / "SKILL.md").read_text(encoding="utf-8")


def test_the_default_is_deny(tmp_path):
    """No config file at all ⇒ the gate is closed (the DEFAULT_CONFIG value)."""
    from hermes_cli.config import cfg_get, load_config_readonly

    assert cfg_get(load_config_readonly(), "auxiliary", "background_review",
                   "bundled_writes") == "deny"


def test_the_batch_path_is_gated_too(tmp_path):
    """A batch is validated before any op runs, so the gate must fire there as well."""
    with _skill_dir(tmp_path), patch("tools.skill_usage.is_bundled", return_value=True):
        assert _create()["success"] is True
        with _background_review():
            result = json.loads(skill_manage(action="", name="", operations=[
                {"action": "patch", "name": "gated-skill",
                 "old_string": "Step 1: do the thing.", "new_string": "Step 1: do it better."},
            ]))

    assert result["success"] is False, result
    assert result.get("_upstream_patch")
