"""Regression test for the review-fork → external_dirs write gate.

Background: ``skill_manage`` write actions today operate on whatever path
``_find_skill`` returns, which searches both ``SKILLS_DIR`` (HERMES_HOME/
skills) and ``skills.external_dirs``. The background self-improvement
review fork is autonomous (no user in the loop) and can patch / edit /
delete skills via the same code path. Consumers (e.g. Walter desktop)
expose external_dirs as a user-curated read-only surface — vault-resident
skills the user has explicitly promoted out of the auto-curated local dir.
Letting the review fork silently mutate those breaks the consumer's
"this is mine now" contract.

The gate added in this PR refuses write actions from the review fork when
the target's containing skills root is not SKILLS_DIR. Foreground writes
(CLI users, gateway-routed agents, subagent delegations) are unaffected.

This file pins:
  - Refusal: 5 mutating actions × external_dir target × review-fork
    context = 5 refusal cases
  - Allowed: 5 mutating actions × external_dir target × foreground
    context = 5 backward-compat cases
  - Allowed: 5 mutating actions × local SKILLS_DIR target × review-fork
    context = 5 local-dir-still-works cases
  - Create:  always allowed even from review fork (lands in SKILLS_DIR
    by construction via _resolve_skill_dir)
"""
from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from tools.skill_provenance import (
    set_current_write_origin,
    reset_current_write_origin,
    BACKGROUND_REVIEW,
)


# Build a minimal SKILL.md frontmatter that passes _validate_frontmatter.
_VALID_SKILL_MD = """---
name: test-skill
description: A skill used by the external_dir review-fork gate regression tests.
---

# Test Skill

Body content.
"""


@pytest.fixture
def isolated_skills_layout(tmp_path, monkeypatch):
    """Set up an isolated SKILLS_DIR + an external_dirs entry.

    SKILLS_DIR (HERMES_HOME/skills) → tmp_path/local-hermes/skills
    External dir                    → tmp_path/external-vault/skills

    Both contain a skill named 'test-skill' so _find_skill resolves it
    in whichever dir we're targeting via fixture parameterisation.
    """
    local_root = tmp_path / "local-hermes"
    local_skills_dir = local_root / "skills"
    external_skills_dir = tmp_path / "external-vault" / "skills"
    local_skills_dir.mkdir(parents=True)
    external_skills_dir.mkdir(parents=True)

    # Point HERMES_HOME at our temporary local root.
    monkeypatch.setenv("HERMES_HOME", str(local_root))

    # Patch SKILLS_DIR module-level constants. Both the read-only
    # tools/skills_tool.py and the mutating tools/skill_manager_tool.py
    # bind SKILLS_DIR at import time, so we patch both.
    import tools.skill_manager_tool as smt
    import tools.skills_tool as st
    monkeypatch.setattr(smt, "SKILLS_DIR", local_skills_dir)
    monkeypatch.setattr(smt, "HERMES_HOME", local_root)
    monkeypatch.setattr(st, "SKILLS_DIR", local_skills_dir)
    monkeypatch.setattr(st, "HERMES_HOME", local_root)

    # Override get_external_skills_dirs so _find_skill / get_all_skills_dirs
    # see our test external dir.
    monkeypatch.setattr(
        "agent.skill_utils.get_external_skills_dirs",
        lambda: [external_skills_dir],
    )

    return {
        "local_root": local_root,
        "local_skills_dir": local_skills_dir,
        "external_skills_dir": external_skills_dir,
    }


def _seed_skill(root: Path, name: str = "test-skill") -> Path:
    """Create a minimal skill directory under *root* and return the path."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(_VALID_SKILL_MD, encoding="utf-8")
    # Add a reference file so write_file/remove_file have a target.
    (skill_dir / "references").mkdir(exist_ok=True)
    (skill_dir / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return skill_dir


@pytest.fixture
def review_fork_context():
    """Set the provenance ContextVar to background_review for the test body."""
    token = set_current_write_origin(BACKGROUND_REVIEW)
    yield
    reset_current_write_origin(token)


# Skip security scan during these tests — it would fail on the bare
# frontmatter without the full guards machinery available, masking the
# actual gate behavior we want to test.
@pytest.fixture(autouse=True)
def _no_security_scan(monkeypatch):
    monkeypatch.setattr(
        "tools.skill_manager_tool._security_scan_skill", lambda _p: None
    )


# ─── Refusal cases: review fork + external_dir target ─────────────────────

def test_edit_skill_refuses_external_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    """skill_manage(action='edit') on an external_dirs skill → refusal."""
    from tools.skill_manager_tool import _edit_skill

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _edit_skill("test-skill", _VALID_SKILL_MD.replace("Body content.", "Modified."))
    assert result["success"] is False
    assert "external skills dir" in result["error"]
    # File unchanged — verify the body still has the original content.
    skill_md = isolated_skills_layout["external_skills_dir"] / "test-skill" / "SKILL.md"
    assert "Modified." not in skill_md.read_text()
    assert "Body content." in skill_md.read_text()


def test_patch_skill_refuses_external_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _patch_skill

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _patch_skill(
        name="test-skill", old_string="Body content.", new_string="Patched."
    )
    assert result["success"] is False
    assert "external skills dir" in result["error"]


def test_delete_skill_refuses_external_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _delete_skill

    skill_dir = _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _delete_skill("test-skill", absorbed_into=None)
    assert result["success"] is False
    assert "external skills dir" in result["error"]
    # Skill dir still exists — delete refused.
    assert skill_dir.exists()


def test_write_file_refuses_external_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _write_file

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _write_file(
        "test-skill", "references/new.md", "new content from fork"
    )
    assert result["success"] is False
    assert "external skills dir" in result["error"]


def test_remove_file_refuses_external_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _remove_file

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _remove_file("test-skill", "references/notes.md")
    assert result["success"] is False
    assert "external skills dir" in result["error"]


# ─── Backward-compat: foreground writes to external_dir still work ────────

def test_edit_skill_allows_external_under_foreground(isolated_skills_layout):
    """Without the review_fork_context fixture, the ContextVar default is
    'foreground' → no refusal, edit proceeds normally."""
    from tools.skill_manager_tool import _edit_skill

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    modified = _VALID_SKILL_MD.replace("Body content.", "Foreground edit.")
    result = _edit_skill("test-skill", modified)
    assert result["success"] is True
    skill_md = isolated_skills_layout["external_skills_dir"] / "test-skill" / "SKILL.md"
    assert "Foreground edit." in skill_md.read_text()


def test_patch_skill_allows_external_under_foreground(isolated_skills_layout):
    from tools.skill_manager_tool import _patch_skill

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _patch_skill(
        name="test-skill", old_string="Body content.", new_string="Foreground patch."
    )
    assert result["success"] is True


def test_delete_skill_allows_external_under_foreground(isolated_skills_layout):
    from tools.skill_manager_tool import _delete_skill

    skill_dir = _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _delete_skill("test-skill", absorbed_into=None)
    assert result["success"] is True
    assert not skill_dir.exists()


def test_write_file_allows_external_under_foreground(isolated_skills_layout):
    from tools.skill_manager_tool import _write_file

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _write_file(
        "test-skill", "references/from-foreground.md", "ok"
    )
    assert result["success"] is True


def test_remove_file_allows_external_under_foreground(isolated_skills_layout):
    from tools.skill_manager_tool import _remove_file

    _seed_skill(isolated_skills_layout["external_skills_dir"])
    result = _remove_file("test-skill", "references/notes.md")
    assert result["success"] is True


# ─── Review fork writes to local SKILLS_DIR still work ───────────────────

def test_edit_skill_allows_local_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    """Review fork CAN edit skills in HERMES_HOME/skills — those are its own.
    The gate only protects external_dirs."""
    from tools.skill_manager_tool import _edit_skill

    _seed_skill(isolated_skills_layout["local_skills_dir"])
    modified = _VALID_SKILL_MD.replace("Body content.", "Fork-edited local skill.")
    result = _edit_skill("test-skill", modified)
    assert result["success"] is True


def test_patch_skill_allows_local_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _patch_skill

    _seed_skill(isolated_skills_layout["local_skills_dir"])
    result = _patch_skill(
        name="test-skill", old_string="Body content.", new_string="Fork patch local."
    )
    assert result["success"] is True


def test_delete_skill_allows_local_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _delete_skill

    skill_dir = _seed_skill(isolated_skills_layout["local_skills_dir"])
    result = _delete_skill("test-skill", absorbed_into=None)
    assert result["success"] is True
    assert not skill_dir.exists()


def test_write_file_allows_local_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _write_file

    _seed_skill(isolated_skills_layout["local_skills_dir"])
    result = _write_file(
        "test-skill", "references/from-fork.md", "ok"
    )
    assert result["success"] is True


def test_remove_file_allows_local_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    from tools.skill_manager_tool import _remove_file

    _seed_skill(isolated_skills_layout["local_skills_dir"])
    result = _remove_file("test-skill", "references/notes.md")
    assert result["success"] is True


# ─── Create is naturally safe — always targets SKILLS_DIR ────────────────

def test_create_skill_always_targets_local_even_under_review_fork(
    isolated_skills_layout, review_fork_context
):
    """_create_skill resolves the target via _resolve_skill_dir which is
    hard-pinned to SKILLS_DIR, so the gate isn't needed and create works
    from the review fork by landing in HERMES_HOME/skills/."""
    from tools.skill_manager_tool import _create_skill

    result = _create_skill("new-skill-from-fork", _VALID_SKILL_MD.replace("test-skill", "new-skill-from-fork"))
    assert result["success"] is True
    # Confirms landing under LOCAL skills dir, not external.
    assert (isolated_skills_layout["local_skills_dir"] / "new-skill-from-fork" / "SKILL.md").exists()
    assert not (isolated_skills_layout["external_skills_dir"] / "new-skill-from-fork").exists()
