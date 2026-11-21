"""A missing *essential* skill must be reseeded, not treated as a user deletion.

`sync_skills` uses the bundled manifest to tell "the user deleted this" from "this
never reached this profile". The rule it applies is "in manifest but not on disk =>
the user deleted it, skip". That is right for optional skills, and wrong for
`ESSENTIAL_SKILLS`: the system prompt unconditionally points at those, so a missing
copy turns the pointer into a dead reference that nothing repairs.

These tests pin both halves of the distinction:

* an essential skill missing from disk is reseeded;
* a non-essential skill missing from disk is still treated as a user deletion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import skills_sync


def _write_skill(root: Path, rel: str, name: str) -> Path:
    """Create ``<root>/<rel>/SKILL.md`` declaring ``name``; return the skill dir."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test fixture skill\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def skill_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated HERMES_HOME + a fake bundled tree holding one essential and one optional skill.

    ``ESSENTIAL_SKILLS`` is replaced by this fixture's own name so the test does not
    depend on which skills upstream currently marks essential.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    bundled = tmp_path / "bundled"
    essential_name = "test-essential-skill"

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    monkeypatch.setattr(skills_sync, "ESSENTIAL_SKILLS", frozenset({essential_name}), raising=False)

    # bundle: one essential, one optional
    _write_skill(bundled, essential_name, essential_name)
    _write_skill(bundled, "test-optional-skill", "test-optional-skill")

    # First sync: everything lands, manifest records both.
    skills_sync.sync_skills(quiet=True)
    live = home / "skills"
    assert (live / essential_name / "SKILL.md").is_file()
    assert (live / "test-optional-skill" / "SKILL.md").is_file()

    return home, live, essential_name


def test_missing_essential_skill_is_reseeded(skill_homes):
    """Deleting an essential skill from disk must not disarm it permanently."""
    home, live, essential_name = skill_homes

    # Simulate the damaged state: the skill is gone from disk but still in the manifest.
    import shutil

    shutil.rmtree(live / essential_name)
    assert not (live / essential_name).exists()

    result = skills_sync.sync_skills(quiet=True)

    assert (live / essential_name / "SKILL.md").is_file(), (
        "essential skill was not reseeded; the system-prompt pointer stays dead"
    )
    assert result["copied"] or result["updated"], f"nothing was recorded as reseeded: {result}"


def test_missing_optional_skill_stays_deleted(skill_homes):
    """A user deletion of a non-essential skill must still be honoured."""
    home, live, essential_name = skill_homes

    import shutil

    shutil.rmtree(live / "test-optional-skill")

    skills_sync.sync_skills(quiet=True)

    assert not (live / "test-optional-skill").exists(), (
        "a deliberately deleted optional skill came back"
    )
