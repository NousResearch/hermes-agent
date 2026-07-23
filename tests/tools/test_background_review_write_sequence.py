"""The exact background-review sequence from the 11:13 failure trace.

Reproduced from ~/.hermes/logs/agent.log, session 20260722_103517_c4b416e4:

    11:13:53  skill_manage patch fix-pr-review          -> refused (not agent-created)
    11:14:01  skill_manage patch dispatch-coding-work   -> refused (not agent-created)
    11:18:19  skill_manage create operations/autonomous-improvement-operations -> ok
    11:18:40  skill_manage write_file
              name='operations/autonomous-improvement-operations'  -> "not found"

Two defects produced that trace:

A. A skill with no ``.usage.json`` record was advertised ``writable: true``.
   Writing one requires ``skill_view`` first (the read-before-write guard), and
   ``skill_view`` calls ``bump_view``, which seeds a record with
   ``created_by=None``. The very act of reading the skill flipped it to
   blocked — a one-way ratchet that made the advertised verdict a lie, and
   permanently marked user skills as refused.

B. ``_create_skill`` returns ``path`` as the category-qualified
   ``"<category>/<name>"``, and the agent reuses it as ``name``. ``_find_skill``
   matched only the directory basename, so every follow-up call on a
   categorized skill failed with "not found".

State here is real: skills, ``.usage.json`` and ``config.yaml`` are written
under the per-test ``HERMES_HOME`` from ``tests/conftest.py``.
"""

import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tools.skill_manager_tool import _find_skill, skill_manage
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    BLOCK_NOT_AGENT_CREATED,
    background_review_block_reason,
    reset_current_write_origin,
    set_current_write_origin,
)
from tools.skills_tool import _skill_view_with_bump, skills_list


def skill_view_as_tool(name: str) -> str:
    """Call skill_view the way the tool registry does — telemetry included.

    ``bump_view``/``bump_use`` live in the registry wrapper, not in
    ``skill_view`` itself. The ratchet only reproduces through the wrapper,
    so tests must go through it.
    """
    return _skill_view_with_bump({"name": name})


SKILL_MD = """\
---
name: {name}
description: A {name} used by the write-sequence regression tests.
---

# {name}

Step 1: Do the thing.
"""

NEW_SKILL_MD = """\
---
name: autonomous-improvement-operations
description: Class-level skill for autonomous self-improvement operations.
---

# autonomous-improvement-operations

Step 1: Check writability before patching.
"""


def _write_skill(root: Path, name: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")
    return skill_dir


@pytest.fixture
def skills_home(monkeypatch):
    """A skills library with no telemetry at all — the pre-ratchet state."""
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    local = get_hermes_home() / "skills"
    local.mkdir(parents=True, exist_ok=True)
    # Same two skills the live trace failed on. Neither has a usage record:
    # they are user-authored skills nothing has touched yet.
    _write_skill(local, "fix-pr-review")
    _write_skill(local, "dispatch-coding-work")

    skills_tool._SKILLS_CACHE.clear()
    skill_utils._EXTERNAL_DIRS_CACHE.clear()
    return local


def _as_review_fork(fn, *args, **kwargs):
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        return fn(*args, **kwargs)
    finally:
        reset_current_write_origin(token)


def _listing_entry(name: str) -> dict:
    result = json.loads(_as_review_fork(skills_list))
    return {s["name"]: s for s in result["skills"]}[name]


# ---------------------------------------------------------------------------
# A — telemetry absence must never read as agent ownership
# ---------------------------------------------------------------------------


def test_untracked_user_skill_is_not_advertised_as_writable(skills_home):
    """No usage record is not evidence of agent authorship.

    ``created_by`` is the only positive claim of ownership. A skill nothing
    has written a record for was authored by the user, so autonomous review
    must treat it as off-limits and say so up front.
    """
    entry = _listing_entry("fix-pr-review")
    assert entry["writable"] is False, (
        "a skill with no usage record must not be advertised as writable — "
        "absence of telemetry is not evidence of agent authorship"
    )
    assert entry["blocked_because"] == BLOCK_NOT_AGENT_CREATED


def test_reading_a_skill_cannot_change_its_advertised_writability(skills_home):
    """The read-induced ratchet: skill_view must not flip the verdict.

    Writing requires skill_view first, and skill_view bumps telemetry. If the
    verdict depends on whether telemetry exists, every advertised-writable
    skill becomes blocked the moment the fork does the reading the write path
    demands of it.
    """
    before = _listing_entry("dispatch-coding-work")

    _as_review_fork(skill_view_as_tool, "dispatch-coding-work")

    import tools.skills_tool as skills_tool
    skills_tool._SKILLS_CACHE.clear()
    after = _listing_entry("dispatch-coding-work")

    assert after["writable"] is before["writable"], (
        "skill_view changed the advertised verdict: the mandatory read "
        f"flipped writable {before['writable']} -> {after['writable']}"
    )
    assert after["blocked_because"] == before["blocked_because"]


def test_the_live_patch_refusal_is_predicted_by_the_listing(skills_home):
    """The 11:13 refusals must be visible before the call, not after it."""
    for name in ("fix-pr-review", "dispatch-coding-work"):
        entry = _listing_entry(name)
        _as_review_fork(skill_view_as_tool, name)  # what the write path forces
        result = json.loads(
            _as_review_fork(
                skill_manage,
                action="patch",
                name=name,
                old_string="Do the thing.",
                new_string="Do the other thing.",
            )
        )
        assert result["success"] is False, f"{name}: guard must still refuse"
        assert entry["writable"] is False, (
            f"{name}: the guard refused a skill the listing called writable"
        )
        assert entry["blocked_because"] == result["blocked_because"]


def test_guard_blocks_untracked_skill_directly(skills_home):
    """Fail closed at the predicate, not only at the listing."""
    reason = background_review_block_reason(
        "fix-pr-review", skills_home / "fix-pr-review"
    )
    assert reason == BLOCK_NOT_AGENT_CREATED


# ---------------------------------------------------------------------------
# B — categorized names returned by create must resolve on the next call
# ---------------------------------------------------------------------------


def test_create_returns_a_path_that_write_file_can_use(skills_home):
    """The exact 11:18 sequence: create in a category, then add a reference.

    ``create`` hands back ``path='operations/autonomous-improvement-operations'``.
    Feeding that straight back as ``name`` is the obvious next move, and it
    must work.
    """
    created = json.loads(
        _as_review_fork(
            skill_manage,
            action="create",
            name="autonomous-improvement-operations",
            category="operations",
            content=NEW_SKILL_MD,
        )
    )
    assert created["success"] is True, created
    assert created["path"] == "operations/autonomous-improvement-operations"

    written = json.loads(
        _as_review_fork(
            skill_manage,
            action="write_file",
            name=created["path"],
            file_path="references/provenance-guard-observability.md",
            file_content="# Observability\n\nWhat the guard refused and why.\n",
        )
    )
    assert written["success"] is True, (
        f"write_file could not resolve the categorized name create returned: "
        f"{written.get('error')}"
    )
    reference = (
        skills_home
        / "operations"
        / "autonomous-improvement-operations"
        / "references"
        / "provenance-guard-observability.md"
    )
    assert reference.exists(), "the reference file was never written"


def test_bare_name_still_resolves_a_categorized_skill(skills_home):
    """Adding categorized lookup must not cost the bare-name path."""
    _as_review_fork(
        skill_manage,
        action="create",
        name="autonomous-improvement-operations",
        category="operations",
        content=NEW_SKILL_MD,
    )
    written = json.loads(
        _as_review_fork(
            skill_manage,
            action="write_file",
            name="autonomous-improvement-operations",
            file_path="references/second.md",
            file_content="# Second\n",
        )
    )
    assert written["success"] is True, written


def test_categorized_lookup_refuses_traversal(skills_home):
    """A slash in the name must not become a path escape."""
    outsider = get_hermes_home() / "outside-skill"
    outsider.mkdir(parents=True, exist_ok=True)
    (outsider / "SKILL.md").write_text(
        SKILL_MD.format(name="outside-skill"), encoding="utf-8"
    )

    for hostile in ("../outside-skill", "operations/../../outside-skill", "/etc"):
        assert _find_skill(hostile) is None, (
            f"{hostile!r} must not resolve — categorized lookup stays inside "
            f"the skills roots"
        )


def test_categorized_lookup_prefers_the_local_skills_dir(skills_home, monkeypatch):
    """Local skills keep precedence over external dirs, as before."""
    import agent.skill_utils as skill_utils
    from hermes_constants import get_config_path

    external = get_hermes_home() / "vault"
    (external / "operations").mkdir(parents=True, exist_ok=True)
    (external / "operations" / "shared-skill").mkdir(parents=True, exist_ok=True)
    (external / "operations" / "shared-skill" / "SKILL.md").write_text(
        SKILL_MD.format(name="shared-skill"), encoding="utf-8"
    )
    (skills_home / "operations").mkdir(parents=True, exist_ok=True)
    (skills_home / "operations" / "shared-skill").mkdir(parents=True, exist_ok=True)
    (skills_home / "operations" / "shared-skill" / "SKILL.md").write_text(
        SKILL_MD.format(name="shared-skill"), encoding="utf-8"
    )
    get_config_path().write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )
    skill_utils._EXTERNAL_DIRS_CACHE.clear()

    found = _find_skill("operations/shared-skill")
    assert found is not None
    assert found["path"] == skills_home / "operations" / "shared-skill", (
        "the local skills dir must win over an external dir"
    )


def test_external_categorized_skill_is_still_protected(skills_home):
    """Resolving a categorized external skill must not make it writable."""
    import agent.skill_utils as skill_utils
    from hermes_constants import get_config_path

    external = get_hermes_home() / "vault"
    ext_skill = external / "operations" / "vendor-skill"
    ext_skill.mkdir(parents=True, exist_ok=True)
    (ext_skill / "SKILL.md").write_text(
        SKILL_MD.format(name="vendor-skill"), encoding="utf-8"
    )
    get_config_path().write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )
    skill_utils._EXTERNAL_DIRS_CACHE.clear()

    result = json.loads(
        _as_review_fork(
            skill_manage,
            action="write_file",
            name="operations/vendor-skill",
            file_path="references/x.md",
            file_content="# x\n",
        )
    )
    assert result["success"] is False
    assert result.get("blocked_because") == "external", result


def test_foreground_categorized_write_still_works(skills_home):
    """Foreground callers get the same resolution, with no ownership guard."""
    created = json.loads(
        skill_manage(
            action="create",
            name="autonomous-improvement-operations",
            category="operations",
            content=NEW_SKILL_MD,
        )
    )
    assert created["success"] is True, created
    written = json.loads(
        skill_manage(
            action="write_file",
            name=created["path"],
            file_path="references/foreground.md",
            file_content="# Foreground\n",
        )
    )
    assert written["success"] is True, written
