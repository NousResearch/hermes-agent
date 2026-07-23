"""One skill, one telemetry key — regardless of how the caller spelled it.

``.usage.json`` is a flat name→record map. Every write into it used the raw
caller string, so a skill addressed two ways grew two records and the guards
read whichever one the caller happened to name. Two repros:

1. ``create(name='aio', category='operations')`` records ownership under
   ``aio``. The first ``write_file(name='operations/aio')`` passes the guard,
   then post-action telemetry calls ``bump_patch('operations/aio')`` — minting
   a SECOND record, keyed by the categorized spelling, with
   ``created_by: null``. The identical second write is now refused, while the
   bare ``aio`` spelling still works, and ``skills_list`` advertises whichever
   key it looked up.

2. A pin written under ``operations/aio`` did not block the bare ``aio``,
   because aliases were derived from the caller's text rather than from the
   skill's actual location on disk.

The fix is one canonical key per skill, derived from the resolved directory
relative to its containing root. Legacy records under other spellings are
still read, and a contradicting legacy record fails closed.

State is real: skills and ``.usage.json`` live under the per-test
``HERMES_HOME`` from ``tests/conftest.py``.
"""

import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tools import skill_usage
from tools.skill_manager_tool import (
    _background_review_write_guard,
    mark_background_review_skill_read,
    skill_manage,
)
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    BLOCK_NOT_AGENT_CREATED,
    BLOCK_PINNED,
    background_review_block_reason,
    canonical_skill_name,
    reset_current_write_origin,
    set_current_write_origin,
    skill_alias_names,
)
from tools.skills_tool import _skill_view_with_bump

AIO_SKILL = """\
---
name: aio
description: Autonomous improvement operations skill for canonical-key tests.
---

# aio

Step 1: Do the thing.
"""


@pytest.fixture
def skills_root():
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    root = get_hermes_home() / "skills"
    root.mkdir(parents=True, exist_ok=True)
    skills_tool._SKILLS_CACHE.clear()
    skill_utils._EXTERNAL_DIRS_CACHE.clear()
    return root


def _as_review_fork(fn, *args, **kwargs):
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        return fn(*args, **kwargs)
    finally:
        reset_current_write_origin(token)


def _create_aio() -> dict:
    return json.loads(
        _as_review_fork(
            skill_manage,
            action="create",
            name="aio",
            category="operations",
            content=AIO_SKILL,
        )
    )


def _write_ref(spelling: str, filename: str) -> dict:
    return json.loads(
        _as_review_fork(
            skill_manage,
            action="write_file",
            name=spelling,
            file_path=f"references/{filename}",
            file_content=f"# {filename}\n",
        )
    )


# ---------------------------------------------------------------------------
# Repro 1 — telemetry must not split by spelling
# ---------------------------------------------------------------------------


def test_repeated_categorized_write_keeps_working(skills_root):
    """Two identical writes by the same categorized name must both pass.

    The first write succeeded and then poisoned the skill: post-action
    telemetry minted an ``operations/aio`` record with ``created_by: null``,
    so the second write was refused as not-agent-created.
    """
    created = _create_aio()
    assert created["success"] is True, created
    spelling = created["path"]
    assert spelling == "operations/aio"

    first = _write_ref(spelling, "one.md")
    assert first["success"] is True, first

    second = _write_ref(spelling, "two.md")
    assert second["success"] is True, (
        "the second identical write was refused — the first write's telemetry "
        f"split this skill's state across spellings: {second.get('error')}"
    )


def test_telemetry_lands_on_one_key_per_skill(skills_root):
    """The flat usage map must gain exactly one entry for one skill."""
    _create_aio()
    _write_ref("operations/aio", "one.md")
    _write_ref("aio", "two.md")

    usage = skill_usage.load_usage()
    keys = [k for k in usage if k.endswith("aio")]
    assert keys == ["aio"], (
        f"one skill produced usage keys {keys} — every operation must record "
        f"against the canonical name"
    )
    assert usage["aio"]["created_by"] == "agent"
    assert usage["aio"]["patch_count"] == 2, usage["aio"]


def test_both_spellings_agree_after_a_write(skills_root):
    """Whatever the first write did, the two spellings must still agree."""
    _create_aio()
    _write_ref("operations/aio", "one.md")
    skill_dir = skills_root / "operations" / "aio"

    verdicts = {
        spelling: background_review_block_reason(spelling, skill_dir)
        for spelling in ("aio", "operations/aio", "./aio")
    }
    assert set(verdicts.values()) == {None}, (
        f"spellings disagree after a write: {verdicts}"
    )


def test_listing_still_advertises_the_skill_as_writable(skills_root):
    """skills_list must not go stale once a categorized write has happened."""
    from tools.skills_tool import _SKILLS_CACHE, skills_list

    _create_aio()
    _write_ref("operations/aio", "one.md")
    _SKILLS_CACHE.clear()

    listing = json.loads(_as_review_fork(skills_list))
    entry = {s["name"]: s for s in listing["skills"]}["aio"]
    assert entry["writable"] is True, (
        f"the listing now refuses a skill the fork created and just wrote to: "
        f"{entry}"
    )


# ---------------------------------------------------------------------------
# Repro 2 — provenance derived from disk, not from caller text
# ---------------------------------------------------------------------------


def test_categorized_pin_blocks_the_bare_name(skills_root):
    """A pin stored under the categorized key must block every spelling.

    Aliases derived only from the caller's string are one-directional: given
    ``aio`` there is nothing to suggest ``operations/aio``. Deriving them from
    the resolved directory closes both directions.
    """
    _create_aio()
    skill_dir = skills_root / "operations" / "aio"

    usage = skill_usage.load_usage()
    usage["operations/aio"] = {"created_by": "agent", "pinned": True}
    skill_usage.save_usage(usage)

    for spelling in ("aio", "operations/aio", "./aio"):
        assert background_review_block_reason(spelling, skill_dir) == BLOCK_PINNED, (
            f"{spelling!r} slipped a pin stored under the categorized key"
        )


def test_aliases_come_from_the_resolved_directory(skills_root):
    """Given the bare name, the categorized spelling must still be derived."""
    _create_aio()
    skill_dir = skills_root / "operations" / "aio"

    aliases = skill_alias_names("aio", skill_dir)
    assert "operations/aio" in aliases, (
        f"aliases for the bare name did not include the on-disk path: {aliases}"
    )
    assert "aio" in aliases


def test_canonical_name_is_the_skill_directory_name(skills_root):
    """Canonical key = the basename, matching every existing provenance store."""
    _create_aio()
    skill_dir = skills_root / "operations" / "aio"
    for spelling in ("aio", "operations/aio", "./aio"):
        assert canonical_skill_name(spelling, skill_dir) == "aio"


# ---------------------------------------------------------------------------
# Legacy compatibility — read old spellings, fail closed on contradiction
# ---------------------------------------------------------------------------


def test_legacy_categorized_ownership_record_is_honoured(skills_root):
    """A pre-existing record under the categorized key still counts."""
    skill_dir = skills_root / "operations" / "legacy"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        AIO_SKILL.replace("aio", "legacy"), encoding="utf-8"
    )
    skill_usage.save_usage({"operations/legacy": {"created_by": "agent"}})

    assert background_review_block_reason("legacy", skill_dir) is None
    assert background_review_block_reason("operations/legacy", skill_dir) is None


def test_contradicting_legacy_record_fails_closed(skills_root):
    """Split legacy state must refuse, never pick the permissive record."""
    skill_dir = skills_root / "operations" / "split"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        AIO_SKILL.replace("aio", "split"), encoding="utf-8"
    )
    skill_usage.save_usage(
        {
            "split": {"created_by": "agent"},
            "operations/split": {"created_by": None, "use_count": 3},
        }
    )

    assert (
        background_review_block_reason("split", skill_dir) == BLOCK_NOT_AGENT_CREATED
    ), "a contradicting legacy record must fail closed"


def test_refusal_reports_the_real_created_by(skills_root):
    """The refusal detail must read the record that actually blocked."""
    skill_dir = skills_root / "operations" / "manual"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        AIO_SKILL.replace("aio", "manual"), encoding="utf-8"
    )
    skill_usage.save_usage({"manual": {"created_by": "user", "use_count": 2}})

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        refusal = _background_review_write_guard("operations/manual", skill_dir, "patch")
    finally:
        reset_current_write_origin(token)

    assert refusal is not None
    assert "created_by='user'" in refusal["error"], (
        f"refusal looked up the caller's spelling instead of the record that "
        f"blocked: {refusal['error']}"
    )


# ---------------------------------------------------------------------------
# Identity leaks at the telemetry MUTATION boundary
#
# Reading aliases is not enough. Every write into .usage.json must land on the
# canonical key, or the write itself creates the split the reader then trips
# over. Both repros below pass their first step and fail the second.
# ---------------------------------------------------------------------------


def _legacy_skill(skills_root: Path, name: str, owner_record: dict) -> Path:
    """A skill whose only usage record is filed under the categorized key."""
    skill_dir = skills_root / "operations" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        AIO_SKILL.replace("aio", name), encoding="utf-8"
    )
    skill_usage.save_usage({f"operations/{name}": owner_record})
    return skill_dir


def test_legacy_agent_record_survives_the_write_it_permitted(skills_root):
    """Repro 1: the first write must not poison the second.

    A legacy record under ``operations/legacy`` says agent, so the guard lets
    the write through. Post-action telemetry then bumped the *canonical* key,
    which had no record — minting a fresh one with ``created_by: None``. That
    new record contradicts the legacy one, so every later write is refused.
    """
    _legacy_skill(skills_root, "legacy", {"created_by": "agent"})
    mark_background_review_skill_read(
        skills_root / "operations" / "legacy" / "SKILL.md"
    )

    first = _write_ref("operations/legacy", "one.md")
    assert first["success"] is True, first

    second = _write_ref("operations/legacy", "two.md")
    assert second["success"] is True, (
        "the write that was permitted destroyed its own permission: "
        f"{second.get('error')}"
    )

    usage = skill_usage.load_usage()
    owners = {k: v.get("created_by") for k, v in usage.items() if "legacy" in k}
    assert set(owners.values()) == {"agent"}, (
        f"canonicalizing the telemetry write dropped the legacy owner: {owners}"
    )


def test_viewing_a_supporting_file_does_not_mint_a_second_key(skills_root):
    """Repro 2: create → write → view supporting file → patch.

    ``skill_view(name='operations/aio', file_path=...)`` echoed the caller's
    spelling back as ``name``, and the tool wrapper bumps telemetry against
    whatever name the payload reports — minting a null ``operations/aio``
    record and blocking every later write.
    """
    created = _create_aio()
    assert created["success"] is True, created
    assert _write_ref("operations/aio", "one.md")["success"] is True

    viewed = json.loads(
        _as_review_fork(
            _skill_view_with_bump,
            {"name": "operations/aio", "file_path": "references/one.md"},
        )
    )
    assert viewed["success"] is True, viewed
    assert viewed["name"] == "aio", (
        f"skill_view echoed the caller spelling as the skill name: {viewed['name']!r}"
    )

    usage = skill_usage.load_usage()
    assert [k for k in usage if k.endswith("aio")] == ["aio"], (
        f"viewing a supporting file split the skill's telemetry: {list(usage)}"
    )

    # Loading SKILL.md is what the read-before-write guard asks for; the point
    # of this test is that the earlier supporting-file read did not block it.
    _as_review_fork(_skill_view_with_bump, {"name": "operations/aio"})
    patched = json.loads(
        _as_review_fork(
            skill_manage,
            action="patch",
            name="operations/aio",
            old_string="Do the thing.",
            new_string="Do the other thing.",
        )
    )
    assert patched["success"] is True, (
        f"a read of a supporting file permanently blocked writing: "
        f"{patched.get('error')}"
    )


@pytest.mark.parametrize("spelling", ["aio", "operations/aio"])
def test_full_cycle_survives_every_spelling(skills_root, spelling):
    """create → write_file → view supporting file → view SKILL.md → patch, twice.

    The SKILL.md view is not padding: the read-before-write guard requires the
    exact target to have been loaded this turn, so a real pass reads it before
    patching. What must not happen is any of these steps splitting the skill's
    telemetry.
    """
    assert _create_aio()["success"] is True

    marker = "Do the thing."
    for round_number in (1, 2):
        written = _write_ref(spelling, f"round{round_number}.md")
        assert written["success"] is True, (round_number, written)

        viewed = json.loads(
            _as_review_fork(
                _skill_view_with_bump,
                {"name": spelling, "file_path": f"references/round{round_number}.md"},
            )
        )
        assert viewed["success"] is True, (round_number, viewed)

        assert json.loads(
            _as_review_fork(_skill_view_with_bump, {"name": spelling})
        )["success"] is True

        next_marker = f"Do the thing (round {round_number})."
        patched = json.loads(
            _as_review_fork(
                skill_manage,
                action="patch",
                name=spelling,
                old_string=marker,
                new_string=next_marker,
            )
        )
        assert patched["success"] is True, (round_number, patched)
        marker = next_marker

    assert [k for k in skill_usage.load_usage() if k.endswith("aio")] == ["aio"]


def test_explicit_non_agent_owner_is_never_consolidated_away(skills_root):
    """Consolidation must not launder a real user-owned record into agent."""
    skill_dir = _legacy_skill(skills_root, "owned", {"created_by": "user", "use_count": 7})

    skill_usage.bump_patch("owned")  # canonical spelling, legacy record elsewhere

    usage = skill_usage.load_usage()
    owners = {k: v.get("created_by") for k, v in usage.items() if "owned" in k}
    assert "agent" not in owners.values(), (
        f"a user-owned skill acquired agent ownership through consolidation: {owners}"
    )
    assert (
        background_review_block_reason("owned", skill_dir) == BLOCK_NOT_AGENT_CREATED
    ), "the explicit user owner must keep blocking autonomous writes"


def test_consolidation_preserves_existing_counters(skills_root):
    """Migrating a legacy record must carry its history, not reset it."""
    _legacy_skill(
        skills_root, "counted", {"created_by": "agent", "use_count": 12, "view_count": 5}
    )

    skill_usage.bump_patch("counted")

    record = skill_usage.get_record("counted")
    assert record["created_by"] == "agent"
    assert record["use_count"] == 12, record
    assert record["view_count"] == 5, record
    assert record["patch_count"] == 1, record


# ---------------------------------------------------------------------------
# Delete consistency
# ---------------------------------------------------------------------------


def test_foreground_pin_guard_survives_the_categorized_spelling(skills_root):
    """The user-facing pin guard is keyed by name too — same bypass class."""
    _create_aio()
    skill_usage.set_pinned("aio", True)

    result = json.loads(skill_manage(action="delete", name="operations/aio"))
    assert result["success"] is False, (
        "a pinned skill was deleted through its categorized spelling"
    )
    assert "pinned" in result["error"].lower()
    assert (skills_root / "operations" / "aio" / "SKILL.md").exists()


def test_curator_archive_uses_the_canonical_name(skills_root):
    """The recoverable archive path must find the skill it was handed."""
    _create_aio()
    mark_background_review_skill_read(skills_root / "operations" / "aio" / "SKILL.md")
    _as_review_fork(
        skill_manage, action="create", name="umbrella", content=AIO_SKILL.replace("aio", "umbrella")
    )
    result = json.loads(
        _as_review_fork(
            skill_manage,
            action="delete",
            name="operations/aio",
            absorbed_into="umbrella",
        )
    )
    assert result["success"] is True, result
    assert skill_usage.get_record("aio")["state"] == "archived", (
        "the archive landed on a different key than the skill's canonical name"
    )


def test_delete_forgets_the_canonical_record(skills_root):
    """A hard delete by categorized name must not leave the record behind."""
    _create_aio()
    result = json.loads(
        skill_manage(action="delete", name="operations/aio")  # foreground
    )
    assert result["success"] is True, result
    assert "aio" not in skill_usage.load_usage(), (
        "the canonical record survived a delete addressed by the categorized "
        "name"
    )
