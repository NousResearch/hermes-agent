"""No spelling of a skill name may reach a guard the canonical name would fail.

A skill is addressable several ways — ``foo``, ``operations/foo``, ``./foo`` —
and ``_find_skill`` resolves all of them to the same directory. Two review
findings live here:

1. The provenance probes for pin, protected-builtin, hub-installed, and
   bundled keyed on the raw caller string alone, while only the ownership
   probe looked at aliases. A pinned agent-owned skill at ``operations/foo``
   therefore refused as ``foo`` and *succeeded* as ``operations/foo``: the
   pin lookup missed, and the ownership lookup found the agent marker under
   the basename alias.

2. ``_find_skill`` ran its exact-relative pass across every root before its
   basename pass, so a bare name could select a top-level skill in an
   external root ahead of a nested skill in the local root — inverting the
   local-wins precedence the function documents.

State is real: skills and ``.usage.json`` are written under the per-test
``HERMES_HOME`` from ``tests/conftest.py``.
"""

import json
from pathlib import Path

import pytest

from hermes_constants import get_config_path, get_hermes_home
from tools import skill_usage
from tools.skill_manager_tool import _background_review_write_guard, _find_skill
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    BLOCK_BUNDLED,
    BLOCK_HUB_INSTALLED,
    BLOCK_NOT_AGENT_CREATED,
    BLOCK_PINNED,
    BLOCK_PROTECTED_BUILTIN,
    background_review_block_reason,
    reset_current_write_origin,
    set_current_write_origin,
)

SKILL_MD = """\
---
name: {name}
description: A {name} used by the alias-guard regression tests.
---

# {name}

Step 1: Do the thing.
"""

# Every spelling that resolves to ~/.hermes/skills/operations/foo.
ALIASES = ["foo", "operations/foo", "./foo"]


def _write_skill(root: Path, relative: str) -> Path:
    skill_dir = root / relative
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_MD.format(name=Path(relative).name), encoding="utf-8"
    )
    return skill_dir


@pytest.fixture
def categorized_skill():
    """An agent-owned skill under a category, reachable by several names."""
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    local = get_hermes_home() / "skills"
    local.mkdir(parents=True, exist_ok=True)
    skill_dir = _write_skill(local, "operations/foo")
    # The record the fork's own create writes: keyed by the bare name.
    (local / ".usage.json").write_text(
        json.dumps({"foo": {"created_by": "agent", "pinned": False}}),
        encoding="utf-8",
    )
    skills_tool._SKILLS_CACHE.clear()
    skill_utils._EXTERNAL_DIRS_CACHE.clear()
    return local, skill_dir


def _guard(alias: str, skill_dir: Path):
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        return _background_review_write_guard(alias, skill_dir, "patch")
    finally:
        reset_current_write_origin(token)


def _pin(name: str) -> None:
    usage = skill_usage.load_usage()
    usage.setdefault(name, {"created_by": "agent"})["pinned"] = True
    skill_usage.save_usage(usage)


# ---------------------------------------------------------------------------
# 1 — every alias must hit every guard
# ---------------------------------------------------------------------------


def test_every_alias_resolves_to_the_same_skill(categorized_skill):
    """Premise check: these spellings really are the same skill."""
    _, skill_dir = categorized_skill
    for alias in ALIASES:
        found = _find_skill(alias)
        assert found is not None, f"{alias!r} should resolve"
        assert found["path"].resolve() == skill_dir.resolve(), (
            f"{alias!r} resolved to {found['path']}, not {skill_dir}"
        )


@pytest.mark.parametrize("alias", ALIASES)
def test_pin_cannot_be_bypassed_by_spelling(categorized_skill, alias):
    """A pinned skill stays pinned under every name that reaches it."""
    _, skill_dir = categorized_skill
    _pin("foo")

    assert background_review_block_reason(alias, skill_dir) == BLOCK_PINNED, (
        f"{alias!r} bypassed the pin guard — pin is a property of the skill, "
        f"not of the string used to address it"
    )
    refusal = _guard(alias, skill_dir)
    assert refusal is not None and refusal["blocked_because"] == BLOCK_PINNED


@pytest.mark.parametrize("alias", ["plan", "operations/plan", "./plan"])
def test_protected_builtin_cannot_be_bypassed_by_spelling(alias):
    """Same for the protected-builtin list, which is keyed by bare name."""
    local = get_hermes_home() / "skills"
    skill_dir = _write_skill(local, "operations/plan")
    (local / ".usage.json").write_text(
        json.dumps({"plan": {"created_by": "agent"}}), encoding="utf-8"
    )
    assert (
        background_review_block_reason(alias, skill_dir) == BLOCK_PROTECTED_BUILTIN
    ), f"{alias!r} bypassed the protected-builtin guard"


@pytest.mark.parametrize(
    "alias", ["bar", "operations/bar", "./bar"]
)
def test_hub_and_bundled_cannot_be_bypassed_by_spelling(alias):
    """Hub and bundled manifests are keyed by bare name too."""
    local = get_hermes_home() / "skills"
    skill_dir = _write_skill(local, "operations/bar")
    (local / ".usage.json").write_text(
        json.dumps({"bar": {"created_by": "agent"}}), encoding="utf-8"
    )

    (local / ".bundled_manifest").write_text("bar:abc\n", encoding="utf-8")
    assert background_review_block_reason(alias, skill_dir) == BLOCK_BUNDLED, (
        f"{alias!r} bypassed the bundled guard"
    )

    (local / ".bundled_manifest").unlink()
    hub = local / ".hub"
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "lock.json").write_text(
        json.dumps({"installed": {"bar": {"version": "1"}}}), encoding="utf-8"
    )
    assert background_review_block_reason(alias, skill_dir) == BLOCK_HUB_INSTALLED, (
        f"{alias!r} bypassed the hub-installed guard"
    )


def test_alias_ownership_does_not_override_a_contradicting_record(categorized_skill):
    """The reviewer's basename-collision note.

    A user skill at ``operations/foo`` must not inherit agent ownership from
    an unrelated top-level ``foo``. When the skill's own record says the user
    wrote it, that record wins over any alias claim.
    """
    local, skill_dir = categorized_skill
    usage = skill_usage.load_usage()
    usage["operations/foo"] = {"created_by": None, "use_count": 4}
    skill_usage.save_usage(usage)

    assert (
        background_review_block_reason("operations/foo", skill_dir)
        == BLOCK_NOT_AGENT_CREATED
    ), (
        "an explicit non-agent record for this skill must not be overridden "
        "by an agent record belonging to a different skill with the same "
        "basename"
    )


def test_agent_owned_categorized_skill_is_still_writable(categorized_skill):
    """The alias probes must not over-block the legitimate create flow."""
    _, skill_dir = categorized_skill
    for alias in ALIASES:
        assert background_review_block_reason(alias, skill_dir) is None, (
            f"{alias!r} should stay writable — the fork created this skill"
        )


# ---------------------------------------------------------------------------
# 2 — root precedence must survive the relative-path lookup
# ---------------------------------------------------------------------------


def test_nested_local_skill_beats_top_level_external(monkeypatch):
    """Local wins, even when only the external root matches the bare name.

    The local copy is nested (``operations/shared``) so only the basename pass
    finds it, while the external copy sits at the root and the relative pass
    finds it immediately. Running every root through the relative pass first
    hands the external skill back and silently shadows the local one.
    """
    import agent.skill_utils as skill_utils

    local = get_hermes_home() / "skills"
    external = get_hermes_home() / "vault"
    local_skill = _write_skill(local, "operations/shared")
    external_skill = _write_skill(external, "shared")
    get_config_path().write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )
    skill_utils._EXTERNAL_DIRS_CACHE.clear()

    found = _find_skill("shared")
    assert found is not None
    assert found["path"].resolve() == local_skill.resolve(), (
        f"bare name resolved to the external root ({found['path']}) instead of "
        f"the nested local skill ({local_skill}) — local must win"
    )
    assert found["path"].resolve() != external_skill.resolve()


def test_external_root_still_reachable_when_local_has_no_match(monkeypatch):
    """Precedence must not turn into "never look in external dirs"."""
    import agent.skill_utils as skill_utils

    local = get_hermes_home() / "skills"
    local.mkdir(parents=True, exist_ok=True)
    external = get_hermes_home() / "vault"
    external_skill = _write_skill(external, "vendor/only-external")
    get_config_path().write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )
    skill_utils._EXTERNAL_DIRS_CACHE.clear()

    for alias in ("only-external", "vendor/only-external"):
        found = _find_skill(alias)
        assert found is not None, f"{alias!r} should still resolve externally"
        assert found["path"].resolve() == external_skill.resolve()
