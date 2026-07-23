"""Background-review skill discovery must advertise what the write guard enforces.

Root cause this file pins down: ``skills_list()`` returns only
name/description/category, while ``_background_review_write_guard`` refuses
writes to bundled, hub-installed, protected, pinned, external, and
non-agent-created skills. The autonomous review fork therefore picks targets
it is not allowed to write, gets a real refusal, and burns its budget
retrying them.

The contract asserted here is a relationship, not a snapshot: for every skill
the review fork can see, the ``writable`` flag it is shown must equal what the
guard actually does to that skill. Both sides read one shared predicate.

State is real: skills, ``.usage.json``, ``.bundled_manifest``,
``.hub/lock.json``, and ``config.yaml`` are written under the per-test
``HERMES_HOME`` from ``tests/conftest.py``.
"""

import json
from pathlib import Path

import pytest

from hermes_constants import get_config_path, get_hermes_home
from tools.skill_manager_tool import (
    _background_review_write_guard,
    mark_background_review_skill_read,
    skill_manage,
)
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    BACKGROUND_REVIEW_BLOCK_REASONS,
    reset_current_write_origin,
    set_current_write_origin,
)
from tools.skills_tool import skills_list


SKILL_BODY = "# {name}\n\nStep 1: Do the thing.\n"


def _write_skill(root: Path, name: str) -> Path:
    """Create a real SKILL.md on disk and return its directory."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A {name} used by the writability tests.\n---\n\n"
        + SKILL_BODY.format(name=name),
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
def review_library(monkeypatch):
    """Build a real skills library covering every guard case.

    Returns ``(expected_blocks, external_dir)`` where ``expected_blocks`` maps
    each skill name to the block reason the guard must produce (``None`` for a
    skill the review fork owns and may write).
    """
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    home = get_hermes_home()
    local = home / "skills"
    local.mkdir(parents=True, exist_ok=True)
    external = home / "vault"
    external.mkdir(parents=True, exist_ok=True)

    for name in (
        "agent-owned",
        "no-usage-record",
        "pinned-skill",
        "plan",
        "hub-skill",
        "bundled-skill",
        "manual-skill",
    ):
        _write_skill(local, name)
    _write_skill(external, "external-skill")

    # created_by provenance — the sidecar the curator reads.
    (local / ".usage.json").write_text(
        json.dumps(
            {
                "agent-owned": {"created_by": "agent", "use_count": 3},
                "pinned-skill": {"created_by": "agent", "pinned": True},
                "plan": {"created_by": "agent"},
                "hub-skill": {"created_by": "agent"},
                "bundled-skill": {"created_by": "agent"},
                "manual-skill": {"created_by": None, "use_count": 9},
            }
        ),
        encoding="utf-8",
    )
    (local / ".bundled_manifest").write_text("bundled-skill:abc123\n", encoding="utf-8")
    hub_dir = local / ".hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "lock.json").write_text(
        json.dumps({"installed": {"hub-skill": {"version": "1.0.0"}}}), encoding="utf-8"
    )

    get_config_path().write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )

    # Both caches key off state we just wrote; drop them so the first read
    # in the test sees the library rather than an empty pre-write scan.
    skills_tool._SKILLS_CACHE.clear()
    skill_utils._EXTERNAL_DIRS_CACHE.clear()

    expected_blocks = {
        "agent-owned": None,
        # No usage record is not a claim of agent authorship. Telemetry
        # absence must fail closed — see test_background_review_write_sequence
        # for the read-induced ratchet this prevents.
        "no-usage-record": "not_agent_created",
        "pinned-skill": "pinned",
        "external-skill": "external",
        "plan": "protected_builtin",
        "hub-skill": "hub_installed",
        "bundled-skill": "bundled",
        "manual-skill": "not_agent_created",
    }
    return expected_blocks, external


def _list_as_review_fork() -> dict:
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        return json.loads(skills_list())
    finally:
        reset_current_write_origin(token)


def test_review_fork_sees_every_skill_it_cannot_write(review_library):
    """Discovery must label each entry, and keep blocked skills visible."""
    expected_blocks, _ = review_library

    result = _list_as_review_fork()
    assert result["success"] is True, result
    entries = {s["name"]: s for s in result["skills"]}

    # Blocked skills stay listed — the fork still needs to read them.
    assert set(entries) == set(expected_blocks), (
        "skills_list must show every skill, blocked ones included"
    )

    for name, expected in expected_blocks.items():
        entry = entries[name]
        assert "writable" in entry, f"{name}: review listing must advertise writability"
        assert entry["blocked_because"] == expected, (
            f"{name}: blocked_because should be {expected!r}, got "
            f"{entry.get('blocked_because')!r}"
        )
        assert entry["writable"] is (expected is None), (
            f"{name}: writable must be the inverse of blocked_because"
        )


def test_advertised_writability_matches_what_the_guard_does(review_library):
    """The invariant the fork's whole target selection rests on.

    Every entry the listing calls writable must survive the guard, and every
    entry it calls blocked must be refused. One shared predicate, so the two
    can never drift.
    """
    expected_blocks, external = review_library
    home = get_hermes_home()

    result = _list_as_review_fork()
    entries = {s["name"]: s for s in result["skills"]}

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        for name, entry in entries.items():
            root = external if name == "external-skill" else home / "skills"
            refusal = _background_review_write_guard(name, root / name, "patch")
            assert entry["writable"] is (refusal is None), (
                f"{name}: listing says writable={entry['writable']} but the guard "
                f"{'refused' if refusal else 'allowed'} the write — "
                f"guard said: {(refusal or {}).get('error')}"
            )
    finally:
        reset_current_write_origin(token)


def test_every_guard_case_still_refuses_end_to_end(review_library):
    """No block reason may be relaxed: each one still stops a real patch."""
    expected_blocks, external = review_library
    home = get_hermes_home()
    blocked = {n: r for n, r in expected_blocks.items() if r is not None}
    assert set(blocked.values()) == set(BACKGROUND_REVIEW_BLOCK_REASONS), (
        "this test must exercise every reason the predicate can return"
    )

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        for name, reason in blocked.items():
            root = external if name == "external-skill" else home / "skills"
            skill_md = root / name / "SKILL.md"
            before = skill_md.read_text(encoding="utf-8")
            # Reading first clears the read-before-write guard, so a failure
            # here can only come from the ownership guard.
            mark_background_review_skill_read(skill_md)
            result = json.loads(
                skill_manage(
                    action="patch",
                    name=name,
                    old_string="Do the thing.",
                    new_string="Do the other thing.",
                )
            )
            assert result["success"] is False, f"{name} ({reason}) must stay unwritable"
            assert result["blocked_because"] == reason, (
                f"{name}: refusal should report reason {reason!r}"
            )
            assert skill_md.read_text(encoding="utf-8") == before, (
                f"{name}: refused patch must not touch the file"
            )
    finally:
        reset_current_write_origin(token)


def test_refusals_name_the_agent_owned_fallback(review_library):
    """A refusal must give the fork somewhere else to go.

    A bare "no" leaves retrying as the only obvious move, which is how the
    pass burns its budget. Every refusal names both owned destinations.
    """
    expected_blocks, external = review_library
    home = get_hermes_home()

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        for name, reason in expected_blocks.items():
            if reason is None:
                continue
            root = external if name == "external-skill" else home / "skills"
            refusal = _background_review_write_guard(name, root / name, "patch")
            error = refusal["error"]
            lower = error.lower()
            assert "do not retry" in lower, (
                f"{name}: refusal must tell the fork not to retry this target"
            )
            assert "create" in lower and "memory" in lower, (
                f"{name}: refusal must name both fallbacks (new skill / memory), "
                f"got: {error}"
            )
            assert "skill_manage" in error, (
                f"{name}: refusal must name the call that creates the new skill"
            )
    finally:
        reset_current_write_origin(token)


def test_foreground_writes_are_untouched_by_the_guard(review_library):
    """The guard is origin-scoped: a user-directed foreground patch still works."""
    home = get_hermes_home()
    skill_md = home / "skills" / "manual-skill" / "SKILL.md"

    result = json.loads(
        skill_manage(
            action="patch",
            name="manual-skill",
            old_string="Do the thing.",
            new_string="Do the other thing.",
        )
    )
    assert result["success"] is True, result
    assert "Do the other thing." in skill_md.read_text(encoding="utf-8")


def test_foreground_listing_carries_no_writability_fields(review_library):
    """Foreground callers pay nothing for the review-only annotation."""
    result = json.loads(skills_list())
    assert result["success"] is True, result
    for entry in result["skills"]:
        assert "writable" not in entry, (
            "writability is a background-review concern; foreground listings "
            "must keep the tier-1 name/description/category shape"
        )
        assert "blocked_because" not in entry
        assert "path" not in entry, "internal scan paths must not leak to callers"
