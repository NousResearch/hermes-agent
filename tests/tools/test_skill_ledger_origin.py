"""Ledger entries record the write ``origin`` and, for review-fork writes, the review ``route``.

``actor`` already says ``curator`` for every background-review write (#143624), so the ledger
could not tell a post-turn review fork from the curator, nor which model made a write. These
tests pin the two additive fields without touching ``actor``.
"""

import json

import pytest

SKILL = """---
name: my-skill
description: test skill
---

# My Skill

Original body.
"""


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + skills dir for skill_manage and the ledger."""
    from agent import skill_utils
    from tools import skill_ledger, skill_manager_tool, skill_usage

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_usage, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    return {"home": home, "skills": skills_dir}


class _Fork:
    """Stands in for the agent running the turn (what ``bind_subagent_parent`` binds)."""

    def __init__(self, model="m-routed", provider="p-routed", **attrs):
        self.model = model
        self.provider = provider
        for key, value in attrs.items():
            setattr(self, key, value)


def _skill_manage(**kwargs):
    from tools.skill_manager_tool import skill_manage

    return json.loads(skill_manage(**kwargs))


def _in_review(fn, agent=None):
    """Run ``fn`` with the background-review write origin bound, inside ``agent``'s turn if given."""
    from agent.subagent_lifecycle import bind_subagent_parent
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        if agent is None:
            return fn()
        with bind_subagent_parent(agent):
            return fn()
    finally:
        reset_current_write_origin(token)


def _create():
    return _skill_manage(action="create", name="my-skill", content=SKILL)


def _rows(action=None):
    from tools import skill_ledger

    rows = skill_ledger.list_entries(skill="my-skill")
    return [r for r in rows if action is None or r["action"] == action]


def test_foreground_write_records_foreground_origin_and_no_route(ledger_env):
    assert _create()["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "foreground"
    assert "route" not in entry
    assert entry["actor"] == "agent"


def test_routed_review_fork_write_records_the_route(ledger_env):
    fork = _Fork(_review_routed=True)
    assert _in_review(_create, fork)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "background_review"
    assert entry.get("route") == {"model": "m-routed", "provider": "p-routed", "routed": True}
    assert entry["actor"] == "curator"  # unchanged: #143624's mapping is not this PR's business


def test_same_model_review_fork_records_routed_false(ledger_env):
    fork = _Fork(model="parent-model", provider="anthropic", _review_routed=False)
    assert _in_review(_create, fork)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("route") == {"model": "parent-model", "provider": "anthropic", "routed": False}


def test_route_is_read_at_write_time_so_a_fallback_switch_is_attributed(ledger_env):
    """The fork's provider can fail over mid-review; each write names the model that issued it."""
    from tools.skill_manager_guards import mark_background_review_skill_read

    fork = _Fork(model="first", provider="p1", _review_routed=True)
    assert _in_review(_create, fork)["success"] is True
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
    fork.model, fork.provider = "fallback", "p2"

    def _patch():
        mark_background_review_skill_read(skill_md)
        return _skill_manage(action="patch", name="my-skill", old_string="Original body.",
                             new_string="Updated body.")

    assert _in_review(_patch, fork)["success"] is True
    assert (_rows("create")[0].get("route") or {}).get("model") == "first"
    assert _rows("patch")[0].get("route") == {"model": "fallback", "provider": "p2", "routed": True}


def test_review_write_without_a_bound_fork_has_origin_but_no_route(ledger_env):
    assert _in_review(_create)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "background_review"
    assert "route" not in entry


_ABSENT = object()


@pytest.mark.parametrize("marker", [_ABSENT, None, 1, 0, "false", "true"], ids=repr)
def test_agent_without_a_boolean_review_fork_marker_gets_no_route(ledger_env, marker):
    """Only a boolean ``_review_routed`` (set by build_cache_parity_fork) marks a review fork; an agent
    without it, or with any other value, yields no route, so a non-boolean never reaches the ledger."""
    fork = _Fork() if marker is _ABSENT else _Fork(_review_routed=marker)
    assert _in_review(_create, fork)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "background_review"
    assert "route" not in entry


def test_curator_consolidation_fork_write_is_origin_curator(ledger_env):
    fork = _Fork(_ledger_origin="curator")
    assert _in_review(_create, fork)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "curator"
    assert "route" not in entry


# (ContextVar override, append_entry actor argument, inside a review-fork turn?, origin, actor).
# The argument wins over the override, and an explicit actor wins over the review write origin.
_EXPLICIT_ACTOR_CASES = [
    ("curator", None, False, "curator", "curator"),
    ("user", None, False, "foreground", "user"),
    (None, "curator", False, "curator", "curator"),
    (None, "user", False, "foreground", "user"),
    ("curator", "user", False, "foreground", "user"),
    ("user", "curator", False, "curator", "curator"),
    ("curator", None, True, "curator", "curator"),
    ("user", None, True, "foreground", "user"),
    (None, "curator", True, "curator", "curator"),
    (None, "user", True, "foreground", "user"),
    ("curator", "user", True, "foreground", "user"),
    ("user", "curator", True, "curator", "curator"),
]


@pytest.mark.parametrize(("override", "argument", "in_review", "origin", "actor"), _EXPLICIT_ACTOR_CASES)
def test_explicit_actor_sets_origin(ledger_env, override, argument, in_review, origin, actor):
    from tools import skill_ledger

    def _append():
        token = skill_ledger.set_ledger_actor(override)
        try:
            return skill_ledger.append_entry("archive", "my-skill", actor=argument)
        finally:
            skill_ledger.reset_ledger_actor(token)

    entry_id = _in_review(_append, _Fork(_review_routed=True)) if in_review else _append()
    assert entry_id
    entry = skill_ledger.get_entry(entry_id)
    assert entry is not None
    assert entry.get("origin") == origin
    assert entry["actor"] == actor
    assert "route" not in entry


class _ExplodingFork:
    _review_routed = True
    provider = "p"

    @property
    def model(self):
        raise RuntimeError("attribute access failed")


def test_route_failure_never_blocks_the_write_or_the_entry(ledger_env):
    assert _in_review(_create, _ExplodingFork())["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "background_review"
    assert "route" not in entry


def test_provenance_failure_keeps_the_entry_and_omits_the_unknown_origin(ledger_env, monkeypatch):
    """Telemetry rule: a provenance failure costs only the new fields, never the entry or the write,
    and an origin that could not be derived is left out rather than guessed as foreground."""
    import tools.skill_provenance as provenance

    def _boom():
        raise RuntimeError("provenance unavailable")

    monkeypatch.setattr(provenance, "get_current_write_origin", _boom)
    assert _create()["success"] is True
    (entry,) = _rows("create")
    assert entry["actor"] == "agent"
    assert "origin" not in entry and "route" not in entry


def _with_write_origin(value, fn):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    token = set_current_write_origin(value)
    try:
        return fn()
    finally:
        reset_current_write_origin(token)


def test_main_agent_turn_origin_is_foreground(ledger_env):
    """turn_context binds ``assistant_tool`` for every main-agent turn; that is a foreground write."""
    assert _with_write_origin("assistant_tool", _create)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "foreground"
    assert "route" not in entry


def test_other_fork_origins_are_recorded_verbatim(ledger_env):
    """A plugin session fork whitelisted for skill_manage is not a foreground write: keep its tag."""
    fork = _Fork(_review_routed=True)
    from agent.subagent_lifecycle import bind_subagent_parent

    def _run():
        with bind_subagent_parent(fork):
            return _create()

    assert _with_write_origin("plugin_fork:demo", _run)["success"] is True
    (entry,) = _rows("create")
    assert entry.get("origin") == "plugin_fork:demo"
    assert "route" not in entry


def test_compaction_keeps_origin_and_route(ledger_env):
    """The size sweep rewrites rows (``compact_ledger``); the new fields must survive the rewrite."""
    from tools import skill_ledger

    assert _in_review(_create, _Fork(_review_routed=True))["success"] is True
    path = skill_ledger.ledger_path()
    row = json.loads(path.read_text(encoding="utf-8"))
    unchanged = {"path": "/same", "sha256": "0" * 64}  # pre-delta shape: compaction must drop it
    row["before"], row["after"] = [unchanged], [*row["after"], unchanged]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    skill_ledger.compact_ledger()

    (entry,) = _rows("create")
    assert unchanged not in entry["after"]  # proves the row was actually rewritten
    assert entry.get("origin") == "background_review"
    assert entry.get("route") == {"model": "m-routed", "provider": "p-routed", "routed": True}


def test_pre_rollback_and_rollback_entries_carry_origin(ledger_env):
    from tools import skill_ledger

    assert _create()["success"] is True
    created = _rows("create")[0]
    ok, msg = skill_ledger.rollback_entry(created["id"])
    assert ok, msg
    assert {r.get("origin") for r in _rows() if r["action"] in {"pre-rollback", "rollback"}} == {"foreground"}
