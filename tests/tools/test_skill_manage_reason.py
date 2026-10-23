"""skill_manage ``reason``: recorded in the audit ledger, optionally required
(``skills.require_reason``), and carried through ``skills.write_approval`` staging
and replay. Driven through the registry handler against a temp HERMES_HOME with
a real config.yaml."""

import json

import pytest


SKILL = """---
name: {name}
description: test skill
---

# Skill

Original body.
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    from agent import skill_utils
    from tools import skill_ledger, skill_manager_tool, skill_usage

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_usage, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    return home


def _configure(home, **skills):
    lines = ["skills:"] + [f"  {k}: {str(v).lower()}" for k, v in skills.items()]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _call(args):
    from tools.registry import registry

    return json.loads(registry.get_entry("skill_manage").handler(args, session_id="sess-1"))


def _create_op(name="probe"):
    return {"operations": [{"action": "create", "name": name, "content": SKILL.format(name=name)}]}


def _ledger_rows(skill="probe"):
    from tools import skill_ledger

    return skill_ledger.list_entries(skill=skill)


def test_reason_is_recorded_in_ledger_evidence(home):
    _configure(home)
    out = _call({**_create_op(), "reason": "  capture the release checklist  "})
    assert out["success"] is True, out

    out = _call({"operations": [{"action": "patch", "name": "probe", "old_string": "Original body.",
                                 "new_string": "New body."}],
                 "reason": "fix a wrong step"})
    assert out["success"] is True, out

    reasons = {r["action"]: r["evidence"].get("reason") for r in _ledger_rows()}
    assert reasons == {"create": "capture the release checklist", "patch": "fix a wrong step"}


def test_reason_is_optional_by_default(home):
    _configure(home)
    assert _call(_create_op())["success"] is True
    assert all("reason" not in r["evidence"] for r in _ledger_rows())


def test_require_reason_refuses_mutation_without_reason(home):
    _configure(home, require_reason=True)
    out = _call(_create_op())
    assert out["success"] is False
    assert "require_reason" in out["error"]
    assert not (home / "skills" / "probe").exists()

    assert _call({**_create_op(), "reason": "new workflow"})["success"] is True


def test_require_reason_also_covers_legacy_flat_shape(home):
    _configure(home, require_reason=True)
    flat = {"action": "create", "name": "probe", "content": SKILL.format(name="probe")}
    assert _call(flat)["success"] is False
    assert _call({**flat, "reason": "why"})["success"] is True


@pytest.mark.parametrize("shape", ["batch", "flat"])
def test_staged_write_keeps_reason_through_approval(home, shape):
    from tools import write_approval as wa
    from tools.skill_manager_tool import apply_skill_pending

    _configure(home, write_approval=True, require_reason=True)
    args = (_create_op() if shape == "batch"
            else {"action": "create", "name": "probe", "content": SKILL.format(name="probe")})
    staged = _call({**args, "reason": "staged why"})
    assert staged.get("staged") is True, staged
    assert not (home / "skills" / "probe").exists()

    record = wa.get_pending(wa.SKILLS, staged["pending_id"])
    assert record["payload"]["reason"] == "staged why"

    out = json.loads(apply_skill_pending(record["payload"]))
    assert out["success"] is True, out
    assert [r["evidence"].get("reason") for r in _ledger_rows()] == ["staged why"]


def test_approved_staged_write_is_not_refused_when_require_reason_turns_on_later(home):
    from tools import write_approval as wa
    from tools.skill_manager_tool import apply_skill_pending

    _configure(home, write_approval=True)
    staged = _call(_create_op())
    assert staged.get("staged") is True, staged

    _configure(home, write_approval=True, require_reason=True)
    record = wa.get_pending(wa.SKILLS, staged["pending_id"])
    out = json.loads(apply_skill_pending(record["payload"]))
    assert out["success"] is True, out


def test_curator_ledger_shows_reason(home, capsys):
    from types import SimpleNamespace

    from hermes_cli.curator import _cmd_ledger

    _configure(home)
    assert _call({**_create_op(), "reason": "capture the release checklist"})["success"] is True
    assert _cmd_ledger(SimpleNamespace(skill="probe", limit=None, compact=False)) == 0
    assert "why: capture the release checklist" in capsys.readouterr().out
