"""Stage-time preflight for pending skill writes (#137276).

With ``skills.write_approval`` on, a proposal the approval replay is guaranteed
to reject — over-budget create description, no-op or unmatched patch,
intra-batch op interference — must fail in the producing turn instead of being
staged and then resurfacing, unappliable, in every later review round.
"""
import json
import os
import shutil
import tempfile

import pytest

SK = ("---\nname: probe\ndescription: Use when probing staged writes. Fires on probe.\n---\n"
      "# Probe\nStep 1.\n")


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="sk_stage_t_")
    home = os.path.join(d, ".hermes")
    os.makedirs(os.path.join(home, "skills"))
    monkeypatch.setenv("HERMES_HOME", home)
    _set_approval(True)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _set_approval(enabled):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("skills", {})["write_approval"] = enabled
    cfg.save_config(c)


def _skill_manage(**kwargs):
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(**kwargs))


def _pending_count():
    from tools import write_approval as wa
    return wa.pending_count("skills")


def _seed_skill(home):
    # Gate off -> the real create path installs the skill directly.
    _set_approval(False)
    try:
        r = _skill_manage(action="create", name="probe", content=SK)
        assert r["success"], r
    finally:
        _set_approval(True)


def test_create_with_over_budget_description_is_refused(hermes_home):
    long_desc = "Probe skill. " + "a" * 60
    r = _skill_manage(action="create", name="probe",
                      content=f"---\nname: probe\ndescription: {long_desc}\n---\n# Probe\nBody.\n")
    assert not r["success"]
    assert "system-prompt budget" in r["error"]
    assert _pending_count() == 0


def test_noop_patch_is_refused(hermes_home):
    _seed_skill(hermes_home)
    r = _skill_manage(action="patch", name="probe", old_string="Step 1.", new_string="Step 1.")
    assert not r["success"]
    assert _pending_count() == 0


def test_unmatched_patch_is_refused(hermes_home):
    _seed_skill(hermes_home)
    r = _skill_manage(action="patch", name="probe",
                      old_string="Not in the file.", new_string="Step 2.")
    assert not r["success"]
    assert "cannot be staged" in r["error"]
    assert _pending_count() == 0


def test_valid_create_still_stages(hermes_home):
    r = _skill_manage(action="create", name="probe", content=SK)
    assert r["success"] and r.get("staged")
    assert _pending_count() == 1


def test_valid_patch_still_stages(hermes_home):
    _seed_skill(hermes_home)
    r = _skill_manage(action="patch", name="probe", old_string="Step 1.", new_string="Step 2.")
    assert r["success"] and r.get("staged")
    assert _pending_count() == 1


def test_batch_with_intra_op_interference_is_refused(hermes_home):
    _seed_skill(hermes_home)
    r = _skill_manage(action="", name="", operations=[
        {"action": "patch", "name": "probe", "old_string": "Step 1.",
         "new_string": "Alpha beta gamma delta epsilon zeta."},
        {"action": "patch", "name": "probe", "old_string": "Step 1.",
         "new_string": "Something else entirely, unmatched."},
    ])
    assert not r["success"]
    assert "operations[1]" in r["error"]
    assert _pending_count() == 0


def test_batch_valid_chain_still_stages(hermes_home):
    _seed_skill(hermes_home)
    r = _skill_manage(action="", name="", operations=[
        {"action": "patch", "name": "probe", "old_string": "Step 1.", "new_string": "Step A."},
        {"action": "patch", "name": "probe", "old_string": "Step A.", "new_string": "Step B."},
    ])
    assert r["success"] and r.get("staged")
    assert _pending_count() == 1


def test_patch_on_missing_target_still_stages(hermes_home):
    # Matchability is only checked against the current state: a skill that does not
    # exist yet may exist by approval time, so refusing here would guess at replay
    # state. The failure stays an approval-time event.
    r = _skill_manage(action="patch", name="not-installed",
                      old_string="Step 1.", new_string="Step 2.")
    assert r["success"] and r.get("staged")
    assert _pending_count() == 1
