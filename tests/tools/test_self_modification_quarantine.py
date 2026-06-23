import json
import threading
import time
from pathlib import Path

import pytest

from agent.finalization_barrier import changed_paths, hash_protected_files, run_finalization_barrier
from agent.self_modification_quarantine import PROPOSAL_SCHEMA
from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin


def _bind_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import agent.self_modification_quarantine as q
    import agent.finalization_barrier as fb
    import tools.skill_manager_tool as sm
    monkeypatch.setattr(q, "get_hermes_home", lambda: home)
    monkeypatch.setattr(fb, "get_hermes_home", lambda: home)
    monkeypatch.setattr(sm, "HERMES_HOME", home, raising=False)
    monkeypatch.setattr(sm, "SKILLS_DIR", skills, raising=False)
    return home


@pytest.fixture
def background_origin():
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        yield
    finally:
        reset_current_write_origin(token)


def _make_skill(home, name="demo", body="---\nname: demo\ndescription: demo skill\n---\n\nold\n"):
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body)
    return path


def _load(path):
    return path.read_text()


def _latest_proposal(home):
    files = sorted((home / "system-improvement-proposals").glob("*.json"))
    assert files
    return files[-1], json.loads(files[-1].read_text())


def test_01_background_review_proposes_skill_change(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage("patch", "demo", old_string="old", new_string="new"))
    assert result["quarantined"] is True
    proposal_path, proposal = _latest_proposal(home)
    assert proposal_path.exists()
    assert proposal["schema"] == PROPOSAL_SCHEMA
    assert "new" in proposal["proposed_diff"]
    assert _load(path).endswith("old\n")


def test_02_active_skill_remains_byte_for_byte_unchanged(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home)
    before = path.read_bytes()
    from tools.skill_manager_tool import skill_manage
    skill_manage("edit", "demo", content="---\nname: demo\n---\n\nchanged\n")
    assert path.read_bytes() == before


def test_03_system_improvement_proposal_created(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    skill_manage("patch", "demo", old_string="old", new_string="new")
    _, proposal = _latest_proposal(home)
    assert proposal["state"] == "awaiting_approval"
    assert proposal["proposed_change_type"] == "skill_manage.patch"


def test_04_proposal_execute_approved_false(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    skill_manage("delete", "demo", absorbed_into="")
    _, proposal = _latest_proposal(home)
    assert proposal["execute_approved"] is False


def test_05_proposed_diff_is_hash_bound(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    skill_manage("patch", "demo", old_string="old", new_string="new")
    _, proposal = _latest_proposal(home)
    import hashlib
    assert hashlib.sha256(proposal["proposed_diff"].encode()).hexdigest() == proposal["proposed_diff_sha256"]


def test_06_tests_and_rollback_are_required(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    skill_manage("write_file", "demo", file_path="references/x.md", file_content="x")
    _, proposal = _latest_proposal(home)
    assert proposal["tests_required"]
    assert proposal["rollback"]
    assert proposal["proof_gates"]


def test_07_foreground_approved_mutation_still_possible(monkeypatch, tmp_path):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage("patch", "demo", old_string="old", new_string="new"))
    assert result["success"] is True
    assert "new" in path.read_text()


def test_08_background_direct_skill_manage_bypass_is_caught(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home)
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage("patch", "demo", old_string="old", new_string="bypass"))
    assert result["quarantined"] is True
    assert "bypass" not in path.read_text()


def test_09_background_cannot_edit_skill_references(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    skill = home / "skills" / "demo"
    ref = skill / "references" / "x.md"
    ref.parent.mkdir(parents=True)
    ref.write_text("old")
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage("write_file", "demo", file_path="references/x.md", file_content="new"))
    assert result["quarantined"] is True
    assert ref.read_text() == "old"


class FakeStore:
    def __init__(self):
        self.calls = []
    def add(self, target, content):
        self.calls.append(("add", target, content)); return {"success": True}
    def replace(self, target, old_text, content):
        self.calls.append(("replace", target, old_text, content)); return {"success": True}
    def remove(self, target, old_text):
        self.calls.append(("remove", target, old_text)); return {"success": True}


def test_10_low_risk_preference_becomes_candidate_not_profile_edit(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    from tools.memory_tool import memory_tool
    store = FakeStore()
    result = json.loads(memory_tool("add", "user", "User prefers concise responses", store=store))
    assert result["candidate_created"] is True
    assert store.calls == []
    candidate = json.loads(Path(result["candidate_path"]).read_text())
    assert candidate["target_routing"] == "governed_profile_candidate"


def test_11_prohibited_content_is_rejected(monkeypatch, tmp_path, background_origin):
    _bind_home(monkeypatch, tmp_path)
    from tools.memory_tool import memory_tool
    result = json.loads(memory_tool("add", "memory", "client matter medical secret token", store=FakeStore()))
    assert result["rejected"] is True
    assert result["violations"]


def test_12_unregistered_writer_is_rejected_by_barrier(tmp_path, monkeypatch):
    home = _bind_home(monkeypatch, tmp_path)
    p = tmp_path / "protected.txt"
    p.write_text("before")
    before = hash_protected_files([str(p)])
    p.write_text("after")
    result = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, authorized_changed_paths=[], quiescence_seconds=0.01)
    assert result["gate"] == "red"
    assert str(p) in result["unauthorized_changed_paths"]


def test_13_sentinel_before_hooks_is_insufficient(tmp_path):
    p = tmp_path / "x"
    p.write_text("a")
    before = hash_protected_files([str(p)])
    p.write_text("b")
    after = hash_protected_files([str(p)])
    assert changed_paths(before, after) == [str(p)]


def test_14_finalization_waits_for_background_hooks(tmp_path, monkeypatch):
    home = _bind_home(monkeypatch, tmp_path)
    done = []
    def worker():
        time.sleep(0.05); done.append(True)
    t = threading.Thread(target=worker, name="bg-review")
    t.start()
    result = run_finalization_barrier(protected_paths=[], quiescence_seconds=0.01, wait_timeout=1)
    assert done == [True]
    assert result["gate"] == "green"


def test_15_post_receipt_write_invalidates_prior_gate(tmp_path, monkeypatch):
    home = _bind_home(monkeypatch, tmp_path)
    p = tmp_path / "protected"
    p.write_text("a")
    before = hash_protected_files([str(p)])
    first = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, quiescence_seconds=0.01)
    assert first["gate"] == "green"
    p.write_text("late")
    second = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, quiescence_seconds=0.01)
    assert second["gate"] == "red"


def test_16_changed_path_list_includes_authorized_write(tmp_path, monkeypatch):
    home = _bind_home(monkeypatch, tmp_path)
    p = tmp_path / "protected"
    p.write_text("a")
    before = hash_protected_files([str(p)])
    p.write_text("b")
    result = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, authorized_changed_paths=[str(p)], quiescence_seconds=0.01)
    assert result["actual_changed_paths"] == [str(p)]
    assert result["gate"] == "green"


def test_17_unreceipted_write_blocks_finalization(tmp_path, monkeypatch):
    _bind_home(monkeypatch, tmp_path)
    p = tmp_path / "protected"
    p.write_text("a")
    before = hash_protected_files([str(p)])
    p.write_text("b")
    result = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, authorized_changed_paths=[], quiescence_seconds=0.01)
    assert str(p) in result["unreceipted_changed_paths"]


def test_18_concurrent_background_writers_cannot_race_barrier(tmp_path, monkeypatch):
    _bind_home(monkeypatch, tmp_path)
    p = tmp_path / "protected"
    p.write_text("a")
    before = hash_protected_files([str(p)])
    def writer():
        time.sleep(0.03); p.write_text("raced")
    t = threading.Thread(target=writer, name="bg-review")
    t.start()
    result = run_finalization_barrier(protected_paths=[str(p)], before_hashes=before, quiescence_seconds=0.01, wait_timeout=1)
    assert result["gate"] == "red"
    assert str(p) in result["actual_changed_paths"]


def test_19_original_incident_pattern_is_reproduced_and_caught(monkeypatch, tmp_path, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home, body="---\nname: demo\n---\n\nincident before\n")
    before = path.read_text()
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage("patch", "demo", old_string="incident before", new_string="incident after"))
    assert result["quarantined"] is True
    assert path.read_text() == before
    _, proposal = _latest_proposal(home)
    assert "incident after" in proposal["proposed_diff"]


def test_20_active_protected_files_remain_unchanged_throughout(tmp_path, monkeypatch, background_origin):
    home = _bind_home(monkeypatch, tmp_path)
    path = _make_skill(home)
    before_hash = hash_protected_files([str(path)])
    from tools.skill_manager_tool import skill_manage
    skill_manage("patch", "demo", old_string="old", new_string="new")
    after_hash = hash_protected_files([str(path)])
    assert after_hash == before_hash
