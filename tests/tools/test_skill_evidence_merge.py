import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import skill_manager_tool as smt
from tools import skill_manager_batch as smb
from tools import write_approval
from tools.skill_evidence import EvidenceMergeError, merge_evidence


BASE = """---
name: test-skill
description: Use when testing evidence merges. Keep evidence additive.
evidence:
  success_count: 11
  fail_count: 1
  steps: []
  evolution: []
---

# Body

Keep this byte-for-byte.
"""


def test_merge_is_additive_and_preserves_body():
    out = merge_evidence(BASE, {"success_count": 1, "fail_count": 2})
    assert "success_count: 12" in out
    assert "fail_count: 3" in out
    assert out.endswith("\n# Body\n\nKeep this byte-for-byte.\n")


def test_merge_steps_and_evolution():
    out = merge_evidence(BASE, {
        "steps": [{"name": "unit", "ok": 2, "fail": 0}],
        "evolution": [{"from": 0, "to": 1, "date": "2026-09-08", "reason": "verified"}],
    })
    assert "name: unit" in out and "version: 1" in out
    assert "reason: verified" in out


def test_merge_rejects_duplicate_and_negative():
    with pytest.raises(EvidenceMergeError):
        merge_evidence("---\nname: x\nname: y\n---\nbody\n", {"success_count": 1})
    with pytest.raises(EvidenceMergeError):
        merge_evidence(BASE, {"success_count": -1})
    with pytest.raises(EvidenceMergeError):
        merge_evidence(BASE, {"steps": [
            {"name": "unit", "ok": 1, "fail": 0},
            {"name": "unit", "ok": 1, "fail": 0},
        ]})
    with pytest.raises(EvidenceMergeError):
        merge_evidence(BASE, {"evolution": [{
            "from": 0, "to": "not-an-int", "date": "2026-09-08", "reason": "bad"}]})


def test_skill_manage_rewrites_temp_skill_without_gate(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", return_value=None):
        result = json.loads(smt.skill_manage(
            action="patch", name="test-skill", evidence_merge={"success_count": 1}))
    assert result["success"] is True
    assert "success_count: 12" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_stale_replay_is_rejected(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        result = smt._act_patch({
            "name": "test-skill", "content": None, "old_string": None, "new_string": None,
            "file_path": None, "replace_all": False,
            "evidence_merge": {"_source_digest": "wrong", "_candidate_content": BASE},
        })
    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed["success"] is False
    assert "stale" in parsed["error"]


def test_staging_captures_candidate_and_digest(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    captured = {}

    def fake_run(build):
        class WA:
            @staticmethod
            def skill_gist(*args, **kwargs):
                captured["gist"] = kwargs
                return "gist"

            @staticmethod
            def skill_pending_diff(record):
                return write_approval.skill_pending_diff(record)
        captured["payload"], _ = build(WA)
        return "staged"

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", side_effect=fake_run):
        result = smt._apply_skill_write_gate(
            "patch", "test-skill", content=None, category=None, file_path=None,
            file_content=None, old_string=None, new_string=None, replace_all=False,
            absorbed_into=None, evidence_merge={"success_count": 1})
    assert result == "staged"
    staged = captured["payload"]["evidence_merge"]
    assert staged["_source_digest"]
    assert "success_count: 12" in staged["_candidate_content"]
    assert captured["gist"]["content"] == staged["_candidate_content"]


def test_evidence_merge_is_patch_only_and_batch_order_is_explicit():
    error = lambda msg, success=False: json.dumps({"success": success, "error": msg})
    flat = smt._apply_skill_write_gate(
        "write_file", "test-skill", content=None, category=None, file_path="x",
        file_content="x", old_string=None, new_string=None, replace_all=False,
        absorbed_into=None, evidence_merge={"success_count": 1})
    assert json.loads(flat)["success"] is False
    _, result = smb._validate_batch_ops(
        [{"action": "write_file", "name": "test-skill", "evidence_merge": {"success_count": 1}}],
        None, error)
    assert json.loads(result)["success"] is False
    _, result = smb._validate_batch_ops(
        [{"action": "patch", "name": "test-skill", "evidence_merge": {"success_count": 1}},
         {"action": "patch", "name": "test-skill", "old_string": "a", "new_string": "b"}],
        None, error)
    assert json.loads(result)["success"] is False


def test_flat_pending_preview_uses_frozen_candidate(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    candidate = BASE.replace("success_count: 11", "success_count: 12")
    with patch.object(write_approval, "_find_skill_path", return_value=skill_dir):
        preview = write_approval.skill_pending_diff({"payload": {
            "action": "patch", "name": "test-skill",
            "evidence_merge": {"_candidate_content": candidate},
        }})
    assert "success_count: 12" in preview
    assert "success_count: 11" in preview
    assert "success_count: 12" != preview.strip()


def test_registered_dispatch_replays_evidence_merge(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    entry = smt.registry.get_entry("skill_manage")
    assert entry is not None and entry.toolset == "skills"
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", return_value=None):
        result = json.loads(entry.handler({
            "action": "patch", "name": "test-skill",
            "evidence_merge": {"success_count": 1}}))
    assert result["success"] is True
    assert "success_count: 12" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_batch_preview_is_frozen_inside_evidence_payload(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    captured = {}

    def fake_run(build):
        class WA:
            @staticmethod
            def skill_pending_diff(record):
                return write_approval.skill_pending_diff(record)

            @staticmethod
            def skill_gist(*args, **kwargs):
                return "gist"

        captured["payload"], _ = build(WA)
        return "staged"

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", side_effect=fake_run):
        result = smb._skill_manage_batch(
            [{"action": "patch", "name": "test-skill",
              "evidence_merge": {"success_count": 1}}],
            None, None, None)
    assert result == "staged"
    op = captured["payload"]["operations"][0]
    assert "_preview" in op["evidence_merge"]
    (skill_dir / "SKILL.md").write_text(BASE.replace("success_count: 11", "success_count: 99"), encoding="utf-8")
    preview = write_approval.skill_pending_diff({"payload": captured["payload"]})
    assert "success_count: 12" in preview
    assert "success_count: 99" not in preview


def test_final_guard_rejects_digest_drift_without_writing(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    candidate = BASE.replace("success_count: 11", "success_count: 12")
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        result = smt._edit_skill("test-skill", candidate, expected_source_digest="wrong")
    assert result["success"] is False
    assert "stale" in result["error"]
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == BASE


def test_evidence_schema_describes_nested_contract():
    schema = smt.SKILL_MANAGE_SCHEMA["parameters"]["properties"]["operations"]["items"]
    evidence = schema["properties"]["evidence_merge"]
    step = evidence["properties"]["steps"]["items"]
    evolution = evidence["properties"]["evolution"]["items"]
    assert step["required"] == ["name", "ok", "fail"]
    assert step["additionalProperties"] is False
    assert evolution["required"] == ["from", "to", "date", "reason"]
    assert evolution["additionalProperties"] is False
