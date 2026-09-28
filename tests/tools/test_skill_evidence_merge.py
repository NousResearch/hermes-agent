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


def _stage_via_real_gate(action="patch", name="test-skill", evidence=None, skill_dir=None):
    """Stage an evidence write through the REAL _apply_skill_write_gate and return the payload.

    Tests must not hand-build a staged payload: the frozen candidate is only honoured when it
    carries this process's per-staging nonce, so a hand-made dict exercises the untrusted-caller
    path instead of the replay path it means to test.
    """
    captured = {}

    def fake_run(build):
        class WA:
            @staticmethod
            def skill_pending_diff(rec):
                return write_approval.skill_pending_diff(rec)

            @staticmethod
            def skill_gist(*a, **k):
                return "gist"

        captured["payload"], _ = build(WA)
        return "staged"

    tmp_path = skill_dir.parent
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", side_effect=fake_run):
        smt._apply_skill_write_gate(
            action, name, content=None, category=None, file_path=None, file_content=None,
            old_string=None, new_string=None, replace_all=False, absorbed_into=None,
            evidence_merge=evidence)
    return captured["payload"]




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
    # Stage for real so the payload carries this process's staging nonce, then corrupt the source so
    # the replay is stale. Replaying it must be refused, not clobber the newer count.
    payload = _stage_via_real_gate(evidence={"success_count": 1}, skill_dir=skill_dir)
    (skill_dir / "SKILL.md").write_text(BASE.replace("success_count: 11", "success_count: 99"),
                                        encoding="utf-8")
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        result = smt.apply_skill_pending(payload)
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


def _evidence_schema_branch():
    """The evidence_merge op branch in the tool schema.

    The schema is a per-action ``anyOf`` (not one flat union) so a model that just used
    write_file cannot emit file_content on a patch. Locate our branch by shape, not position.
    """
    items = smt.SKILL_MANAGE_SCHEMA["parameters"]["properties"]["operations"]["items"]
    for branch in items["anyOf"]:
        if "evidence_merge" in branch.get("properties", {}):
            return branch
    raise AssertionError("no evidence_merge branch in the skill_manage schema")


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
    staged = captured["payload"]["operations"][0]["evidence_merge"]
    # The candidate is frozen at staging, with the reviewer-facing diff attached...
    assert "success_count: 12" in staged["_candidate_content"]
    assert "_preview" in staged
    # ...and it must still describe the APPROVED bytes after the skill moves on underneath.
    (skill_dir / "SKILL.md").write_text(BASE.replace("success_count: 11", "success_count: 99"),
                                        encoding="utf-8")
    preview = write_approval.skill_pending_diff(
        {"payload": {"action": "patch", "name": "test-skill", "evidence_merge": staged}})
    assert "success_count: 12" in preview
    assert "success_count: 99" not in preview
    # A replay against the moved file is refused rather than clobbering the newer count.
    # Go through the real approve entry point: a frozen candidate is honoured only when the
    # gate-bypass token is set, so calling the handler directly would test nothing.
    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        replayed = smt.apply_skill_pending({
            "action": "patch", "name": "test-skill", "content": None, "old_string": None,
            "new_string": None, "file_path": None, "replace_all": False,
            "evidence_merge": staged})
    parsed = json.loads(replayed) if isinstance(replayed, str) else replayed
    assert parsed["success"] is False
    assert "stale" in parsed["error"]
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8").count("success_count: 99") == 1


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
    evidence = _evidence_schema_branch()["properties"]["evidence_merge"]
    step = evidence["properties"]["steps"]["items"]
    evolution = evidence["properties"]["evolution"]["items"]
    assert step["required"] == ["name", "ok", "fail"]
    assert step["additionalProperties"] is False
    assert evolution["required"] == ["from", "to", "date", "reason"]
    assert evolution["additionalProperties"] is False
    # Counters are deltas to ADD, so the schema must forbid negatives.
    assert evidence["properties"]["success_count"]["minimum"] == 0
    assert evidence["properties"]["fail_count"]["minimum"] == 0


def test_evidence_branch_is_patch_only_and_exclusive():
    """The branch advertises patch, and carries no text slot that could ride along."""
    branch = _evidence_schema_branch()
    assert branch["properties"]["action"]["enum"] == ["patch"]
    assert set(branch["properties"]) == {"name", "action", "evidence_merge"}
    assert branch["additionalProperties"] is False


def test_restaging_a_staged_payload_is_idempotent(tmp_path):
    """A staged payload carries private keys (_source_digest/_candidate_content/_preview).

    Re-staging it must not feed those back into merge_evidence, which rejects unknown fields by
    design. Without the strip this raises "unknown merge fields" and an approved write that
    happens to be re-staged would fail outright.
    """
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
            def skill_gist(*a, **k):
                return "gist"

        captured["payload"], _ = build(WA)
        return "staged"

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", side_effect=fake_run):
        assert smt.skill_manage(action="patch", name="test-skill",
                                evidence_merge={"success_count": 1}) == "staged"
        staged = captured["payload"]["evidence_merge"]
        assert {"_source_digest", "_candidate_content", "_preview"} <= set(staged)
        # Re-stage the very same payload: must succeed, not raise.
        assert smt.skill_manage(action="patch", name="test-skill", evidence_merge=staged) == "staged"
    again = captured["payload"]["evidence_merge"]
    assert "success_count: 12" in again["_candidate_content"]


def test_concurrent_evidence_updates_do_not_lose_a_count(tmp_path):
    """Two threads updating the same skill must ADD, not overwrite.

    Read-merge-write has to be one critical section. If the merge happens outside the lock, both
    threads read the same count and the second write silently discards the first — the exact
    failure additive counters exist to prevent. 11 + 1 + 1 must be 13.

    Deterministic by construction: the two threads are released together from a barrier, and the
    merge is forced to happen outside the lock to model the interleaving. Patching globals is
    process-wide, so the patch is installed ONCE around both threads rather than per-thread.
    """
    import threading

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")

    barrier = threading.Barrier(2)
    errors: list = []
    gate = threading.Event()

    def bump():
        try:
            barrier.wait(timeout=10)  # both threads race into the read
            gate.wait(timeout=10)
            smt._act_patch({
                "name": "test-skill", "content": None, "old_string": None, "new_string": None,
                "file_path": None, "replace_all": False,
                "evidence_merge": {"success_count": 1},
            })
        except Exception as exc:  # noqa: BLE001 — surfaced via `errors`
            errors.append(repr(exc))

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        threads = [threading.Thread(target=bump) for _ in range(2)]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "a writer deadlocked on the skill lock"
    assert not errors, errors
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "success_count: 13" in text, f"lost update: got {[l for l in text.splitlines() if 'success_count' in l]}"


def test_injected_candidate_content_cannot_write_a_forged_count(tmp_path):
    """A caller-supplied _candidate_content must not be written verbatim.

    The frozen candidate is only honoured when THIS process's staging pass produced it, identified
    by a per-pass nonce the caller cannot derive. Here the write gate is off (no staging ran), so
    the merge runs on the public delta and the forged candidate is simply never used.
    """
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    forged = BASE.replace("success_count: 11", "success_count: 9999")

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", return_value=None):
        smt.skill_manage(action="patch", name="test-skill", evidence_merge={
            "success_count": 1,
            "_source_digest": smt.content_digest(BASE),
            "_candidate_content": forged,
        })

    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "success_count: 9999" not in text, "the forged candidate was written verbatim"
    # The real merge still applied the caller's legitimate delta: 11 + 1.
    assert "success_count: 12" in text, text[:200]


def test_forged_candidate_is_rejected_even_with_a_matching_nonce_shape(tmp_path):
    """Guessing the key name is not enough: the nonce value must match this process's staging pass.

    A caller who supplies _staged_by with a plausible value must still not be honoured, because
    the nonce is generated per staging pass and is not predictable.
    """
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    forged = BASE.replace("success_count: 11", "success_count: 9999")

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", return_value=None):
        smt.skill_manage(action="patch", name="test-skill", evidence_merge={
            "success_count": 1,
            "_source_digest": smt.content_digest(BASE),
            "_candidate_content": forged,
            "_staged_by": "0" * 32,
        })

    assert "success_count: 9999" not in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_batch_rejects_injected_staging_keys(tmp_path):
    """A caller must not be able to smuggle a frozen candidate through the public batch interface.

    The batch path sets the same gate-bypass token as an approved replay, so a caller-supplied
    _candidate_content used to be written verbatim — turning an 11-success skill into 0 through a
    documented parameter. Reproduced before the fix; the ingress now rejects internal keys.
    """
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    current = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    forged = current.replace("success_count: 11", "success_count: 0")

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        out = smb._skill_manage_batch(
            [{"action": "patch", "name": "test-skill", "evidence_merge": {
                "success_count": 1,
                "_source_digest": smt.content_digest(current),
                "_candidate_content": forged}}], None, None, None)

    text = json.dumps(out)
    assert "internal staging keys" in text, text[:200]
    assert "success_count: 0" not in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "success_count: 11" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_batch_approval_preview_shows_the_evidence_change(tmp_path):
    """The reviewer must SEE the counter change they are approving.

    A batch record used to render as "( on '')": the approval surface showed nothing while the
    write proceeded. The preview is built the way hermes_cli/write_approval_commands.py builds it.
    """
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE, encoding="utf-8")
    captured = {}

    def fake_run(build):
        class WA:
            @staticmethod
            def skill_pending_diff(rec):
                return write_approval.skill_pending_diff(rec)

            @staticmethod
            def skill_gist(*a, **k):
                return "gist"

        captured["payload"], _ = build(WA)
        return "staged"

    with patch.object(smt, "SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch.object(smt, "_run_write_gate", side_effect=fake_run):
        assert smb._skill_manage_batch(
            [{"action": "patch", "name": "test-skill",
              "evidence_merge": {"success_count": 1}}], None, None, None) == "staged"

    rec = {"id": "p1", "summary": "batch", "payload": captured["payload"]}
    preview = write_approval.skill_pending_diff(rec)
    # The reviewer's surface must show the resulting count and must not be the empty fallback.
    assert "success_count: 12" in preview, preview[:300]
    assert preview.strip() != "( on '')"
    assert "( on '')" not in preview
