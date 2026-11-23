"""Staging-time duplicate notice for the background review (tools/skill_manager_guards.py).

One long session re-fires the review fork every ``skills.creation_nudge_interval`` iterations; each
fork is a fresh context, so it re-derives the lesson it staged twenty minutes earlier and the queue
fills with records the user has to reconcile by hand. These tests pin the notice: a restatement
stages nothing and names the record that already holds the lesson, a genuinely different patch still
stages, and the wiring in ``_run_write_gate`` returns the notice instead of a record.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

LINE_A = "- **Boolean exists() is not a state model.** A permission fault reads as absence and the cleanup path then acts as though nothing were there."
LINE_B = "- **os.kill(pid, 0) folds deniable liveness into death.** EPERM means alive under another user; only ESRCH means gone."
LINE_C = "- **A cadence gate needs two candidate fixtures.** One candidate cannot distinguish a passing gate from a gate that never ran."


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_dupe_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def as_background_review(monkeypatch):
    from tools import write_approval as wa
    monkeypatch.setattr(wa, "current_origin", lambda: "background_review")


@pytest.fixture
def as_foreground(monkeypatch):
    from tools import write_approval as wa
    monkeypatch.setattr(wa, "current_origin", lambda: "foreground")


def _queue():
    from tools import write_approval as wa
    return wa.list_pending(wa.SKILLS)


def _stage_first():
    from tools import write_approval as wa
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- **Existing bullet.** anchor line the fork re-emits.",
        "new_string": "- **Existing bullet.** anchor line the fork re-emits.\n" + LINE_A + "\n" + LINE_B,
    }]}
    return wa.stage_write(wa.SKILLS, payload, summary="batch(1 ops: patch) on requesting-code-review",
                          origin="background_review")


def _restatement(anchor="- **Different anchor.** a later fork anchors somewhere else."):
    return {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": anchor,
        "new_string": anchor + "\n" + LINE_A + "\n" + LINE_B,
    }]}


def test_restatement_stages_nothing_and_names_the_record(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    first = _stage_first()
    notice = duplicate_background_write_notice(_restatement())
    assert notice is not None
    parsed = json.loads(notice)
    assert parsed["staged"] is False and parsed["duplicate_of"] == first["id"]
    assert parsed["coverage_pct"] == 100 and parsed["new_lines"] == []
    records = _queue()
    assert len(records) == 1 and len(records[0]["payload"]["operations"]) == 1


def test_three_forks_leave_one_record(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    first = _stage_first()
    for _ in range(2):
        assert duplicate_background_write_notice(_restatement()) is not None
    records = _queue()
    assert len(records) == 1 and records[0]["id"] == first["id"]
    assert len(records[0]["payload"]["operations"]) == 1, "the queued record must not grow"


def test_uncovered_lines_are_reported_back(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    mixed = _restatement()
    mixed["operations"][0]["new_string"] += "\n" + LINE_C
    parsed = json.loads(duplicate_background_write_notice(mixed))
    assert parsed["staged"] is False
    assert parsed["new_lines"] == [LINE_C], "the fork must be told which line is new"
    assert "stage a patch adding just those lines" in parsed["message"]
    assert "that record's approval" in parsed["message"]
    assert "covered by one approval" not in parsed["message"], "uncovered lines make that false"
    assert ".." not in parsed["message"], "a bullet terminator plus the list's own full stop"
    assert len(_queue()) == 1


def test_distinct_patch_is_staged(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    other = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- **Anchor.** live text.",
        "new_string": "- **Anchor.** live text.\n" + LINE_C,
    }]}
    assert duplicate_background_write_notice(other) is None


def test_foreground_write_is_never_checked(hermes_home, as_foreground):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    assert duplicate_background_write_notice(_restatement()) is None


def test_flat_restatement_is_caught(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    flat = {"action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
            "old_string": "- **Third anchor.** elsewhere in the file.",
            "new_string": "- **Third anchor.** elsewhere in the file.\n" + LINE_A + "\n" + LINE_B}
    assert duplicate_background_write_notice(flat) is not None


def test_creation_payloads_are_out_of_scope(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    create = {"action": "batch", "operations": [{
        "action": "create", "name": "requesting-code-review", "content": LINE_A + "\n" + LINE_B,
    }]}
    assert duplicate_background_write_notice(create) is None


def test_malformed_record_payload_does_not_raise(hermes_home, as_background_review, monkeypatch):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    monkeypatch.setattr(wa, "list_pending", lambda subsystem: [
        {"id": "deadbeef", "origin": "background_review", "payload": ["not", "a", "dict"]},
    ])
    assert duplicate_background_write_notice(_restatement()) is None


def test_write_gate_returns_the_notice_instead_of_a_record(hermes_home, as_background_review, monkeypatch):
    from tools import skill_manager_tool, write_approval as wa
    first = _stage_first()
    monkeypatch.setattr(wa, "evaluate_gate",
                        lambda subsystem: SimpleNamespace(allow=False, blocked=False, message="staged for approval"))
    result = skill_manager_tool._run_write_gate(lambda w: (_restatement(), "batch(1 ops: patch)"))
    assert result is not None
    parsed = json.loads(result)
    assert parsed["staged"] is False and parsed["duplicate_of"] == first["id"]
    assert len(_queue()) == 1, "the gate must not stage a second record"


def test_write_gate_stages_a_distinct_patch(hermes_home, as_background_review, monkeypatch):
    from tools import skill_manager_tool, write_approval as wa
    _stage_first()
    monkeypatch.setattr(wa, "evaluate_gate",
                        lambda subsystem: SimpleNamespace(allow=False, blocked=False, message="staged for approval"))
    other = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- **Anchor.** live text.",
        "new_string": "- **Anchor.** live text.\n" + LINE_C,
    }]}
    result = skill_manager_tool._run_write_gate(lambda w: (other, "batch(1 ops: patch)"))
    assert result is not None
    parsed = json.loads(result)
    assert parsed["staged"] is True and parsed["pending_id"] != ""
    assert len(_queue()) == 2


def test_same_skill_other_reference_file_is_not_a_match(hermes_home, as_background_review):
    """A patch to references/x.md must not be suppressed by the same lines queued against SKILL.md."""
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    ref = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "references/x.md",
        "old_string": "- **Reference anchor.** different file.",
        "new_string": "- **Reference anchor.** different file.\n" + LINE_A + "\n" + LINE_B,
    }]}
    assert duplicate_background_write_notice(ref) is None


def test_record_side_coverage_is_filtered_per_file(hermes_home, as_background_review):
    """Lines queued for a sibling reference file must not count as covered for SKILL.md."""
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [
        {"action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
         "old_string": "- a.", "new_string": "- a.\n" + LINE_A},
        {"action": "patch", "name": "requesting-code-review", "file_path": "references/notes.md",
         "old_string": "- b.", "new_string": "- b.\n" + LINE_A + "\n" + LINE_B},
    ]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B,
    }]}
    # 50% coverage: LINE_A is queued for SKILL.md, LINE_B is not
    assert duplicate_background_write_notice(payload) is None


def test_one_repeated_bullet_is_suppressed(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    one = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- **Anchor.** text.", "new_string": "- **Anchor.** text.\n" + LINE_B,
    }]}
    assert duplicate_background_write_notice(one) is not None


def test_lines_spread_across_records_are_recognised(hermes_home, as_background_review):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    for line in (LINE_A, LINE_B):
        wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [{
            "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
            "old_string": "- a.", "new_string": "- a.\n" + line,
        }]}, summary="probe", origin="background_review")
    both = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B,
    }]}
    parsed = json.loads(duplicate_background_write_notice(both))
    assert parsed["staged"] is False and parsed["coverage_pct"] == 100
    assert "2 queued records" in parsed["message"]


def test_anchor_lines_do_not_count_as_added(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    # The anchor is itself a line the queued record adds: with anchors counted this payload scores
    # 2/3 (67%) and is refused, so the assertion discriminates.
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": LINE_A, "new_string": LINE_A + "\n" + LINE_B + "\n" + LINE_C,
    }]}
    assert duplicate_background_write_notice(payload) is None


def test_review_summary_reports_nothing_for_a_notice():
    from agent import background_review as bg
    notice = {"success": True, "staged": False, "duplicate_of": "a1b2c3d4",
              "coverage_pct": 100, "new_lines": [],
              "message": "Not staged: 100% of this patch's added lines are already queued for x/SKILL.md"}
    detail = {"tool": "skill_manage"}
    assert bg._action_lines(notice, detail, verbose=False) == []
    assert bg._action_lines(notice, detail, verbose=True) == []


def test_path_spellings_and_name_less_ops_still_match(hermes_home, as_background_review):
    """The applier accepts '/SKILL.md', './SKILL.md' and ops inheriting the payload's name."""
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [
        {"action": "patch", "name": "requesting-code-review", "file_path": "./SKILL.md",
         "old_string": "- a.", "new_string": "- a.\n" + LINE_A + "\n" + LINE_B},
    ]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "name": "requesting-code-review", "operations": [
        {"action": "patch", "file_path": "/SKILL.md", "old_string": "- c.",
         "new_string": "- c.\n" + LINE_A + "\n" + LINE_B},
    ]}
    assert duplicate_background_write_notice(payload) is not None


def test_origin_lookup_failure_does_not_raise(hermes_home, monkeypatch):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()

    def boom():
        raise RuntimeError("provenance store unreadable")

    monkeypatch.setattr(wa, "current_origin", boom)
    assert duplicate_background_write_notice(_restatement()) is None


def test_long_uncovered_list_is_truncated_with_a_count(hermes_home, as_background_review):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    queued = [f"- **Queued rule {i}.** A forty-plus character lesson line, numbered {i}, that the earlier record carries." for i in range(12)]
    fresh = [f"- **Fresh rule {i}.** A forty-plus character lesson line, numbered {i}, that no record carries." for i in range(8)]
    wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- a.", "new_string": "- a.\n" + "\n".join(queued),
    }]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + "\n".join(queued + fresh),
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert parsed["coverage_pct"] == 60
    assert len(parsed["new_lines"]) == 6 and "and 2 more" in parsed["message"]


def test_chiefly_when_one_record_carries_every_covered_line(hermes_home, as_background_review):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [
        {"action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
         "old_string": "- a.", "new_string": "- a.\n" + LINE_A + "\n" + LINE_B},
        {"action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
         "old_string": "- z.", "new_string": "- z.\n" + LINE_C},
    ]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B + "\n" + LINE_C,
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert "chiefly in pending record" in parsed["message"]
    assert "2 queued records" not in parsed["message"]
    assert "covered by one approval" in parsed["message"]


def test_partly_and_plural_when_carriers_split(hermes_home, as_background_review):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    for line in (LINE_A, LINE_B):
        wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [{
            "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
            "old_string": "- a.", "new_string": "- a.\n" + line,
        }]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B,
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert "partly in pending record" in parsed["message"]
    assert "2 queued records carry these lines" in parsed["message"]
    assert "covered by one approval" not in parsed["message"]
    assert "covered once those records are approved" in parsed["message"]


def test_padded_file_path_is_normalised(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    padded = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": " SKILL.md ",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B,
    }]}
    assert duplicate_background_write_notice(padded) is not None


def test_uncovered_lines_across_two_carriers_pluralise_the_closing(hermes_home, as_background_review):
    from tools import write_approval as wa
    from tools.skill_manager_guards import duplicate_background_write_notice
    for line in (LINE_A, LINE_B):
        wa.stage_write(wa.SKILLS, {"action": "batch", "operations": [{
            "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
            "old_string": "- a.", "new_string": "- a.\n" + line,
        }]}, summary="probe", origin="background_review")
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B + "\n" + LINE_C,
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert parsed["coverage_pct"] == 66
    assert "those records' approvals" in parsed["message"]
    assert "that record's approval" not in parsed["message"]
    assert "covered by one approval" not in parsed["message"]


def test_short_novel_line_is_reported_not_buried(hermes_home, as_background_review):
    """Coverage counts 40+ char lines only; a short new rule must still be named back."""
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    short_rule = "- **No secrets.** keep them out"
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B + "\n" + short_rule,
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert parsed["coverage_pct"] == 100
    assert short_rule in parsed["new_lines"], "the short line must be named, not swallowed"
    assert short_rule.rstrip(".") in parsed["message"], "named in the message, not only the JSON"
    assert "counted added lines" in parsed["message"], "the lead must not claim uncounted text is queued"
    assert "Every line is already queued" not in parsed["message"]


def test_mixed_uncovered_long_and_short_lines_are_both_named(hermes_home, as_background_review):
    from tools.skill_manager_guards import duplicate_background_write_notice
    _stage_first()
    short_rule = "- **No secrets.** keep them out"
    payload = {"action": "batch", "operations": [{
        "action": "patch", "name": "requesting-code-review", "file_path": "SKILL.md",
        "old_string": "- c.", "new_string": "- c.\n" + LINE_A + "\n" + LINE_B + "\n" + LINE_C + "\n" + short_rule,
    }]}
    parsed = json.loads(duplicate_background_write_notice(payload))
    assert LINE_C in parsed["new_lines"] and short_rule in parsed["new_lines"]
    assert LINE_C.rstrip(".") in parsed["message"] and short_rule.rstrip(".") in parsed["message"]
    assert "counted added lines" in parsed["message"]
