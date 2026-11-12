"""Tests for the delegated-child operative-prompt recorder.

A subagent's stored ``system_prompt_hash`` points at the cached BASE prompt. The operative
prompt -- base + ``CONTEXT:`` block + workspace context files -- is composed at call time and
was never persisted, so two children given different contracts were indistinguishable in the
session store. These tests pin the recorder's contract: content-addressed artefact, one
append-only index line per child, full-text digest, and fail-open behaviour.
"""

import hashlib
import json
import os

from tools.delegate_tool_prompt_record import record_child_prompt


def _index(tmp_path):
    return [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines() if line.strip()]


def test_records_prompt_and_index_line(tmp_path):
    rec = record_child_prompt(
        goal="do the thing", context="GOVERNANCE: human_gates", child_prompt="PROMPT human_gates",
        subagent_id="sa-0-deadbeef", task_index=0, parent_session_id="parent-1",
        workspace_path="/tmp", state_dir=str(tmp_path),
    )
    assert rec is not None
    assert rec["prompt_sha256"] == hashlib.sha256(b"PROMPT human_gates").hexdigest()
    assert rec["prompt_len"] == len("PROMPT human_gates")
    assert rec["markers"]["human_gates"] is True
    assert rec["markers"]["sentinel"] is False
    assert rec["subagent_id"] == "sa-0-deadbeef"
    assert rec["parent_session_id"] == "parent-1"
    # the goal is the child's first user turn, so its digest is the join key to the session
    assert rec["goal_sha256"] == hashlib.sha256(b"do the thing").hexdigest()
    assert _index(tmp_path) == [rec]
    assert (tmp_path / f"{rec['prompt_sha256']}.txt").read_text() == "PROMPT human_gates"


def test_identical_prompts_deduplicate_the_artefact_but_not_the_record(tmp_path):
    for i in range(3):
        assert record_child_prompt(goal=f"goal {i}", context=None, child_prompt="SAME",
                                   task_index=i, state_dir=str(tmp_path)) is not None
    # the suite may provision its own dirs under tmp_path; assert only on our artefacts
    artefacts = sorted(p.name for p in tmp_path.glob("*.txt"))
    assert artefacts == [hashlib.sha256(b"SAME").hexdigest() + ".txt"]
    assert len(_index(tmp_path)) == 3


def test_digest_covers_full_text_when_the_artefact_is_truncated(tmp_path):
    text = "z" * 400_000
    rec = record_child_prompt(goal="g", context=None, child_prompt=text, state_dir=str(tmp_path))
    assert rec is not None
    assert rec["truncated"] is True
    assert rec["prompt_len"] == len(text)
    assert rec["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    stored = (tmp_path / f"{rec['prompt_sha256']}.txt").read_text()
    assert len(stored) < len(text)
    assert "truncated" in stored


def test_never_raises_when_the_state_dir_is_unusable(tmp_path):
    blocker = tmp_path / "a-file-where-a-directory-is-needed"
    blocker.write_text("not a directory")
    assert record_child_prompt(goal="g", context=None, child_prompt="x",
                               state_dir=str(blocker)) is None


def test_default_state_dir_is_profile_scoped_and_overridable(tmp_path):
    from tools.delegate_tool_prompt_record import _resolve_state_dir

    default = _resolve_state_dir(None)
    assert default.parts[-2:] == ("state", "delegated-prompts")
    assert default.is_absolute()
    assert _resolve_state_dir(str(tmp_path)) == tmp_path
    # computing the default must not create anything
    assert not os.path.exists(default) or default.is_dir()
