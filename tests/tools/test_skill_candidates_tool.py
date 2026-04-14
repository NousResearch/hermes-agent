from __future__ import annotations

import json
from pathlib import Path

from plugins.memory import load_memory_provider
from tools.registry import registry
import tools.skill_candidates_tool  # ensure registration side effect


def _seed_candidate(tmp_path: Path):
    provider = load_memory_provider("layered")
    provider.initialize(session_id="skill-candidate-tool", hermes_home=str(tmp_path), platform="cli")
    for _ in range(3):
        provider.on_session_end([
            {"role": "user", "content": "Please implement a bugfix with tests."},
            {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
        ])
    provider.shutdown()


def test_skill_candidates_tool_registered():
    entry = registry.get_entry("skill_candidates")
    assert entry is not None
    assert entry.toolset == "skills"


def test_skill_candidates_tool_list_returns_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    result = json.loads(registry.dispatch("skill_candidates", {"action": "list"}))

    assert result["success"] is True
    assert result["candidates"]
    assert any(item["skill_name"] == "write-failing-tests-first-then-verify-tests-pass" for item in result["candidates"])


def test_skill_candidates_tool_inspect_returns_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    result = json.loads(registry.dispatch("skill_candidates", {"action": "inspect", "name": "write-failing-tests-first-then-verify-tests-pass"}))

    assert result["success"] is True
    assert result["candidate"]["skill_name"] == "write-failing-tests-first-then-verify-tests-pass"
    assert result["candidate"]["skill_draft_path"].endswith("write-failing-tests-first-then-verify-tests-pass.md")
    assert result["candidate"]["evidence"]["promotion_rationale"]
    assert result["candidate"]["evidence"]["sample_evidence"]
    assert result["candidate"]["evidence"]["verification_hints"]


def test_skill_candidates_tool_approve_installs_skill(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    result = json.loads(registry.dispatch("skill_candidates", {"action": "approve", "name": "write-failing-tests-first-then-verify-tests-pass"}))

    assert result["success"] is True
    assert result["strategy"] == "create"
    assert result["installed_skill_path"].endswith("skills/write-failing-tests-first-then-verify-tests-pass/SKILL.md")
    assert (tmp_path / "skills" / "write-failing-tests-first-then-verify-tests-pass" / "SKILL.md").exists()


def test_skill_candidates_tool_approve_returns_duplicate_skip_strategy(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    details = json.loads(registry.dispatch("skill_candidates", {"action": "inspect", "name": "write-failing-tests-first-then-verify-tests-pass"}))
    package_skill = Path(details["candidate"]["publish_ready_dir"]) / "SKILL.md"
    skills_dir = tmp_path / "skills" / "write-failing-tests-first-then-verify-tests-pass"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text(package_skill.read_text())

    result = json.loads(registry.dispatch("skill_candidates", {"action": "approve", "name": "write-failing-tests-first-then-verify-tests-pass"}))

    assert result["success"] is True
    assert result["strategy"] == "duplicate_skip"


def test_skill_candidates_tool_reject_updates_status(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    result = json.loads(registry.dispatch("skill_candidates", {
        "action": "reject",
        "name": "write-failing-tests-first-then-verify-tests-pass",
        "reason": "manual_reject",
    }))

    assert result["success"] is True
    assert result["review_status"] == "rejected"
    assert result["review_gate_reason"] == "manual_reject"
