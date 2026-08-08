import json
from pathlib import Path

import pytest

from tools.skillopt import SkillOptError, promote_skill_candidate


BASE_SKILL = """---
name: demo-skill
description: Demo skill for tests.
---

# Demo

Do the old thing.
"""

BETTER_SKILL = """---
name: demo-skill
description: Demo skill for tests.
---

# Demo

Do the better thing with a concrete verification step.
"""

INVALID_SKILL = """# Missing frontmatter

Nope.
"""

RENAMED_SKILL = """---
name: other-skill
description: Renamed skill for tests.
---

# Demo

Do the renamed thing.
"""


def _make_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(BASE_SKILL, encoding="utf-8")
    return skill_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_promotes_candidate_when_score_strictly_improves(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "candidate.md"
    candidate.write_text(BETTER_SKILL, encoding="utf-8")

    result = promote_skill_candidate(
        skill_dir,
        candidate,
        baseline_score=0.72,
        candidate_score=0.81,
    )

    assert result.accepted is True
    assert result.decision == "accepted"
    assert "better thing" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert Path(result.backup_path).exists()
    history = _read_jsonl(skill_dir / ".skillopt" / "history.jsonl")
    assert history[-1]["candidate_score"] == 0.81
    assert history[-1]["baseline_score"] == 0.72


def test_rejects_candidate_when_score_does_not_improve(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "candidate.md"
    candidate.write_text(BETTER_SKILL, encoding="utf-8")

    result = promote_skill_candidate(
        skill_dir,
        candidate,
        baseline_score=0.81,
        candidate_score=0.81,
    )

    assert result.accepted is False
    assert result.decision == "rejected"
    assert "old thing" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    rejected = _read_jsonl(skill_dir / ".skillopt" / "rejected.jsonl")
    assert "did not beat baseline" in rejected[-1]["reason"]


def test_rejects_invalid_skill_document_before_score_gate(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "invalid.md"
    candidate.write_text(INVALID_SKILL, encoding="utf-8")

    result = promote_skill_candidate(
        skill_dir,
        candidate,
        baseline_score=0.1,
        candidate_score=0.9,
    )

    assert result.accepted is False
    assert "candidate validation failed" in result.reason
    assert "old thing" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_rejects_candidate_when_validator_fails(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "candidate.md"
    candidate.write_text(BETTER_SKILL, encoding="utf-8")

    result = promote_skill_candidate(
        skill_dir,
        candidate,
        baseline_score=0.2,
        candidate_score=0.9,
        validator="python -c 'import os; print(os.environ[\"HERMES_SKILLOPT_CANDIDATE\"]); raise SystemExit(3)'",
    )

    assert result.accepted is False
    assert result.validator_exit_code == 3
    assert "validator failed" in result.reason


def test_rejects_candidate_that_renames_skill_identity(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "renamed.md"
    candidate.write_text(RENAMED_SKILL, encoding="utf-8")

    result = promote_skill_candidate(
        skill_dir,
        candidate,
        baseline_score=0.2,
        candidate_score=0.9,
    )

    assert result.accepted is False
    assert "name mismatch" in result.reason
    assert "old thing" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    rejected = _read_jsonl(skill_dir / ".skillopt" / "rejected.jsonl")
    assert "other-skill" in rejected[-1]["reason"]


def test_rejects_candidate_that_is_live_skill_file(tmp_path):
    skill_dir = _make_skill(tmp_path)

    result = promote_skill_candidate(
        skill_dir,
        skill_dir / "SKILL.md",
        baseline_score=0.2,
        candidate_score=0.9,
    )

    assert result.accepted is False
    assert "quarantined file distinct" in result.reason
    assert not (skill_dir / ".skillopt" / "backups").exists()


def test_rejects_non_finite_scores_before_history_write(tmp_path):
    skill_dir = _make_skill(tmp_path)
    candidate = tmp_path / "candidate.md"
    candidate.write_text(BETTER_SKILL, encoding="utf-8")

    with pytest.raises(SkillOptError, match="finite numbers"):
        promote_skill_candidate(
            skill_dir,
            candidate,
            baseline_score=0.2,
            candidate_score=float("nan"),
        )

    assert not (skill_dir / ".skillopt").exists()
