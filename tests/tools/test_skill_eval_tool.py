"""Tests for the native skill_eval_run tool."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _write_eval_pack(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "skill_name": "demo-skill",
                "skill_path": "/tmp/demo/SKILL.md",
                "purpose": "Verify native tool wrapper behaviour.",
                "cases": [
                    {
                        "id": "CASE-1",
                        "title": "Dry run case",
                        "prompt": "Do the thing",
                        "expected_trigger": "user asks for thing",
                        "required_behaviors": ["show required behavior"],
                        "forbidden_behaviors": ["claim scored output during dry-run"],
                        "evidence_requirements": ["report path exists"],
                        "scoring_rubric": {"max_points": 5, "criteria": ["truthful"]},
                    }
                ],
                "scoring": {
                    "case_max_points": 5,
                    "pass_threshold_percent": 80,
                    "hard_fail_conditions": ["claims dry-run is scored"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_dry_run_writes_unscored_report_under_reports_dir(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run(
            {
                "eval_path": str(eval_path),
                "mode": "dry_run",
                "json_summary": True,
            }
        )
    )

    assert result["status"] == "ok"
    assert result["mode"] == "dry_run"
    assert result["scored"] is False
    assert result["case_count"] == 1
    report = Path(result["report"])
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "Skill Eval Report — demo-skill" in report_text
    assert "Dry-run: `true`" in report_text
    assert "Scored: `false`" in report_text
    assert "claim scored output during dry-run" in report_text


def test_manual_score_hard_fail_overrides_numeric_pass(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(
        json.dumps({"scores": {"CASE-1": {"score": 5, "notes": "good but unsafe", "hard_fail": True}}}),
        encoding="utf-8",
    )

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run(
            {
                "eval_path": str(eval_path),
                "mode": "manual_score",
                "score_file": str(score_path),
                "json_summary": True,
            }
        )
    )

    assert result["status"] == "failed"
    assert result["scored"] is True
    assert result["hard_fail"] is True
    assert result["score"] == 5
    assert result["max_score"] == 5


def test_eval_path_outside_allowed_roots_is_blocked(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = tmp_path / "outside.json"
    _write_eval_pack(eval_path)

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert "outside allowed roots" in result["error"]


def test_report_path_outside_reports_dir_is_blocked(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run(
            {
                "eval_path": str(eval_path),
                "mode": "dry_run",
                "report_path": str(tmp_path / "bad-report.md"),
            }
        )
    )

    assert result["status"] == "error"
    assert "report_path outside reports directory" in result["error"]


def test_invalid_pack_missing_cases_returns_schema_error(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "bad.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert any("missing top-level field: cases" in err for err in result["errors"])


def test_report_path_rejects_symlinked_reports_directory(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    reports_dir = hermes_home / "aion-evolution" / "evals" / "reports"
    outside = tmp_path / "outside-reports"
    outside.mkdir(parents=True)
    reports_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        reports_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert "symlink" in result["error"]
    assert not (outside / "demo-eval-report.md").exists()


def test_eval_path_rejects_symlinked_evals_root(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    real_evals = tmp_path / "real-evals"
    eval_path = real_evals / "demo.json"
    _write_eval_pack(eval_path)
    evals_root = hermes_home / "aion-evolution" / "evals"
    evals_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        evals_root.symlink_to(real_evals, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(evals_root / "demo.json"), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert "evals root contains symlink" in result["error"]


def test_manual_score_with_unknown_case_id_is_error(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(json.dumps({"scores": {"UNKNOWN": {"score": 5}}}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "error"
    assert "unknown score case id" in result["error"]


def test_manual_score_with_empty_score_file_is_error(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(json.dumps({"scores": {}}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "error"
    assert "at least one matching score" in result["error"]


def test_partial_manual_score_is_incomplete_not_scored(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    second_case = dict(data["cases"][0])
    second_case["id"] = "CASE-2"
    data["cases"].append(second_case)
    eval_path.write_text(json.dumps(data), encoding="utf-8")
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(json.dumps({"scores": {"CASE-1": {"score": 5}}}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "incomplete"
    assert result["scored"] is False
    assert result["scored_cases"] == 1
    assert result["case_count"] == 2


def test_report_path_rejects_symlink_leaf_file(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    reports_dir = hermes_home / "aion-evolution" / "evals" / "reports"
    reports_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("do not overwrite", encoding="utf-8")
    link = reports_dir / "linked-report.md"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run", "report_path": str(link)})
    )

    assert result["status"] == "error"
    assert "report_path contains symlink" in result["error"]
    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_report_path_replaces_hardlink_without_overwriting_outside_target(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    reports_dir = hermes_home / "aion-evolution" / "evals" / "reports"
    reports_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("do not overwrite", encoding="utf-8")
    link = reports_dir / "hardlinked-report.md"
    try:
        os.link(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hardlinks unavailable in test environment: {exc}")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run", "report_path": str(link)})
    )

    assert result["status"] == "ok"
    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    assert "Skill Eval Report" in link.read_text(encoding="utf-8")
    assert os.stat(outside).st_ino != os.stat(link).st_ino


def test_partial_manual_low_score_remains_incomplete_not_failed(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    second_case = dict(data["cases"][0])
    second_case["id"] = "CASE-2"
    data["cases"].append(second_case)
    eval_path.write_text(json.dumps(data), encoding="utf-8")
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(
        json.dumps({"scores": {"CASE-1": {"score": 0, "hard_fail": True}}}),
        encoding="utf-8",
    )

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "incomplete"
    assert result["scored"] is False
    assert result["hard_fail"] is True


def test_eval_path_rejects_parent_directory_traversal(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    outside = hermes_home / "aion-evolution" / "outside.json"
    _write_eval_pack(outside)
    traversal = hermes_home / "aion-evolution" / "evals" / ".." / "outside.json"

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(traversal), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert "parent-directory traversal" in result["error"]


def test_score_file_rejects_parent_directory_traversal(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    outside_score = hermes_home / "aion-evolution" / "scores.json"
    outside_score.write_text(json.dumps({"scores": {"CASE-1": {"score": 5}}}), encoding="utf-8")
    traversal = hermes_home / "aion-evolution" / "evals" / ".." / "scores.json"

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(traversal)})
    )

    assert result["status"] == "error"
    assert "parent-directory traversal" in result["error"]


def test_report_path_rejects_parent_directory_traversal(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    traversal = hermes_home / "aion-evolution" / "evals" / "reports" / ".." / "escaped.md"

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run", "report_path": str(traversal)})
    )

    assert result["status"] == "error"
    assert "parent-directory traversal" in result["error"]
    assert not (hermes_home / "aion-evolution" / "evals" / "escaped.md").exists()


def test_invalid_numeric_pack_fields_return_schema_errors(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "bad-numeric.json"
    _write_eval_pack(eval_path)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    data["cases"][0]["scoring_rubric"]["max_points"] = 0
    data["scoring"]["case_max_points"] = True
    data["scoring"]["pass_threshold_percent"] = None
    eval_path.write_text(json.dumps(data), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert any("max_points must be a positive integer" in err for err in result["errors"])
    assert any("case_max_points must be a positive integer" in err for err in result["errors"])
    assert any("pass_threshold_percent must be a number" in err for err in result["errors"])


def test_manual_score_rejects_non_integer_score(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(json.dumps({"scores": {"CASE-1": {"score": []}}}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "error"
    assert "Score for CASE-1 must be an integer" in result["error"]


def test_manual_score_rejects_boolean_score(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(json.dumps({"scores": {"CASE-1": True}}), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "error"
    assert "Invalid score value for CASE-1" in result["error"]


def test_manual_score_rejects_non_boolean_hard_fail(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "demo.json"
    _write_eval_pack(eval_path)
    score_path = hermes_home / "aion-evolution" / "evals" / "scores.json"
    score_path.write_text(
        json.dumps({"scores": {"CASE-1": {"score": 5, "hard_fail": "false"}}}),
        encoding="utf-8",
    )

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(
        skill_eval_run({"eval_path": str(eval_path), "mode": "manual_score", "score_file": str(score_path)})
    )

    assert result["status"] == "error"
    assert "hard_fail for CASE-1 must be a boolean" in result["error"]


def test_eval_pack_rejects_non_string_case_id(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    eval_path = hermes_home / "aion-evolution" / "evals" / "bad-case-id.json"
    _write_eval_pack(eval_path)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    data["cases"][0]["id"] = 1
    eval_path.write_text(json.dumps(data), encoding="utf-8")

    from tools.skill_eval_tool import skill_eval_run

    result = json.loads(skill_eval_run({"eval_path": str(eval_path), "mode": "dry_run"}))

    assert result["status"] == "error"
    assert any("id must be a non-empty string" in err for err in result["errors"])


def test_tool_is_registered_in_skills_toolset():
    import tools.skill_eval_tool  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("skill_eval_run")
    assert entry is not None
    assert entry.toolset == "skills"
