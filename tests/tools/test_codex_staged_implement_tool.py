import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import toolsets
from tools import codex_staged_implement_tool as tool
from tools.registry import registry


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _call(**kwargs):
    defaults = {
        "task": "make a small change",
        "continue_policy": "stop-on-review-needed",
        "dirty_baseline_policy": "require-clean",
    }
    defaults.update(kwargs)
    return json.loads(tool.codex_staged_implement(defaults))


def test_tool_schema_promotes_staged_implement_as_default_channel():
    schema = registry.get_schema("codex_staged_implement")

    assert schema is not None
    description = schema["description"]
    assert "Default Hermes channel" in description
    assert "codex-yuna exec" in description
    assert "allowed_files" in description
    assert "allowed_globs" in description
    assert "dirty worktrees" in description
    assert "candidate work only" in description
    assert "not that the task is complete" in description
    assert "verify the diff" in description
    assert schema["parameters"]["properties"]["dirty_baseline_policy"]["enum"] == ["require-clean"]


def test_empty_scope_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)

    result = _call(workdir=str(repo), allowed_files=[], allowed_globs=[])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


@pytest.mark.parametrize(
    "scope",
    [
        {"allowed_files": ["/tmp/outside.txt"]},
        {"allowed_files": ["../outside.txt"]},
        {"allowed_globs": ["*"]},
        {"allowed_globs": ["**/*.py"]},
        {"allowed_globs": ["**"]},
        {"allowed_files": [".git/config"]},
        {"allowed_files": [".venv/bin/python"]},
        {"allowed_files": ["node_modules/pkg/index.js"]},
    ],
)
def test_invalid_scope_paths_are_rejected(tmp_path, scope):
    repo = _clean_repo(tmp_path)

    result = _call(workdir=str(repo), **scope)

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_symlink_escape_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-ish\n", encoding="utf-8")
    (repo / "link.txt").symlink_to(outside)

    result = _call(workdir=str(repo), allowed_files=["link.txt"])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_dangling_symlink_escape_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)
    outside = tmp_path / "missing-outside.txt"
    (repo / "dangling.txt").symlink_to(outside)
    _git(repo, "add", "dangling.txt")
    _git(repo, "commit", "-m", "add dangling symlink")

    result = _call(workdir=str(repo), allowed_files=["dangling.txt"])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_symlink_directory_prefix_for_new_file_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (repo / "linkdir").symlink_to(outside_dir, target_is_directory=True)
    _git(repo, "add", "linkdir")
    _git(repo, "commit", "-m", "add symlink dir")

    result = _call(workdir=str(repo), allowed_files=["linkdir/new_file.py"])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_symlink_directory_prefix_for_glob_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (repo / "linkdir").symlink_to(outside_dir, target_is_directory=True)
    _git(repo, "add", "linkdir")
    _git(repo, "commit", "-m", "add symlink dir")

    result = _call(workdir=str(repo), allowed_globs=["linkdir/*.py"])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_dirty_worktree_rejects_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["dirty_check"]["is_clean"] is False
    assert result["dirty_check"]["dirty_count"] == 1
    assert result["dirty_check"]["dirty_paths"] == ["dirty.txt"]
    assert result["dirty_baseline_policy"] == "require-clean"
    assert result["runner_exit_code"] is None
    assert calls == []


@pytest.mark.parametrize("dirty_policy", ["allow-listed-owned", "fail-on-overlap"])
def test_non_require_clean_dirty_policy_is_rejected_without_runner_call(tmp_path, monkeypatch, dirty_policy):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("dirty owned\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        workdir=str(repo),
        allowed_files=["README.md"],
        dirty_baseline_policy=dirty_policy,
    )

    assert result["status"] == "unsupported_dirty_policy"
    assert result["dirty_baseline_policy"] == dirty_policy
    assert result["runner_exit_code"] is None
    assert calls == []


def test_runner_argv_and_plan_generation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    captured = {}

    def fake_runner(argv):
        captured["argv"] = argv
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "changed_files": ["README.md"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        workdir=str(repo),
        task="update readme",
        allowed_files=["README.md"],
        verify_cmd_ids=["diff-check"],
    )

    assert result["status"] == "ready_for_review"
    assert result["next_required_action"] == "run_hermes_verification"
    assert captured["argv"][0].endswith("python") or "python" in captured["argv"][0]
    assert captured["argv"][1].endswith("scripts/runtime/codex_stage_runner.py")
    assert "--plan-file" in captured["argv"]
    assert "--raw-dir" in captured["argv"]
    assert "--workdir" not in captured["argv"]
    assert "--plan" not in captured["argv"]
    assert "--prompt" not in captured["argv"]
    assert "codex-yuna exec" not in " ".join(captured["argv"])
    assert result["resolved_workdir"] == str(repo.resolve())
    assert result["resolved_allowlist"]["files"] == ["README.md"]
    assert result["changed_files"] == ["README.md"]
    assert result["candidate_id"]
    assert result["candidate_disposition"] == "pending_review"
    assert result["completion_trusted"] is True
    assert result["final_present"] is False
    assert result["limit_exceeded"] is False
    assert result["out_of_scope_files"] == []

    plan_path = Path(result["stage_plan_path"])
    raw_dir = Path(result["raw_dir"])
    assert not plan_path.resolve().is_relative_to(repo.resolve())
    assert not raw_dir.resolve().is_relative_to(repo.resolve())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["repo"] == str(repo.resolve())
    assert plan["continue_policy"] == "stop-on-review-needed"
    assert plan["dirty_baseline_policy"] == "require-clean"
    assert len(plan["slices"]) == 1
    stage_slice = plan["slices"][0]
    assert Path(stage_slice["prompt_file"]).read_text(encoding="utf-8") == "update readme"
    assert stage_slice["allowed_files"] == ["README.md"]
    assert stage_slice["verify_cmd_ids"] == ["diff-check"]


def test_runner_review_needed_mapping(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "stopped",
                    "reason": "slice_review_needed",
                    "changed_files": ["README.md"],
                    "stopped_slice": "slice-1",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "review_needed"
    assert result["next_required_action"] == "run_read_only_review"
    assert result["stopped_slice"] == "slice-1"


def test_stopped_blocked_by_allowlist_mapping(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=1,
            stdout=json.dumps(
                {
                    "status": "stopped",
                    "reason": "slice_blocked_by_allowlist",
                    "slice_results": [
                        {
                            "impl_guard_result": {
                                "changed_files": ["README.md"],
                                "status": "blocked_by_allowlist",
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "blocked_by_allowlist"
    assert result["changed_files"] == ["README.md"]
    assert result["candidate_disposition"] == "rejected"


def test_candidate_lifecycle_fields_from_impl_guard_payload(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "changed_files": ["README.md"],
                    "slice_results": [
                        {
                            "impl_guard_result": {
                                "status": "completed",
                                "final_file": "/tmp/final.txt",
                                "trusted_completion": True,
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])
    repeat = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "ready_for_review"
    assert result["candidate_id"] == repeat["candidate_id"]
    assert result["completion_trusted"] is True
    assert result["final_present"] is True
    assert result["limit_exceeded"] is False
    assert result["out_of_scope_files"] == []
    assert result["candidate_disposition"] == "pending_review"


def test_limit_or_scope_issues_require_takeover_and_do_not_trust_completion(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "slice_results": [
                        {
                            "impl_guard_result": {
                                "status": "completed",
                                "limit_reason": "stdout flood",
                                "out_of_scope_files": ["other.py"],
                                "allowlist_violations": ["README.md"],
                                "final_missing": True,
                                "trusted_completion": False,
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "ready_for_review"
    assert result["completion_trusted"] is False
    assert result["final_present"] is False
    assert result["limit_exceeded"] is True
    assert result["out_of_scope_files"] == ["README.md", "other.py"]
    assert result["candidate_disposition"] == "takeover_required"


def test_scope_issue_without_limit_requires_takeover_and_untrusted_completion(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "slice_results": [
                        {
                            "impl_guard_result": {
                                "status": "completed",
                                "allowlist_violations": ["other.py"],
                                "trusted_completion": True,
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "ready_for_review"
    assert result["completion_trusted"] is False
    assert result["limit_exceeded"] is False
    assert result["out_of_scope_files"] == ["other.py"]
    assert result["candidate_disposition"] == "takeover_required"


def test_final_issue_requires_takeover_and_untrusted_completion(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "slice_results": [
                        {
                            "impl_guard_result": {
                                "status": "completed",
                                "final_missing": True,
                                "trusted_completion": True,
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "ready_for_review"
    assert result["completion_trusted"] is False
    assert result["final_present"] is False
    assert result["candidate_disposition"] == "takeover_required"


def test_large_runner_output_is_bounded(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(returncode=1, stdout="x" * 20000, stderr="y" * 20000)

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "malformed"
    assert result["candidate_id"]
    assert result["candidate_disposition"] == "takeover_required"
    assert result["completion_trusted"] is False
    assert result["limit_exceeded"] is True
    assert len(result["runner_stdout_preview"]) <= 8000
    assert len(result["runner_stderr_preview"]) <= 8000
    assert "x" * 9000 not in json.dumps(result)


def test_runner_unusable_keeps_bounded_preview(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(returncode=1, stdout=json.dumps({"status": "unusable"}), stderr="boom" * 3000)

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "runner_unusable"
    assert 0 < len(result["runner_stderr_preview"]) <= 8000


def test_nonzero_completed_runner_is_not_ready_for_review(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=7,
            stdout=json.dumps({"status": "completed", "verification_status": "passed"}),
            stderr="unexpected failure",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "runner_unusable"
    assert result["runner_exit_code"] == 7
    assert result["runner_stderr_preview"] == "unexpected failure"


def test_stage_temp_files_ignore_repo_tmpdir(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    repo_tmp = repo / "tmp"
    repo_tmp.mkdir()
    captured = {}

    def fake_runner(argv):
        captured["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "completed"}), stderr="")

    monkeypatch.setenv("TMPDIR", str(repo_tmp))
    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "ready_for_review"
    assert not Path(result["stage_plan_path"]).resolve().is_relative_to(repo.resolve())
    assert not Path(result["raw_dir"]).resolve().is_relative_to(repo.resolve())


def test_stage_file_creation_error_returns_bounded_json(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def boom(**kwargs):
        raise RuntimeError("unsafe_stage_temp_dir")

    monkeypatch.setattr(tool, "_write_stage_files", boom)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "runner_unusable"
    assert result["runner_exit_code"] is None
    assert result["next_required_action"] is None


def test_verify_none_defers_to_hermes_without_completion_claim(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "completed", "verification_status": "passed"}),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        workdir=str(repo),
        allowed_files=["README.md"],
        verify_cmd_ids=["none"],
    )

    assert result["status"] == "review_needed"
    assert result["verification_policy"] == "deferred_to_hermes"
    assert result["next_required_action"] == "run_read_only_review"


def test_verify_none_rejects_high_risk_task(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv):
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        workdir=str(repo),
        task="delete files and restart deployment",
        allowed_files=["README.md"],
        verify_cmd_ids=["none"],
    )

    assert result["status"] == "rejected_verify_policy"


def test_policy_rejections(tmp_path):
    repo = _clean_repo(tmp_path)

    assert _call(
        workdir=str(repo),
        allowed_files=["README.md"],
        continue_policy="keep-going",
    )["status"] == "rejected_verify_policy"
    assert _call(
        workdir=str(repo),
        allowed_files=["README.md"],
        dirty_baseline_policy="allow-dirty",
    )["status"] == "unsupported_dirty_policy"
    assert _call(
        workdir=str(repo),
        allowed_files=["README.md"],
        verify_cmd_ids=["pytest"],
    )["status"] == "unsupported_verify_cmd_id"


def test_tool_registration_and_core_exposure():
    entry = registry.get_entry("codex_staged_implement")

    assert entry is not None
    assert entry.toolset == "codex_staged_implement"
    assert "codex_staged_implement" in toolsets._HERMES_CORE_TOOLS
    assert "codex_staged_implement" in toolsets.TOOLSETS["hermes-acp"]["tools"]
    assert "codex_staged_implement" in toolsets.TOOLSETS["hermes-api-server"]["tools"]
