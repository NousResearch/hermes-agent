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


def _commit_files(repo: Path, paths: list[str]) -> None:
    for path in paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# test fixture\n", encoding="utf-8")
    _git(repo, "add", *paths)
    _git(repo, "commit", "-m", "add fixture files")


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
    assert schema["parameters"]["properties"]["mode"]["enum"] == [
        "execute",
        "execute_inferred",
        "dry_run_plan",
    ]
    assert "execute_inferred only" in schema["parameters"]["properties"]["mode"]["description"]
    assert schema["parameters"]["properties"]["dirty_baseline_policy"]["enum"] == ["require-clean"]


def test_empty_scope_is_rejected(tmp_path):
    repo = _clean_repo(tmp_path)

    result = _call(workdir=str(repo), allowed_files=[], allowed_globs=[])

    assert result["status"] == "rejected_scope"
    assert result["runner_exit_code"] is None


def test_execute_omitted_scope_does_not_infer_or_invoke_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"])
    calls = []

    def fake_infer(repo, task_text):
        raise AssertionError("execute mode must not infer omitted scope")

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_infer_scope_from_task", fake_infer)
    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), task="tighten terminal tool policy")

    assert result["status"] == "rejected_scope"
    assert result["error"] == "scope is required"
    assert result["runner_exit_code"] is None
    assert calls == []


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


def test_dirty_worktree_paths_are_bounded_without_losing_total_count(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    for index in range(105):
        (repo / f"dirty-{index:03d}.txt").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["dirty_check"]["dirty_count"] == 105
    assert len(result["dirty_check"]["dirty_paths"]) == 100
    assert result["dirty_check"]["dirty_paths_truncated"] is True
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dirty_worktree_paths_at_limit_are_not_marked_truncated(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    for index in range(100):
        (repo / f"dirty-{index:03d}.txt").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["dirty_check"]["dirty_count"] == 100
    assert len(result["dirty_check"]["dirty_paths"]) == 100
    assert result["dirty_check"]["dirty_paths_truncated"] is False
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_with_explicit_scope_is_read_only_and_does_not_invoke_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)
    before_status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update readme",
        allowed_files=["README.md"],
        verify_cmd_ids=["diff-check"],
    )
    after_status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "low"
    assert result["needs_user_confirmation"] is True
    assert result["resolved_workdir"] == str(repo.resolve())
    assert result["resolved_allowlist"]["files"] == ["README.md"]
    assert result["stage_plan_path"] is None
    assert result["raw_dir"] is None
    assert result["runner_exit_code"] is None
    assert result["next_required_action"] == "confirm_or_execute_with_explicit_scope"
    assert result["proposed_allowlist"] == {"files": ["README.md"], "globs": []}
    proposed = result["proposed_stage_plan"]
    assert proposed["repo"] == str(repo.resolve())
    assert proposed["continue_policy"] == "stop-on-review-needed"
    assert proposed["dirty_baseline_policy"] == "require-clean"
    assert proposed["slices"] == [
        {
            "id": "slice-1",
            "goal": "update readme",
            "prompt_file": "<create-outside-repo-before-execution>",
            "allowed_files": ["README.md"],
            "allowed_globs": [],
            "verify_cmd_ids": ["diff-check"],
            "dirty_baseline_policy": "require-clean",
        }
    ]
    assert calls == []
    assert after_status == before_status == ""


def test_dry_run_plan_uncertain_scope_returns_needs_scope_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="make the tool better")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_scope"
    assert result["needs_user_confirmation"] is True
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


@pytest.mark.parametrize(
    ("task", "expected_files"),
    [
        (
            "tighten terminal tool policy handling",
            ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"],
        ),
        (
            "add process registry Codex metadata coverage",
            ["tools/process_registry.py", "tests/tools/test_process_registry.py"],
        ),
        (
            "adjust stage runner timeout reporting",
            ["scripts/runtime/codex_stage_runner.py", "tests/scripts/test_codex_stage_runner.py"],
        ),
        (
            "harden impl guard final-output checks",
            ["scripts/runtime/codex_impl_guard.py", "tests/scripts/test_codex_impl_guard.py"],
        ),
        (
            "update review guard policy checks",
            ["scripts/runtime/codex_review_guard.py", "tests/scripts/test_codex_review_guard.py"],
        ),
        (
            "fix review packet summary metadata",
            ["scripts/runtime/codex_review_packet.py", "tests/scripts/test_codex_review_packet.py"],
        ),
    ],
)
def test_dry_run_plan_infers_known_template_scope_without_runner_call(
    tmp_path, monkeypatch, task, expected_files
):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, expected_files)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task=task)

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "low"
    assert result["needs_user_confirmation"] is True
    assert result["resolved_allowlist"] == {"files": expected_files, "globs": []}
    assert result["proposed_allowlist"] == {"files": expected_files, "globs": []}
    assert result["proposed_stage_plan"]["slices"][0]["allowed_files"] == expected_files
    assert result["proposed_stage_plan"]["slices"][0]["allowed_globs"] == []
    assert result["next_required_action"] == "confirm_inferred_scope_or_execute_with_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_inferred_missing_files_fail_closed_without_globs(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["scripts/runtime/codex_review_guard.py"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="update review guard policy checks")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported_template"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["resolved_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_ambiguous_omitted_scope_returns_needs_scope_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="make the runtime safer")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_scope"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_cross_component_omitted_scope_returns_needs_split(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(
        repo,
        [
            "tools/terminal_tool.py",
            "tests/tools/test_terminal_tool.py",
            "scripts/runtime/codex_impl_guard.py",
            "tests/scripts/test_codex_impl_guard.py",
        ],
    )
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="tighten terminal tool policy and harden impl guard",
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_split"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "split_task_or_provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_infers_single_existing_docs_path_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["docs/plans/example-plan.md"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update docs/plans/example-plan.md with Phase 15 notes",
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "low"
    assert result["proposed_allowlist"] == {"files": ["docs/plans/example-plan.md"], "globs": []}
    assert result["proposed_stage_plan"]["slices"][0]["allowed_globs"] == []
    assert result["next_required_action"] == "confirm_inferred_scope_or_execute_with_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_rejects_docs_path_traversal_before_inference(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update docs/../tools/terminal_tool.py with notes",
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported_template"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["resolved_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_multiple_docs_paths_need_split(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["docs/a.md", "docs/b.md"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update docs/a.md and docs/b.md",
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_split"
    assert result["proposed_stage_plan"] is None
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_inference_never_outputs_broad_globs(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="tighten terminal tool policy")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "low"
    assert result["resolved_allowlist"]["globs"] == []
    assert result["proposed_allowlist"]["globs"] == []
    assert result["proposed_stage_plan"]["slices"][0]["allowed_globs"] == []
    assert calls == []


def test_execute_inferred_known_template_invokes_runner_with_inferred_allowlist(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    expected_files = ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"]
    _commit_files(repo, expected_files)
    captured = {}

    def fake_runner(argv):
        captured["argv"] = argv
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "completed",
                    "verification_status": "passed",
                    "changed_files": expected_files,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="execute_inferred",
        workdir=str(repo),
        task="tighten terminal tool policy handling",
    )

    assert result["status"] == "ready_for_review"
    assert result["resolved_allowlist"] == {"files": expected_files, "globs": []}
    assert result["changed_files"] == expected_files
    assert result["runner_exit_code"] == 0
    assert captured["argv"][0].endswith("python") or "python" in captured["argv"][0]
    assert "--plan-file" in captured["argv"]
    plan_path = Path(captured["argv"][captured["argv"].index("--plan-file") + 1])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["slices"][0]["allowed_files"] == expected_files
    assert plan["slices"][0]["allowed_globs"] == []


@pytest.mark.parametrize(
    "scope",
    [
        {"allowed_files": ["README.md"]},
        {"allowed_globs": ["docs/*.md"]},
        {"allowed_files": []},
        {"allowed_globs": []},
    ],
)
def test_execute_inferred_rejects_explicit_scope_instead_of_bypassing_inference(
    tmp_path, monkeypatch, scope
):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["docs/example.md", "tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"])
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="execute_inferred",
        workdir=str(repo),
        task="tighten terminal tool policy handling",
        **scope,
    )

    assert result["status"] == "rejected_scope"
    assert result["error"] == "execute_inferred requires omitted scope for inference"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_inferred_explicit_scope_cannot_rescue_unsupported_task(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="execute_inferred",
        workdir=str(repo),
        task="make the runtime safer",
        allowed_files=["README.md"],
    )

    assert result["status"] == "rejected_scope"
    assert result["error"] == "execute_inferred requires omitted scope for inference"
    assert result["runner_exit_code"] is None
    assert calls == []


@pytest.mark.parametrize(
    ("task", "expected_error"),
    [
        ("make the runtime safer", "scope is required"),
        ("tighten terminal tool policy and harden impl guard", "needs_split"),
        ("update docs/../tools/terminal_tool.py with notes", "unsupported_template"),
    ],
)
def test_execute_inferred_uncertain_split_or_unsupported_scope_does_not_invoke_runner(
    tmp_path, monkeypatch, task, expected_error
):
    repo = _clean_repo(tmp_path)
    _commit_files(
        repo,
        [
            "tools/terminal_tool.py",
            "tests/tools/test_terminal_tool.py",
            "scripts/runtime/codex_impl_guard.py",
            "tests/scripts/test_codex_impl_guard.py",
        ],
    )
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="execute_inferred", workdir=str(repo), task=task)

    assert result["status"] == "rejected_scope"
    assert result["error"] == expected_error
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_inferred_dirty_inferred_scope_does_not_invoke_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"])
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="execute_inferred",
        workdir=str(repo),
        task="tighten terminal tool policy handling",
    )

    assert result["status"] == "dirty_worktree"
    assert result["resolved_allowlist"] == {
        "files": ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"],
        "globs": [],
    }
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_invalid_scope_returns_unsupported_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update python files",
        allowed_globs=["**/*.py"],
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported"
    assert result["needs_user_confirmation"] is True
    assert result["proposed_stage_plan"] is None
    assert result["next_required_action"] == "provide_narrow_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_dirty_worktree_is_not_classified_low(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update readme",
        allowed_files=["README.md"],
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported"
    assert result["dirty_check"]["is_clean"] is False
    assert result["proposed_stage_plan"] is None
    assert result["next_required_action"] == "clean_worktree_before_execution"
    assert result["runner_exit_code"] is None
    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"continue_policy": "keep-going", "allowed_files": ["README.md"]}, "unsupported_continue_policy"),
        ({"dirty_baseline_policy": "allow-listed-owned", "allowed_files": ["README.md"]}, "unsupported_dirty_policy"),
        ({"verify_cmd_ids": ["pytest"], "allowed_files": ["README.md"]}, "unsupported_verify_cmd_id"),
        ({"verify_cmd_ids": ["none"], "task": "delete files and restart deployment", "allowed_files": ["README.md"]}, "verify_none_high_risk_task"),
    ],
)
def test_dry_run_plan_policy_rejections_keep_dry_run_contract(tmp_path, monkeypatch, kwargs, reason):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), **kwargs)

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported"
    assert result["needs_user_confirmation"] is True
    assert result["resolved_workdir"] == str(repo.resolve())
    assert result["dirty_check"]["is_clean"] is True
    assert result["proposed_stage_plan"] is None
    assert result["next_required_action"] == "provide_supported_policy"
    assert result["reason"] == reason
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_mode_preserves_phase13_policy_rejection(tmp_path):
    repo = _clean_repo(tmp_path)

    result = _call(
        mode="execute",
        workdir=str(repo),
        allowed_files=["README.md"],
        continue_policy="keep-going",
    )

    assert result["status"] == "rejected_verify_policy"
    assert result["runner_exit_code"] is None


@pytest.mark.parametrize(
    ("verify_ids", "reason"),
    [
        ([[]], "unsupported_verify_cmd_id"),
        ([{}], "unsupported_verify_cmd_id"),
        (["none", "diff-check"], "rejected_verify_policy"),
    ],
)
def test_dry_run_plan_verify_id_type_errors_fail_closed(tmp_path, monkeypatch, verify_ids, reason):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    def fake_write_stage_files(**kwargs):
        raise AssertionError("dry_run_plan must not write stage files")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)
    monkeypatch.setattr(tool, "_write_stage_files", fake_write_stage_files)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="update readme",
        allowed_files=["README.md"],
        verify_cmd_ids=verify_ids,
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported"
    assert result["reason"] == reason
    assert result["proposed_stage_plan"] is None
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
