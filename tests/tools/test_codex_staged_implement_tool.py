import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import toolsets
from tools import codex_staged_implement_tool as tool
from tools import codex_workflow_run_tool as workflow
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


@pytest.fixture(autouse=True)
def _default_impl_preflight_ready(monkeypatch, request):
    if request.node.name.startswith("test_impl_preflight"):
        return
    if request.node.name.startswith("test_execute_blocks_before_stage_files_when_impl_preflight"):
        return
    monkeypatch.setenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", "1")
    monkeypatch.setattr(
        tool.shutil,
        "which",
        lambda name, path=None: f"/tmp/{name}" if name in {"codex-yuna", "node"} else None,
    )


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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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
    assert result["dirty_state_id"] == result["dirty_check"]["dirty_state_id"]
    assert result["auto_resolvable_classes"] == []
    assert result["requires_user_decision"] is True
    assert result["resume_strategy"] == "isolated_worktree_recommended"
    assert calls == []


def test_dirty_state_id_is_stable_and_changes_with_bounded_dirty_evidence(tmp_path):
    repo = _clean_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    first = tool._dirty_check(repo)
    repeat = tool._dirty_check(repo)
    (repo / "second.txt").write_text("dirty\n", encoding="utf-8")
    changed = tool._dirty_check(repo)

    assert first["dirty_state_id"] == repeat["dirty_state_id"]
    assert first["dirty_state_id"] != changed["dirty_state_id"]
    assert len(first["dirty_state_id"]) == 24


def test_cache_only_dirty_metadata_is_auto_resolvable_without_mutation_or_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["dirty_path_classes"] == {
        "source": [],
        "test": [],
        "docs": [],
        "cache": [".pytest_cache/v/cache/nodeids"],
        "unknown": [],
    }
    assert result["auto_resolvable_classes"] == ["cache"]
    assert result["requires_user_decision"] is False
    assert result["unsafe_reasons"] == []
    assert result["resume_strategy"] == "clean_worktree_required"
    assert result["runner_exit_code"] is None
    assert cache_path.exists()
    assert calls == []


def test_source_and_unknown_dirty_metadata_requires_decision_or_isolation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "tools").mkdir()
    (repo / "tools" / "dirty_tool.py").write_text("dirty\n", encoding="utf-8")
    (repo / "scratch.tmp").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["auto_resolvable_classes"] == []
    assert result["requires_user_decision"] is True
    assert result["resume_strategy"] == "isolated_worktree_recommended"
    assert "source_dirty" in result["unsafe_reasons"]
    assert "unknown_dirty" in result["unsafe_reasons"]
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dirty_unsafe_reasons_include_conservative_porcelain_and_path_signals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    large_path = repo / "artifacts" / "large.bin"
    large_path.parent.mkdir(parents=True)
    large_path.write_bytes(b"\0" * (tool._LARGE_DIRTY_FILE_BYTES + 1))
    lines = [
        "UU conflicted.tmp",
        "R  old.txt -> renamed.txt",
        " D deleted.txt",
        "T  mode-ish.sh",
        " M .gitmodules",
        "?? secrets/API_TOKEN.txt",
        "?? data/customer.csv",
        "?? image.png",
        "?? artifacts/large.bin",
    ]
    classes = tool._dirty_path_classes(lines)

    reasons = tool._dirty_unsafe_reasons(repo, lines, classes)

    assert "conflict_status" in reasons
    assert "rename_status" in reasons
    assert "delete_status" in reasons
    assert "typechange_or_chmod_status" in reasons
    assert "submodule_status_or_metadata" in reasons
    assert "secret_path_evidence" in reasons
    assert "real_data_path_evidence" in reasons
    assert "binary_path_evidence" in reasons
    assert "large_file_evidence" in reasons


def test_dirty_worktree_returns_actionable_fail_closed_resolver(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_dirty.py").write_text("dirty\n", encoding="utf-8")
    _git(repo, "add", "tests/test_dirty.py")
    (repo / "scratch.tmp").write_text("untracked\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "dirty_worktree"
    assert result["reason"] == "dirty_worktree"
    assert result["recommended_next_action"] == "clean_worktree_before_execution"
    assert result["next_required_action"] == "clean_worktree_before_execution"
    assert result["authorization_required"] is True
    assert {option["id"] for option in result["dirty_resolution_options"]} == {
        "commit_or_checkpoint_current_changes",
        "create_isolated_worktree",
        "manually_clean_worktree",
    }
    assert all(option["authorization_required"] is True for option in result["dirty_resolution_options"])
    assert all("command" not in option and "argv" not in option for option in result["dirty_resolution_options"])
    assert result["dirty_path_classes"] == {
        "source": [],
        "test": ["tests/test_dirty.py"],
        "docs": ["README.md"],
        "cache": [],
        "unknown": ["scratch.tmp"],
    }
    assert result["diff_stat"]["truncated"] is False
    assert "README.md" in "\n".join(result["diff_stat"]["unstaged"])
    assert "tests/test_dirty.py" in "\n".join(result["diff_stat"]["staged"])
    assert "diff --git" not in json.dumps(result["diff_stat"])
    assert "@@" not in json.dumps(result["diff_stat"])
    assert calls == []


def test_dirty_path_classes_are_coarse_path_classes_not_ownership_guesses():
    assert tool._dirty_path_classes(
        [
            " M tools/codex_staged_implement_tool.py",
            "A  tests/tools/test_codex_staged_implement_tool.py",
            "?? docs/plans/codex-impl-guard-safe-plan.zh-CN.md",
            " D .pytest_cache/v/cache/nodeids",
            "R  old.txt -> renamed.unknown",
            "UU conflicted.tmp",
        ]
    ) == {
        "source": ["tools/codex_staged_implement_tool.py"],
        "test": ["tests/tools/test_codex_staged_implement_tool.py"],
        "docs": ["docs/plans/codex-impl-guard-safe-plan.zh-CN.md"],
        "cache": [".pytest_cache/v/cache/nodeids"],
        "unknown": ["renamed.unknown", "conflicted.tmp"],
    }


def test_dirty_diff_stat_is_bounded(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []
    long_path = "very-long-path-" + "x" * 260 + ".txt"
    many_lines = "\n".join(f" {long_path}-{index:03d} | 1 +" for index in range(45))

    def fake_git(repo_arg, *args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=many_lines, stderr="")

    monkeypatch.setattr(tool, "_git", fake_git)

    result = tool._dirty_diff_stat(repo)

    assert calls == [("diff", "--stat", "--no-ext-diff"), ("diff", "--stat", "--no-ext-diff", "--cached")]
    assert 0 < len(result["unstaged"]) <= 40
    assert 0 < len(result["staged"]) <= 40
    assert all(len(line) <= 180 for line in result["unstaged"] + result["staged"])
    assert sum(len(line) + 1 for line in result["unstaged"]) <= 4000
    assert sum(len(line) + 1 for line in result["staged"]) <= 4000
    assert result["max_lines_per_section"] == 40
    assert result["max_line_chars"] == 180
    assert result["max_total_chars"] == 4000
    assert result["truncated"] is True
    assert "diff --git" not in json.dumps(result)
    assert "@@" not in json.dumps(result)


def test_dirty_diff_stat_failure_is_non_blocking(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_git(repo_arg, *args):
        calls.append(args)
        return SimpleNamespace(returncode=2, stdout="", stderr="fatal: bad revision")

    monkeypatch.setattr(tool, "_git", fake_git)

    result = tool._dirty_diff_stat(repo)

    assert result["unstaged"] == []
    assert result["staged"] == []
    assert result["truncated"] is False
    assert result["error"] == "diff_stat_unavailable"
    assert calls == [("diff", "--stat", "--no-ext-diff"), ("diff", "--stat", "--no-ext-diff", "--cached")]


def test_dirty_worktree_paths_are_bounded_without_losing_total_count(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    for index in range(105):
        (repo / f"dirty-{index:03d}.txt").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "explicit"
    assert "inferred_template" not in result
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

    def fake_runner(argv, *, env=None):
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
    ("task", "expected_files", "expected_template"),
    [
        (
            "tighten terminal tool policy handling",
            ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"],
            "terminal",
        ),
        (
            "add process registry Codex metadata coverage",
            ["tools/process_registry.py", "tests/tools/test_process_registry.py"],
            "process_registry",
        ),
        (
            "adjust stage runner timeout reporting",
            ["scripts/runtime/codex_stage_runner.py", "tests/scripts/test_codex_stage_runner.py"],
            "stage_runner",
        ),
        (
            "harden impl guard final-output checks",
            ["scripts/runtime/codex_impl_guard.py", "tests/scripts/test_codex_impl_guard.py"],
            "impl_guard",
        ),
        (
            "update review guard policy checks",
            ["scripts/runtime/codex_review_guard.py", "tests/scripts/test_codex_review_guard.py"],
            "review_guard",
        ),
        (
            "fix review packet summary metadata",
            ["scripts/runtime/codex_review_packet.py", "tests/scripts/test_codex_review_packet.py"],
            "review_packet",
        ),
    ],
)
def test_dry_run_plan_infers_known_template_scope_without_runner_call(
    tmp_path, monkeypatch, task, expected_files, expected_template
):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, expected_files)
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task=task)

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "low"
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == expected_template
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

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="update review guard policy checks")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "unsupported_template"
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == "review_guard"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["resolved_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_ambiguous_omitted_scope_returns_needs_scope_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="dry_run_plan", workdir=str(repo), task="make the runtime safer")

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_scope"
    assert result["scope_source"] == "inferred"
    assert "inferred_template" not in result
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

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "inferred"
    assert "inferred_template" not in result
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["next_required_action"] == "split_task_or_provide_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_template_plus_docs_path_need_split_keeps_template_audit(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py", "docs/a.md"])
    calls = []

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("dry_run_plan must not invoke runner")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(
        mode="dry_run_plan",
        workdir=str(repo),
        task="tighten terminal tool policy and update docs/a.md",
    )

    assert result["status"] == "dry_run_plan"
    assert result["risk_classification"] == "needs_split"
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == "terminal"
    assert result["proposed_stage_plan"] is None
    assert result["proposed_allowlist"] == {"files": [], "globs": []}
    assert result["runner_exit_code"] is None
    assert calls == []


def test_dry_run_plan_infers_single_existing_docs_path_without_runner_call(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["docs/plans/example-plan.md"])
    calls = []

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == "docs_path"
    assert result["proposed_allowlist"] == {"files": ["docs/plans/example-plan.md"], "globs": []}
    assert result["proposed_stage_plan"]["slices"][0]["allowed_globs"] == []
    assert result["next_required_action"] == "confirm_inferred_scope_or_execute_with_explicit_scope"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_inferred_docs_path_invokes_runner_with_docs_template_metadata(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    expected_files = ["docs/plans/example-plan.md"]
    _commit_files(repo, expected_files)
    captured = {}

    def fake_runner(argv, *, env=None):
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
        task="update docs/plans/example-plan.md with Phase 17 notes",
    )

    assert result["status"] == "ready_for_review"
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == "docs_path"
    assert result["resolved_allowlist"] == {"files": expected_files, "globs": []}
    assert result["changed_files"] == expected_files
    plan_path = Path(captured["argv"][captured["argv"].index("--plan-file") + 1])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["slices"][0]["allowed_files"] == expected_files
    assert plan["slices"][0]["allowed_globs"] == []


def test_dry_run_plan_rejects_docs_path_traversal_before_inference(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py"])
    calls = []

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "inferred"
    assert result["inferred_template"] == "terminal"
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

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "explicit"
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_inferred_explicit_scope_cannot_rescue_unsupported_task(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_runner(argv, *, env=None):
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
    ("task", "expected_error", "expected_template"),
    [
        ("make the runtime safer", "scope is required", None),
        ("tighten terminal tool policy and harden impl guard", "needs_split", None),
        ("update docs/../tools/terminal_tool.py with notes", "unsupported_template", None),
        ("update review guard policy checks", "unsupported_template", "review_guard"),
    ],
)
def test_execute_inferred_uncertain_split_or_unsupported_scope_does_not_invoke_runner(
    tmp_path, monkeypatch, task, expected_error, expected_template
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

    def fake_runner(argv, *, env=None):
        calls.append(argv)
        raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(mode="execute_inferred", workdir=str(repo), task=task)

    assert result["status"] == "rejected_scope"
    assert result["error"] == expected_error
    assert result["scope_source"] == "inferred"
    if expected_template is None:
        assert "inferred_template" not in result
    else:
        assert result["inferred_template"] == expected_template
    assert result["runner_exit_code"] is None
    assert calls == []


def test_execute_inferred_dirty_inferred_scope_does_not_invoke_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    _commit_files(repo, ["tools/terminal_tool.py", "tests/tools/test_terminal_tool.py"])
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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


def test_execute_blocks_before_stage_files_when_impl_preflight_missing_sandbox_env(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    monkeypatch.delenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", raising=False)
    monkeypatch.setattr(
        tool.shutil,
        "which",
        lambda name, path=None: f"/tmp/{name}" if name in {"codex-yuna", "node"} else None,
    )
    monkeypatch.setattr(
        tool,
        "_write_stage_files",
        lambda **kwargs: calls.append("write")
        or (_ for _ in ()).throw(AssertionError("must not write stage files")),
    )
    monkeypatch.setattr(
        tool,
        "_run_runner",
        lambda *args, **kwargs: calls.append("runner") or (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = _call(workdir=str(repo), allowed_files=["README.md"], allowed_globs=[], verify_cmd_ids=["diff-check"])
    encoded = json.dumps(result)

    assert result["status"] == "runner_unusable"
    assert result["reason"] == "preflight_blocked"
    assert result["implementation_preflight"]["status"] == "blocked"
    assert result["implementation_preflight"]["blockers"] == ["sandbox_not_verified"]
    assert calls == []
    assert all(isinstance(value, bool) for value in result["implementation_preflight"]["checks"].values())
    assert "PATH" not in encoded
    assert "HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED" not in encoded


@pytest.mark.parametrize(
    ("missing_name", "expected_blocker"),
    [("codex-yuna", "missing_codex_bin"), ("node", "missing_node_bin")],
)
def test_execute_blocks_before_stage_files_when_impl_preflight_missing_codex_or_node(
    tmp_path, monkeypatch, missing_name, expected_blocker
):
    repo = _clean_repo(tmp_path)
    calls = []

    monkeypatch.setenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", "1")
    monkeypatch.setattr(
        tool.shutil,
        "which",
        lambda name, path=None: None if name == missing_name else f"/tmp/{name}",
    )
    monkeypatch.setattr(
        tool,
        "_write_stage_files",
        lambda **kwargs: calls.append("write")
        or (_ for _ in ()).throw(AssertionError("must not write stage files")),
    )
    monkeypatch.setattr(
        tool,
        "_run_runner",
        lambda *args, **kwargs: calls.append("runner") or (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = _call(workdir=str(repo), allowed_files=["README.md"], allowed_globs=[], verify_cmd_ids=["diff-check"])

    assert result["status"] == "runner_unusable"
    assert result["reason"] == "preflight_blocked"
    assert result["implementation_preflight"]["blockers"] == [expected_blocker]
    assert calls == []


def test_impl_preflight_uses_same_env_snapshot_for_checks_and_runner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    env_snapshot = {
        "PATH": "/tmp/snapshot-path",
        "HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED": "1",
    }
    observed_paths = []
    captured = {}

    monkeypatch.setattr(tool, "_implementation_env", lambda: env_snapshot)

    def fake_which(name, path=None):
        observed_paths.append(path)
        return f"/tmp/{name}" if name in {"codex-yuna", "node"} else None

    def fake_runner(argv, *, env=None):
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "completed", "verification_status": "passed"}), stderr="")

    monkeypatch.setattr(tool.shutil, "which", fake_which)
    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"], allowed_globs=[], verify_cmd_ids=["diff-check"])

    assert result["status"] == "ready_for_review"
    assert observed_paths == [env_snapshot["PATH"], env_snapshot["PATH"]]
    assert captured["env"] is env_snapshot


@pytest.mark.parametrize(
    ("missing_name", "env_value", "expected_blocker"),
    [
        (None, None, "sandbox_not_verified"),
        ("codex-yuna", "1", "missing_codex_bin"),
        ("node", "1", "missing_node_bin"),
    ],
)
def test_impl_preflight_blockers_match_workflow_preflight_short_codes(
    tmp_path, monkeypatch, missing_name, env_value, expected_blocker
):
    repo = _clean_repo(tmp_path)
    calls = []
    if env_value is None:
        monkeypatch.delenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", raising=False)
    else:
        monkeypatch.setenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", env_value)

    def fake_which(name, path=None):
        return None if name == missing_name else f"/tmp/{name}"

    monkeypatch.setattr(tool.shutil, "which", fake_which)
    monkeypatch.setattr(workflow.shutil, "which", fake_which)
    monkeypatch.setattr(
        tool,
        "_write_stage_files",
        lambda **kwargs: calls.append("write")
        or (_ for _ in ()).throw(AssertionError("must not write stage files")),
    )
    monkeypatch.setattr(
        tool,
        "_run_runner",
        lambda *args, **kwargs: calls.append("runner") or (_ for _ in ()).throw(AssertionError("must not run")),
    )

    direct = _call(workdir=str(repo), allowed_files=["README.md"], allowed_globs=[], verify_cmd_ids=["diff-check"])
    workflow_preflight = workflow._codex_preflight(repo)
    shared_blockers = {"missing_codex_bin", "missing_node_bin", "sandbox_not_verified"}
    direct_shared = set(direct["implementation_preflight"]["blockers"]) & shared_blockers
    workflow_shared = set(workflow_preflight["blockers"]) & shared_blockers

    assert direct["status"] == "runner_unusable"
    assert direct["reason"] == "preflight_blocked"
    assert direct["implementation_preflight"]["blockers"] == [expected_blocker]
    assert direct_shared == workflow_shared == {expected_blocker}
    assert all(isinstance(value, bool) for value in direct["implementation_preflight"]["checks"].values())
    assert all(isinstance(value, bool) for value in workflow_preflight["checks"].values())
    assert calls == []


def test_impl_preflight_reports_multiple_blockers_without_writing_or_leaking_paths(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    monkeypatch.delenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", raising=False)
    monkeypatch.setattr(tool.shutil, "which", lambda name, path=None: None)
    monkeypatch.setattr(
        tool,
        "_write_stage_files",
        lambda **kwargs: calls.append("write")
        or (_ for _ in ()).throw(AssertionError("must not write stage files")),
    )
    monkeypatch.setattr(
        tool,
        "_run_runner",
        lambda *args, **kwargs: calls.append("runner") or (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = _call(workdir=str(repo), allowed_files=["README.md"], allowed_globs=[], verify_cmd_ids=["diff-check"])
    encoded = json.dumps(result)

    assert result["status"] == "runner_unusable"
    assert result["reason"] == "preflight_blocked"
    assert result["implementation_preflight"]["blockers"] == [
        "missing_codex_bin",
        "missing_node_bin",
        "sandbox_not_verified",
    ]
    assert calls == []
    assert "PATH" not in encoded
    assert "HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED" not in encoded
    assert "/tmp/codex-yuna" not in encoded
    assert "/tmp/node" not in encoded


def test_runner_argv_and_plan_generation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    captured = {}

    def fake_runner(argv, *, env=None):
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
    assert result["scope_source"] == "explicit"
    assert "inferred_template" not in result
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
        return SimpleNamespace(returncode=1, stdout=json.dumps({"status": "unusable"}), stderr="boom" * 3000)

    monkeypatch.setattr(tool, "_run_runner", fake_runner)

    result = _call(workdir=str(repo), allowed_files=["README.md"])

    assert result["status"] == "runner_unusable"
    assert 0 < len(result["runner_stderr_preview"]) <= 8000


def test_nonzero_completed_runner_is_not_ready_for_review(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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

    def fake_runner(argv, *, env=None):
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
    assert "codex_staged_implement" in toolsets.TOOLSETS
    assert "codex_staged_implement" in toolsets.resolve_toolset("codex_staged_implement")
    assert "codex_staged_implement" in toolsets.TOOLSETS["hermes-acp"]["tools"]
    assert "codex_staged_implement" in toolsets.TOOLSETS["hermes-api-server"]["tools"]


def test_codex_staged_implement_is_enabled_for_default_gateway_tool_resolution():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_platform_tools

    configurable_names = {name for name, _label, _description in CONFIGURABLE_TOOLSETS}
    assert "codex_staged_implement" in configurable_names

    enabled = _get_platform_tools({}, "qqbot")

    assert "codex_staged_implement" in enabled


def test_codex_staged_implement_schema_is_exposed_for_default_gateway_tools():
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions

    enabled = _get_platform_tools({}, "qqbot")
    schemas = get_tool_definitions(enabled_toolsets=sorted(enabled), quiet_mode=True)

    assert "codex_staged_implement" in {schema["function"]["name"] for schema in schemas}
