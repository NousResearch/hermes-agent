"""Tests for the Codex Goal Lane helper."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "skills"
    / "autonomous-ai-agents"
    / "kanban-codex-lane"
    / "scripts"
    / "codex_goal_lane.py"
)


def _load_lane_module():
    spec = importlib.util.spec_from_file_location("codex_goal_lane", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], text=True, check=True, capture_output=True)
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    commit = _run_git(repo, "commit", "-m", "initial")
    assert commit.returncode == 0, commit.stderr
    return repo


def _json_from_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def _run_simulated_lane(lane, repo: Path, state_root: Path, capsys):
    code = lane.main(
        [
            "run",
            "--repo",
            str(repo),
            "--task-id",
            "T-123",
            "--goal",
            "Create a simulated candidate diff.",
            "--state-root",
            str(state_root),
            "--mode",
            "goal",
            "--autonomy",
            "yolo",
            "--simulate",
        ]
    )
    assert code == 0
    payload = _json_from_stdout(capsys)
    return payload["run_id"], payload["state"]


def test_preflight_reports_missing_builder_and_reviewer_tools(monkeypatch, tmp_path):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    real_which = lane.shutil.which

    def fake_which(tool: str):
        if tool in {"codex", "claude"}:
            return None
        return real_which(tool)

    monkeypatch.setattr(lane.shutil, "which", fake_which)

    result = lane.preflight_checks(
        repo,
        require_codex=True,
        require_claude=True,
        simulate=False,
    )

    failed_names = {item["name"] for item in result["failed"]}
    assert result["success"] is False
    assert {"codex_cli", "claude_cli"} <= failed_names


def test_simulated_run_creates_isolated_worktree_and_state(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"

    run_id, state = _run_simulated_lane(lane, repo, state_root, capsys)

    worktree = Path(state["worktree"])
    assert run_id.startswith("T-123-")
    assert state["status"] == "codex_completed"
    assert state["autonomy"] == "yolo"
    assert state["mode"] == "goal"
    assert worktree.exists()
    assert (worktree / "codex_lane_simulated_output.txt").exists()
    assert (state_root / run_id / "state.json").exists()
    assert state["codex_lane"]["used"] is True
    assert state["codex_lane"]["run_id"] == run_id
    assert state["codex_lane"]["reviewer"] == "claude-code"
    assert state["codex_lane"]["hermes_verification"] == "required"
    assert state["codex_lane"]["worktree"] == str(worktree)

    code = lane.main(["status", "--run-id", run_id, "--state-root", str(state_root)])
    assert code == 0
    status = _json_from_stdout(capsys)["state"]
    assert status["status"] == "codex_completed"

    code = lane.main(
        ["logs", "--run-id", run_id, "--state-root", str(state_root), "--phase", "builder"]
    )
    assert code == 0
    logs = _json_from_stdout(capsys)["logs"]
    assert "SIMULATED Codex builder completed" in logs["builder"]


def test_run_rejects_existing_repo_root_as_worktree(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)

    code = lane.main(
        [
            "run",
            "--repo",
            str(repo),
            "--task-id",
            "T-bad-worktree",
            "--goal",
            "Do not run in the shared checkout.",
            "--worktree",
            str(repo),
            "--simulate",
        ]
    )

    assert code == 1
    payload = _json_from_stdout(capsys)
    assert "refusing to use the repo root" in payload["error"]


def test_review_and_verify_accept_only_after_hermes_verification(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"
    run_id, _state = _run_simulated_lane(lane, repo, state_root, capsys)

    code = lane.main(["review", "--run-id", run_id, "--state-root", str(state_root), "--simulate"])
    assert code == 0
    reviewed = _json_from_stdout(capsys)["state"]
    assert reviewed["status"] == "review_passed"

    verifier = f"{shlex.quote(sys.executable)} -c \"print('hermes verified')\""
    code = lane.main(
        [
            "verify",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--command",
            verifier,
            "--accept",
        ]
    )
    assert code == 0
    verified = _json_from_stdout(capsys)["state"]
    assert verified["status"] == "accepted"
    assert verified["codex_lane"]["result"] == "accepted"
    assert verified["codex_lane"]["tests_run"][0]["owner"] == "hermes"


def test_accept_requires_successful_claude_review_phase(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"
    run_id, _state = _run_simulated_lane(lane, repo, state_root, capsys)

    verifier = f"{shlex.quote(sys.executable)} -c \"print('hermes verified')\""
    code = lane.main(
        [
            "verify",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--command",
            verifier,
            "--accept",
        ]
    )

    assert code == 2
    state = _json_from_stdout(capsys)["state"]
    assert state["status"] == "human_review"
    assert "before a successful Claude review" in state["codex_lane"]["rejected_reason"]


def test_verify_refuses_acceptance_without_independent_evidence(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"
    run_id, _state = _run_simulated_lane(lane, repo, state_root, capsys)

    code = lane.main(["verify", "--run-id", run_id, "--state-root", str(state_root), "--accept"])

    assert code == 2
    state = _json_from_stdout(capsys)["state"]
    assert state["status"] == "human_review"
    assert "without an independent verification" in state["codex_lane"]["rejected_reason"]


def test_verify_failure_parks_lane_in_human_review(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"
    run_id, _state = _run_simulated_lane(lane, repo, state_root, capsys)
    code = lane.main(["review", "--run-id", run_id, "--state-root", str(state_root), "--simulate"])
    assert code == 0
    _json_from_stdout(capsys)

    failing = f"{shlex.quote(sys.executable)} -c \"raise SystemExit(7)\""
    code = lane.main(
        [
            "verify",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--command",
            failing,
            "--accept",
        ]
    )

    assert code == 3
    state = _json_from_stdout(capsys)["state"]
    assert state["status"] == "human_review"
    assert state["codex_lane"]["tests_run"][0]["exit_code"] == 7


def test_simulated_timeout_and_stop_are_recorded(tmp_path, capsys):
    lane = _load_lane_module()
    repo = _make_repo(tmp_path)
    state_root = tmp_path / "state"

    code = lane.main(
        [
            "run",
            "--repo",
            str(repo),
            "--task-id",
            "T-timeout",
            "--goal",
            "Simulate a timeout.",
            "--state-root",
            str(state_root),
            "--simulate",
            "--simulate-result",
            "timeout",
        ]
    )
    assert code == 0
    state = _json_from_stdout(capsys)["state"]
    assert state["status"] == "timed_out"
    assert state["codex_lane"]["result"] == "timed_out"

    run_id = state["run_id"]
    code = lane.main(["stop", "--run-id", run_id, "--state-root", str(state_root)])
    assert code == 0
    stopped = _json_from_stdout(capsys)["state"]
    assert stopped["status"] == "stopped"
    assert stopped["codex_lane"]["result"] == "human_review"


def test_codex_command_and_claude_verdict_parsing_are_conservative(tmp_path):
    lane = _load_lane_module()

    yolo = lane.codex_command(tmp_path, "yolo")
    full_auto = lane.codex_command(tmp_path, "full-auto")

    assert "--dangerously-bypass-approvals-and-sandbox" in yolo
    assert "--sandbox" in full_auto
    assert "workspace-write" in full_auto
    assert lane.parse_claude_verdict('{"verdict": "pass"}') == ("pass", "top-level verdict")
    assert lane.parse_claude_verdict('{"result": "{\\"verdict\\": \\"fail\\"}"}')[0] == "fail"
    assert (
        lane.parse_claude_verdict(
            '{"content": "review text {\\"verdict\\": \\"pass\\"} with no failures"}'
        )[0]
        == "pass"
    )
    assert lane.parse_claude_verdict("review completed without verdict")[0] == "unknown"


def test_live_codex_wait_success_sets_pending_verification(monkeypatch, tmp_path):
    lane = _load_lane_module()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    state = {
        "worktree": str(worktree),
        "logs": {"builder": str(tmp_path / "builder.log")},
        "autonomy": "yolo",
        "state_path": str(tmp_path / "state.json"),
        "phases": [],
    }

    monkeypatch.setattr(
        lane,
        "codex_command",
        lambda _worktree, _autonomy: [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('live_marker.txt').write_text('ok')",
        ],
    )

    lane.start_live_codex(state, "unused prompt", timeout=10, wait=True)

    assert state["status"] == "codex_completed"
    assert state["result"] == "pending_verification"
    assert state["codex_exit_code"] == 0
    assert state["pid"] is None
    assert (worktree / "live_marker.txt").read_text(encoding="utf-8") == "ok"
