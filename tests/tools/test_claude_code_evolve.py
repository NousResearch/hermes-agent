"""Tests for Claude Code history evolution helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.claude_code_evolve import (
    MemoryCandidate,
    SkillCandidate,
    ToolInvocation,
    _looks_like_correction,
    _project_dir_for_path,
    analyze_claude_code_history,
    discover_claude_projects,
    iter_jsonl_events,
    mine_skill_candidates,
    select_scope,
    write_memory_candidate,
    write_skill_candidate,
)


def _write_jsonl(project_dir: Path, session_id: str, events: list[dict], malformed: bool = False) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    target = project_dir / f"{session_id}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        if malformed:
            handle.write("{ definitely not json }\n")
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _user_event(ts: str, cwd: str, content: str, session_id: str = "sess-1") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "role": "user",
            "content": content,
        },
    }


def _assistant_tool_event(
    ts: str,
    cwd: str,
    tool_name: str,
    tool_input: dict,
    session_id: str = "sess-1",
) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu-{tool_name.lower()}",
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
        },
    }


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True, check=True)
    (repo / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo, capture_output=True, check=True)


def test_iter_jsonl_events_ignores_malformed_lines(tmp_path):
    projects_root = tmp_path / ".claude" / "projects"
    project_path = tmp_path / "repo"
    project_dir = _project_dir_for_path(projects_root, project_path)
    _write_jsonl(
        project_dir,
        "sess-1",
        [
            _user_event(
                ts="2026-04-10T10:00:00Z",
                cwd=str(project_path),
                content="hello",
            )
        ],
        malformed=True,
    )

    projects = discover_claude_projects(projects_root)
    events = list(iter_jsonl_events(projects, since=datetime(2026, 4, 1, tzinfo=timezone.utc)))

    assert len(events) == 1
    assert events[0].event_type == "user"
    assert events[0].cwd == project_path.resolve(strict=False)


def test_correction_detector_flags_expected_markers():
    assert _looks_like_correction("don't run tests automatically")
    assert _looks_like_correction("别自动跑测试")
    assert _looks_like_correction("改成先给我 pytest 命令")
    assert not _looks_like_correction("please review this branch")


def test_repo_scope_includes_main_active_and_historical_worktrees(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    active_worktree = repo / ".worktrees" / "hermes-active"
    active_worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", str(active_worktree), "-b", "hermes/hermes-active", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
    )

    historical_worktree = repo / ".worktrees" / "hermes-old"

    projects_root = tmp_path / ".claude" / "projects"
    now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    for cwd in (repo, active_worktree, historical_worktree):
        project_dir = _project_dir_for_path(projects_root, cwd)
        _write_jsonl(
            project_dir,
            f"session-{cwd.name}",
            [
                _user_event(now.isoformat().replace("+00:00", "Z"), str(cwd), "start"),
                _assistant_tool_event(
                    (now + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                    str(cwd),
                    "Read",
                    {"file_path": str(cwd / "README.md"), "offset": 0, "limit": 100},
                    session_id=f"session-{cwd.name}",
                ),
            ],
        )

    discovered = discover_claude_projects(projects_root)
    summary = select_scope(
        anchor_path=active_worktree,
        scope="repo",
        projects_root=projects_root,
        discovered_projects=discovered,
        since=now - timedelta(days=7),
    )

    assert summary.repo_root == repo.resolve(strict=False)
    assert len(summary.analysis_projects) == 3
    assert _project_dir_for_path(projects_root, repo) in summary.memory_target_project_dirs
    assert _project_dir_for_path(projects_root, active_worktree) in summary.memory_target_project_dirs
    assert _project_dir_for_path(projects_root, historical_worktree) in summary.memory_target_project_dirs


def test_skill_miner_requires_three_occurrences_across_two_days():
    base_time = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    project_dir = Path("/tmp/fake-project")
    tool_a = ToolInvocation(
        tool_name="Read",
        args_shape="file_path,limit,offset",
        timestamp=base_time,
        day_key="2026-04-10",
        session_id="s1",
        project_dir=project_dir,
        project_cwd=project_dir,
    )
    tool_b = ToolInvocation(
        tool_name="Grep",
        args_shape="path,pattern",
        timestamp=base_time + timedelta(seconds=1),
        day_key="2026-04-10",
        session_id="s1",
        project_dir=project_dir,
        project_cwd=project_dir,
    )
    good_segments = (
        (tool_a, tool_b),
        (
            tool_a,
            ToolInvocation(
                tool_name="Grep",
                args_shape="path,pattern",
                timestamp=base_time + timedelta(minutes=5),
                day_key="2026-04-10",
                session_id="s2",
                project_dir=project_dir,
                project_cwd=project_dir,
            ),
        ),
        (
            ToolInvocation(
                tool_name="Read",
                args_shape="file_path,limit,offset",
                timestamp=base_time + timedelta(days=1),
                day_key="2026-04-11",
                session_id="s3",
                project_dir=project_dir,
                project_cwd=project_dir,
            ),
            ToolInvocation(
                tool_name="Grep",
                args_shape="path,pattern",
                timestamp=base_time + timedelta(days=1, seconds=1),
                day_key="2026-04-11",
                session_id="s3",
                project_dir=project_dir,
                project_cwd=project_dir,
            ),
        ),
    )
    bad_segments = good_segments[:2]

    assert mine_skill_candidates(bad_segments) == ()
    candidates = mine_skill_candidates(good_segments)
    assert any(candidate.frequency >= 3 and candidate.distinct_days >= 2 for candidate in candidates)


def test_apply_writes_idempotently(tmp_path):
    projects_root = tmp_path / ".claude" / "projects"
    skills_root = tmp_path / ".claude" / "skills"
    project_dir = projects_root / "encoded-project"

    memory_candidate = MemoryCandidate(
        type="feedback",
        name="feedback-manual-tests",
        description="Repeated user correction: manual tests",
        body="# Evolved Memory: manual tests\n",
        target_project_dirs=(project_dir,),
        evidence=("2026-04-12: don't run tests automatically",),
        frequency=2,
    )
    skill_candidate = SkillCandidate(
        slug="read-grep",
        title="Read -> Grep",
        description="Observed repeatedly",
        body="# Evolved Workflow: Read -> Grep\n",
        evidence=("2026-04-12: /tmp/project",),
        frequency=3,
        distinct_days=2,
        source_projects=("/tmp/project",),
    )

    first_memory = write_memory_candidate(project_dir, memory_candidate, projects_root=projects_root)
    second_memory = write_memory_candidate(project_dir, memory_candidate, projects_root=projects_root)
    first_skill = write_skill_candidate(skills_root, skill_candidate)
    second_skill = write_skill_candidate(skills_root, skill_candidate)

    assert first_memory == second_memory
    assert first_skill == second_skill
    assert sorted((project_dir / "memory").glob("*.md")) == [first_memory]
    assert sorted(skills_root.glob("*/SKILL.md")) == [first_skill]


def test_analyze_history_finds_repeated_corrections_and_tools(tmp_path):
    projects_root = tmp_path / ".claude" / "projects"
    project_path = tmp_path / "repo"
    project_dir = _project_dir_for_path(projects_root, project_path)
    project_path.mkdir()

    day1 = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 4, 11, 9, 0, tzinfo=timezone.utc)
    sessions = [
        ("s1", day1, "don't run tests automatically"),
        ("s2", day1 + timedelta(hours=2), "don't run tests automatically"),
        ("s3", day2, "don't run tests automatically"),
    ]
    for session_id, start, correction in sessions:
        _write_jsonl(
            project_dir,
            session_id,
            [
                _user_event(start.isoformat().replace("+00:00", "Z"), str(project_path), "review this"),
                _assistant_tool_event(
                    (start + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                    str(project_path),
                    "Read",
                    {"file_path": str(project_path / "README.md"), "offset": 0, "limit": 120},
                    session_id=session_id,
                ),
                _assistant_tool_event(
                    (start + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
                    str(project_path),
                    "Grep",
                    {"path": str(project_path), "pattern": "pytest"},
                    session_id=session_id,
                ),
                _user_event(
                    (start + timedelta(minutes=3)).isoformat().replace("+00:00", "Z"),
                    str(project_path),
                    correction,
                    session_id=session_id,
                ),
            ],
        )

    result = analyze_claude_code_history(
        anchor_path=project_path,
        since=day1 - timedelta(days=1),
        scope="cwd",
        projects_root=projects_root,
    )

    assert result.memory_candidates
    assert result.skill_candidates
    assert result.memory_candidates[0].frequency == 3


def _tool_result_user_event(ts: str, cwd: str, tool_use_id: str, session_id: str = "sess-1") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "ok",
                }
            ],
        },
    }


def test_tool_result_user_events_do_not_split_segments(tmp_path):
    projects_root = tmp_path / ".claude" / "projects"
    project_path = tmp_path / "repo"
    project_dir = _project_dir_for_path(projects_root, project_path)
    project_path.mkdir()

    day1 = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 4, 11, 9, 0, tzinfo=timezone.utc)
    day3 = datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc)

    for session_id, start in (("s1", day1), ("s2", day2), ("s3", day3)):
        ts = lambda offset: (start + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")
        _write_jsonl(
            project_dir,
            session_id,
            [
                _assistant_tool_event(ts(0), str(project_path), "Read",
                                      {"file_path": "a", "offset": 0, "limit": 1},
                                      session_id=session_id),
                _tool_result_user_event(ts(1), str(project_path), "toolu-read", session_id=session_id),
                _assistant_tool_event(ts(2), str(project_path), "Grep",
                                      {"path": "a", "pattern": "x"},
                                      session_id=session_id),
                _tool_result_user_event(ts(3), str(project_path), "toolu-grep", session_id=session_id),
                _assistant_tool_event(ts(4), str(project_path), "Edit",
                                      {"file_path": "a", "old_string": "x", "new_string": "y"},
                                      session_id=session_id),
            ],
        )

    result = analyze_claude_code_history(
        anchor_path=project_path,
        since=day1 - timedelta(days=1),
        scope="cwd",
        projects_root=projects_root,
    )

    assert result.skill_candidates, "tool_result user events must not split tool segments"
    top = result.skill_candidates[0]
    assert top.frequency >= 3
    assert top.distinct_days >= 2


def test_real_user_text_still_splits_segments(tmp_path):
    projects_root = tmp_path / ".claude" / "projects"
    project_path = tmp_path / "repo"
    project_dir = _project_dir_for_path(projects_root, project_path)
    project_path.mkdir()

    start = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
    ts = lambda offset: (start + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")
    _write_jsonl(
        project_dir,
        "sess",
        [
            _assistant_tool_event(ts(0), str(project_path), "Read",
                                  {"file_path": "a"}, session_id="sess"),
            _user_event(ts(1), str(project_path), "now please grep", session_id="sess"),
            _assistant_tool_event(ts(2), str(project_path), "Grep",
                                  {"path": "a", "pattern": "x"}, session_id="sess"),
        ],
    )

    result = analyze_claude_code_history(
        anchor_path=project_path,
        since=start - timedelta(days=1),
        scope="cwd",
        projects_root=projects_root,
    )

    assert result.skill_candidates == ()


def test_repo_scope_includes_subdirectory_sessions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    subdir = repo / "pkg"
    subdir.mkdir()

    projects_root = tmp_path / ".claude" / "projects"
    subdir_project_dir = _project_dir_for_path(projects_root, subdir)
    now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    _write_jsonl(
        subdir_project_dir,
        "session-pkg",
        [_user_event(now.isoformat().replace("+00:00", "Z"), str(subdir), "hi",
                     session_id="session-pkg")],
    )

    summary = select_scope(
        anchor_path=repo,
        scope="repo",
        projects_root=projects_root,
        since=now - timedelta(days=7),
    )

    assert subdir_project_dir in {p.project_dir for p in summary.analysis_projects}


def test_repo_scope_flags_removed_nested_paths_as_historical(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    nested_missing = repo / "phantom-worktree"
    projects_root = tmp_path / ".claude" / "projects"
    nested_project_dir = _project_dir_for_path(projects_root, nested_missing)

    now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    _write_jsonl(
        nested_project_dir,
        "session-removed",
        [_user_event(now.isoformat().replace("+00:00", "Z"), str(nested_missing), "hi",
                     session_id="session-removed")],
    )

    unrelated = tmp_path / "external-feature"
    unrelated_project_dir = _project_dir_for_path(projects_root, unrelated)
    _write_jsonl(
        unrelated_project_dir,
        "session-unrelated",
        [_user_event(now.isoformat().replace("+00:00", "Z"), str(unrelated), "hi",
                     session_id="session-unrelated")],
    )

    summary = select_scope(
        anchor_path=repo,
        scope="repo",
        projects_root=projects_root,
        since=now - timedelta(days=7),
    )

    assert nested_project_dir in {p.project_dir for p in summary.analysis_projects}
    assert nested_project_dir in summary.historical_worktree_project_dirs
    assert unrelated_project_dir not in {p.project_dir for p in summary.analysis_projects}
