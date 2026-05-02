"""Tests for the copilot_remote semantic delegation tool."""

import json

import pytest

from copilot_remote.models import RepoEntry
from hermes_state import SessionDB
from tools.copilot_remote_tool import COPILOT_REMOTE_SCHEMA, copilot_remote


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / ".hermes" / "state.db"
    db_path.parent.mkdir(parents=True)
    test_db = SessionDB(db_path=db_path)
    real_close = test_db.close
    test_db.close = lambda: None
    monkeypatch.setattr("tools.copilot_remote_tool._get_db", lambda: test_db)
    yield test_db
    real_close()


def test_launch_explicit_repo_dry_run(db):
    result = json.loads(
        copilot_remote(
            {
                "action": "launch",
                "prompt": "Build a static webpage about the Macy Conferences",
                "repo": "repo-name",
                "repo_path": "/workspace/repos/corp_it/repo-name",
                "dry_run": True,
            },
            task_id="slack-session-1",
        )
    )

    assert result["success"] is True
    assert result["action"] == "launch"
    assert result["job"]["repo"] == "repo-name"
    assert result["job"]["state"] == "done"
    # Dry-run never spawns the Copilot subprocess, so the launcher cannot
    # extract a real reconnect handle. The tool must NOT fabricate one
    # from the Hermes job UUID — the launcher does not pass that into
    # Copilot via --resume, so it would be a non-functional command.
    assert result["job"]["connect_handle"] is None
    assert result["job"]["connect_command"] is None
    assert result["job"]["resume_command"] is None
    # Without a connect handle there is no web_url.
    assert result["job"]["web_url"] is None

    jobs = db.list_copilot_remote(state="done")
    assert len(jobs) == 1
    assert jobs[0]["repo_slug"] == "repo-name"


def test_launch_routes_repo_and_stores_connect_handle(db, monkeypatch):
    routed_repo = RepoEntry(
        slug="repo-name",
        path="/workspace/repos/corp_it/repo-name",
    )
    monkeypatch.setattr("tools.copilot_remote_tool._route_repo", lambda prompt: routed_repo)

    def fake_launch(repo, prompt, *, session_id, model=None, dry_run=False, on_complete=None):
        assert repo.slug == "repo-name"
        assert "new static webpage" in prompt
        assert dry_run is False
        return {
            "session_id": session_id,
            "connect_id": "task-123",
            "cmd": ["copilot"],
            "proc": None,
            "prompt_delivery_status": "already-submitted",
            "prompt_delivery_warning": None,
        }

    monkeypatch.setattr("copilot_remote.launcher.launch_copilot", fake_launch)

    result = json.loads(
        copilot_remote(
            {
                "action": "launch",
                "prompt": "Please build a new static webpage for the Macy Conferences",
            },
            task_id="slack-session-2",
        )
    )

    assert result["success"] is True
    assert result["job"]["repo"] == "repo-name"
    assert result["job"]["connect_handle"] == "task-123"
    assert result["job"]["connect_command"] == "copilot --connect=task-123"
    # repo_path is not a real git clone in the test environment, so the shared
    # GitHub task URL helper cannot derive an origin-backed web_url.
    assert result["job"]["web_url"] is None

    jobs = db.list_copilot_remote(state="running")
    assert len(jobs) == 1
    # Post-v12: the launcher-discovered connect handle lives in its own
    # column; `signal_ref` is reserved for caller-supplied metadata.
    assert jobs[0]["connect_handle"] == "task-123"


def test_launch_routes_repo_with_web_url(db, monkeypatch):
    """When the repo path is a real git clone and connect handle exists, web_url should be present."""
    routed_repo = RepoEntry(
        slug="repo-name",
        path="/workspace/repos/corp_it/repo-name",
    )
    monkeypatch.setattr("tools.copilot_remote_tool._route_repo", lambda prompt: routed_repo)
    monkeypatch.setattr(
        "tools.copilot_remote_tool.build_github_task_web_url",
        lambda repo_path, repo_slug, connect_handle: (
            f"https://github.com/RosenblattAI/{repo_slug}/tasks/{connect_handle}"
        ),
    )

    def fake_launch(repo, prompt, *, session_id, model=None, dry_run=False, on_complete=None):
        return {
            "session_id": session_id,
            "connect_id": "task-456",
            "cmd": ["copilot"],
            "proc": None,
            "prompt_delivery_status": "already-submitted",
            "prompt_delivery_warning": None,
        }

    monkeypatch.setattr("copilot_remote.launcher.launch_copilot", fake_launch)

    result = json.loads(
        copilot_remote(
            {
                "action": "launch",
                "prompt": "Please build a new static webpage for the Macy Conferences",
            },
            task_id="slack-session-3",
        )
    )

    assert result["success"] is True
    assert result["job"]["connect_handle"] == "task-456"
    assert result["job"]["web_url"] == (
        "https://github.com/RosenblattAI/repo-name/tasks/task-456"
    )


def test_launch_requires_prompt(db):
    result = json.loads(copilot_remote({"action": "launch", "repo": "repo-name"}))

    assert result["success"] is False
    assert "prompt is required" in result["error"]


def test_list_and_show(db):
    db.create_copilot_remote(
        job_id="job-1",
        repo_slug="repo-name",
        repo_path="/workspace/repos/corp_it/repo-name",
        prompt="Build page",
        connect_handle="task-1",
    )

    listing = json.loads(copilot_remote({"action": "list"}))
    assert listing["success"] is True
    assert listing["jobs"][0]["job_id"] == "job-1"

    shown = json.loads(copilot_remote({"action": "show", "job_id": "job-1"}))
    assert shown["success"] is True
    assert shown["job"]["resume_command"] == "copilot --resume=task-1"
    # repo_path is not a real git clone in the test environment.
    assert shown["job"]["web_url"] is None


def test_list_skips_web_url_lookup(db, monkeypatch):
    db.create_copilot_remote(
        job_id="job-2",
        repo_slug="repo-name",
        repo_path="/workspace/repos/corp_it/repo-name",
        prompt="Build page",
        connect_handle="task-2",
    )

    def _unexpected_web_url(*args, **kwargs):
        raise AssertionError("list should not compute web_url")

    monkeypatch.setattr(
        "tools.copilot_remote_tool.build_github_task_web_url",
        _unexpected_web_url,
    )

    listing = json.loads(copilot_remote({"action": "list"}))

    assert listing["success"] is True
    assert listing["jobs"][0]["job_id"] == "job-2"
    assert listing["jobs"][0]["web_url"] is None


def test_hermes_slack_toolset_exposes_copilot_remote():
    from toolsets import resolve_toolset

    assert "copilot_remote" in resolve_toolset("hermes-slack")


def test_schema_discourages_terminal_copilot_probes():
    description = COPILOT_REMOTE_SCHEMA["description"]

    assert "terminal Copilot probes" in description
    assert "call this tool directly" in description


def test_schema_marks_copilot_remote_as_default_implementation_tool():
    description = COPILOT_REMOTE_SCHEMA["description"]

    assert "default implementation tool" in description
    assert "code-writing" in description
    assert "website-building" in description
