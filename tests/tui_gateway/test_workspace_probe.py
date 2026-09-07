"""Pure and hermetic contracts for the read-only workspace HUD probe."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import workspace_probe
from tui_gateway.workspace_probe import (
    _dirty_from_status,
    _upstream_counts,
    empty_workspace,
    normalize_github_origin,
    parse_pull_request_payload,
)


def test_normalize_github_origin_drops_credentials_and_non_github_hosts():
    assert normalize_github_origin(
        "https://user:super-secret@github.com/NousResearch/hermes-agent.git?token=super-secret"
    ) == {
        "owner": "NousResearch",
        "repo": "hermes-agent",
        "full_name": "NousResearch/hermes-agent",
    }
    assert normalize_github_origin("git@github.com:NousResearch/hermes-agent.git")["full_name"] == (
        "NousResearch/hermes-agent"
    )
    assert normalize_github_origin("https://gitlab.com/NousResearch/hermes-agent.git") is None


def test_git_status_and_upstream_counts_keep_direction_and_unknown_state():
    assert _upstream_counts("2 3") == {"ahead": 2, "behind": 3}
    assert _dirty_from_status("## main...origin/main\n") is False
    assert _dirty_from_status("## main...origin/main\n M changed.py\n") is True
    assert _dirty_from_status("") is None


def test_parse_pull_request_payload_bounds_state_and_redacts_title():
    parsed = parse_pull_request_payload(
        {
            "number": 12,
            "state": "OPEN",
            "title": "Ship https://user:super-secret@github.com/a/b token=super-secret",
            "headRefName": "feature/hud",
        }
    )

    assert parsed == {
        "number": 12,
        "state": "open",
        "title": "Ship [url] [redacted]",
        "head_ref_name": "feature/hud",
    }
    assert "super-secret" not in str(parsed)
    assert parse_pull_request_payload({"number": 0, "state": "OPEN"}) is None
    assert parse_pull_request_payload({"number": True}) is None


def test_run_gh_requests_only_a_read_only_pr_view(tmp_path):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "number": 42,
            "state": "OPEN",
            "title": "Review https://user:super-secret@github.com/a/b",
            "headRefName": "feature/hud",
        }),
    )
    repo = {"owner": "NousResearch", "repo": "hermes-agent", "full_name": "NousResearch/hermes-agent"}

    with patch.object(workspace_probe.subprocess, "run", return_value=completed) as run:
        result = workspace_probe._run_gh(str(tmp_path), repo)

    assert result == {
        "number": 42,
        "state": "open",
        "title": "Review [url]",
        "head_ref_name": "feature/hud",
    }
    argv = run.call_args.args[0]
    assert argv[:4] == ["gh", "pr", "view", "--repo"]
    assert argv[4] == "NousResearch/hermes-agent"
    assert "super-secret" not in str(result)


def test_probe_composes_read_only_workspace_facts_and_current_pr(tmp_path):
    calls = []

    def run_git(cwd, *args):
        calls.append(args)
        values = {
            ("branch", "--show-current"): "feature/hud",
            ("config", "--get", "remote.origin.url"): (
                "https://user:super-secret@github.com/NousResearch/hermes-agent.git"
            ),
            ("status", "--porcelain=v1", "--branch", "--untracked-files=normal"): (
                "## feature/hud...origin/feature/hud\n M changed.py\n"
            ),
            ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"): "2 1",
        }
        return values[args]

    gh_result = {
        "number": 42,
        "state": "open",
        "title": "HUD",
        "head_ref_name": "feature/hud",
    }

    canonical_root = tmp_path / "canonical"
    with patch.object(workspace_probe.git_probe, "repo_root", return_value=str(tmp_path)), patch.object(
        workspace_probe.git_probe, "common_repo_root", return_value=str(canonical_root)
    ), patch.object(workspace_probe.git_probe, "run_git", side_effect=run_git), patch.object(
        workspace_probe, "_run_gh", return_value=gh_result
    ) as run_gh:
        result = workspace_probe._probe(str(tmp_path))

    assert "super-secret" not in str(result)
    assert result == {
        "git_root": str(canonical_root.resolve()),
        "github": {
            "owner": "NousResearch",
            "repo": "hermes-agent",
            "full_name": "NousResearch/hermes-agent",
        },
        "branch": "feature/hud",
        "dirty": True,
        "upstream": {"ahead": 2, "behind": 1},
        "pull_request": {"number": 42, "state": "open", "title": "HUD"},
    }
    assert run_gh.call_args.args == (str(tmp_path.resolve()), {
        "owner": "NousResearch",
        "repo": "hermes-agent",
        "full_name": "NousResearch/hermes-agent",
    })
    assert all(command[0] in {"branch", "config", "status", "rev-list"} for command in calls)


def test_resolve_caches_probe_results_and_returns_nested_copies(tmp_path):
    payload = {
        "git_root": str(tmp_path),
        "github": {"owner": "NousResearch", "repo": "hermes-agent", "full_name": "NousResearch/hermes-agent"},
        "branch": "main",
        "dirty": False,
        "upstream": {"ahead": 0, "behind": 0},
        "pull_request": None,
    }

    with patch.object(workspace_probe, "_probe", return_value=payload) as probe:
        first = workspace_probe.resolve(str(tmp_path))
        first["github"]["owner"] = "mutated"
        second = workspace_probe.resolve(str(tmp_path))

    assert probe.call_count == 1
    assert second["github"]["owner"] == "NousResearch"


def test_workspace_rpc_is_registered_as_a_long_handler():
    from tui_gateway import server

    assert "workspace.info" in server._methods
    assert "workspace.info" in server._LONG_HANDLERS


def test_empty_workspace_is_stable_and_contains_no_probe_details():
    first = empty_workspace()
    first["github"] = {"owner": "mutated"}
    second = empty_workspace()

    assert second == {
        "git_root": None,
        "github": None,
        "branch": None,
        "dirty": None,
        "upstream": None,
        "pull_request": None,
    }
