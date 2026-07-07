"""Tests for the GitHub native tools and helper layer."""

import json
import pytest
from unittest.mock import patch, MagicMock

from tools.github_tools import (
    _get_github_token,
    check_github_requirements,
    parse_owner_repo,
    get_github_error_message,
    github_get_issue_tool,
    github_list_issues_tool,
    github_get_pull_request_tool,
    github_list_pull_requests_tool,
    github_add_issue_comment_tool,
)


def test_check_github_requirements():
    # 1. Gating
    with patch("tools.github_tools._get_github_token", return_value=""):
        assert check_github_requirements() is False

    with patch("tools.github_tools._get_github_token", return_value="token123"):
        assert check_github_requirements() is True


def test_parse_owner_repo():
    # 2. Repository parsing
    # Standard format
    assert parse_owner_repo("octocat/hello-world") == ("octocat", "hello-world")
    assert parse_owner_repo(" octocat/hello-world  ") == ("octocat", "hello-world")
    
    # Trailing .git
    assert parse_owner_repo("octocat/hello-world.git") == ("octocat", "hello-world")

    # URL formats
    assert parse_owner_repo("https://github.com/octocat/hello-world") == ("octocat", "hello-world")
    assert parse_owner_repo("http://github.com/octocat/hello-world.git") == ("octocat", "hello-world")
    assert parse_owner_repo("github.com/octocat/hello-world") == ("octocat", "hello-world")

    # Invalid formats
    with pytest.raises(ValueError, match="Repository identifier cannot be empty"):
        parse_owner_repo("")

    with pytest.raises(ValueError, match="Invalid repository identifier"):
        parse_owner_repo("invalid-format")

    with pytest.raises(ValueError, match="Invalid repository identifier"):
        parse_owner_repo("too/many/slashes/here")


def test_get_github_error_message():
    # 3. GitHub error extraction
    # JSON error payload from GitHub
    api_result_with_msg = {
        "success": True,
        "ok": False,
        "status": 404,
        "json": {"message": "Not Found"},
    }
    assert get_github_error_message(api_result_with_msg) == "GitHub API error (404): Not Found"

    # Fallback to text preview
    api_result_text = {
        "success": True,
        "ok": False,
        "status": 500,
        "text_preview": "Internal server error details",
    }
    assert get_github_error_message(api_result_text) == "GitHub API error (500): Internal server error details"

    # Execution failure
    api_failed = {
        "success": False,
        "error": "Connection timed out",
    }
    assert get_github_error_message(api_failed) == "Connection timed out"


@patch("tools.github_tools.github_api_request")
def test_github_get_issue_success(mock_api_req):
    # 4. github_get_issue success case and 7. Output shaping
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": True,
        "status": 200,
        "json": {
            "number": 42,
            "title": "Fix memory leak",
            "state": "open",
            "user": {"login": "octocat"},
            "assignees": [{"login": "alice"}],
            "labels": [{"name": "bug"}],
            "html_url": "https://github.com/octocat/hello-world/issues/42",
            "body": "Detailed description of bug",
            "extra_noise": "should be filtered out",
        }
    })

    res_str = github_get_issue_tool("octocat/hello-world", 42)
    data = json.loads(res_str)
    
    assert "error" not in data
    assert data["number"] == 42
    assert data["title"] == "Fix memory leak"
    assert data["state"] == "open"
    assert data["author"] == "octocat"
    assert data["assignees"] == ["alice"]
    assert data["labels"] == ["bug"]
    assert data["html_url"] == "https://github.com/octocat/hello-world/issues/42"
    assert data["body"] == "Detailed description of bug"
    assert "extra_noise" not in data


@patch("tools.github_tools.github_api_request")
def test_github_list_issues_filtering_and_shaping(mock_api_req):
    # 4. github_list_issues success
    # 6. Issue-vs-PR filtering (PRs filtered out)
    # 7. Output shaping (summary details)
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": True,
        "status": 200,
        "json": [
            {
                "number": 1,
                "title": "Bug report",
                "state": "open",
                "user": {"login": "user1"},
                "html_url": "https://github.com/owner/repo/issues/1",
            },
            {
                "number": 2,
                "title": "Feature branch PR",
                "state": "open",
                "user": {"login": "user2"},
                "html_url": "https://github.com/owner/repo/pull/2",
                "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/2"},
            }
        ]
    })

    res_str = github_list_issues_tool("owner/repo", "open")
    issues = json.loads(res_str)
    
    assert isinstance(issues, list)
    assert len(issues) == 1
    assert issues[0]["number"] == 1
    assert issues[0]["title"] == "Bug report"
    assert "pull_request" not in issues[0]


@patch("tools.github_tools.github_api_request")
def test_github_get_pull_request_success(mock_api_req):
    # 4. github_get_pull_request success case
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": True,
        "status": 200,
        "json": {
            "number": 101,
            "title": "Add feature",
            "state": "closed",
            "user": {"login": "developer"},
            "draft": False,
            "merged": True,
            "base": {"ref": "main", "sha": "basesha123"},
            "head": {"ref": "feat-branch", "sha": "headsha456"},
            "html_url": "https://github.com/owner/repo/pull/101",
            "body": "PR description",
        }
    })

    res_str = github_get_pull_request_tool("owner/repo", 101)
    data = json.loads(res_str)
    
    assert "error" not in data
    assert data["number"] == 101
    assert data["merged"] is True
    assert data["base"]["ref"] == "main"
    assert data["head"]["ref"] == "feat-branch"


@patch("tools.github_tools.github_api_request")
def test_github_list_pull_requests_success(mock_api_req):
    # 4. github_list_pull_requests success case
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": True,
        "status": 200,
        "json": [
            {
                "number": 5,
                "title": "Fix typos",
                "state": "open",
                "user": {"login": "editor"},
                "html_url": "https://github.com/owner/repo/pull/5",
                "draft": True,
            }
        ]
    })

    res_str = github_list_pull_requests_tool("owner/repo")
    prs = json.loads(res_str)
    
    assert isinstance(prs, list)
    assert len(prs) == 1
    assert prs[0]["number"] == 5
    assert prs[0]["draft"] is True


def test_wrapper_tool_invalid_repo_input():
    # 5. Wrapper tool failure cases - invalid repo
    res_str = github_get_issue_tool("invalidrepo", 1)
    res = json.loads(res_str)
    assert "error" in res
    assert "Invalid repository identifier" in res["error"]


@patch("tools.github_tools.github_api_request")
def test_wrapper_tool_non_200_response(mock_api_req):
    # 5. Wrapper tool failure cases - non-200
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": False,
        "status": 404,
        "json": {"message": "Not Found"},
    })

    res_str = github_get_issue_tool("owner/repo", 999)
    res = json.loads(res_str)
    
    assert "error" in res
    assert "GitHub API error (404): Not Found" in res["error"]


@patch("tools.github_tools.github_api_request")
def test_wrapper_tool_failed_http_request(mock_api_req):
    # 5. Wrapper tool failure cases - HTTP failed
    mock_api_req.return_value = json.dumps({
        "success": False,
        "error": "SSL verification failed",
        "error_type": "SSLError",
    })

    res_str = github_list_issues_tool("owner/repo")
    res = json.loads(res_str)
    
    assert "error" in res
    assert "SSL verification failed" in res["error"]


@patch("tools.github_tools.github_api_request")
def test_github_add_issue_comment_success(mock_api_req):
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": True,
        "status": 201,
        "json": {
            "id": 123456,
            "html_url": "https://github.com/octocat/hello-world/issues/42#issuecomment-123456",
            "body": "This is a comment body",
            "created_at": "2026-07-07T09:00:00Z",
            "updated_at": "2026-07-07T09:00:00Z",
            "user": {"login": "octocat"},
            "extra_ignored_field": "noise"
        }
    })

    res_str = github_add_issue_comment_tool("octocat/hello-world", 42, "This is a comment body")
    data = json.loads(res_str)

    assert "error" not in data
    assert data["id"] == 123456
    assert data["html_url"] == "https://github.com/octocat/hello-world/issues/42#issuecomment-123456"
    assert data["body"] == "This is a comment body"
    assert data["created_at"] == "2026-07-07T09:00:00Z"
    assert data["updated_at"] == "2026-07-07T09:00:00Z"
    assert data["author"] == "octocat"
    assert "extra_ignored_field" not in data

    mock_api_req.assert_called_once_with(
        "POST",
        "/repos/octocat/hello-world/issues/42/comments",
        json_body={"body": "This is a comment body"}
    )


def test_github_add_issue_comment_invalid_repo():
    res_str = github_add_issue_comment_tool("invalidrepo", 42, "hello")
    res = json.loads(res_str)
    assert "error" in res
    assert "Invalid repository identifier" in res["error"]


@patch("tools.github_tools.github_api_request")
def test_github_add_issue_comment_non_200_response(mock_api_req):
    mock_api_req.return_value = json.dumps({
        "success": True,
        "ok": False,
        "status": 422,
        "json": {"message": "Validation Failed"},
    })

    res_str = github_add_issue_comment_tool("owner/repo", 42, "hello")
    res = json.loads(res_str)
    
    assert "error" in res
    assert "GitHub API error (422): Validation Failed" in res["error"]


@patch("tools.github_tools.github_api_request")
def test_github_add_issue_comment_failed_http_request(mock_api_req):
    mock_api_req.return_value = json.dumps({
        "success": False,
        "error": "Connection timed out",
        "error_type": "ConnectTimeout",
    })

    res_str = github_add_issue_comment_tool("owner/repo", 42, "hello")
    res = json.loads(res_str)
    
    assert "error" in res
    assert "Connection timed out" in res["error"]
