#!/usr/bin/env python3
"""Native GitHub tools package built on the secure `http_request` foundation.

Provides helper methods, error shaping, and core GitHub integrations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from tools.http_request import http_request_tool
from tools.registry import registry, tool_result, tool_error

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"


def _get_github_token() -> str:
    """Helper to resolve GitHub token from environment.
    
    Checks GITHUB_TOKEN from Hermes configuration or environment.
    """
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(GITHUB_TOKEN_ENV_VAR)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(GITHUB_TOKEN_ENV_VAR, "")
    return (val or "").strip()


def check_github_requirements() -> bool:
    """Check if GitHub tool prerequisites are met (i.e., token is available)."""
    return bool(_get_github_token())


def parse_owner_repo(repo_str: str) -> Tuple[str, str]:
    """Parse owner and repository name from a string (owner/repo or GitHub URL).
    
    Raises ValueError if the format is invalid.
    """
    clean_str = repo_str.strip()
    if not clean_str:
        raise ValueError("Repository identifier cannot be empty")

    # 1. Handle GitHub URLs
    if "github.com" in clean_str.lower():
        try:
            # Add scheme if missing so urlparse works correctly
            url_to_parse = clean_str
            if not re.match(r"^[a-z]+://", clean_str.lower()):
                url_to_parse = "https://" + clean_str
            
            parsed = urlparse(url_to_parse)
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2:
                # Extract owner and repo (ignoring trailing .git if present)
                owner = path_parts[0]
                repo = path_parts[1]
                if repo.lower().endswith(".git"):
                    repo = repo[:-4]
                return owner, repo
        except Exception as exc:
            raise ValueError(f"Failed to parse URL '{repo_str}': {exc}")

    # 2. Handle standard owner/repo format
    parts = [p.strip() for p in clean_str.split("/") if p.strip()]
    if len(parts) == 2:
        owner, repo = parts
        if repo.lower().endswith(".git"):
            repo = repo[:-4]
        return owner, repo

    raise ValueError(
        f"Invalid repository identifier: '{repo_str}'. "
        "Must be in 'owner/repo' format or a valid GitHub URL."
    )


def github_api_request(
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Execute a request against the GitHub REST API using the http_request_tool foundation.
    
    Includes standard GitHub headers and bearer token authorization.
    """
    # 1. Construct URL
    base_url = DEFAULT_GITHUB_API_BASE.rstrip("/")
    clean_path = path.lstrip("/")
    url = f"{base_url}/{clean_path}"

    # 2. Setup standard GitHub headers
    req_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if headers:
        req_headers.update(headers)

    # 3. Delegate to secure http_request_tool
    return http_request_tool(
        method=method,
        url=url,
        headers=req_headers,
        query=query,
        json_body=json_body,
        auth_mode="bearer_env",
        auth_token_env=GITHUB_TOKEN_ENV_VAR,
    )


def get_github_error_message(result: Dict[str, Any]) -> str:
    """Extract a user-friendly error message from a GitHub tool result dictionary.
    
    Assumes result is the parsed dictionary output from github_api_request.
    """
    if not result.get("success"):
        return result.get("error", "Unknown tool execution failure")
    
    if not result.get("ok"):
        status = result.get("status")
        # Try to parse GitHub error message
        json_data = result.get("json")
        if isinstance(json_data, dict) and "message" in json_data:
            return f"GitHub API error ({status}): {json_data['message']}"
        
        # Fallback to text preview
        preview = result.get("text_preview", "").strip()
        if preview:
            # If the response itself is short, show it
            if len(preview) < 200:
                return f"GitHub API error ({status}): {preview}"
            return f"GitHub API error ({status}): {preview[:200]}..."
        
        return f"GitHub API error ({status})"
    
    return ""


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def github_get_issue_tool(repository: str, issue_number: int) -> str:
    """Retrieve detailed information about a specific issue in a GitHub repository."""
    try:
        owner, repo = parse_owner_repo(repository)
        path = f"/repos/{owner}/{repo}/issues/{issue_number}"
        res_str = github_api_request("GET", path)
        res_data = json.loads(res_str)
        
        if not res_data.get("success"):
            return res_str
        if not res_data.get("ok"):
            return tool_error(get_github_error_message(res_data), status=res_data.get("status"))
            
        issue = res_data.get("json")
        if not isinstance(issue, dict):
            return tool_error("Invalid response payload from GitHub API")
            
        formatted = {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "state": issue.get("state"),
            "author": issue.get("user", {}).get("login") if isinstance(issue.get("user"), dict) else None,
            "assignees": [a.get("login") for a in issue.get("assignees") or [] if isinstance(a, dict)],
            "labels": [l.get("name") for l in issue.get("labels") or [] if isinstance(l, dict)],
            "html_url": issue.get("html_url"),
            "body": issue.get("body"),
        }
        return tool_result(formatted)
    except Exception as e:
        logger.exception("github_get_issue_tool failed")
        return tool_error(str(e))


def github_list_issues_tool(repository: str, state: str = "open", per_page: int = 30) -> str:
    """List issues in a GitHub repository (excludes pull requests)."""
    try:
        owner, repo = parse_owner_repo(repository)
        path = f"/repos/{owner}/{repo}/issues"
        query = {"state": state, "per_page": per_page}
        res_str = github_api_request("GET", path, query=query)
        res_data = json.loads(res_str)
        
        if not res_data.get("success"):
            return res_str
        if not res_data.get("ok"):
            return tool_error(get_github_error_message(res_data), status=res_data.get("status"))
            
        issues = res_data.get("json")
        if not isinstance(issues, list):
            return tool_error("Invalid response payload from GitHub API")
            
        formatted = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            # GitHub's issues endpoint returns PRs too; filter them out
            if "pull_request" in issue:
                continue
            formatted.append({
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "author": issue.get("user", {}).get("login") if isinstance(issue.get("user"), dict) else None,
                "html_url": issue.get("html_url"),
            })
        return tool_result(formatted)
    except Exception as e:
        logger.exception("github_list_issues_tool failed")
        return tool_error(str(e))


def github_get_pull_request_tool(repository: str, pull_number: int) -> str:
    """Retrieve detailed information about a specific pull request in a GitHub repository."""
    try:
        owner, repo = parse_owner_repo(repository)
        path = f"/repos/{owner}/{repo}/pulls/{pull_number}"
        res_str = github_api_request("GET", path)
        res_data = json.loads(res_str)
        
        if not res_data.get("success"):
            return res_str
        if not res_data.get("ok"):
            return tool_error(get_github_error_message(res_data), status=res_data.get("status"))
            
        pr = res_data.get("json")
        if not isinstance(pr, dict):
            return tool_error("Invalid response payload from GitHub API")
            
        formatted = {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "author": pr.get("user", {}).get("login") if isinstance(pr.get("user"), dict) else None,
            "draft": pr.get("draft", False),
            "merged": pr.get("merged", False),
            "base": {
                "ref": pr.get("base", {}).get("ref") if isinstance(pr.get("base"), dict) else None,
                "sha": pr.get("base", {}).get("sha") if isinstance(pr.get("base"), dict) else None,
            },
            "head": {
                "ref": pr.get("head", {}).get("ref") if isinstance(pr.get("head"), dict) else None,
                "sha": pr.get("head", {}).get("sha") if isinstance(pr.get("head"), dict) else None,
            },
            "html_url": pr.get("html_url"),
            "body": pr.get("body"),
        }
        return tool_result(formatted)
    except Exception as e:
        logger.exception("github_get_pull_request_tool failed")
        return tool_error(str(e))


def github_list_pull_requests_tool(repository: str, state: str = "open", per_page: int = 30) -> str:
    """List pull requests in a GitHub repository."""
    try:
        owner, repo = parse_owner_repo(repository)
        path = f"/repos/{owner}/{repo}/pulls"
        query = {"state": state, "per_page": per_page}
        res_str = github_api_request("GET", path, query=query)
        res_data = json.loads(res_str)
        
        if not res_data.get("success"):
            return res_str
        if not res_data.get("ok"):
            return tool_error(get_github_error_message(res_data), status=res_data.get("status"))
            
        pulls = res_data.get("json")
        if not isinstance(pulls, list):
            return tool_error("Invalid response payload from GitHub API")
            
        formatted = []
        for pr in pulls:
            if not isinstance(pr, dict):
                continue
            formatted.append({
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "author": pr.get("user", {}).get("login") if isinstance(pr.get("user"), dict) else None,
                "html_url": pr.get("html_url"),
                "draft": pr.get("draft", False),
            })
        return tool_result(formatted)
    except Exception as e:
        logger.exception("github_list_pull_requests_tool failed")
        return tool_error(str(e))


def github_add_issue_comment_tool(repository: str, issue_number: int, body: str) -> str:
    """Add a comment to an issue in a GitHub repository."""
    try:
        owner, repo = parse_owner_repo(repository)
        path = f"/repos/{owner}/{repo}/issues/{issue_number}/comments"
        json_payload = {"body": body}
        res_str = github_api_request("POST", path, json_body=json_payload)
        res_data = json.loads(res_str)
        
        if not res_data.get("success"):
            return res_str
        if not res_data.get("ok"):
            return tool_error(get_github_error_message(res_data), status=res_data.get("status"))
            
        comment = res_data.get("json")
        if not isinstance(comment, dict):
            return tool_error("Invalid response payload from GitHub API")
            
        formatted = {
            "id": comment.get("id"),
            "html_url": comment.get("html_url"),
            "body": comment.get("body"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "author": comment.get("user", {}).get("login") if isinstance(comment.get("user"), dict) else None,
        }
        return tool_result(formatted)
    except Exception as e:
        logger.exception("github_add_issue_comment_tool failed")
        return tool_error(str(e))


# -----------------------------------------------------------------------------------------------------------
# Schemas & Handlers for Registry
# ---------------------------------------------------------------------------

GITHUB_GET_ISSUE_SCHEMA = {
    "name": "github_get_issue",
    "description": "Retrieve detailed information about a specific issue in a GitHub repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository identifier, either in 'owner/repo' format or as a full GitHub URL.",
            },
            "issue_number": {
                "type": "integer",
                "description": "The issue number to retrieve.",
            },
        },
        "required": ["repository", "issue_number"],
    },
}

GITHUB_LIST_ISSUES_SCHEMA = {
    "name": "github_list_issues",
    "description": "List issues in a GitHub repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository identifier, either in 'owner/repo' format or as a full GitHub URL.",
            },
            "state": {
                "type": "string",
                "description": "Filter by issue state.",
                "enum": ["open", "closed", "all"],
                "default": "open",
            },
            "per_page": {
                "type": "integer",
                "description": "Number of items to retrieve per page (max 100, default 30).",
                "default": 30,
            },
        },
        "required": ["repository"],
    },
}

GITHUB_GET_PULL_REQUEST_SCHEMA = {
    "name": "github_get_pull_request",
    "description": "Retrieve detailed information about a specific pull request in a GitHub repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository identifier, either in 'owner/repo' format or as a full GitHub URL.",
            },
            "pull_number": {
                "type": "integer",
                "description": "The pull request number to retrieve.",
            },
        },
        "required": ["repository", "pull_number"],
    },
}

GITHUB_LIST_PULL_REQUESTS_SCHEMA = {
    "name": "github_list_pull_requests",
    "description": "List pull requests in a GitHub repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository identifier, either in 'owner/repo' format or as a full GitHub URL.",
            },
            "state": {
                "type": "string",
                "description": "Filter by pull request state.",
                "enum": ["open", "closed", "all"],
                "default": "open",
            },
            "per_page": {
                "type": "integer",
                "description": "Number of items to retrieve per page (max 100, default 30).",
                "default": 30,
            },
        },
        "required": ["repository"],
    },
}

GITHUB_ADD_ISSUE_COMMENT_SCHEMA = {
    "name": "github_add_issue_comment",
    "description": "Add a comment to an issue in a GitHub repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Repository identifier, either in 'owner/repo' format or as a full GitHub URL.",
            },
            "issue_number": {
                "type": "integer",
                "description": "The issue number to add the comment to.",
            },
            "body": {
                "type": "string",
                "description": "The comment text body (Markdown supported).",
            },
        },
        "required": ["repository", "issue_number", "body"],
    },
}


def _handle_github_get_issue(args: dict, **kwargs) -> str:
    return github_get_issue_tool(
        repository=args.get("repository", ""),
        issue_number=int(args.get("issue_number")),
    )


def _handle_github_list_issues(args: dict, **kwargs) -> str:
    return github_list_issues_tool(
        repository=args.get("repository", ""),
        state=args.get("state", "open"),
        per_page=int(args.get("per_page", 30)),
    )


def _handle_github_get_pull_request(args: dict, **kwargs) -> str:
    return github_get_pull_request_tool(
        repository=args.get("repository", ""),
        pull_number=int(args.get("pull_number")),
    )


def _handle_github_list_pull_requests(args: dict, **kwargs) -> str:
    return github_list_pull_requests_tool(
        repository=args.get("repository", ""),
        state=args.get("state", "open"),
        per_page=int(args.get("per_page", 30)),
    )


def _handle_github_add_issue_comment(args: dict, **kwargs) -> str:
    return github_add_issue_comment_tool(
        repository=args.get("repository", ""),
        issue_number=int(args.get("issue_number")),
        body=args.get("body", ""),
    )


# Register tools under the "github" toolset
registry.register(
    name="github_get_issue",
    toolset="github",
    schema=GITHUB_GET_ISSUE_SCHEMA,
    handler=_handle_github_get_issue,
    check_fn=check_github_requirements,
    requires_env=[GITHUB_TOKEN_ENV_VAR],
    emoji="🐙",
)

registry.register(
    name="github_list_issues",
    toolset="github",
    schema=GITHUB_LIST_ISSUES_SCHEMA,
    handler=_handle_github_list_issues,
    check_fn=check_github_requirements,
    requires_env=[GITHUB_TOKEN_ENV_VAR],
    emoji="🐙",
)

registry.register(
    name="github_get_pull_request",
    toolset="github",
    schema=GITHUB_GET_PULL_REQUEST_SCHEMA,
    handler=_handle_github_get_pull_request,
    check_fn=check_github_requirements,
    requires_env=[GITHUB_TOKEN_ENV_VAR],
    emoji="🐙",
)

registry.register(
    name="github_list_pull_requests",
    toolset="github",
    schema=GITHUB_LIST_PULL_REQUESTS_SCHEMA,
    handler=_handle_github_list_pull_requests,
    check_fn=check_github_requirements,
    requires_env=[GITHUB_TOKEN_ENV_VAR],
    emoji="🐙",
)

registry.register(
    name="github_add_issue_comment",
    toolset="github",
    schema=GITHUB_ADD_ISSUE_COMMENT_SCHEMA,
    handler=_handle_github_add_issue_comment,
    check_fn=check_github_requirements,
    requires_env=[GITHUB_TOKEN_ENV_VAR],
    emoji="🐙",
)
