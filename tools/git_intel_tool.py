"""
git_intel_tool.py — Git repository intelligence toolset for Hermes Agent

Pure Python stdlib only: subprocess, os, json, datetime, collections.
No external dependencies. No API keys required.
Works with any local git repository.

Tools:
  git_repo_summary     — Overview of repo (branches, remotes, latest commit, stats)
  git_log              — Commit history with filters (author, date, limit, path)
  git_diff_stats       — Line-level diff statistics between any two refs
  git_contributors     — Contributor leaderboard with commit counts and line stats
  git_file_history     — Full history and blame summary for a specific file
  git_branch_compare   — What separates two branches (ahead/behind + unique commits)
"""

import subprocess
import os
import json
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out after 30 seconds", 1
    except FileNotFoundError:
        return "", "git not found — please install git", 1
    except Exception as e:
        return "", str(e), 1


def _find_repo(path: str) -> str | None:
    """Walk up from path to find the .git directory."""
    path = os.path.abspath(path)
    while True:
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _validate_repo(repo_path: str) -> tuple[str | None, str | None]:
    """Return (resolved_repo_path, error_message)."""
    resolved = _find_repo(repo_path)
    if not resolved:
        return None, f"No git repository found at or above: {repo_path}"
    return resolved, None


def _parse_log_line(line: str) -> dict:
    """Parse a single --format line from git log."""
    parts = line.split("\x1f")
    if len(parts) < 6:
        return {}
    return {
        "hash": parts[0],
        "hash_short": parts[1],
        "author": parts[2],
        "email": parts[3],
        "date": parts[4],
        "subject": parts[5],
    }


# ---------------------------------------------------------------------------
# Tool 1: git_repo_summary
# ---------------------------------------------------------------------------

def git_repo_summary(repo_path: str = ".") -> dict:
    """
    Return a high-level summary of a git repository.

    Args:
        repo_path: Path to the repository (or any subdirectory). Defaults to cwd.

    Returns:
        dict with keys: repo_path, current_branch, branches, remotes,
        latest_commit, total_commits, contributors_count, tracked_files,
        repo_size_kb, tags_count, is_dirty
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    result = {}

    # Current branch
    out, _, rc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    result["current_branch"] = out if rc == 0 else "unknown"

    # All branches
    out, _, rc = _run(["git", "branch", "-a", "--format=%(refname:short)"], cwd=repo)
    branches = [b for b in out.splitlines() if b] if rc == 0 else []
    result["branches"] = branches
    result["branch_count"] = len(branches)

    # Remotes
    out, _, rc = _run(["git", "remote", "-v"], cwd=repo)
    remotes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "(fetch)" in line:
            remotes[parts[0]] = parts[1]
    result["remotes"] = remotes

    # Latest commit
    fmt = "%H\x1f%h\x1f%an\x1f%ae\x1f%ai\x1f%s"
    out, _, rc = _run(["git", "log", "-1", f"--format={fmt}"], cwd=repo)
    if rc == 0 and out:
        parsed = _parse_log_line(out)
        result["latest_commit"] = parsed
    else:
        result["latest_commit"] = None

    # Total commits
    out, _, rc = _run(["git", "rev-list", "--count", "HEAD"], cwd=repo)
    result["total_commits"] = int(out) if rc == 0 and out.isdigit() else 0

    # Contributors count
    out, _, rc = _run(["git", "shortlog", "-sn", "--no-merges", "HEAD"], cwd=repo)
    result["contributors_count"] = len(out.splitlines()) if rc == 0 else 0

    # Tracked files count
    out, _, rc = _run(["git", "ls-files"], cwd=repo)
    result["tracked_files"] = len(out.splitlines()) if rc == 0 else 0

    # Tags
    out, _, rc = _run(["git", "tag"], cwd=repo)
    result["tags_count"] = len(out.splitlines()) if rc == 0 and out else 0

    # Dirty working tree?
    out, _, rc = _run(["git", "status", "--porcelain"], cwd=repo)
    result["is_dirty"] = bool(out) if rc == 0 else False

    # Repo size (approximate via .git folder)
    git_dir = os.path.join(repo, ".git")
    total = 0
    for dirpath, _, filenames in os.walk(git_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    result["repo_size_kb"] = round(total / 1024, 1)

    result["repo_path"] = repo
    return result


# ---------------------------------------------------------------------------
# Tool 2: git_log
# ---------------------------------------------------------------------------

def git_log(
    repo_path: str = ".",
    limit: int = 20,
    author: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    path_filter: Optional[str] = None,
    branch: Optional[str] = "HEAD",
    no_merges: bool = True,
) -> dict:
    """
    Retrieve commit history with optional filters.

    Args:
        repo_path:    Path to repo or subdirectory.
        limit:        Max number of commits to return (1-500, default 20).
        author:       Filter by author name or email substring.
        since:        Only commits after this date (e.g. "2024-01-01", "2 weeks ago").
        until:        Only commits before this date.
        path_filter:  Limit to commits touching this file or directory.
        branch:       Branch/ref to log (default HEAD).
        no_merges:    Exclude merge commits (default True).

    Returns:
        dict with keys: commits (list), count, filters_applied
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    limit = max(1, min(limit, 500))

    fmt = "%H\x1f%h\x1f%an\x1f%ae\x1f%ai\x1f%s"
    cmd = ["git", "log", f"--format={fmt}", f"-{limit}", branch]

    if no_merges:
        cmd.append("--no-merges")
    if author:
        cmd.extend(["--author", author])
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])
    if path_filter:
        cmd.extend(["--", path_filter])

    out, err_msg, rc = _run(cmd, cwd=repo)
    if rc != 0:
        return {"error": err_msg or "git log failed"}

    commits = []
    for line in out.splitlines():
        parsed = _parse_log_line(line)
        if parsed:
            commits.append(parsed)

    return {
        "commits": commits,
        "count": len(commits),
        "filters_applied": {
            "author": author,
            "since": since,
            "until": until,
            "path_filter": path_filter,
            "branch": branch,
            "no_merges": no_merges,
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: git_diff_stats
# ---------------------------------------------------------------------------

def git_diff_stats(
    repo_path: str = ".",
    base: str = "HEAD~1",
    target: str = "HEAD",
    path_filter: Optional[str] = None,
) -> dict:
    """
    Show line-level diff statistics between two git refs.

    Args:
        repo_path:   Path to repo or subdirectory.
        base:        Base ref (commit hash, branch, tag). Default: HEAD~1.
        target:      Target ref. Default: HEAD.
        path_filter: Limit diff to this file or directory.

    Returns:
        dict with keys: base, target, files_changed, insertions, deletions,
        net_change, file_stats (list of per-file stats), summary
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    cmd = ["git", "diff", "--stat", "--stat-width=200", base, target]
    if path_filter:
        cmd.extend(["--", path_filter])

    out, err_msg, rc = _run(cmd, cwd=repo)
    if rc != 0:
        return {"error": err_msg or f"Could not diff {base}..{target}"}

    # Also get machine-readable numstat
    cmd2 = ["git", "diff", "--numstat", base, target]
    if path_filter:
        cmd2.extend(["--", path_filter])
    numstat_out, _, _ = _run(cmd2, cwd=repo)

    file_stats = []
    total_ins = 0
    total_del = 0

    for line in numstat_out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            ins_str, del_str, fname = parts
            ins = int(ins_str) if ins_str.isdigit() else 0
            dels = int(del_str) if del_str.isdigit() else 0
            total_ins += ins
            total_del += dels
            file_stats.append({
                "file": fname,
                "insertions": ins,
                "deletions": dels,
                "net": ins - dels,
            })

    # Sort by total change descending
    file_stats.sort(key=lambda x: x["insertions"] + x["deletions"], reverse=True)

    return {
        "base": base,
        "target": target,
        "files_changed": len(file_stats),
        "insertions": total_ins,
        "deletions": total_del,
        "net_change": total_ins - total_del,
        "file_stats": file_stats,
        "summary": out.splitlines()[-1] if out else "",
    }


# ---------------------------------------------------------------------------
# Tool 4: git_contributors
# ---------------------------------------------------------------------------

def git_contributors(
    repo_path: str = ".",
    branch: str = "HEAD",
    limit: int = 20,
    since: Optional[str] = None,
) -> dict:
    """
    Generate a contributor leaderboard with commit counts and line statistics.

    Args:
        repo_path: Path to repo or subdirectory.
        branch:    Branch/ref to analyze (default HEAD).
        limit:     Max contributors to return (default 20).
        since:     Only count commits after this date (e.g. "6 months ago").

    Returns:
        dict with keys: contributors (list sorted by commits desc),
        total_contributors, total_commits, period
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    limit = max(1, min(limit, 100))

    # Get commit counts per author
    cmd = ["git", "shortlog", "-sne", "--no-merges", branch]
    if since:
        cmd.extend(["--since", since])
    out, err_msg, rc = _run(cmd, cwd=repo)
    if rc != 0:
        return {"error": err_msg or "git shortlog failed"}

    contributors = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "  42\tName <email>"
        parts = line.split("\t", 1)
        if len(parts) == 2:
            count_str = parts[0].strip()
            name_email = parts[1].strip()
            count = int(count_str) if count_str.isdigit() else 0
            # Split name and email
            if "<" in name_email:
                name = name_email.split("<")[0].strip()
                email = name_email.split("<")[1].rstrip(">")
            else:
                name = name_email
                email = ""
            contributors.append({"name": name, "email": email, "commits": count})

    total_commits = sum(c["commits"] for c in contributors)

    # Add line stats for top contributors (expensive, limit to top 10)
    for contrib in contributors[:10]:
        cmd_lines = [
            "git", "log", "--author", contrib["email"] or contrib["name"],
            "--no-merges", "--numstat", "--format=", branch,
        ]
        if since:
            cmd_lines.extend(["--since", since])
        lines_out, _, lines_rc = _run(cmd_lines, cwd=repo)
        ins_total = 0
        del_total = 0
        if lines_rc == 0:
            for l in lines_out.splitlines():
                parts = l.split("\t")
                if len(parts) == 3:
                    i = int(parts[0]) if parts[0].isdigit() else 0
                    d = int(parts[1]) if parts[1].isdigit() else 0
                    ins_total += i
                    del_total += d
        contrib["lines_added"] = ins_total
        contrib["lines_removed"] = del_total
        contrib["lines_net"] = ins_total - del_total
        contrib["commit_pct"] = round(contrib["commits"] / total_commits * 100, 1) if total_commits else 0

    return {
        "contributors": contributors[:limit],
        "total_contributors": len(contributors),
        "total_commits": total_commits,
        "period": f"since {since}" if since else "all time",
        "branch": branch,
    }


# ---------------------------------------------------------------------------
# Tool 5: git_file_history
# ---------------------------------------------------------------------------

def git_file_history(
    file_path: str,
    repo_path: str = ".",
    limit: int = 30,
    show_blame_summary: bool = True,
) -> dict:
    """
    Show the complete commit history for a specific file, with optional blame summary.

    Args:
        file_path:          Path to the file (relative to repo root or absolute).
        repo_path:          Path to repo or any subdirectory.
        limit:              Max commits to return (default 30).
        show_blame_summary: Include a blame summary (lines per author).

    Returns:
        dict with keys: file, commits, total_commits, blame_summary (if requested),
        first_commit, latest_commit, authors
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    # Resolve file path relative to repo
    if os.path.isabs(file_path):
        rel_path = os.path.relpath(file_path, repo)
    else:
        rel_path = file_path

    limit = max(1, min(limit, 500))

    fmt = "%H\x1f%h\x1f%an\x1f%ae\x1f%ai\x1f%s"
    cmd = ["git", "log", f"--format={fmt}", f"-{limit}", "--follow", "--", rel_path]
    out, err_msg, rc = _run(cmd, cwd=repo)
    if rc != 0:
        return {"error": err_msg or f"Could not get history for {rel_path}"}

    commits = []
    authors = defaultdict(int)
    for line in out.splitlines():
        parsed = _parse_log_line(line)
        if parsed:
            commits.append(parsed)
            authors[parsed["author"]] += 1

    result = {
        "file": rel_path,
        "commits": commits,
        "total_commits": len(commits),
        "authors": dict(sorted(authors.items(), key=lambda x: -x[1])),
        "first_commit": commits[-1] if commits else None,
        "latest_commit": commits[0] if commits else None,
    }

    # Blame summary
    if show_blame_summary and commits:
        blame_cmd = ["git", "blame", "--line-porcelain", "--", rel_path]
        blame_out, _, blame_rc = _run(blame_cmd, cwd=repo)
        if blame_rc == 0:
            blame_counts = defaultdict(int)
            for line in blame_out.splitlines():
                if line.startswith("author "):
                    blame_counts[line[7:]] += 1
            result["blame_summary"] = dict(
                sorted(blame_counts.items(), key=lambda x: -x[1])
            )
        else:
            result["blame_summary"] = {}

    return result


# ---------------------------------------------------------------------------
# Tool 6: git_branch_compare
# ---------------------------------------------------------------------------

def git_branch_compare(
    repo_path: str = ".",
    base: str = "main",
    target: str = "HEAD",
) -> dict:
    """
    Compare two branches: how far ahead/behind they are and what unique commits each has.

    Args:
        repo_path: Path to repo or subdirectory.
        base:      Base branch/ref (default "main").
        target:    Target branch/ref to compare against base (default "HEAD").

    Returns:
        dict with keys: base, target, ahead, behind,
        commits_ahead (list), commits_behind (list),
        common_ancestor, diff_stats
    """
    repo, err = _validate_repo(repo_path)
    if err:
        return {"error": err}

    # Find merge base
    out, err_msg, rc = _run(["git", "merge-base", base, target], cwd=repo)
    if rc != 0:
        return {"error": f"Cannot find common ancestor of {base} and {target}: {err_msg}"}
    common_ancestor = out

    # Ahead: commits in target but not in base
    fmt = "%H\x1f%h\x1f%an\x1f%ae\x1f%ai\x1f%s"
    out, _, rc = _run(
        ["git", "log", f"--format={fmt}", "--no-merges", f"{base}..{target}"],
        cwd=repo,
    )
    commits_ahead = []
    if rc == 0 and out:
        for line in out.splitlines():
            parsed = _parse_log_line(line)
            if parsed:
                commits_ahead.append(parsed)

    # Behind: commits in base but not in target
    out, _, rc = _run(
        ["git", "log", f"--format={fmt}", "--no-merges", f"{target}..{base}"],
        cwd=repo,
    )
    commits_behind = []
    if rc == 0 and out:
        for line in out.splitlines():
            parsed = _parse_log_line(line)
            if parsed:
                commits_behind.append(parsed)

    # Diff stats between the two
    diff = git_diff_stats(repo_path=repo, base=base, target=target)

    return {
        "base": base,
        "target": target,
        "common_ancestor": common_ancestor[:12],
        "ahead": len(commits_ahead),
        "behind": len(commits_behind),
        "commits_ahead": commits_ahead,
        "commits_behind": commits_behind,
        "diff_stats": diff,
        "verdict": (
            "up to date" if not commits_ahead and not commits_behind
            else f"{target} is {len(commits_ahead)} ahead, {len(commits_behind)} behind {base}"
        ),
    }


# ---------------------------------------------------------------------------
# Hermes tool registry shims
# (These are the functions Hermes will call via its tool dispatch)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "git_repo_summary",
        "description": (
            "Get a high-level overview of a git repository: current branch, all branches, "
            "remotes, latest commit, total commits, contributor count, tracked file count, "
            "repo size, tags, and whether the working tree is dirty. "
            "Use this to quickly understand any git repo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the git repository or any subdirectory. Defaults to current working directory.",
                    "default": ".",
                }
            },
            "required": [],
        },
    },
    {
        "name": "git_log",
        "description": (
            "Retrieve commit history for a git repository with optional filters. "
            "Filter by author, date range, branch, or file path. "
            "Returns structured commit data: hash, author, date, subject."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repo.", "default": "."},
                "limit": {"type": "integer", "description": "Max commits to return (1-500).", "default": 20},
                "author": {"type": "string", "description": "Filter by author name or email substring."},
                "since": {"type": "string", "description": "Only commits after this date, e.g. '2024-01-01' or '2 weeks ago'."},
                "until": {"type": "string", "description": "Only commits before this date."},
                "path_filter": {"type": "string", "description": "Only commits touching this file or directory."},
                "branch": {"type": "string", "description": "Branch or ref to log.", "default": "HEAD"},
                "no_merges": {"type": "boolean", "description": "Exclude merge commits.", "default": True},
            },
            "required": [],
        },
    },
    {
        "name": "git_diff_stats",
        "description": (
            "Show line-level diff statistics between any two git refs (commits, branches, tags). "
            "Returns files changed, insertions, deletions, net change, and per-file breakdown. "
            "Useful for understanding the scope of a PR, release, or set of changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repo.", "default": "."},
                "base": {"type": "string", "description": "Base ref (commit, branch, tag).", "default": "HEAD~1"},
                "target": {"type": "string", "description": "Target ref.", "default": "HEAD"},
                "path_filter": {"type": "string", "description": "Limit diff to this file or directory."},
            },
            "required": [],
        },
    },
    {
        "name": "git_contributors",
        "description": (
            "Generate a contributor leaderboard for a git repository. "
            "Shows commit counts, lines added/removed, and contribution percentage per author. "
            "Can be filtered by time period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repo.", "default": "."},
                "branch": {"type": "string", "description": "Branch to analyze.", "default": "HEAD"},
                "limit": {"type": "integer", "description": "Max contributors to return.", "default": 20},
                "since": {"type": "string", "description": "Only count commits after this date, e.g. '6 months ago'."},
            },
            "required": [],
        },
    },
    {
        "name": "git_file_history",
        "description": (
            "Show the full commit history for a specific file in a git repository. "
            "Includes all commits that touched the file, author breakdown, "
            "and an optional blame summary showing lines-per-author."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (relative to repo root or absolute)."},
                "repo_path": {"type": "string", "description": "Path to repo.", "default": "."},
                "limit": {"type": "integer", "description": "Max commits to return.", "default": 30},
                "show_blame_summary": {"type": "boolean", "description": "Include blame summary (lines per author).", "default": True},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "git_branch_compare",
        "description": (
            "Compare two git branches or refs. "
            "Shows how many commits each side has that the other doesn't (ahead/behind), "
            "lists the unique commits on each side, and provides diff statistics. "
            "Useful for understanding PR scope or branch divergence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repo.", "default": "."},
                "base": {"type": "string", "description": "Base branch/ref.", "default": "main"},
                "target": {"type": "string", "description": "Target branch/ref to compare.", "default": "HEAD"},
            },
            "required": [],
        },
    },
]
