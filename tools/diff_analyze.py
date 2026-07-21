#!/usr/bin/env python3
"""
Diff Analyze Tool - Analyze Git diffs

Provides Git diff analysis: generate diffs, summarize changes, and identify file types.
"""

import json
import os
import subprocess
import re
from typing import Dict, List, Optional


def _run_git(args: List[str], cwd: Optional[str] = None) -> Dict[str, any]:
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _parse_numstat(numstat_output: str) -> Dict[str, int]:
    """Parse git diff --numstat output for accurate per-file line counts.

    Each line: ``insertions\\tdeletions\\tpath``
    Binary files show ``-\\t-\\tpath``.
    """
    stats = {"files_changed": 0, "insertions": 0, "deletions": 0}
    for line in numstat_output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins_str, dels_str = parts[0], parts[1]
        if ins_str == "-" or dels_str == "-":
            # binary file — count as 1 changed file, 0 lines
            stats["files_changed"] += 1
            continue
        try:
            stats["insertions"] += int(ins_str)
            stats["deletions"] += int(dels_str)
            stats["files_changed"] += 1
        except ValueError:
            continue
    return stats


def _analyze_diff_details(diff_output: str) -> List[Dict[str, str]]:
    """Analyze individual file changes in diff."""
    files = []
    current_file = None
    current_changes = {"additions": 0, "deletions": 0}
    
    for line in diff_output.split("\n"):
        if line.startswith("diff --git"):
            if current_file:
                files.append({
                    "file": current_file,
                    "additions": current_changes["additions"],
                    "deletions": current_changes["deletions"],
                })
            match = re.search(r"diff --git a/(.+) b/(.+)", line)
            if match:
                current_file = match.group(2) if match.group(2) != "/dev/null" else match.group(1)
            current_changes = {"additions": 0, "deletions": 0}
        elif line.startswith("+") and not line.startswith("+++"):
            current_changes["additions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current_changes["deletions"] += 1
    
    if current_file:
        files.append({
            "file": current_file,
            "additions": current_changes["additions"],
            "deletions": current_changes["deletions"],
        })
    
    return files


def diff_analyze(
    base: Optional[str] = None,
    target: Optional[str] = None,
    file_filter: Optional[str] = None,
    summary: bool = True,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Analyze Git diffs between commits, branches, or working tree.

    Args:
        base: Base commit/branch (default: working tree vs HEAD)
        target: Target commit/branch (default: HEAD)
        file_filter: Filter by file pattern (e.g., *.py)
        summary: Return summary only instead of full diff
        cwd: Working directory for git operation
        task_id: Optional task ID for tracking

    Returns:
        JSON string with diff analysis results
    """
    if cwd:
        if not os.path.isdir(cwd):
            return json.dumps({
                "success": False,
                "error": f"Working directory not found: {cwd}",
            })
        if not os.access(cwd, os.R_OK):
            return json.dumps({
                "success": False,
                "error": f"Working directory not readable: {cwd}",
            })
    
    repo_root = cwd or os.getcwd()

    diff_output = ""
    numstat_output = ""

    if base and target:
        range_spec = f"{base}..{target}"
    elif base:
        range_spec = f"{base}..HEAD"
    else:
        range_spec = ""

    # Fetch unified diff for context / file_changes
    if range_spec:
        result = _run_git(["diff", "--no-color", range_spec], repo_root)
    else:
        result = _run_git(["diff", "--no-color"], repo_root)
        if not result["success"] or not result.get("stdout"):
            result = _run_git(["diff", "--no-color", "HEAD"], repo_root)
    diff_output = result.get("stdout", "") if result["success"] else ""

    # Fetch --numstat for accurate line-count stats
    if range_spec:
        numstat_result = _run_git(["diff", "--numstat", range_spec], repo_root)
    else:
        numstat_result = _run_git(["diff", "--numstat"], repo_root)
        if not numstat_result["success"] or not numstat_result.get("stdout"):
            numstat_result = _run_git(["diff", "--numstat", "HEAD"], repo_root)
    numstat_output = numstat_result.get("stdout", "") if numstat_result["success"] else ""

    if file_filter and diff_output:
        filtered_lines = []
        pattern = file_filter.replace("*", ".*").replace("?", ".")
        for line in diff_output.split("\n"):
            if re.search(pattern, line) or line.startswith("diff") or line.startswith("index"):
                filtered_lines.append(line)
            elif any(ext in line for ext in file_filter.split(",")):
                filtered_lines.append(line)
        diff_output = "\n".join(filtered_lines)

    stats = _parse_numstat(numstat_output) if numstat_output else {"files_changed": 0, "insertions": 0, "deletions": 0}
    
    file_changes = _analyze_diff_details(diff_output) if not summary else []
    
    file_types: Dict[str, int] = {}
    for fc in file_changes:
        ext = os.path.splitext(fc["file"])[1] if "." in fc["file"] else "no_ext"
        file_types[ext] = file_types.get(ext, 0) + 1
    
    response = {
        "success": True,
        "base": base or "HEAD",
        "target": target or "working tree",
        "stats": stats,
        "file_types": file_types,
    }
    
    if summary:
        response["summary"] = f"{stats['files_changed']} files changed, {stats['insertions']} insertions, {stats['deletions']} deletions"
    else:
        response["diff"] = diff_output[:50000] if len(diff_output) > 50000 else diff_output
        response["file_changes"] = file_changes[:100]
    
    return json.dumps(response, ensure_ascii=False)


def check_diff_analyze_requirements() -> bool:
    """Diff analyze tool requires git to be installed."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


DIFF_ANALYZE_SCHEMA = {
    "name": "diff_analyze",
    "description": (
        "Analyze Git diffs between commits, branches, or working tree. Summarize changes by file, additions, and deletions.\n\n"
        "Parameters:\n"
        "- base: Base commit/branch (default: HEAD)\n"
        "- target: Target commit/branch (default: working tree)\n"
        "- file_filter: Filter by file pattern (e.g., *.py)\n"
        "- summary: Return summary only instead of full diff"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base": {
                "type": "string",
                "description": "Base commit/branch (default: HEAD)",
            },
            "target": {
                "type": "string",
                "description": "Target commit/branch (default: working tree)",
            },
            "file_filter": {
                "type": "string",
                "description": "Filter by file pattern (e.g., *.py, *.js)",
            },
            "summary": {
                "type": "boolean",
                "description": "Return summary only instead of full diff",
                "default": True,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for git operation",
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
    },
}


from tools.registry import registry

registry.register(
    name="diff_analyze",
    toolset="git",
    schema=DIFF_ANALYZE_SCHEMA,
    handler=lambda args, **kw: diff_analyze(
        base=args.get("base"),
        target=args.get("target"),
        file_filter=args.get("file_filter"),
        summary=args.get("summary", True),
        cwd=args.get("cwd"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_diff_analyze_requirements,
    emoji="📊",
)