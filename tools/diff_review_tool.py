#!/usr/bin/env python3
"""
Diff Review Tool - Automated code diff analysis for hermes-agent.

Runs `git diff` (or accepts a raw diff string) and returns a structured
review: changed files, line counts, and flagged patterns such as debug
prints, hardcoded secrets, and missing error handling.
"""

import json
import logging
import re
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patterns that warrant a warning in the review
# ---------------------------------------------------------------------------
_WARN_PATTERNS: List[tuple] = [
    (r"print\s*\(", "debug print statement"),
    (r"console\.log\s*\(", "console.log statement"),
    (r"(?i)(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]", "possible hardcoded secret"),
    (r"except\s*:", "bare except clause"),
    (r"except\s+Exception\s*:", "broad Exception catch"),
    (r"TODO|FIXME|HACK|XXX", "unresolved TODO/FIXME"),
]


def _get_git_diff(base: str = "HEAD", target: Optional[str] = None) -> str:
    """Run git diff and return the output as a string."""
    cmd = ["git", "diff", base]
    if target:
        cmd.append(target)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("git diff exited with code %d: %s", result.returncode, result.stderr)
            return ""
        return result.stdout
    except FileNotFoundError:
        logger.error("git not found in PATH")
        return ""
    except subprocess.TimeoutExpired:
        logger.error("git diff timed out")
        return ""


def _parse_diff(diff_text: str) -> Dict[str, Any]:
    """Parse a unified diff into structured data."""
    files: List[Dict[str, Any]] = []
    current_file: Optional[Dict[str, Any]] = None
    warnings: List[Dict[str, str]] = []
    total_added = 0
    total_removed = 0
    line_number = 0

    for raw_line in diff_text.splitlines():
        # New file section
        if raw_line.startswith("diff --git"):
            if current_file:
                files.append(current_file)
            current_file = {
                "path": "",
                "added": 0,
                "removed": 0,
            }
            continue

        if raw_line.startswith("+++ b/") and current_file is not None:
            current_file["path"] = raw_line[6:]
            continue

        if raw_line.startswith("@@"):
            # Extract new-file line number from @@ -a,b +c,d @@
            match = re.search(r"\+(\d+)", raw_line)
            if match:
                line_number = int(match.group(1)) - 1
            continue

        if current_file is None:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            line_number += 1
            current_file["added"] += 1
            total_added += 1
            added_content = raw_line[1:]
            for pattern, label in _WARN_PATTERNS:
                if re.search(pattern, added_content):
                    warnings.append({
                        "file": current_file.get("path", "unknown"),
                        "line": line_number,
                        "issue": label,
                    })

        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            current_file["removed"] += 1
            total_removed += 1
        else:
            if not raw_line.startswith(("---", "+++")):
                line_number += 1

    if current_file:
        files.append(current_file)

    return {
        "files_changed": len(files),
        "total_added": total_added,
        "total_removed": total_removed,
        "files": files,
        "warnings": warnings,
        "warning_count": len(warnings),
    }


def diff_review_tool(
    base: str = "HEAD",
    target: Optional[str] = None,
    diff_text: Optional[str] = None,
) -> str:
    """
    Review a git diff for common issues.

    Args:
        base: base git ref (default: HEAD — shows working tree changes).
        target: optional target ref (e.g. 'origin/main').
        diff_text: raw unified diff string. If provided, skips git diff.

    Returns:
        JSON string with structured review results.
    """
    if diff_text is None:
        diff_text = _get_git_diff(base=base, target=target)

    if not diff_text.strip():
        return json.dumps({
            "status": "no_diff",
            "message": "No differences found.",
            "files_changed": 0,
            "total_added": 0,
            "total_removed": 0,
            "files": [],
            "warnings": [],
            "warning_count": 0,
        }, ensure_ascii=False)

    review = _parse_diff(diff_text)
    review["status"] = "ok"
    return json.dumps(review, ensure_ascii=False, indent=2)


def check_diff_review_requirements() -> bool:
    """Requires git to be available in PATH."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DIFF_REVIEW_SCHEMA = {
    "name": "diff_review",
    "description": (
        "Review a git diff for common code quality issues. "
        "Detects added/removed lines per file, flags debug prints, "
        "hardcoded secrets, bare except clauses, and unresolved TODOs. "
        "By default diffs the working tree against HEAD. "
        "Pass target='origin/main' to review changes before a PR."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base": {
                "type": "string",
                "description": "Base git ref to diff from (default: HEAD).",
                "default": "HEAD",
            },
            "target": {
                "type": "string",
                "description": "Target git ref (optional). E.g. 'origin/main'.",
            },
            "diff_text": {
                "type": "string",
                "description": "Raw unified diff string. If provided, skips running git.",
            },
        },
        "required": [],
    },
}

# --- Registry ---
from tools.registry import registry

registry.register(
    name="diff_review",
    toolset="diff_review",
    schema=DIFF_REVIEW_SCHEMA,
    handler=lambda args, **kw: diff_review_tool(
        base=args.get("base", "HEAD"),
        target=args.get("target"),
        diff_text=args.get("diff_text"),
    ),
    check_fn=check_diff_review_requirements,
    emoji="🔍",
)
