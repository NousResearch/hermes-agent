"""Terminal Output Economy and Summarization.

Parses command outputs (e.g. pytest, npm, builds) into compact structured summaries,
saving full stdout/stderr streams to ArtifactStore to conserve context window tokens.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from workstation.artifacts import ArtifactRef, ArtifactStore

logger = logging.getLogger(__name__)


def summarize_terminal_output(
    stdout: str,
    exit_code: int = 0,
    *,
    task_id: Optional[str] = None,
    command: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    max_tail_lines: int = 15,
) -> Dict[str, Any]:
    """Produce a token-efficient summary of terminal execution."""
    lines = stdout.splitlines()
    total_lines = len(lines)
    tail = "\n".join(lines[-max_tail_lines:]) if total_lines > max_tail_lines else stdout

    # Parsers for common development tools
    test_summary: Dict[str, Any] = {}

    # Pytest parser
    # Example: 187 passed, 2 failed, 8 skipped in 3.45s
    pytest_match = re.search(r"(=+\s*)?(\d+)\s+passed.*in\s+([\d\.]+s)", stdout)
    if pytest_match:
        test_summary["tool"] = "pytest"
        passed_m = re.search(r"(\d+)\s+passed", stdout)
        failed_m = re.search(r"(\d+)\s+failed", stdout)
        skipped_m = re.search(r"(\d+)\s+skipped", stdout)
        if passed_m:
            test_summary["passed"] = int(passed_m.group(1))
        if failed_m:
            test_summary["failed"] = int(failed_m.group(1))
        if skipped_m:
            test_summary["skipped"] = int(skipped_m.group(1))

    # Generic error extractor
    error_lines = [
        line.strip()
        for line in lines
        if any(err_kw in line.lower() for err_kw in ("error:", "fatal:", "exception:", "failed:", "failure"))
    ][:5]

    stdout_ref = None
    if task_id and len(stdout) > 500:
        store = artifact_store or ArtifactStore()
        cmd_slug = re.sub(r"[^\w]+", "_", (command or "terminal").strip()[:20]).strip("_")
        ref = store.store(
            task_id=task_id,
            name=f"terminal_{cmd_slug}.log",
            content=stdout,
            schema="terminal_output",
        )
        stdout_ref = ref.ref

    summary: Dict[str, Any] = {
        "exit_code": exit_code,
        "success": exit_code == 0,
        "total_lines": total_lines,
        "tail": tail,
    }

    if test_summary:
        summary["test_results"] = test_summary
    if error_lines:
        summary["errors"] = error_lines
    if stdout_ref:
        summary["stdout_ref"] = stdout_ref

    return summary
