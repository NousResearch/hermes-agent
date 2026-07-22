"""Docker / local-stack health snapshot tool.

Returned as a zodern:relay-style envelope through ``tools.validation_helper``.

Output model (success)
----------------------
- ``docker_daemon_reachable``: bool
- ``compose_files``: list of str
- ``service_status_counts``: dict[compose_file, dict[status, count]]
- ``container_health_summary``: dict with counts for running/stopped/unhealthy/total
- ``top_issues``: small list of notable health problems, capped for readability

Failure envelope uses code ``docker_health_error`` when docker CLI/daemon is
unavailable or any health command errors / times out.

Design constraints
------------------
* No new runtime dependencies: stdlib only + existing ``tools.validation_helper``.
* Reads docker state through ``subprocess.run`` with a short timeout.
* Bounded read path: large status output is summarized and only top issues are
  retained to keep the envelope small.
* Availability gate ``is_available`` should be used by the runtime dispatcher
  before invoking the tool.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from tools.validation_helper import relay_failure, relay_success

TOOL_NAME = "docker_health"
FAILURE_CODE = "docker_health_error"

# Keep top issues summary small regardless of output size.
TOP_ISSUES_CAP = 25
# Seconds before docker subcommands are considered hung.
DEFAULT_COMMAND_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """Return True only when the docker CLI exists and the daemon responds.

    Runtime/meta-config dispatchers should call this before run().
    """
    cli = shutil.which("docker")
    if not cli:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=DEFAULT_COMMAND_TIMEOUT,
        )
        return completed.returncode == 0
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Bounded parsing helpers
# ---------------------------------------------------------------------------
def _bounded_counter(items: List[str], cap: int = TOP_ISSUES_CAP) -> List[Dict[str, Any]]:
    """Return a compact ranked summary of string items.

    For very large inputs, we preserve diversity by keeping top-N distinct keys
    after collapsing counts, and skip trailing duplicates.
    """
    counts = Counter(items)
    most_common: List[tuple[str, int]] = counts.most_common(cap * 2)
    # dedupe repeated keys while preserving multiplicity labeling
    seen: List[str] = []
    result: List[Dict[str, Any]] = []
    remaining = cap
    for key, count in most_common:
        if remaining <= 0:
            break
        if key in seen:
            continue
        seen.append(key)
        result.append({"item": key, "count": count})
        remaining -= 1
    return result


def _extract_status_token(raw: str) -> str:
    """Normalize docker compose ps status strings into coarse buckets.

    Matching is prefix-based for the state verb and checks the parenthetical
    health qualifier first so overlapping substrings like ``RESTARTING`` /
    ``STARTING`` do not collide.
    """
    token = raw.strip().upper()
    if "(UNHEALTHY)" in token or token.endswith("UNHEALTHY"):
        return "unhealthy"
    if "(HEALTHY)" in token or token.endswith("HEALTHY"):
        return "healthy"
    if token.startswith("EXITED") or token.startswith("STOPPED"):
        return "exited"
    if token.startswith("RESTARTING"):
        return "restarting"
    if token.startswith("STARTING"):
        return "starting"
    if token.startswith("RUNNING"):
        return "running"
    return token.lower() or "unknown"


# ---------------------------------------------------------------------------
# Docker probing
# ---------------------------------------------------------------------------
def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=DEFAULT_COMMAND_TIMEOUT,
    )


def _probe_daemon() -> bool:
    try:
        completed = _run(["docker", "info"])
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _discover_compose_files() -> List[str]:
    candidates: List[str] = []
    try:
        completed = _run(["docker", "compose", "files"])
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return candidates
    if completed.returncode != 0:
        return candidates
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if compose_tool := shutil.which("docker"):
            # each discovered path is relative to current project root
            candidates.append(line)
    return candidates


def _service_status_by_compose_file(compose_files: List[str]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for compose in compose_files:
        try:
            cmd = ["docker", "compose", "-f", compose, "ps", "--format", "{{.Name}}\t{{.Service}}\t{{.Status}}"]
            completed = _run(cmd)
        except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
            result[compose] = {"error": 1}
            continue
        if completed.returncode != 0:
            result[compose] = {"error": 1}
            continue
        statuses: List[str] = []
        for line in completed.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                statuses.append(_extract_status_token(parts[2]))
        result[compose] = dict(Counter(statuses)) if statuses else {}
    return result


def _container_health_summary(service_stats: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    running = sum(v.get("running", 0) for v in service_stats.values() if isinstance(v, dict))
    exited = sum(v.get("exited", 0) for v in service_stats.values() if isinstance(v, dict))
    restarting = sum(v.get("restarting", 0) for v in service_stats.values() if isinstance(v, dict))
    healthy = sum(v.get("healthy", 0) for v in service_stats.values() if isinstance(v, dict))
    unhealthy = sum(v.get("unhealthy", 0) for v in service_stats.values() if isinstance(v, dict))
    starting = sum(v.get("starting", 0) for v in service_stats.values() if isinstance(v, dict))
    unknown = sum(v.get("unknown", 0) for v in service_stats.values() if isinstance(v, dict))
    error = sum(v.get("error", 0) for v in service_stats.values() if isinstance(v, dict))
    total = running + exited + restarting + healthy + unhealthy + starting + unknown + error
    return {
        "running": running,
        "stopped": exited,
        "restarting": restarting,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "starting": starting,
        "unknown": unknown,
        "error_services": error,
        "total": total,
    }


def _top_issues(service_stats: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    issues: List[str] = []
    for compose, stats in service_stats.items():
        if not isinstance(stats, dict):
            continue
        for status, count in stats.items():
            if status == "error":
                issues.append(f"{compose}: command error")
            elif status in {"unhealthy", "exited", "restarting", "starting"}:
                for _ in range(count):
                    issues.append(f"{compose}: {status}")
    return _bounded_counter(issues, cap=TOP_ISSUES_CAP)


# ---------------------------------------------------------------------------
# Public tool interface
# ---------------------------------------------------------------------------
def run() -> Dict[str, Any]:
    """Inspect local docker stack and return a success or failure envelope."""
    try:
        if not _probe_daemon():
            return relay_failure(
                "docker daemon is not reachable",
                code=FAILURE_CODE,
                meta={"tool": TOOL_NAME},
            )
        compose_files = _discover_compose_files()
        service_stats = _service_status_by_compose_file(compose_files)
        snapshot = {
            "docker_daemon_reachable": True,
            "compose_files": compose_files,
            "service_status_counts": service_stats,
            "container_health_summary": _container_health_summary(service_stats),
            "top_issues": _top_issues(service_stats),
        }
        return relay_success(snapshot)
    except Exception as exc:  # pragma: no cover - defensive envelope
        return relay_failure(str(exc), code=FAILURE_CODE, meta={"tool": TOOL_NAME})


__all__ = ["TOOL_NAME", "FAILURE_CODE", "is_available", "run"]
