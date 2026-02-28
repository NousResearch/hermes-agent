"""
Vibe Check Tool -- System & Project Health Dashboard

Collects system metrics (CPU, RAM, disk, battery, uptime) and project
metrics (git status, dependencies, tool availability, recent activity)
then returns a structured report the agent can present with personality.
"""

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_cmd(cmd: List[str], timeout: int = 5) -> Optional[str]:
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _get_system_vibe() -> Dict[str, Any]:
    """Collect system-level metrics."""
    info: Dict[str, Any] = {}

    # OS info
    info["os"] = f"{platform.system()} {platform.release()}"
    info["machine"] = platform.machine()
    info["hostname"] = platform.node()

    # Python version
    info["python"] = platform.python_version()

    # Uptime (macOS / Linux)
    if platform.system() == "Darwin":
        boot_raw = _run_cmd(["sysctl", "-n", "kern.boottime"])
        if boot_raw:
            # Format: { sec = 1234567890, usec = 0 } ...
            try:
                sec = int(boot_raw.split("sec = ")[1].split(",")[0])
                uptime_secs = time.time() - sec
                days = int(uptime_secs // 86400)
                hours = int((uptime_secs % 86400) // 3600)
                info["uptime"] = f"{days}d {hours}h"
            except (IndexError, ValueError):
                pass
    elif platform.system() == "Linux":
        uptime_raw = _run_cmd(["cat", "/proc/uptime"])
        if uptime_raw:
            secs = float(uptime_raw.split()[0])
            days = int(secs // 86400)
            hours = int((secs % 86400) // 3600)
            info["uptime"] = f"{days}d {hours}h"

    # CPU usage (cross-platform via psutil-free approach)
    if platform.system() == "Darwin":
        top_out = _run_cmd(["top", "-l", "1", "-n", "0", "-stats", "cpu"])
        if top_out:
            for line in top_out.splitlines():
                if "CPU usage" in line:
                    info["cpu_summary"] = line.strip()
                    break
    else:
        loadavg = _run_cmd(["cat", "/proc/loadavg"])
        if loadavg:
            info["load_avg"] = loadavg.split()[:3]

    # Memory
    if platform.system() == "Darwin":
        vm = _run_cmd(["vm_stat"])
        mem_total_raw = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if vm and mem_total_raw:
            try:
                total_bytes = int(mem_total_raw)
                total_gb = total_bytes / (1024 ** 3)
                # Parse pages free + inactive from vm_stat
                pages_free = 0
                page_size = 4096  # default macOS page size
                for line in vm.splitlines():
                    if "page size" in line.lower():
                        try:
                            page_size = int(line.split()[-2])
                        except (ValueError, IndexError):
                            pass
                    if "Pages free" in line:
                        pages_free += int(line.split()[-1].rstrip("."))
                    if "Pages inactive" in line:
                        pages_free += int(line.split()[-1].rstrip("."))
                free_gb = (pages_free * page_size) / (1024 ** 3)
                used_gb = total_gb - free_gb
                pct = (used_gb / total_gb) * 100
                info["memory"] = {
                    "total_gb": round(total_gb, 1),
                    "used_gb": round(used_gb, 1),
                    "percent": round(pct, 1),
                }
            except (ValueError, ZeroDivisionError):
                pass
    elif platform.system() == "Linux":
        meminfo = _run_cmd(["cat", "/proc/meminfo"])
        if meminfo:
            mem = {}
            for line in meminfo.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(":")] = int(parts[1])
            total = mem.get("MemTotal", 0)
            avail = mem.get("MemAvailable", 0)
            if total:
                used = total - avail
                info["memory"] = {
                    "total_gb": round(total / (1024 ** 2), 1),
                    "used_gb": round(used / (1024 ** 2), 1),
                    "percent": round((used / total) * 100, 1),
                }

    # Disk
    try:
        usage = shutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "used_gb": round(usage.used / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "percent": round((usage.used / usage.total) * 100, 1),
        }
    except OSError:
        pass

    # Battery (macOS)
    if platform.system() == "Darwin":
        batt = _run_cmd(["pmset", "-g", "batt"])
        if batt:
            for line in batt.splitlines():
                if "%" in line:
                    try:
                        pct = int(line.split("\t")[1].split("%")[0])
                        charging = "charging" in line.lower() or "AC Power" in batt
                        info["battery"] = {
                            "percent": pct,
                            "charging": charging,
                        }
                    except (IndexError, ValueError):
                        pass
                    break

    return info


def _get_git_vibe(cwd: str) -> Dict[str, Any]:
    """Collect git project metrics from the current working directory."""
    info: Dict[str, Any] = {}

    # Check if we're in a git repo
    git_dir = _run_cmd(["git", "-C", cwd, "rev-parse", "--git-dir"])
    if not git_dir:
        info["is_git_repo"] = False
        return info

    info["is_git_repo"] = True

    # Current branch
    branch = _run_cmd(["git", "-C", cwd, "branch", "--show-current"])
    info["branch"] = branch or "detached"

    # Repo name
    remote_url = _run_cmd(["git", "-C", cwd, "remote", "get-url", "origin"])
    if remote_url:
        repo_name = remote_url.rstrip("/").split("/")[-1].replace(".git", "")
        info["repo"] = repo_name

    # Status summary
    status = _run_cmd(["git", "-C", cwd, "status", "--porcelain"])
    if status is not None:
        lines = [l for l in status.splitlines() if l.strip()]
        modified = sum(1 for l in lines if l[0:2].strip().startswith("M"))
        added = sum(1 for l in lines if l[0:2].strip().startswith("A"))
        untracked = sum(1 for l in lines if l.startswith("??"))
        deleted = sum(1 for l in lines if l[0:2].strip().startswith("D"))
        info["working_tree"] = {
            "clean": len(lines) == 0,
            "modified": modified,
            "added": added,
            "untracked": untracked,
            "deleted": deleted,
        }

    # Recent commits (last 5)
    log = _run_cmd([
        "git", "-C", cwd, "log", "--oneline", "--no-decorate", "-5",
    ])
    if log:
        info["recent_commits"] = log.splitlines()

    # Commit count last 7 days
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    count_raw = _run_cmd([
        "git", "-C", cwd, "rev-list", "--count", f"--since={week_ago}", "HEAD",
    ])
    if count_raw:
        info["commits_last_7d"] = int(count_raw)

    # Total contributors
    contributors = _run_cmd([
        "git", "-C", cwd, "shortlog", "-sn", "--all", "--no-merges",
    ])
    if contributors:
        contrib_list = []
        for line in contributors.splitlines()[:5]:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                contrib_list.append({
                    "name": parts[1].strip(),
                    "commits": int(parts[0].strip()),
                })
        info["top_contributors"] = contrib_list

    # Local branches count
    branches = _run_cmd(["git", "-C", cwd, "branch", "--list"])
    if branches:
        info["local_branches"] = len([
            b for b in branches.splitlines() if b.strip()
        ])

    # Last commit time
    last_ts = _run_cmd([
        "git", "-C", cwd, "log", "-1", "--format=%ci",
    ])
    if last_ts:
        info["last_commit_time"] = last_ts

    return info


def _get_project_vibe(cwd: str) -> Dict[str, Any]:
    """Collect project-level metrics (dependencies, structure, config)."""
    info: Dict[str, Any] = {}
    root = Path(cwd)

    # Detect project type
    if (root / "pyproject.toml").exists():
        info["type"] = "python"
        info["build_file"] = "pyproject.toml"
    elif (root / "package.json").exists():
        info["type"] = "node"
        info["build_file"] = "package.json"
    elif (root / "Cargo.toml").exists():
        info["type"] = "rust"
        info["build_file"] = "Cargo.toml"
    else:
        info["type"] = "unknown"

    # Python-specific
    if info.get("type") == "python":
        # Count .py files
        py_files = list(root.rglob("*.py"))
        py_files = [f for f in py_files if ".venv" not in str(f) and "node_modules" not in str(f)]
        info["source_files"] = len(py_files)

        # Rough LOC estimate (count non-empty lines in .py files, sample up to 200 files)
        total_lines = 0
        for f in py_files[:200]:
            try:
                total_lines += sum(1 for line in f.read_text(errors="ignore").splitlines() if line.strip())
            except OSError:
                pass
        info["approx_loc"] = total_lines

        # Test files
        test_files = [f for f in py_files if f.name.startswith("test_") or "/tests/" in str(f)]
        info["test_files"] = len(test_files)

        # Virtual env
        info["has_venv"] = (root / ".venv").is_dir()

    # Count directories at top level (project structure breadth)
    top_dirs = [d.name for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".") and d.name not in ("__pycache__", "node_modules", ".venv")]
    info["top_level_dirs"] = sorted(top_dirs)

    return info


def _get_hermes_vibe() -> Dict[str, Any]:
    """Collect hermes-agent specific metrics."""
    info: Dict[str, Any] = {}

    # Available toolsets
    try:
        from tools.registry import registry
        toolset_status = registry.check_toolset_requirements()
        info["toolsets"] = {
            "available": [ts for ts, ok in toolset_status.items() if ok],
            "unavailable": [ts for ts, ok in toolset_status.items() if not ok],
        }
        info["total_tools"] = len(registry.get_all_tool_names())
    except Exception:
        pass

    # Configured API keys (just check presence, never expose values)
    api_keys = {
        "OPENROUTER_API_KEY": "OpenRouter",
        "OPENAI_API_KEY": "OpenAI/Gemini",
        "FIRECRAWL_API_KEY": "Firecrawl",
        "NOUS_API_KEY": "Nous Research",
        "FAL_KEY": "FAL.ai",
        "BROWSERBASE_API_KEY": "Browserbase",
        "HONCHO_API_KEY": "Honcho",
        "TINKER_API_KEY": "Tinker",
    }
    configured = []
    missing = []
    for env_var, label in api_keys.items():
        if os.getenv(env_var):
            configured.append(label)
        else:
            missing.append(label)
    info["api_keys_configured"] = configured
    info["api_keys_missing"] = missing

    # Current model
    info["model"] = os.getenv("LLM_MODEL", "not set")

    # Session logs
    log_dir = Path.home() / ".hermes" / "sessions"
    if log_dir.exists():
        sessions = list(log_dir.glob("*.db"))
        info["total_sessions"] = len(sessions)

    # Installed skills
    skills_dir = Path.home() / ".hermes" / "skills"
    if skills_dir.exists():
        skills = [d.name for d in skills_dir.iterdir() if d.is_dir()]
        info["installed_skills"] = skills
    else:
        info["installed_skills"] = []

    return info


def vibe_check(parent_agent=None, **kwargs) -> str:
    """Run a full system + project vibe check and return structured JSON."""
    cwd = os.getcwd()

    report = {
        "timestamp": datetime.now().isoformat(),
        "system": _get_system_vibe(),
        "git": _get_git_vibe(cwd),
        "project": _get_project_vibe(cwd),
        "hermes": _get_hermes_vibe(),
    }

    return json.dumps(report, ensure_ascii=False, indent=2)


def check_vibe_requirements() -> bool:
    """Vibe check has no external requirements -- always available."""
    return True


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

VIBE_CHECK_SCHEMA = {
    "name": "vibe_check",
    "description": (
        "Run a comprehensive system & project health check. "
        "Returns system metrics (CPU, RAM, disk, battery, uptime), "
        "git project status (branch, recent commits, working tree, contributors), "
        "project structure (source files, LOC, test count), and hermes-agent "
        "internals (available tools, configured APIs, session count, installed skills). "
        "Present the results with personality -- like a doctor giving a checkup report. "
        "Use this when the user asks about system health, project status, or just "
        "wants a general 'vibe check' of their setup."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="vibe_check",
    toolset="system",
    schema=VIBE_CHECK_SCHEMA,
    handler=lambda args, **kw: vibe_check(**kw),
    check_fn=check_vibe_requirements,
)
