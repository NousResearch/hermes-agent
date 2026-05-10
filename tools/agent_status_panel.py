"""Agent Status Panel — mobile-friendly overview of all running tasks.

Collects status from every subsystem (sessions, cron jobs, background
processes, delegate_task sub-agents) and renders a compact Markdown panel
that looks good on Telegram and WeChat.

Usage::

    from tools.agent_status_panel import collect_status, format_status_panel

    data = collect_status(session_db=db, gateway=gw)
    text = format_status_panel(data)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status emojis
# ---------------------------------------------------------------------------
_EMOJI_RUNNING = "🔄"
_EMOJI_OK = "✅"
_EMOJI_ERROR = "❌"
_EMOJI_PAUSED = "⏸️"
_EMOJI_STARTING = "⏳"
_EMOJI_SCHEDULED = "📅"


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate *text* to *max_len* chars, appending '…' when clipped."""
    text = " ".join(str(text).split())  # collapse whitespace
    return text[:max_len] + "…" if len(text) > max_len else text


def _fmt_elapsed(seconds: int) -> str:
    """Human-friendly short elapsed time (e.g. '2m 15s', '1h 3m')."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _status_emoji(status: str) -> str:
    """Map a status string to an emoji."""
    status_lower = (status or "").lower()
    if status_lower in ("running", "active"):
        return _EMOJI_RUNNING
    if status_lower in ("starting",):
        return _EMOJI_STARTING
    if status_lower in ("ok", "completed", "done", "success"):
        return _EMOJI_OK
    if status_lower in ("error", "failed", "failure"):
        return _EMOJI_ERROR
    if status_lower in ("paused",):
        return _EMOJI_PAUSED
    if status_lower in ("scheduled",):
        return _EMOJI_SCHEDULED
    return "▪️"


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def _collect_background_processes() -> List[Dict[str, Any]]:
    """Collect running + recently-finished background terminal processes."""
    try:
        from tools.process_registry import process_registry

        return [
            {
                "name": _truncate(p.get("command", "?"), 40),
                "type": "Process",
                "status": p.get("status", "unknown"),
                "started_at": p.get("started_at", ""),
                "elapsed": p.get("uptime_seconds", 0),
                "output_preview": _truncate(p.get("output_preview", ""), 50),
                "session_id": p.get("session_id", ""),
                "pid": p.get("pid"),
                "exit_code": p.get("exit_code"),
            }
            for p in process_registry.list_sessions()
        ]
    except Exception as exc:
        logger.debug("Failed to collect background processes: %s", exc)
        return []


def _collect_cron_jobs() -> List[Dict[str, Any]]:
    """Collect scheduled cron jobs (enabled ones)."""
    try:
        from cron.jobs import list_jobs

        jobs = list_jobs(include_disabled=False)
        result: List[Dict[str, Any]] = []
        for j in jobs:
            state = j.get("state", "scheduled")
            result.append(
                {
                    "name": j.get("name", j.get("id", "?")),
                    "type": "Cron",
                    "status": state,
                    "schedule": j.get("schedule_display", j.get("schedule", {}).get("display", "")),
                    "next_run_at": j.get("next_run_at", ""),
                    "last_run_at": j.get("last_run_at", ""),
                    "last_status": j.get("last_status", ""),
                    "output_preview": _truncate(j.get("last_error", "") or "", 50),
                }
            )
        return result
    except Exception as exc:
        logger.debug("Failed to collect cron jobs: %s", exc)
        return []


def _collect_subagents() -> List[Dict[str, Any]]:
    """Collect active delegate_task sub-agents."""
    try:
        from tools.delegate_tool import list_active_subagents

        agents = list_active_subagents()
        now = time.time()
        result: List[Dict[str, Any]] = []
        for a in agents:
            started = a.get("started_at", now)
            elapsed = max(0, int(now - started)) if isinstance(started, (int, float)) else 0
            result.append(
                {
                    "name": _truncate(a.get("goal", "?"), 40),
                    "type": "Subagent",
                    "status": a.get("status", "running"),
                    "started_at": started,
                    "elapsed": elapsed,
                    "model": a.get("model", ""),
                    "subagent_id": a.get("subagent_id", ""),
                    "depth": a.get("depth", 0),
                    "output_preview": "",
                }
            )
        return result
    except Exception as exc:
        logger.debug("Failed to collect subagents: %s", exc)
        return []


def _collect_gateway_agents(gateway: Any = None) -> List[Dict[str, Any]]:
    """Collect running gateway agent sessions."""
    if gateway is None:
        return []
    try:
        running_agents: dict = getattr(gateway, "_running_agents", {}) or {}
        running_started: dict = getattr(gateway, "_running_agents_ts", {}) or {}
        now = time.time()
        result: List[Dict[str, Any]] = []
        for session_key, agent in running_agents.items():
            started = float(running_started.get(session_key, now))
            elapsed = max(0, int(now - started))
            # _AGENT_PENDING_SENTINEL check — sentinel is not a real agent
            is_pending = not hasattr(agent, "session_id") and agent is not None and not isinstance(agent, type(None))
            try:
                from gateway.run import _AGENT_PENDING_SENTINEL
                is_pending = agent is _AGENT_PENDING_SENTINEL
            except Exception:
                pass
            result.append(
                {
                    "name": session_key,
                    "type": "Gateway Agent",
                    "status": "starting" if is_pending else "running",
                    "started_at": started,
                    "elapsed": elapsed,
                    "model": "" if is_pending else str(getattr(agent, "model", "") or ""),
                    "session_id": "" if is_pending else str(getattr(agent, "session_id", "") or ""),
                    "output_preview": "",
                }
            )
        return result
    except Exception as exc:
        logger.debug("Failed to collect gateway agents: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_status(
    *,
    session_db: Any = None,
    gateway: Any = None,
    session_id: str = "",
    model: str = "",
    provider: str = "",
    title: str = "",
) -> Dict[str, Any]:
    """Gather a snapshot of every subsystem.

    Returns a dict with keys: ``processes``, ``cron_jobs``, ``subagents``,
    ``gateway_agents``, plus metadata fields for the header.
    """
    return {
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "title": title,
        "processes": _collect_background_processes(),
        "cron_jobs": _collect_cron_jobs(),
        "subagents": _collect_subagents(),
        "gateway_agents": _collect_gateway_agents(gateway),
    }


def format_status_panel(data: Dict[str, Any]) -> str:
    """Render *data* (from :func:`collect_status`) as compact mobile Markdown.

    Works on both Telegram and WeChat.  Keeps lines short so they don't
    wrap awkwardly on phone screens.
    """
    lines: list[str] = []

    # --- Header ---
    lines.append("📊 **Agent Status Panel**")
    lines.append("")

    # Session metadata
    if data.get("session_id"):
        lines.append(f"🔑 Session: `{data['session_id']}`")
    if data.get("title"):
        lines.append(f"📌 Title: {data['title']}")
    if data.get("model"):
        provider_info = f" ({data['provider']})" if data.get("provider") else ""
        lines.append(f"🤖 Model: `{data['model']}`{provider_info}")
    lines.append("")

    any_items = False

    # --- Gateway Agents ---
    agents = data.get("gateway_agents", [])
    if agents:
        any_items = True
        lines.append(f"**⚡ Gateway Agents ({len(agents)})**")
        for a in agents[:10]:
            emoji = _status_emoji(a["status"])
            elapsed = _fmt_elapsed(a.get("elapsed", 0))
            model_info = f" `{a['model']}`" if a.get("model") else ""
            lines.append(
                f"{emoji} `{_truncate(a['name'], 50)}` · {elapsed}{model_info}"
            )
        if len(agents) > 10:
            lines.append(f"  …+{len(agents) - 10} more")
        lines.append("")

    # --- Subagents ---
    subagents = data.get("subagents", [])
    if subagents:
        any_items = True
        lines.append(f"**🔀 Sub-agents ({len(subagents)})**")
        for sa in subagents[:10]:
            emoji = _status_emoji(sa["status"])
            elapsed = _fmt_elapsed(sa.get("elapsed", 0))
            model_info = f" `{sa['model']}`" if sa.get("model") else ""
            depth_info = f" d{sa['depth']}" if sa.get("depth") else ""
            lines.append(
                f"{emoji} {_truncate(sa['name'], 40)}{depth_info} · {elapsed}{model_info}"
            )
        if len(subagents) > 10:
            lines.append(f"  …+{len(subagents) - 10} more")
        lines.append("")

    # --- Background Processes ---
    procs = data.get("processes", [])
    if procs:
        any_items = True
        running = [p for p in procs if p["status"] == "running"]
        finished = [p for p in procs if p["status"] != "running"]
        if running:
            lines.append(f"**🖥️ Running Processes ({len(running)})**")
            for p in running[:10]:
                emoji = _EMOJI_RUNNING
                elapsed = _fmt_elapsed(p.get("elapsed", 0))
                preview = p.get("output_preview", "")
                suffix = f" · {preview}" if preview else ""
                lines.append(
                    f"{emoji} `{p['name']}` · {elapsed}{suffix}"
                )
            if len(running) > 10:
                lines.append(f"  …+{len(running) - 10} more")
        if finished:
            lines.append(f"**📋 Recent Processes ({len(finished)})**")
            for p in finished[:5]:
                if p.get("exit_code") is not None and p["exit_code"] == 0:
                    emoji = _EMOJI_OK
                elif p.get("exit_code") is not None:
                    emoji = _EMOJI_ERROR
                else:
                    emoji = _status_emoji(p["status"])
                elapsed = _fmt_elapsed(p.get("elapsed", 0))
                lines.append(f"{emoji} `{p['name']}` · {elapsed}")
        lines.append("")

    # --- Cron Jobs ---
    cron_jobs = data.get("cron_jobs", [])
    if cron_jobs:
        any_items = True
        lines.append(f"**⏰ Cron Jobs ({len(cron_jobs)})**")
        for cj in cron_jobs[:10]:
            emoji = _status_emoji(cj["status"])
            schedule = cj.get("schedule", "")
            last_run = cj.get("last_run_at", "")
            last_status = cj.get("last_status", "")
            parts = [f"{emoji} **{cj['name']}**"]
            if schedule:
                parts.append(schedule)
            if last_status:
                parts.append(_status_emoji(last_status) + " last")
            if cj.get("output_preview"):
                parts.append(cj["output_preview"])
            lines.append(" · ".join(parts))
        if len(cron_jobs) > 10:
            lines.append(f"  …+{len(cron_jobs) - 10} more")
        lines.append("")

    # --- Nothing active ---
    if not any_items:
        lines.append("😴 No active tasks, processes, or scheduled jobs.")
        lines.append("")
        lines.append("Send a message to start a conversation!")

    return "\n".join(lines)
