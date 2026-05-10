"""Sloop dashboard for Hermes Agent.

Provides the `sloop status` view — shows loop health, recent execution results,
success rate, average duration, and an ASCII failure trend chart.

Designed to back both the CLI `/sloop status` command and the agent tool.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── ASCII Chart Rendering ────────────────────────────────────────────────────

def _ascii_bar_chart(
    values: List[int],
    labels: Optional[List[str]] = None,
    width: int = 40,
    height: int = 8,
    title: str = "",
) -> str:
    """Render a simple vertical bar chart as ASCII art.

    Args:
        values: Data values (one per bar).
        labels: Optional x-axis labels (one per bar).
        width: Max chart width in characters.
        height: Max bar height in lines.
        title: Optional chart title.

    Returns:
        Multi-line string with the chart.
    """
    if not values:
        return "(no data)"

    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1

    lines = []
    if title:
        lines.append(title)

    # Bar chars
    blocks = " ▁▂▃▄▅▆▇█"
    num_levels = len(blocks) - 1  # 9 levels per row

    for row in range(height, 0, -1):
        threshold = (row / height) * max_val
        line_parts = []
        for v in values:
            if v >= threshold:
                # How full is this bar at this row?
                fill = min((v / max_val) * height, height) - (row - 1)
                if fill >= 1:
                    line_parts.append("█")
                elif fill > 0:
                    idx = int(fill * num_levels)
                    line_parts.append(blocks[min(idx, num_levels)])
                else:
                    line_parts.append(" ")
            else:
                line_parts.append(" ")
        row_str = " ".join(line_parts)
        # Right-align with y-axis label
        if row == height:
            lines.append(f"{max_val:>4} │{row_str}")
        elif row == 1:
            lines.append(f"    0│{row_str}")
        else:
            lines.append(f"     │{row_str}")

    # X-axis
    lines.append("     └" + "─" * (len(values) * 2))
    if labels:
        label_line = "      "
        for lbl in labels:
            label_line += lbl[:1].ljust(2)
        lines.append(label_line)

    return "\n".join(lines)


def _sparkline(values: List[int]) -> str:
    """Generate a sparkline string from values."""
    if not values:
        return "·"
    blocks = " ▁▂▃▄▅▆▇█"
    max_val = max(values) if max(values) > 0 else 1
    return "".join(
        blocks[min(int((v / max_val) * (len(blocks) - 1)), len(blocks) - 1)]
        for v in values
    )


def _status_icon(rate: float) -> str:
    """Return a status icon based on success rate."""
    if rate >= 95:
        return "🟢"
    elif rate >= 80:
        return "🟡"
    elif rate >= 50:
        return "🟠"
    else:
        return "🔴"


def _health_word(rate: float) -> str:
    """Return a health status word."""
    if rate >= 95:
        return "HEALTHY"
    elif rate >= 80:
        return "DEGRADED"
    elif rate >= 50:
        return "UNHEALTHY"
    else:
        return "CRITICAL"


# ── Dashboard Rendering ──────────────────────────────────────────────────────

def render_sloop_status(
    success_rate: Dict[str, Any],
    avg_duration_ms: Optional[float],
    failure_trend: List[Dict[str, Any]],
    recent_entries: List[Dict[str, Any]],
    error_types: List[Dict[str, Any]],
    consecutive_failures: Optional[Dict[str, int]] = None,
) -> str:
    """Render the full sloop status dashboard as a formatted string.

    All data is passed in as arguments so this function stays pure (no DB access).
    """
    rate = success_rate.get("rate_percent", 0)
    total = success_rate.get("total", 0)
    hours = success_rate.get("hours", 24)
    successes = success_rate.get("successes", 0)
    failures = success_rate.get("failures", 0)

    out = []
    out.append("")
    out.append("┌────────────────────────────────────────────────────────────────────┐")
    out.append("│                    🔁 Sloop Loop Status                           │")
    out.append("└────────────────────────────────────────────────────────────────────┘")
    out.append("")

    # Overall health
    icon = _status_icon(rate)
    health = _health_word(rate)
    out.append(f"  {icon} Overall Health: {health}")
    out.append(f"  📊 Success Rate:  {rate}% ({successes}/{total} in last {hours}h)")

    if avg_duration_ms is not None:
        if avg_duration_ms > 60000:
            dur_str = f"{avg_duration_ms / 60000:.1f}m"
        elif avg_duration_ms > 1000:
            dur_str = f"{avg_duration_ms / 1000:.1f}s"
        else:
            dur_str = f"{avg_duration_ms:.0f}ms"
        out.append(f"  ⏱  Avg Duration:  {dur_str}")

    out.append("")

    # Consecutive failures warning
    if consecutive_failures:
        warned_jobs = {jid: cnt for jid, cnt in consecutive_failures.items() if cnt >= 3}
        if warned_jobs:
            out.append("  ⚠️  CONSECUTIVE FAILURES (auto-pause threshold: 3):")
            for jid, cnt in warned_jobs.items():
                out.append(f"     🔴 {jid}: {cnt} consecutive failures — job paused!")
            out.append("")

    # Failure trend chart
    out.append("  ── Failure Trend (last 12h) ──")
    if failure_trend:
        fail_values = [b.get("failures", 0) for b in failure_trend]
        total_values = [b.get("total", 0) for b in failure_trend]
        chart = _ascii_bar_chart(
            fail_values,
            height=6,
            title="  Failures per hour",
        )
        out.append(chart)
        spark = _sparkline(fail_values)
        out.append(f"  Sparkline: {spark}")
    else:
        out.append("  (no data)")
    out.append("")

    # Recent executions
    out.append("  ── Recent Executions ──")
    if recent_entries:
        out.append(f"  {'Time':<20} {'Status':<8} {'Duration':<10} {'Job':<20} {'Error'}")
        out.append("  " + "─" * 72)
        for entry in recent_entries[:15]:
            ts = entry.get("timestamp", "?")[:19]
            ok = "✅" if entry.get("success") else "❌"
            dur = entry.get("duration_ms")
            dur_str = f"{dur:.0f}ms" if dur else "-"
            name = (entry.get("job_name") or entry.get("job_id", "?"))[:20]
            err = (entry.get("error_type") or "")[:30]
            out.append(f"  {ts:<20} {ok:<8} {dur_str:<10} {name:<20} {err}")
    else:
        out.append("  (no recent executions)")
    out.append("")

    # Error breakdown
    if error_types:
        out.append("  ── Top Error Types ──")
        for et in error_types:
            out.append(f"  ❌ {et['error_type']}: {et['count']} occurrences")
        out.append("")

    return "\n".join(out)


def get_sloop_dashboard_data(
    feedback_collector=None,
    job_ids: Optional[List[str]] = None,
    hours: int = 24,
) -> Dict[str, Any]:
    """Gather all data needed for the sloop dashboard.

    Args:
        feedback_collector: Module or object with query functions.
                           If None, imports from tools.feedback_collector.
        job_ids: Optional list of job IDs to include. None = all.
        hours: Time window in hours.

    Returns:
        Dict with all dashboard data.
    """
    if feedback_collector is None:
        from tools import feedback_collector as fc
    else:
        fc = feedback_collector

    success_rate = fc.get_success_rate(hours=hours)
    avg_duration = fc.get_avg_duration(hours=hours)
    failure_trend = fc.get_failure_trend(buckets=min(hours, 12), bucket_minutes=60)
    recent = fc.get_recent_feedback(limit=20)
    error_types = fc.get_error_types(hours=hours)

    # Gather consecutive failures per job
    consecutive = {}
    if job_ids:
        for jid in job_ids:
            consecutive[jid] = fc.get_consecutive_failures(jid)
    else:
        # Get unique job_ids from recent entries
        seen = set()
        for entry in recent:
            jid = entry.get("job_id")
            if jid and jid not in seen:
                seen.add(jid)
                consecutive[jid] = fc.get_consecutive_failures(jid)

    return {
        "success_rate": success_rate,
        "avg_duration_ms": avg_duration,
        "failure_trend": failure_trend,
        "recent_entries": recent,
        "error_types": error_types,
        "consecutive_failures": consecutive,
    }


def sloop_dashboard_tool(action: str = "status", hours: int = 24, **kwargs) -> str:
    """Tool entry point for the sloop dashboard."""
    try:
        if action == "status":
            data = get_sloop_dashboard_data(hours=hours)
            rendered = render_sloop_status(
                success_rate=data["success_rate"],
                avg_duration_ms=data["avg_duration_ms"],
                failure_trend=data["failure_trend"],
                recent_entries=data["recent_entries"],
                error_types=data["error_types"],
                consecutive_failures=data["consecutive_failures"],
            )
            return json.dumps(
                {"success": True, "display": rendered, "data": data},
                indent=2,
            )

        elif action == "data":
            data = get_sloop_dashboard_data(hours=hours)
            return json.dumps({"success": True, **data}, indent=2)

        return json.dumps(
            {"success": False, "error": f"Unknown action: {action}"}
        )
    except Exception as e:
        logger.exception("sloop_dashboard error")
        return json.dumps({"success": False, "error": str(e)})


# ── Registry ─────────────────────────────────────────────────────────────────

from tools.registry import registry

SLOOP_DASHBOARD_SCHEMA = {
    "name": "sloop_dashboard",
    "description": (
        "View the Sloop loop system health dashboard. "
        "Use action='status' for a formatted display with success rate, "
        "failure trends, and recent execution history. "
        "Use action='data' for raw JSON data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: status, data (default: status)",
            },
            "hours": {
                "type": "integer",
                "description": "Time window in hours (default: 24)",
            },
        },
        "required": [],
    },
}


registry.register(
    name="sloop_dashboard",
    toolset="sloop",
    schema=SLOOP_DASHBOARD_SCHEMA,
    handler=lambda args, **kw: sloop_dashboard_tool(
        action=args.get("action", "status"),
        hours=args.get("hours", 24),
    ),
    check_fn=lambda: True,
    emoji="📈",
)
