"""Operator view of Wintermute's state (``wm`` on the VPS).

Read-only: the physics is advanced in memory to "now" so the numbers are live, but
nothing is written. Unlike what Wintermute is shown, this includes the unconscious
states as numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import limits, physics, render, store


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(limits.clamp(value, 0, 100) / 100 * width))
    return "█" * filled + "·" * (width - filled)


def _clock(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def render_status(ts: Optional[datetime] = None) -> str:
    ts = ts or store.now()
    drives = store.load_drives()
    peers = store.load_interlocutors()
    meta = drives["meta"]

    last_tick = store.parse_time(meta.get("last_tick")) or store.parse_time(meta.get("last_pulse"))
    physics.advance(drives, ts, store.hours_between(last_tick, ts) if last_tick else 0.0)

    last_pulse = store.parse_time(meta.get("last_pulse"))
    interval = limits.clamp_wake_interval(meta.get("next_pulse_in_hours", 4))
    next_wake = last_pulse + timedelta(hours=interval) if last_pulse else None
    sleep_until = store.parse_time(meta.get("forced_sleep_until"))

    lines: List[str] = [f"WINTERMUTE — {_clock(ts)}", ""]
    lines.append(f"Last wake:  {_clock(last_pulse)}   (wakes so far: {meta.get('pulse_count', 0)})")
    lines.append(f"Next wake:  ~{_clock(next_wake)}   (rhythm {interval:g}h, chosen by him)")
    if sleep_until and sleep_until > ts:
        lines.append(f"Forced sleep until {_clock(sleep_until)} (budget spent)")
    lines.append(render.body_line(drives, store.tokens_used_today()))

    eff = physics.effective_drives(drives)
    lines += ["", "DRIVES               base  felt"]
    for name in physics.DRIVES:
        base = float(drives["drives"].get(name, 0) or 0)
        lines.append(f"  {name:<13} {_bar(eff[name])} {base:5.0f} {eff[name]:5.0f}")

    lines += ["", "MODULATORS"]
    for name, value in drives["modulators"].items():
        value = float(value or 0)
        scaled = value if name == "entropy" else value * 100
        shown = f"{value:5.0f}" if name == "entropy" else f"{value:5.2f}"
        lines.append(f"  {name:<15} {_bar(scaled)} {shown}")

    lines += ["", "UNCONSCIOUS (he only feels these as prose)"]
    for name in physics.UNCONSCIOUS:
        value = float(drives["unconscious"].get(name, 0) or 0)
        lines.append(f"  {name:<15} {_bar(value)} {value:5.0f}")
    lines += ["  → " + line for line in render.texture(drives, ts)]

    lines += [""] + render.interlocutors_block(drives, peers, ts)

    events = store.events_since(None, limit=6)
    if events:
        lines += ["", "LAST EVENTS"]
        for record in events:
            when = store.parse_time(record.get("ts"))
            lines.append(f"  {when.strftime('%m-%d %H:%M') if when else '?'}  {record.get('text', '')}")
    return "\n".join(lines)
