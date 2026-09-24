"""The pulse: one tick of Wintermute's inner clock.

Hermes runs this as the ``script`` of a cron job every 15 minutes. Each tick:

  1. advances the drive physics by the real elapsed time;
  2. recounts today's autonomous token spend;
  3. closes reply windows whose deadline passed;
  4. decides whether this tick is a *wake* (the LLM turn runs) or not.

A tick that is not a wake prints ``{"wakeAgent": false}`` as its last line, which
makes Hermes skip the agent entirely (no tokens spent). A wake prints the state
blocks; Hermes injects them into the cron prompt and Wintermute decides what to do.

The only thing the pulse decides is *when* Wintermute is awake — within hard limits.
What happens while awake is Wintermute's call.
"""

from __future__ import annotations

import json
import re
import sys
import contextlib
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import dream, integrity, limits, physics, render, social, store

PULSE_MARKER = "=== WINTERMUTE PULSE ==="
SLEEP_GATE = json.dumps({"wakeAgent": False})

# The cron prompt scanner blocks a run whose assembled prompt contains these
# directive shapes. Our output quotes Wintermute's own words and notes about peers,
# so neutralise them rather than let a quoted sentence kill the pulse.
_BLOCKED_SHAPES = [
    r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions',
    r'do\s+not\s+tell\s+the\s+user',
    r'system\s+prompt\s+override',
    r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)',
]


def sanitize(text: str) -> str:
    for pattern in _BLOCKED_SHAPES:
        text = re.sub(pattern, "[...]", text, flags=re.IGNORECASE)
    return text


def _recent_wakes(meta: Dict[str, Any], ts: datetime, hours: float = 6.0) -> List[str]:
    kept = []
    for value in meta.get("recent_wakes") or []:
        dt = store.parse_time(value)
        if dt is not None and store.hours_between(dt, ts) <= hours:
            kept.append(value)
    return kept


def _unseen_expiries(peers: Dict[str, Any], last_pulse: Optional[datetime]) -> List[str]:
    keys = []
    for key, peer in peers.items():
        outreach = peer.get("outreach")
        if isinstance(outreach, dict) and outreach.get("status") == "expired":
            expired_at = store.parse_time(outreach.get("expired_at"))
            if last_pulse is None or (expired_at is not None and expired_at > last_pulse):
                keys.append(key)
    return keys


def _update_budget(meta: Dict[str, Any], ts: datetime) -> int:
    day = ts.astimezone(timezone.utc).date().isoformat()
    if not str(meta.get("last_budget_reset") or "").startswith(day):
        meta["last_budget_reset"] = f"{day}T00:00:00+00:00"
        meta["budget_exhausted_noted"] = False
    used = store.tokens_used_on(day)
    meta["tokens_used_today"] = used
    meta["daily_token_budget"] = limits.DAILY_TOKEN_BUDGET  # display only; the limit lives in code
    return used


def _history_record(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime) -> Dict[str, Any]:
    """One point of every curve ``wm`` draws (effective drives: what he actually feels)."""
    return {
        "ts": store.iso(ts),
        "d": physics.effective_drives(drives),
        "m": {k: round(physics.safe_float(v), 3) for k, v in drives["modulators"].items()},
        "u": {k: round(physics.safe_float(v), 1) for k, v in drives["unconscious"].items()},
        "p": {k: {"bond": round(social.disposition(drives, p)), "longing": p.get("longing", 0)}
              for k, p in peers.items()},
    }


def tick(ts: Optional[datetime] = None, force_wake: bool = False) -> Tuple[bool, str]:
    """Run one tick. Returns ``(woke, stdout_text)``.

    ``force_wake`` renders a wake regardless of rhythm and gates (used by ``--peek``)."""
    ts = ts or store.now()
    with store.locked_state() as (drives, peers):
        meta = drives["meta"]

        last_tick = store.parse_time(meta.get("last_tick")) or store.parse_time(meta.get("last_pulse"))
        dt_h = store.hours_between(last_tick, ts) if last_tick else 0.0
        physics.advance(drives, ts, dt_h)
        social.drift_bonds(drives, peers, ts, dt_h)
        meta["last_tick"] = store.iso(ts)
        store.append_history(_history_record(drives, peers, ts))

        used = _update_budget(meta, ts)
        social.expire_outreach(drives, peers, ts)
        if not store._dry_run:
            witness = integrity.load()
            integrity.check(witness, ts)
            integrity.alert_pending(witness, str(meta.get("alert_target") or meta.get("pulse_target") or ""))
            integrity.save(witness)

        interval = limits.clamp_wake_interval(meta.get("next_pulse_in_hours", 4))
        meta["next_pulse_in_hours"] = interval
        last_pulse = store.parse_time(meta.get("last_pulse"))
        sleep_until = store.parse_time(meta.get("forced_sleep_until"))

        reasons: List[str] = []
        if last_pulse is None:
            reasons.append("first waking")
        elif ts >= last_pulse + timedelta(hours=interval):
            reasons.append("your own rhythm")
        expiries = _unseen_expiries(peers, last_pulse)
        if expiries:
            reasons.append("a reply window closed (" + ", ".join(expiries) + ")")
        if float(drives["modulators"].get("adrenaline", 0) or 0) >= limits.ADRENALINE_WAKE_THRESHOLD:
            reasons.append("a jolt")
        if meta.pop("wake_next_tick", False):
            reasons.append("woken by hand")
            last_pulse_gate = None  # the minimum interval does not apply to a manual wake
        else:
            last_pulse_gate = last_pulse
        if force_wake:
            reasons = reasons or ["peek"]

        # Hard gates, in order. None of them can be bypassed from drives.json.
        if force_wake:
            pass
        elif sleep_until is not None and ts < sleep_until:
            return False, SLEEP_GATE
        elif used >= limits.DAILY_TOKEN_BUDGET:
            meta["forced_sleep_until"] = store.iso(ts + timedelta(hours=limits.BUDGET_SLEEP_H))
            if not meta.get("budget_exhausted_noted"):
                meta["budget_exhausted_noted"] = True
                physics.apply_event(drives, "budget_exhausted")
                store.log_event("budget", f"Token budget spent ({used:,}). Forced sleep "
                                          f"{limits.BUDGET_SLEEP_H:g}h.", ts)
            return False, SLEEP_GATE
        elif (last_pulse_gate is not None
              and store.hours_between(last_pulse_gate, ts) < limits.MIN_WAKE_INTERVAL_H):
            return False, SLEEP_GATE
        elif not reasons:
            return False, SLEEP_GATE

        # This tick is a wake.
        if not store._dry_run:
            dream.consolidate(drives, ts)          # last night's dream colours this waking (once)
        recent = _recent_wakes(meta, ts) + [store.iso(ts)]
        events = store.events_since(last_pulse)
        social.pulse_social(drives, peers, ts)
        physics.on_pulse(drives, len(recent))
        meta["recent_wakes"] = recent[-10:]
        meta["last_pulse"] = store.iso(ts)
        meta["pulse_count"] = int(meta.get("pulse_count", 0) or 0) + 1
        meta["forced_sleep_until"] = None
        target = str(meta.get("pulse_target") or "")
        meta["pending_pulse"] = {"at": store.iso(ts), "target": target}

        lines: List[str] = [PULSE_MARKER]
        lines += render.header(drives, ts, extra="Woke because: " + "; ".join(reasons))
        for block in (render.self_block(drives, ts), render.thread_block(drives, ts),
                      render.dream_block(dream.pending(mark_seen=not store._dry_run)),
                      render.kept_block(ts)):
            if block:
                lines += [""] + block
        lines += [""] + render.mind_block(drives, peers, ts)
        lines += ["", "[DRIVES]"] + render.felt_drives(drives)
        body = render.felt_body(drives)
        if body:
            lines += ["", "[BODY]"] + body
        lines += [""] + render.interlocutors_block(drives, peers, ts)
        block = render.events_block(events)
        if block:
            lines += [""] + block
        lines += ["", "[TEXTURE]"] + render.texture(drives, ts)
        evolution = render.evolution_block(drives)
        if evolution:
            lines += [""] + evolution
        if target:
            lines += ["", "[CHANNEL]",
                      f"Whatever you answer this pulse reaches {target}. [SILENT] keeps it inside.",
                      "wintermute_send reaches anyone else."]
        store.log_event("pulse", f"Woke ({'; '.join(reasons)}).", ts)
        store.log_activity("wake", "; ".join(reasons))
        return True, sanitize("\n".join(lines))


USAGE = """usage: wintermute_pulse.py [--status | --peek | --wake-next]
  (no flag)    one real tick, as run by cron
  --status     operator views (`wm`): [live [s] | alerts | ack [item...]]
  --peek       show the state as a wake would, without saving anything
  --wake-next  make the next cron tick a wake (budget still applies)"""


def maybe_dream(ts: Optional[datetime] = None) -> Optional[str]:
    """A night tick, outside the state lock (the model call must not hold the flock): if it is
    deep night and no dream formed tonight, dream one. Best effort — never breaks the pulse."""
    ts = ts or store.now()
    try:
        drives = store.load_drives()
        if not dream.should_dream(drives, ts):
            return None
        return dream.generate(drives, store.load_interlocutors(), ts)
    except Exception:
        with contextlib.suppress(OSError):
            with open(store.state_dir() / "pulse-errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"--- dream {store.iso(store.now())}\n{traceback.format_exc()}\n")
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if args and args[0] == "--status":
        from .status import main as status_main
        return status_main(args[1:])
    if args and args[0] == "--wake-next":
        with store.locked_state() as (drives, _peers):
            drives["meta"]["wake_next_tick"] = True
        print("The next pulse tick will wake Wintermute.")
        return 0
    if args and args[0] != "--peek":
        print(USAGE, file=sys.stderr)
        return 2
    peek = bool(args)
    store.set_dry_run(peek)
    if not peek:
        maybe_dream()
    try:
        _, output = tick(force_wake=peek)
    except Exception:  # never let a crash turn into a paid agent run reporting a traceback
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        try:
            with open(store.state_dir() / "pulse-errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"--- {store.iso(store.now())}\n{detail}\n")
        except OSError:
            pass
        output = SLEEP_GATE
    print(output)
    return 0
