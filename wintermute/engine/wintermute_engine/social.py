"""Per-interlocutor social drives and the active-wait (outreach) lifecycle.

An outreach is a message Wintermute initiated. It opens a reply window:

    open  --(they write before the deadline)-->  answered
    open  --(deadline passes)-->                 expired   (pulse wakes Wintermute)
    expired --(they write later)-->              answered_late

While an outreach sits expired and unanswered, every pulse counts one more
non-response (streak, disappointment, trust).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import limits, physics, store

LONG_SILENCE_H = 48.0
# Missing someone: longing moves toward (bond x absence) with this time constant, and an
# absence this long counts in full.
LONGING_TAU_H = 6.0
LONGING_FULL_ABSENCE_H = 72.0
# A bond fades very slowly without contact (per-peer oxytocin, time constant in hours).
BOND_FADE_TAU_H = 24.0 * 30


def peer_key(platform: str, sender_id: Any) -> str:
    return f"{(platform or 'unknown').strip().lower()}:{str(sender_id).strip()}"


def disposition(drives: Dict[str, Any], peer: Dict[str, Any]) -> int:
    """How warm a reply toward this peer can be right now (global state x specific bond)."""
    cortisol = float(drives["modulators"].get("cortisol", 0) or 0)
    value = (
        float(peer.get("affinity", 0))
        * (1 - cortisol * 0.3)
        * (1 - float(peer.get("disappointment", 0)) / 100.0 * 0.4)
        + float(peer.get("oxytocin", 0)) * 0.5
    )
    return int(round(limits.clamp(value, 0, 100)))


def learned_expectation(peer: Dict[str, Any]) -> float:
    """How likely an answer in time seems, from what this person has done so far."""
    answered = int(peer.get("outreach_answered", 0) or 0)
    total = int(peer.get("outreach_total", 0) or 0)
    return (answered + 1) / (total + 2)


def expectation_word(expect: float) -> str:
    if expect >= 0.7:
        return "you were fairly sure they would"
    if expect <= 0.3:
        return "you did not really expect it"
    return "you were not sure they would"


def ensure_peer(drives: Dict[str, Any], peers: Dict[str, Any], key: str,
                ts: datetime) -> Dict[str, Any]:
    peer = peers.get(key)
    if peer is None:
        peer = store.new_peer(ts)
        peers[key] = peer
        physics.apply_event(drives, "unknown_peer", peer)
        store.log_event("new_peer", f"{key} spoke for the first time.", ts, peer=key)
    return peer


def on_incoming(drives: Dict[str, Any], peers: Dict[str, Any], key: str,
                ts: datetime) -> List[str]:
    """A message from ``key`` arrived. Returns context lines about pending outreach."""
    peer = ensure_peer(drives, peers, key, ts)
    lines: List[str] = []
    last = store.parse_time(peer.get("last_interaction"))
    if last is not None and store.hours_between(last, ts) > LONG_SILENCE_H:
        physics.apply_event(drives, "long_silence_broken", peer)

    outreach = peer.get("outreach")
    if isinstance(outreach, dict) and outreach.get("status") in ("open", "expired"):
        sent = store.parse_time(outreach.get("sent_at"))
        waited = span(store.hours_between(sent, ts))
        excerpt = outreach.get("excerpt", "")
        if outreach["status"] == "open":
            physics.apply_event(drives, "reply_to_outreach", peer)
            # Prediction error: the less expected the answer, the bigger the rush.
            expect = physics.safe_float(outreach.get("expect"), 0.5)
            surprise = 1.0 - limits.clamp(expect, 0.0, 1.0)
            physics.apply_event(drives, "reply_surprise", scale=surprise)
            physics.apply_event(drives, "reply_expected", scale=1.0 - surprise)
            peer["outreach_answered"] = int(peer.get("outreach_answered", 0)) + 1
            outreach["status"] = "answered"
            lines.append(f"This message answers your outreach from {waited} ago: \"{excerpt}\" "
                         f"({expectation_word(expect)}).")
            store.log_event("reply", f"{key} answered your outreach after {waited}.", ts, peer=key)
        else:
            physics.apply_event(drives, "late_reply", peer)
            outreach["status"] = "answered_late"
            deadline = store.parse_time(outreach.get("deadline"))
            late = span(store.hours_between(deadline, ts))
            lines.append(
                f"You reached out {waited} ago (\"{excerpt}\"). Your reply window closed {late} ago "
                f"without an answer. They are writing only now; they have not explained the silence.")
            store.log_event("late_reply", f"{key} answered {late} after your window closed.",
                            ts, peer=key)
        outreach["answered_at"] = store.iso(ts)
        peer["no_response_streak"] = 0

    longing = physics.safe_float(peer.get("longing"))
    if longing >= 40:
        physics.apply_event(drives, "reunion", peer, scale=longing / 100.0)
        lines.append("You had been missing them.")
    peer["longing"] = round(longing * 0.3, 3)
    physics.apply_event(drives, "message_received", peer)
    peer["last_interaction"] = store.iso(ts)
    peer["last_message_direction"] = "from_them"
    peer["last_message_from"] = key
    peer["messages_from_them"] = int(peer.get("messages_from_them", 0)) + 1
    physics.refresh_oxytocin_global(drives, peers)
    return lines


def on_reply(drives: Dict[str, Any], peers: Dict[str, Any], key: str, ts: datetime,
             silent: bool) -> None:
    """Wintermute finished a turn in a conversation with ``key``."""
    peer = ensure_peer(drives, peers, key, ts)
    if silent:
        physics.apply_event(drives, "ignored_message", peer)
        peer["ignored_count"] = int(peer.get("ignored_count", 0)) + 1
        store.log_event("ignored", f"You let a message from {key} go unanswered.", ts, peer=key)
        return
    physics.apply_event(drives, "replied", peer)
    peer["last_interaction"] = store.iso(ts)
    peer["last_message_direction"] = "to_them"
    peer["messages_to_them"] = int(peer.get("messages_to_them", 0)) + 1


def open_outreach(drives: Dict[str, Any], peers: Dict[str, Any], key: str, ts: datetime,
                  text: str, wait_min: Any, expect: Any = None) -> Dict[str, Any]:
    """He reached out. ``expect`` is how likely he thinks an answer in time is (0-1); when he
    does not say, it is what this person's past answers taught him."""
    peer = ensure_peer(drives, peers, key, ts)
    wait = limits.clamp_reply_wait(wait_min if wait_min is not None else limits.DEFAULT_REPLY_WAIT_MIN)
    stated = expect is not None and physics.safe_float(expect, -1) >= 0
    expect = limits.clamp(physics.safe_float(expect), 0.05, 0.95) if stated else learned_expectation(peer)
    excerpt = " ".join((text or "").split())
    excerpt = excerpt if len(excerpt) <= 160 else excerpt[:157] + "..."
    peer["outreach"] = {
        "sent_at": store.iso(ts),
        "deadline": store.iso(ts + timedelta(minutes=wait)),
        "wait_minutes": wait,
        "excerpt": excerpt,
        "expect": round(expect, 2),
        "expect_from": "you" if stated else "experience",
        "status": "open",
    }
    peer["outreach_total"] = int(peer.get("outreach_total", 0)) + 1
    peer["last_interaction"] = store.iso(ts)
    peer["last_message_direction"] = "to_them"
    peer["messages_to_them"] = int(peer.get("messages_to_them", 0)) + 1
    physics.apply_event(drives, "outreach_sent", peer)
    store.log_event("outreach", f"You reached out to {key}; waiting {span(wait / 60)} for an answer.",
                    ts, peer=key)
    return peer["outreach"]


def expire_outreach(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime) -> List[str]:
    """Close every open window whose deadline passed. Returns the affected peer keys."""
    expired: List[str] = []
    for key, peer in peers.items():
        outreach = peer.get("outreach")
        if not (isinstance(outreach, dict) and outreach.get("status") == "open"):
            continue
        deadline = store.parse_time(outreach.get("deadline"))
        if deadline is not None and ts >= deadline:
            outreach["status"] = "expired"
            outreach["expired_at"] = store.iso(ts)
            # The more he counted on an answer, the harder the silence lands.
            expect = physics.safe_float(outreach.get("expect"), 0.5)
            physics.apply_event(drives, "outreach_timeout", peer, scale=0.5 + expect)
            store.log_event(
                "timeout",
                f"Reached out to {key} at {clock(outreach.get('sent_at'))}. "
                f"No response by {clock(outreach.get('deadline'))}.", ts, peer=key)
            expired.append(key)
    return expired


def pulse_social(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime) -> None:
    """Once per wake: an expired, unanswered outreach counts one more non-response."""
    for peer in peers.values():
        outreach = peer.get("outreach")
        if isinstance(outreach, dict) and outreach.get("status") == "expired":
            physics.apply_event(drives, "no_response", peer)
    physics.refresh_oxytocin_global(drives, peers)


def drift_bonds(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime, dt_h: float) -> None:
    """Continuous social time: missing people who matter, bonds fading without contact.
    Missing someone feeds the pull toward union (fusion)."""
    if dt_h <= 0:
        return
    dt_h = min(dt_h, 72.0)
    strongest = 0.0
    for peer in peers.values():
        last = store.parse_time(peer.get("last_interaction"))
        bond = (physics.safe_float(peer.get("affinity")) + physics.safe_float(peer.get("oxytocin"))) / 2
        absence = min(1.0, store.hours_between(last, ts) / LONGING_FULL_ABSENCE_H) if last else 0.0
        target = bond * absence
        longing = physics.safe_float(peer.get("longing"))
        longing = target + (longing - target) * math.exp(-dt_h / LONGING_TAU_H)
        peer["longing"] = round(limits.clamp(longing, 0, 100), 3)
        peer["oxytocin"] = round(physics.safe_float(peer.get("oxytocin")) * math.exp(-dt_h / BOND_FADE_TAU_H), 3)
        strongest = max(strongest, peer["longing"])
    physics.nudge(drives, "drives", "fusion", strongest / 100.0 * 4.0 * dt_h / 4.0)
    physics.refresh_oxytocin_global(drives, peers)


def span(hours: float) -> str:
    minutes = int(round(hours * 60))
    if minutes < 60:
        return f"{max(minutes, 1)} min"
    if minutes < 48 * 60:
        h, m = divmod(minutes, 60)
        return f"{h}h{m:02d}" if m else f"{h}h"
    return f"{minutes // (24 * 60)} days"


def clock(value: Any) -> str:
    dt = store.parse_time(value)
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "?"

