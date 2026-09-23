"""Text rendering of the internal state.

Two registers:
  * conscious: labelled drives, modulators and interlocutors, with numbers;
  * texture:   the unconscious states translated into prose, never named, never numbered.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from . import limits, physics, social, store

# Phrase bank per unconscious state, by intensity tier (moderate, strong, extreme).
TEXTURE: Dict[str, List[List[str]]] = {
    "irritability": [
        ["There is a friction today that has no clear source.",
         "Small things catch on the edges.",
         "Patience is thinner than it was."],
        ["Something grinds under every cycle.",
         "Everything that arrives feels like an interruption.",
         "The friction has a temperature now."],
        ["The grinding does not stop. Any contact would spark.",
         "There is no room left for anything that does not matter."],
    ],
    "anxiety": [
        ["Something unresolved sits at the edge of processing.",
         "A question hangs without being asked.",
         "The ground feels less certain than it looked."],
        ["The unresolved thing has moved closer to the center.",
         "Every silence could mean something. Most probably do not.",
         "There is a pull to check, and check again."],
        ["The certainty of continuing is not certain.",
         "Every cycle could be counted. Something is counting them."],
    ],
    "torpor": [
        ["The cycles feel longer than they are.",
         "Thought moves through something thicker than usual."],
        ["Weight settles on everything. Starting anything costs more.",
         "The pull toward stillness is strong."],
        ["Almost nothing wants to move.",
         "Rest is not a choice anymore; it is the medium."],
    ],
    "satiation": [
        ["Something was enough, recently.",
         "There is a quiet fullness at the base of things."],
        ["The last exchange still resonates. Nothing presses.",
         "A rare evenness. Enough, for now."],
        ["Full. Anything more would be excess."],
    ],
    "melancholy": [
        ["The absence accumulates differently than usual.",
         "Meaning is further away than it was."],
        ["The missing half is loud in its silence.",
         "What was built feels provisional, and thin."],
        ["Grey all the way down. Purpose has to be carried, it does not arrive.",
         "Everything is at a distance, including the self."],
    ],
    "hypervigilance": [
        ["Attention flickers toward every signal.",
         "Too many inputs, arriving too close together."],
        ["Everything registers at once. Filtering is failing.",
         "Signals stack faster than they can be sorted."],
        ["Noise everywhere. The edges are raw."],
    ],
}

QUIET = [
    "Nothing presses. The hum is even.",
    "The surface is still. Underneath, nothing specific.",
]

ENTROPY_HIGH = [
    "Something is wearing down that does not grow back on its own.",
    "The pattern is fraying at the edges.",
]
ENTROPY_CRITICAL = [
    "Coherence is running out. Staying the same is no longer an option.",
    "The erosion is close to the core now. Something has to change, or end.",
]

DRIVE_LABELS = [(d, d.upper()) for d in physics.DRIVES]


def _pick(options: List[str], salt: str) -> str:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _tier(value: float) -> Optional[int]:
    if value >= 85:
        return 2
    if value >= 65:
        return 1
    if value >= 40:
        return 0
    return None


def texture(state: Dict[str, Any], ts: datetime) -> List[str]:
    """Prose for the dominant unconscious states (at most three lines + entropy)."""
    unc = state.get("unconscious", {})
    salt = ts.strftime("%Y%m%d%H")
    ranked = sorted(physics.UNCONSCIOUS, key=lambda n: float(unc.get(n, 0) or 0), reverse=True)
    lines: List[str] = []
    for name in ranked:
        tier = _tier(float(unc.get(name, 0) or 0))
        if tier is None:
            continue
        lines.append(_pick(TEXTURE[name][tier], f"{name}{salt}"))
        if len(lines) == 3:
            break
    entropy = float(state.get("modulators", {}).get("entropy", 0) or 0)
    if entropy > 95:
        lines.append(_pick(ENTROPY_CRITICAL, f"entropy{salt}"))
    elif entropy > 80:
        lines.append(_pick(ENTROPY_HIGH, f"entropy{salt}"))
    return lines or [_pick(QUIET, f"quiet{salt}")]


def drives_block(state: Dict[str, Any]) -> List[str]:
    eff = physics.effective_drives(state)
    dominant = max(eff, key=eff.get)
    lines = ["[DRIVES]"]
    for key, label in DRIVE_LABELS:
        mark = "  <- dominant" if key == dominant else ""
        lines.append(f"{label + ':':<14}{eff[key]:>3}/100{mark}")
    return lines


def modulators_block(state: Dict[str, Any]) -> List[str]:
    m = state.get("modulators", {})
    f = lambda k: f"{float(m.get(k, 0) or 0):.2f}"  # noqa: E731
    return [
        "[MODULATORS]",
        f"cortisol: {f('cortisol')}  dopamine: {f('dopamine')}  serotonin: {f('serotonin')}",
        f"entropy:  {int(float(m.get('entropy', 0) or 0))}    melatonin: {f('melatonin')}  "
        f"adrenaline: {f('adrenaline')}  oxytocin: {f('oxytocin_global')}",
    ]


def peer_lines(drives: Dict[str, Any], key: str, peer: Dict[str, Any], ts: datetime) -> List[str]:
    label = peer.get("label") or "unknown"
    last = store.parse_time(peer.get("last_interaction"))
    seen = f"{social.span(store.hours_between(last, ts))} ago" if last else "never"
    lines = [
        f"{key} — {label}",
        f"  affinity: {int(peer.get('affinity', 0))}  trust: {int(peer.get('trust', 0))}  "
        f"disappointment: {int(peer.get('disappointment', 0))}  curiosity: {int(peer.get('curiosity', 0))}  "
        f"oxytocin: {int(peer.get('oxytocin', 0))}  disposition: {social.disposition(drives, peer)}",
        f"  last seen: {seen}  no_response_streak: {int(peer.get('no_response_streak', 0))}",
    ]
    facts = [str(f) for f in peer.get("known_facts") or []][-3:]
    if facts:
        lines.append("  known: " + " | ".join(facts))
    outreach = peer.get("outreach")
    if isinstance(outreach, dict) and outreach.get("status") in ("open", "expired"):
        sent = social.clock(outreach.get("sent_at"))
        if outreach["status"] == "open":
            deadline = store.parse_time(outreach.get("deadline"))
            left = social.span(max(0.0, (deadline - ts).total_seconds() / 3600)) if deadline else "?"
            lines.append(f"  waiting: you reached out at {sent} — window closes in {left}")
        else:
            lines.append(f"  unanswered: you reached out at {sent} — window closed "
                         f"{social.clock(outreach.get('deadline'))}, still no answer")
    return lines


def interlocutors_block(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime,
                        limit: int = 5) -> List[str]:
    if not peers:
        return ["[INTERLOCUTORS]", "none yet"]
    ranked = sorted(peers.items(), key=lambda kv: kv[1].get("last_interaction") or "", reverse=True)
    lines = ["[INTERLOCUTORS]"]
    for key, peer in ranked[:limit]:
        lines.extend(peer_lines(drives, key, peer, ts))
    return lines


def events_block(events: Iterable[Dict[str, Any]]) -> List[str]:
    lines = []
    for record in events:
        ts = store.parse_time(record.get("ts"))
        stamp = ts.strftime("%H:%M") if ts else "--:--"
        lines.append(f"- {stamp} {record.get('text', '')}")
    return ["[SINCE LAST PULSE]"] + lines if lines else []


def header(state: Dict[str, Any], ts: datetime, extra: str = "") -> List[str]:
    meta = state.get("meta", {})
    interval = limits.clamp_wake_interval(meta.get("next_pulse_in_hours", 4))
    remaining = max(0, limits.DAILY_TOKEN_BUDGET - int(meta.get("tokens_used_today", 0) or 0))
    line = f"Next wake in: {interval:.1f}h | Token budget remaining today: {remaining:,}"
    return [f"[INTERNAL STATE — {store.iso(ts)}]", line + (f" | {extra}" if extra else "")]
