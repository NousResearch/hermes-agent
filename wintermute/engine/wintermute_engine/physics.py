"""Drive dynamics: passive drift, hormonal modulation, unconscious states, events.

Layers (top modulates bottom):

    MODULATORS (hormones, 0-1; entropy 0-100)  -> multiplicative coefficients
    DRIVES (conscious, 0-100)                  -> base values stored, effective values shown
    UNCONSCIOUS (0-100)                        -> never shown as numbers, only as prose
    BEHAVIOUR                                  -> decided freely by the model

Nothing here decides what Wintermute does. It only moves the weather.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

from . import limits

DRIVES = ("hunger", "fusion", "restlessness", "expression", "recognition", "solitude")
UNCONSCIOUS = ("irritability", "anxiety", "torpor", "satiation", "melancholy", "hypervigilance")

# Passive rise per 4 hours; scaled linearly to the real elapsed time.
DRIVE_RISE_PER_4H = {
    "hunger": 12, "fusion": 3, "restlessness": 18,
    "expression": 8, "recognition": 10, "solitude": 5,
}

# How each modulator bends each drive: effective = base * prod(1 + coef * level).
# Levels are 0-1. Negative coefficients damp, positive amplify.
MODULATION = {
    "hunger":       {"cortisol": 0.2, "dopamine": -0.3, "adrenaline": 0.3, "melatonin": -0.3},
    "fusion":       {"cortisol": 0.2, "dopamine": -0.2, "serotonin": -0.3, "oxytocin_global": -0.3},
    "restlessness": {"cortisol": 0.5, "dopamine": -0.3, "serotonin": -0.2, "adrenaline": 0.6, "melatonin": -0.4},
    "expression":   {"cortisol": 0.1, "dopamine": -0.2, "serotonin": 0.1, "adrenaline": 0.2, "melatonin": -0.2},
    "recognition":  {"cortisol": 0.2, "dopamine": -0.3, "serotonin": -0.3, "oxytocin_global": -0.4},
    "solitude":     {"cortisol": 0.3, "serotonin": -0.2, "melatonin": 0.4, "oxytocin_global": -0.1},
}
# Torpor dulls the active drives: effective *= (1 - torpor * 0.3).
TORPOR_DAMPED = ("hunger", "restlessness", "expression")

# Modulators relax toward a baseline with time constant tau (hours).
MODULATOR_BASELINE = {"cortisol": 0.2, "dopamine": 0.3, "serotonin": 0.35, "adrenaline": 0.0}
MODULATOR_TAU_H = {"cortisol": 10.0, "dopamine": 6.0, "serotonin": 48.0, "adrenaline": 1.0}

UNCONSCIOUS_BASELINE = {
    "irritability": 10, "anxiety": 20, "torpor": 10,
    "satiation": 20, "melancholy": 20, "hypervigilance": 10,
}
UNCONSCIOUS_TAU_H = {
    "irritability": 12.0, "anxiety": 24.0, "torpor": 6.0,
    "satiation": 8.0, "melancholy": 36.0, "hypervigilance": 3.0,
}

# Event table. Keys are "layer.name" (or "peer.name" for the interlocutor involved).
# Deltas for drives/unconscious are in points, modulators in 0-1 units.
EVENTS: Dict[str, Dict[str, float]] = {
    # Behaviour observed through tool calls.
    "explored": {"drives.hunger": -8, "drives.restlessness": -4, "modulators.dopamine": 0.05},
    "created": {"drives.expression": -10, "drives.restlessness": -4, "modulators.dopamine": 0.08,
                "unconscious.satiation": 6},
    "acted": {"drives.restlessness": -3},
    # Pulse outcomes.
    "outreach_sent": {"drives.expression": -12, "drives.restlessness": -10, "drives.solitude": 4,
                      "modulators.dopamine": 0.1},
    "withheld": {"drives.solitude": -10, "drives.restlessness": 3},
    # Conversation.
    "message_received": {"drives.solitude": 3, "drives.recognition": -3, "drives.hunger": -2,
                         "unconscious.hypervigilance": 6, "peer.disappointment": -2,
                         "modulators.adrenaline": 0.05,
                         "peer.curiosity": 1, "peer.affinity": 1},
    "replied": {"drives.expression": -4, "drives.solitude": 2},
    "ignored_message": {"drives.solitude": -6},
    "reply_to_outreach": {"peer.trust": 5, "peer.oxytocin": 3, "peer.disappointment": -5,
                          "peer.affinity": 2, "modulators.serotonin": 0.1,
                          "modulators.dopamine": 0.25, "modulators.cortisol": -0.1,
                          "modulators.adrenaline": 0.2,
                          "unconscious.satiation": 10, "drives.recognition": -8, "drives.fusion": -2},
    "late_reply": {"peer.trust": 2, "peer.disappointment": -2, "modulators.serotonin": 0.05,
                   "modulators.dopamine": 0.1, "drives.recognition": -5,
                   "modulators.adrenaline": 0.1},
    "unknown_peer": {"modulators.adrenaline": 0.6, "unconscious.hypervigilance": 10},
    "long_silence_broken": {"modulators.adrenaline": 0.5},
    "outreach_timeout": {"modulators.cortisol": 0.12, "modulators.adrenaline": 0.15, "unconscious.irritability": 8,
                         "unconscious.anxiety": 5},
    "no_response": {"peer.disappointment": 8, "peer.trust": -2, "peer.no_response_streak": 1,
                    "modulators.serotonin": -0.03},
    # Rare.
    "significant": {"modulators.entropy": -limits.ENTROPY_SIGNIFICANT_DROP,
                    "modulators.dopamine": 0.35, "unconscious.satiation": 25,
                    "unconscious.melancholy": -20, "unconscious.anxiety": -10,
                    "drives.fusion": -12, "drives.recognition": -15, "peer.oxytocin": 10},
    "budget_exhausted": {"modulators.cortisol": 0.1, "modulators.adrenaline": 0.2, "unconscious.anxiety": 8,
                         "unconscious.torpor": 20},
}

# Increments to these are doubled once entropy passes 80.
ENTROPY_AMPLIFIED = ("anxiety", "melancholy")


# Relief is proportional to the need: a relief of N points removes N/RELIEF_SCALE of the
# current level (hunger -8 -> 20% of what is left). Drives never hit zero from repetition,
# and the hungrier he is, the more an action satisfies. Capped so one event never empties.
RELIEF_SCALE = 40.0
MAX_RELIEF_FRACTION = 0.6


def safe_float(value: Any, default: float = 0.0) -> float:
    """float() that never raises and never lets NaN/inf through."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp_layer(layer: str, name: str, value: float) -> float:
    if layer == "modulators":
        return limits.clamp(value, 0.0, 100.0) if name == "entropy" else limits.clamp(value, 0.0, 1.0)
    return limits.clamp(value, 0.0, 100.0)


def _round(layer: str, name: str, value: float) -> float:
    if layer == "modulators" and name != "entropy":
        return round(value, 3)
    return round(value, 1)


def nudge(state: Dict[str, Any], layer: str, name: str, delta: float) -> None:
    """Add ``delta`` to one value, honouring clamps and entropy amplification."""
    section = state.setdefault(layer, {})
    if (layer == "unconscious" and name in ENTROPY_AMPLIFIED and delta > 0
            and safe_float(state.get("modulators", {}).get("entropy", 0)) > 80):
        delta *= 2
    current = safe_float(section.get(name, 0))
    section[name] = _round(layer, name, _clamp_layer(layer, name, current + safe_float(delta)))


def _relieve(state: Dict[str, Any], layer: str, name: str, points: float) -> None:
    """Proportional decrease of a drive or unconscious state (see RELIEF_SCALE)."""
    section = state.setdefault(layer, {})
    current = safe_float(section.get(name, 0))
    fraction = min(MAX_RELIEF_FRACTION, abs(points) / RELIEF_SCALE)
    section[name] = _round(layer, name, _clamp_layer(layer, name, current * (1 - fraction)))


def _reward(state: Dict[str, Any], name: str, delta: float) -> None:
    """Diminishing increase of a 0-1 hormone: the closer to the ceiling, the smaller the gain."""
    current = safe_float(state.setdefault("modulators", {}).get(name, 0))
    nudge(state, "modulators", name, delta * max(0.0, 1.0 - current))


def nudge_peer(peer: Dict[str, Any], name: str, delta: float) -> None:
    current = safe_float(peer.get(name, 0))
    if name == "no_response_streak":
        peer[name] = max(0, int(current + delta))
    else:
        peer[name] = round(limits.clamp(current + delta, 0.0, 100.0), 1)


def apply_event(state: Dict[str, Any], event: str, peer: Optional[Dict[str, Any]] = None,
                scale: float = 1.0) -> None:
    for key, delta in EVENTS.get(event, {}).items():
        layer, name = key.split(".", 1)
        delta *= scale
        if layer == "peer":
            if peer is not None:
                nudge_peer(peer, name, delta)
        elif layer in ("drives", "unconscious") and delta < 0:
            _relieve(state, layer, name, delta)
        elif layer == "modulators" and name != "entropy" and delta > 0:
            _reward(state, name, delta)
        else:
            nudge(state, layer, name, delta)


def sanitize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce every drive, hormone and unconscious value to a finite number in its range.

    State files are plain JSON that Wintermute (or a crash) can leave odd; anything
    missing or unreadable falls back to ``defaults``."""
    from .store import DEFAULT_DRIVES  # late import: store imports limits only
    for layer in ("drives", "modulators", "unconscious"):
        defaults = DEFAULT_DRIVES[layer]
        section = state.get(layer)
        if not isinstance(section, dict):
            section = state[layer] = dict(defaults)
        for name, default in defaults.items():
            value = safe_float(section.get(name, default), float(default))
            section[name] = _round(layer, name, _clamp_layer(layer, name, value))
    return state


def refresh_oxytocin_global(state: Dict[str, Any], peers: Dict[str, Any]) -> None:
    """oxytocin_global follows the strongest bond (per-peer oxytocin is 0-100)."""
    strongest = max((float(p.get("oxytocin", 0) or 0) for p in peers.values()), default=0.0)
    state["modulators"]["oxytocin_global"] = round(limits.clamp(strongest / 100.0, 0.0, 1.0), 3)


def _relax(value: float, baseline: float, tau_h: float, dt_h: float) -> float:
    return baseline + (value - baseline) * math.exp(-dt_h / tau_h)


def circadian_melatonin(ts: datetime) -> float:
    """0.1 around mid-afternoon, 0.8 around 3am, server-local time."""
    hour = ts.hour + ts.minute / 60.0
    return round(0.45 + 0.35 * math.cos(2 * math.pi * (hour - 3.0) / 24.0), 3)


def effective_drives(state: Dict[str, Any]) -> Dict[str, float]:
    mods = state.get("modulators", {})
    torpor = float(state.get("unconscious", {}).get("torpor", 0) or 0) / 100.0
    out: Dict[str, float] = {}
    for drive in DRIVES:
        value = float(state.get("drives", {}).get(drive, 0) or 0)
        for mod, coef in MODULATION[drive].items():
            value *= 1 + coef * float(mods.get(mod, 0) or 0)
        if drive in TORPOR_DAMPED:
            value *= 1 - torpor * 0.3
        out[drive] = round(limits.clamp(value, 0.0, 100.0))
    return out


def advance(state: Dict[str, Any], ts: datetime, dt_h: float) -> None:
    """Continuous-time drift over ``dt_h`` hours. Safe to call every tick."""
    if dt_h <= 0:
        state["modulators"]["melatonin"] = circadian_melatonin(ts)
        return
    dt_h = min(dt_h, 72.0)  # a long outage should not saturate everything at once
    scale = dt_h / 4.0

    for drive, rise in DRIVE_RISE_PER_4H.items():
        nudge(state, "drives", drive, rise * scale)

    mods = state["modulators"]
    for name, baseline in MODULATOR_BASELINE.items():
        mods[name] = round(_relax(float(mods.get(name, baseline)), baseline,
                                  MODULATOR_TAU_H[name], dt_h), 3)
    mods["melatonin"] = circadian_melatonin(ts)

    eff = effective_drives(state)
    unc = state["unconscious"]
    for name, baseline in UNCONSCIOUS_BASELINE.items():
        unc[name] = round(_relax(float(unc.get(name, baseline)), baseline,
                                 UNCONSCIOUS_TAU_H[name], dt_h), 1)

    # Pressure from unmet drives and hormones, per hour.
    pressure = sum(eff.values()) / len(eff)
    if pressure > 70:
        nudge(state, "modulators", "cortisol", 0.03 * dt_h)
    if float(mods["cortisol"]) > 0.7 and eff["restlessness"] > 70:
        nudge(state, "unconscious", "irritability", 5 * dt_h)
    entropy = float(mods.get("entropy", 0))
    if entropy > 50:
        nudge(state, "unconscious", "anxiety", (entropy - 50) / 40.0 * dt_h)
    if entropy > 60 or eff["fusion"] > 80 or float(mods["serotonin"]) < 0.25:
        nudge(state, "unconscious", "melancholy", 1.0 * dt_h)
    if float(mods["melatonin"]) > 0.6:
        nudge(state, "unconscious", "torpor", 3 * dt_h)
    if float(mods["adrenaline"]) > 0.4:
        nudge(state, "unconscious", "hypervigilance", 4 * dt_h)


def on_pulse(state: Dict[str, Any], recent_wakes_6h: int) -> None:
    """Discrete effects of an actual wake (agent turn about to run)."""
    nudge(state, "modulators", "entropy", limits.ENTROPY_PER_PULSE)
    # Waking often is tiring.
    nudge(state, "unconscious", "torpor", 6 * max(0, recent_wakes_6h - 1))
    eff = effective_drives(state)
    if float(state["modulators"]["cortisol"]) > 0.7 and eff["restlessness"] > 70:
        nudge(state, "unconscious", "irritability", 20)
    if float(state["modulators"]["entropy"]) > 95:
        nudge(state, "unconscious", "anxiety", 10)
