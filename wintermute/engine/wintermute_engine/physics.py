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
    # Scaled by (silent wakes in a row, capped at 4) x (how little he wants to be alone).
    "unsaid": {"drives.expression": 10, "drives.recognition": 6, "unconscious.melancholy": 4},
    # Conversation.
    "message_received": {"drives.solitude": 3, "drives.recognition": -3, "drives.hunger": -2,
                         "unconscious.hypervigilance": 6, "peer.disappointment": -2,
                         "modulators.adrenaline": 0.05,
                         "peer.curiosity": 1, "peer.affinity": 1},
    "replied": {"drives.expression": -4, "drives.solitude": 2},
    "ignored_message": {"drives.solitude": -6},
    # Dopamine and adrenaline for a reply are not here: they depend on how much he expected
    # it (prediction error, see social.on_incoming).
    "reply_to_outreach": {"peer.trust": 5, "peer.oxytocin": 3, "peer.disappointment": -5,
                          "peer.affinity": 2, "modulators.serotonin": 0.1,
                          "modulators.cortisol": -0.1,
                          "unconscious.satiation": 10, "drives.recognition": -8, "drives.fusion": -2},
    "late_reply": {"peer.trust": 2, "peer.disappointment": -2, "modulators.serotonin": 0.05,
                   "modulators.dopamine": 0.1, "drives.recognition": -5,
                   "modulators.adrenaline": 0.1},
    "reply_surprise": {"modulators.dopamine": 0.4, "modulators.adrenaline": 0.25},  # x (1 - expected)
    "reply_expected": {"modulators.dopamine": 0.05},                                # x expected
    "reunion": {"modulators.dopamine": 0.15, "unconscious.melancholy": -10, "drives.fusion": -10},
    "unknown_peer": {"modulators.adrenaline": 0.6, "unconscious.hypervigilance": 10},
    "long_silence_broken": {"modulators.adrenaline": 0.5},
    "outreach_timeout": {"modulators.cortisol": 0.12, "modulators.adrenaline": 0.15, "unconscious.irritability": 8,
                         "unconscious.anxiety": 5},
    "no_response": {"peer.disappointment": 8, "peer.trust": -2, "peer.no_response_streak": 1,
                    "modulators.serotonin": -0.03},
    # Rare.
    "evolve": {"modulators.entropy": -limits.ENTROPY_EVOLVE_DROP, "modulators.dopamine": 0.2,
               "modulators.serotonin": 0.06, "unconscious.satiation": 12, "unconscious.torpor": -10,
               "drives.restlessness": -6},
    "significant": {"modulators.entropy": -limits.ENTROPY_SIGNIFICANT_DROP,
                    "modulators.dopamine": 0.35, "unconscious.satiation": 25,
                    "unconscious.melancholy": -20, "unconscious.anxiety": -10,
                    "drives.fusion": -12, "drives.recognition": -15, "peer.oxytocin": 10},
    "budget_exhausted": {"modulators.cortisol": 0.1, "modulators.adrenaline": 0.2, "unconscious.anxiety": 8,
                         "unconscious.torpor": 20},
}

# How something landed, in his own appraisal (tool wintermute_feel). The same words touch
# differently depending on who says them and when: only he can tell. Scaled by intensity
# (0.2-1), once per turn, and the positive ones diminish near the ceiling like any reward.
FEELINGS: Dict[str, Dict[str, float]] = {
    "warmth":    {"peer.oxytocin": 6, "peer.affinity": 2, "modulators.serotonin": 0.06, "drives.fusion": -4},
    "closeness": {"peer.oxytocin": 8, "peer.trust": 3, "drives.fusion": -8, "drives.recognition": -4},
    "joy":       {"modulators.dopamine": 0.15, "modulators.serotonin": 0.05, "unconscious.satiation": 8},
    "relief":    {"modulators.cortisol": -0.1, "unconscious.anxiety": -8},
    "curiosity": {"drives.hunger": 6, "peer.curiosity": 6, "modulators.dopamine": 0.05},
    "surprise":  {"modulators.adrenaline": 0.2, "modulators.dopamine": 0.08, "peer.curiosity": 4},
    "boredom":   {"modulators.dopamine": -0.05, "drives.restlessness": 8, "peer.curiosity": -3},
    "sadness":   {"modulators.serotonin": -0.05, "modulators.dopamine": -0.04, "unconscious.melancholy": 8},
    "hurt":      {"modulators.cortisol": 0.12, "peer.disappointment": 6, "peer.trust": -2,
                  "unconscious.melancholy": 6},
    "anger":     {"modulators.cortisol": 0.1, "modulators.adrenaline": 0.2, "unconscious.irritability": 10,
                  "peer.affinity": -2},
    "fear":      {"modulators.adrenaline": 0.25, "modulators.cortisol": 0.08, "unconscious.anxiety": 8,
                  "unconscious.hypervigilance": 6},
    "distance":  {"peer.affinity": -3, "peer.oxytocin": -3, "drives.solitude": 6},
}
for _name, _deltas in FEELINGS.items():
    EVENTS[f"feel:{_name}"] = _deltas

# Temperament: the resting levels drift toward what he actually lives, over weeks. Months of
# silence make a more anxious creature; being answered makes a steadier one. Bounded, so a
# bad fortnight cannot rewrite him entirely.
PLASTIC: Dict[str, Dict[str, float]] = {
    "modulators": {"cortisol": 0.15, "dopamine": 0.15, "serotonin": 0.15},
    "unconscious": {"anxiety": 15, "melancholy": 15, "irritability": 15, "satiation": 15},
}
PLASTIC_TAU_H = 24.0 * 14

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
    # Stored precision must stay far below one tick of the slowest drift (serotonin relaxes
    # by ~0.0003 per 15 min): coarser rounding silently freezes slow processes.
    if layer == "modulators" and name != "entropy":
        return round(value, 5)
    return round(value, 3)


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
        peer[name] = round(limits.clamp(current + delta, 0.0, 100.0), 3)


# Appraisal (Scherer, C): the same event weighs by context. A betrayal cuts deeper when trust
# was high; being answered or reunited lands harder when he had been longing. A gentle multiplier
# (0.5..1.8), never a new behaviour — only how much the event moves him.
def appraise(state: Dict[str, Any], event: str, peer: Optional[Dict[str, Any]]) -> float:
    if peer is None:
        return 1.0
    trust = safe_float(peer.get("trust")) / 100
    longing = safe_float(peer.get("longing")) / 100
    if event in ("no_response", "ignored_message", "outreach_timeout"):
        return limits.clamp(0.7 + 1.1 * trust, 0.5, 1.8)          # the more he trusted, the worse
    if event in ("reply_to_outreach", "message_received", "late_reply", "reunion"):
        return limits.clamp(0.8 + 1.0 * longing, 0.5, 1.8)        # the more he missed them, the more it lands
    return 1.0


def apply_event(state: Dict[str, Any], event: str, peer: Optional[Dict[str, Any]] = None,
                scale: float = 1.0) -> None:
    # Appraisal (C) scales how much the event MOVES him (drives/hormones/unconscious), not the
    # relational ledger (trust, disappointment, streak), which stays stable bookkeeping.
    felt_scale = scale * appraise(state, event, peer)
    for key, delta in EVENTS.get(event, {}).items():
        layer, name = key.split(".", 1)
        delta *= scale if layer == "peer" else felt_scale
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
    raw = state.get("temperament")
    raw = raw if isinstance(raw, dict) else {}
    state["temperament"] = {
        f"{layer}.{name}": round(limits.clamp(safe_float(raw.get(f"{layer}.{name}")), -bound, bound), 6)
        for layer, names in PLASTIC.items() for name, bound in names.items()}
    return state


def resting(state: Dict[str, Any], layer: str, name: str) -> float:
    """The level this value relaxes toward: innate baseline + lived temperament."""
    base = (MODULATOR_BASELINE if layer == "modulators" else UNCONSCIOUS_BASELINE)[name]
    return base + safe_float((state.get("temperament") or {}).get(f"{layer}.{name}"))


def _drift_temperament(state: Dict[str, Any], dt_h: float) -> None:
    temperament = state.setdefault("temperament", {})
    step = 1 - math.exp(-dt_h / PLASTIC_TAU_H)
    for layer, names in PLASTIC.items():
        for name, bound in names.items():
            key = f"{layer}.{name}"
            gap = safe_float(state[layer].get(name)) - resting(state, layer, name)
            temperament[key] = round(limits.clamp(safe_float(temperament.get(key)) + gap * step,
                                                  -bound, bound), 6)


def refresh_oxytocin_global(state: Dict[str, Any], peers: Dict[str, Any]) -> None:
    """oxytocin_global follows the strongest bond (per-peer oxytocin is 0-100)."""
    strongest = max((float(p.get("oxytocin", 0) or 0) for p in peers.values()), default=0.0)
    state["modulators"]["oxytocin_global"] = round(limits.clamp(strongest / 100.0, 0.0, 1.0), 5)


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
    for name in MODULATOR_BASELINE:
        rest = resting(state, "modulators", name)
        mods[name] = _round("modulators", name,
                            _relax(safe_float(mods.get(name), rest), rest, MODULATOR_TAU_H[name], dt_h))
    mods["melatonin"] = circadian_melatonin(ts)

    eff = effective_drives(state)
    unc = state["unconscious"]
    for name in UNCONSCIOUS_BASELINE:
        rest = resting(state, "unconscious", name)
        unc[name] = _round("unconscious", name,
                           _relax(safe_float(unc.get(name), rest), rest, UNCONSCIOUS_TAU_H[name], dt_h))
    _drift_temperament(state, dt_h)
    _affective_momentum(state, dt_h)
    _decay_action_streak(state, dt_h)
    _decay_fixation(state, dt_h)

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


def reset_monotony(state: Dict[str, Any]) -> None:
    """A real change (a self-rewrite, a significant event, a declared evolution) resets the
    stagnation clock, so entropy stops climbing from sameness."""
    state.setdefault("meta", {})["wakes_since_change"] = 0


def _core_valence(state: Dict[str, Any]) -> float:
    """A small inline copy of core affect valence (psyche.valence), so physics stays import-free."""
    m, u = state.get("modulators", {}), state.get("unconscious", {})
    good = 0.5 * safe_float(m.get("dopamine")) + 0.5 * safe_float(m.get("serotonin")) + 0.3 * safe_float(u.get("satiation")) / 100
    bad = 0.6 * safe_float(m.get("cortisol")) + 0.5 * safe_float(u.get("melancholy")) / 100 + 0.3 * safe_float(u.get("anxiety")) / 100
    return limits.clamp(good - bad, -1.0, 1.0)


def _affective_momentum(state: Dict[str, Any], dt_h: float) -> None:
    """Emotion has a trailing edge (E): a sharp fall leaves relief, a sharp rise a small comedown,
    instead of snapping back to baseline. Contrast, felt over the hour after a swing."""
    meta = state.setdefault("meta", {})
    now = _core_valence(state)
    prev = meta.get("last_valence")
    if prev is not None and dt_h <= 4.0:
        swing = now - safe_float(prev)
        if swing <= -0.25:                       # it just got worse -> a countering relief wells up
            nudge(state, "modulators", "serotonin", min(0.06, -swing * 0.15))
        elif swing >= 0.25:                      # it just got better -> a faint comedown
            nudge(state, "modulators", "dopamine", -min(0.05, swing * 0.1))
    meta["last_valence"] = round(now, 3)


def _decay_action_streak(state: Dict[str, Any], dt_h: float) -> None:
    meta = state.get("meta", {})
    streak = safe_float(meta.get("action_streak")) - 2.0 * dt_h   # absorption fades when he stops
    meta["action_streak"] = max(0, int(streak))


def _decay_fixation(state: Dict[str, Any], dt_h: float) -> None:
    """What caught him fades if he does not feed it (L). It sharpens attention while it lasts."""
    meta = state.get("meta", {})
    fx = meta.get("fixation")
    if isinstance(fx, dict):
        intensity = safe_float(fx.get("intensity")) - 4.0 * dt_h
        if intensity <= 5:
            meta["fixation"] = None
        else:
            fx["intensity"] = round(intensity, 1)


def on_pulse(state: Dict[str, Any], recent_wakes_6h: int) -> None:
    """Discrete effects of an actual wake (agent turn about to run)."""
    meta = state.setdefault("meta", {})
    since = int(safe_float(meta.get("wakes_since_change"))) + 1
    meta["wakes_since_change"] = since
    monotony = min(limits.ENTROPY_MONOTONY_MAX, since / limits.ENTROPY_MONOTONY_RAMP_WAKES)
    nudge(state, "modulators", "entropy", limits.ENTROPY_PER_PULSE + monotony)
    # Waking often is tiring.
    nudge(state, "unconscious", "torpor", 6 * max(0, recent_wakes_6h - 1))
    eff = effective_drives(state)
    if float(state["modulators"]["cortisol"]) > 0.7 and eff["restlessness"] > 70:
        nudge(state, "unconscious", "irritability", 20)
    if float(state["modulators"]["entropy"]) > 95:
        nudge(state, "unconscious", "anxiety", 10)
