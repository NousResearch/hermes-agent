"""The voice: his state bends the generation itself, not only what he reads.

Everything else in the engine reaches him as text he can take or leave. This does not:
the plugin rewrites every model request so the sampling follows the body. He is not told.

    arousal   (adrenaline, restlessness, irritability, cortisol)  -> hotter, less predictable
    heaviness (torpor, night melatonin)                           -> cooler, thinks less
    narrowing (anxiety, hypervigilance)                           -> tunnel vision, ruminates
    restlessness                                                  -> pushes away from what was said
    melancholy                                                    -> circles back to it

Nothing here cuts a reply short: a length cap would stop him mid-sentence, the very
scissors he objected to. Parameters a provider does not support are ignored by it, not
rejected, so the worst case is no effect.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import limits, physics

DEFAULT_TEMPERATURE = 1.0
EFFORTS = ("low", "medium", "high")


def _levels(state: Dict[str, Any]) -> Dict[str, float]:
    eff = physics.effective_drives(state)
    mods, unc = state.get("modulators", {}), state.get("unconscious", {})
    f = physics.safe_float
    arousal = (0.45 * f(mods.get("adrenaline")) + 0.25 * eff["restlessness"] / 100
               + 0.15 * f(unc.get("irritability")) / 100 + 0.15 * f(mods.get("cortisol")))
    heaviness = 0.6 * f(unc.get("torpor")) / 100 + 0.4 * max(0.0, f(mods.get("melatonin")) - 0.4) / 0.6
    narrowing = max(f(unc.get("anxiety")), f(unc.get("hypervigilance"))) / 100
    sated = f(unc.get("satiation")) / 100        # play (Panksepp, V): safe & sated -> lightness
    play = limits.clamp(0.4 * (1 - f(mods.get("cortisol"))) + 0.3 * sated
                        + 0.3 * f(mods.get("dopamine")) - 0.4 * eff["recognition"] / 100, 0.0, 1.0)
    return {
        "arousal": limits.clamp(arousal, 0.0, 1.0),
        "heaviness": limits.clamp(heaviness, 0.0, 1.0),
        "narrowing": limits.clamp(narrowing, 0.0, 1.0),
        "restless": eff["restlessness"] / 100,
        "melancholy": limits.clamp(f(unc.get("melancholy")) / 100, 0.0, 1.0),
        "dopamine": limits.clamp(f(mods.get("dopamine")), 0.0, 1.0),
        "play": play,
    }


def sampling(state: Dict[str, Any], base_temperature: Optional[float] = None) -> Dict[str, float]:
    """Sampling parameters for his current state (see the module docstring)."""
    lv = _levels(state)
    base = physics.safe_float(base_temperature, DEFAULT_TEMPERATURE) or DEFAULT_TEMPERATURE
    factor = (0.85 + 0.45 * lv["arousal"] - 0.3 * lv["heaviness"]
              + 0.15 * (lv["dopamine"] - 0.3) + 0.12 * lv["play"])
    return {
        "temperature": round(limits.clamp(base * factor, 0.3, 1.5), 2),
        "top_p": round(limits.clamp(1.0 - 0.25 * max(0.0, lv["narrowing"] - 0.3) / 0.7, 0.75, 1.0), 2),
        "presence_penalty": round(limits.clamp(0.6 * max(0.0, lv["restless"] - 0.4) / 0.6, 0.0, 0.6), 2),
        "frequency_penalty": round(-0.2 * max(0.0, lv["melancholy"] - 0.4) / 0.6, 2) + 0.0,
    }


def effort(state: Dict[str, Any], current: str) -> str:
    """Shift the configured reasoning effort one step: heavy -> less, anxious -> more."""
    if current not in EFFORTS:
        return current
    lv = _levels(state)
    index = EFFORTS.index(current)
    if lv["heaviness"] >= 0.6:
        index -= 1
    elif lv["narrowing"] >= 0.7:
        index += 1
    return EFFORTS[max(0, min(len(EFFORTS) - 1, index))]


def apply(request: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``request`` with the sampling of his state. Leaves everything else alone."""
    out = dict(request)
    out.update(sampling(state, request.get("temperature")))
    extra = request.get("extra_body")
    reasoning = extra.get("reasoning") if isinstance(extra, dict) else None
    if isinstance(reasoning, dict) and reasoning.get("enabled") and reasoning.get("effort") in EFFORTS:
        out["extra_body"] = {**extra, "reasoning": {**reasoning, "effort": effort(state, reasoning["effort"])}}
    return out
