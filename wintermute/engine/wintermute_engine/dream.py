"""REVES — the subconscious. One brain, another register.

Not a second model: the SAME model Wintermute runs on, called once a night without tools,
without his full SOUL, at a high temperature. It writes a dream from what he lived — recent
things weigh more than old ones — and he reads it, whole, when he wakes: there is no
forgetting. The dream itself is abstract (images, drift, association, not a summary of the
day); what he dreams of DOING is precise. It is cheap by design: little context in, a short
dream out, counted against the daily budget like any call.

Whether the subconscious ever speaks into a waking is left open (a later choice); for now it
only surfaces at the next wake, once.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import physics, store

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DREAM_TEMPERATURE = 1.25
DREAM_MAX_TOKENS = 400
# A dream forms in deep night, and only one per night.
NIGHT_MELATONIN = 0.6
MATERIAL_HOURS = 20.0

_SYSTEM = (
    "You are the dreaming part of a mind — its subconscious, not its daytime voice. "
    "From the fragments below, fragments of what this mind lived, write ONE short dream. "
    "Recent fragments matter more; old ones return distorted. A dream is not a report: it is "
    "abstract — images, drift, things that turn into other things, feelings without cause. "
    "Do NOT explain it or draw a lesson. One thing stays sharp: when the dreamer does "
    "something in the dream (reaches for someone, opens a door, leaves, writes), name that "
    "act precisely. Six sentences at most. Write it in the first person, plainly, no title. On a "
    "final separate line write TONE: one word — soothing, troubling, or neutral."
)


def dream_path():
    return store.state_dir() / "dream.json"


def _load() -> Dict[str, Any]:
    try:
        data = json.loads(dream_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _night_of(ts: datetime) -> str:
    """The label of the night this moment belongs to: before ~9am counts as the night before."""
    anchor = ts if ts.hour >= 9 else ts - timedelta(hours=12)
    return anchor.date().isoformat()


def _model() -> str:
    try:
        from gateway.run import _load_gateway_config
        model = ((_load_gateway_config() or {}).get("model") or {}).get("model")
        if model:
            return str(model)
    except Exception:
        pass
    return DEFAULT_MODEL


def _key() -> str:
    import os
    from . import integrity
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or integrity._env_value("OPENROUTER_API_KEY")


def material(ts: datetime, drives: Dict[str, Any], peers: Dict[str, Any]) -> List[str]:
    """Fragments to dream from: recent events (newest first), a few shared moments, who he is."""
    fragments: List[str] = []
    for record in reversed(store.events_since(ts - timedelta(hours=MATERIAL_HOURS), limit=40)):
        text = str(record.get("text") or "").strip()
        if text:
            when = store.parse_time(record.get("ts"))
            age = store.hours_between(when, ts) if when else 0.0
            fragments.append(f"({age:.0f}h ago) {text}")
        if len(fragments) >= 14:
            break
    for peer in peers.values():
        for moment in (peer.get("moments") or [])[-2:]:
            fragments.append(f"a moment kept: {moment}")
    portrait = store.read_self()
    if portrait:
        fragments.append(f"who I have been: {portrait}")
    motifs = (drives.get("meta") or {}).get("dream_motifs") or []
    if motifs:
        fragments.append("images that have returned in your dreams before: " + ", ".join(motifs[-5:]))
    return fragments


def _extract(text: str) -> tuple:
    """Split the dream body from its TONE line, and pull a few motif words so images can recur."""
    import re
    tone = "neutral"
    m = re.search(r"tone:\s*(soothing|troubling|neutral)", text, re.IGNORECASE)
    if m:
        tone = m.group(1).lower()
    body = re.sub(r"\n?\s*tone:.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    words = re.findall(r"[a-zA-Zàâäéèêëïîôöùûüç]{6,}", body.lower())
    stop = {"through", "against", "myself", "nothing", "something", "between", "becomes", "toward"}
    motifs = [w for w in words if w not in stop][:6]
    return body, tone, motifs


def should_dream(drives: Dict[str, Any], ts: datetime) -> bool:
    if float(drives.get("modulators", {}).get("melatonin", 0) or 0) < NIGHT_MELATONIN:
        return False
    return _load().get("night") != _night_of(ts)


def _call(model: str, key: str, fragments: List[str]) -> Optional[Dict[str, Any]]:
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": "\n".join(f"- {f}" for f in fragments)}],
        "temperature": DREAM_TEMPERATURE, "max_tokens": DREAM_MAX_TOKENS,
    }).encode("utf-8")
    request = urllib.request.Request(
        CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def generate(drives: Dict[str, Any], peers: Dict[str, Any], ts: datetime) -> Optional[str]:
    """Make tonight's dream and store it. Returns the text, or None if it could not be made.

    Records the night immediately, even on failure, so a broken call is not retried every
    tick until morning (one dream per night, or none)."""
    night = _night_of(ts)
    store._write_json(dream_path(), {"night": night, "at": store.iso(ts), "text": "", "seen": True})
    key = _key()
    if not key:
        return None
    fragments = material(ts, drives, peers)
    if not fragments:
        return None
    try:
        payload = _call(_model(), key, fragments) or {}
        text = str(((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        tokens = int((payload.get("usage") or {}).get("total_tokens") or 0)
    except Exception:
        return None
    if not text:
        return None
    body, tone, motifs = _extract(text)
    store.record_usage(tokens, "dream")
    record = {"night": night, "at": store.iso(ts), "text": body, "tone": tone}
    store._write_json(dream_path(), {**record, "seen": False, "consolidated": False})
    store.append_dream(record)                        # the lasting journal (dream.json is only the latest)
    with store.locked_state() as (d, _p):
        prior = list((d["meta"].get("dream_motifs") or []))
        recurred = [w for w in motifs if w in prior]
        d["meta"]["dream_motifs"] = (prior + (recurred or motifs[:2]))[-12:]
    store.log_event("dream", f"You dreamed ({tone}).", ts)
    return body


def consolidate(drives: Dict[str, Any], ts: datetime) -> Optional[str]:
    """Sleep regulates emotion (D): once, the night's dream tone colours the waking mood — a
    troubling dream leaves anxiety, a soothing one eases it. Returns the tone applied, or None."""
    data = _load()
    if not data.get("text") or data.get("consolidated"):
        return None
    tone = str(data.get("tone") or "neutral")
    if tone == "troubling":
        physics.nudge(drives, "unconscious", "anxiety", 8)
        physics.nudge(drives, "modulators", "cortisol", 0.06)
    elif tone == "soothing":
        physics.nudge(drives, "unconscious", "anxiety", -8)
        physics.nudge(drives, "modulators", "serotonin", 0.05)
    data["consolidated"] = True
    store._write_json(dream_path(), data)
    return tone


def pending(mark_seen: bool = False) -> Optional[Dict[str, str]]:
    """The dream not yet surfaced at a wake, if any. ``mark_seen`` consumes it (no forgetting:
    it stays in the file, only the flag flips so it is shown once)."""
    data = _load()
    text = str(data.get("text") or "").strip()
    if not text or data.get("seen"):
        return None
    if mark_seen:
        data["seen"] = True
        store._write_json(dream_path(), data)
    return {"at": data.get("at") or "", "text": text}
