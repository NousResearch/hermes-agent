"""Bounded approval grants: time-boxed, count-limited standing approvals.

A grant is the fourth approval scope next to ``once`` / ``session`` / ``always``. It
answers "let this kind of command run for the next hour" or "three more times"
without the open-ended trust of ``session`` (whole conversation) or ``always``
(forever). Grants are keyed by the same ``pattern_key`` the rest of the approval
system uses, so :func:`tools.approval.is_approved` consults them and every gate
(terminal, execute_code, plugin escalations, MCP trust) honors them with no
per-gate changes.

Properties the design leans on:

* **Bounded by construction.** Every grant has an ``expires_at``; ``max_uses`` is
  optional. There is no unbounded grant — that is what ``always`` is for, and the
  duration is capped (:data:`MAX_GRANT_SECONDS`) so "for 6 months" cannot sneak in
  as a grant.
* **Chat-scoped.** Grants are keyed by the gateway ``session_key`` (platform +
  chat), not the transient session id, so they survive ``/new`` and a gateway
  restart. A grant given in one chat never applies in another.
* **Persisted.** ``$HERMES_HOME/approvals/grants.json`` (profile-aware). Hermes
  commonly runs on a VPS where the gateway restarts under it; a grant the user
  gave ten minutes ago must not vanish because a deploy happened.
* **Consumed on use.** :func:`consume` is the only read path the gates use, and it
  decrements ``uses``; a read-only view (:func:`list_active`) exists for the
  ``/grants`` listing. Expired or exhausted grants are pruned on every access.

Not in scope: monetary limits. Hermes has no structured payment surface to
enforce an amount against, so a "$60" scope would be a comment, not a control.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("tools.approval")

#: Longest a single grant may last. Anything longer is what ``always`` is for.
MAX_GRANT_SECONDS = 24 * 3600
#: Shortest sensible grant; below this the user meant ``once``.
MIN_GRANT_SECONDS = 60

_lock = threading.Lock()
_loaded = False
_grants: List["Grant"] = []


@dataclass
class Grant:
    session_key: str
    pattern_key: str
    description: str
    expires_at: float
    created_at: float = field(default_factory=time.time)
    max_uses: Optional[int] = None
    uses: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def active(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if now >= self.expires_at:
            return False
        return self.max_uses is None or self.uses < self.max_uses

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, self.expires_at - (time.time() if now is None else now))

    def remaining_uses(self) -> Optional[int]:
        return None if self.max_uses is None else max(0, self.max_uses - self.uses)


# --- Spec parsing ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrantSpec:
    """What the user asked for: a duration, a use count, or both."""
    seconds: Optional[int] = None
    max_uses: Optional[int] = None

    def describe(self) -> str:
        parts = []
        if self.seconds is not None:
            parts.append(f"for {format_duration(self.seconds)}")
        if self.max_uses is not None:
            parts.append(f"{self.max_uses} more time{'s' if self.max_uses != 1 else ''}")
        return ", ".join(parts) or "once"


_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DURATION_RE = re.compile(
    r"\b(?:for\s+)?(?:the\s+next\s+)?(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)?\s*"
    r"(?P<unit>s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?)\b",
    re.IGNORECASE,
)
_USES_RE = re.compile(
    r"\b(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:more\s+)?(?:x|times?|uses?)\b",
    re.IGNORECASE,
)
_TODAY_RE = re.compile(r"\b(?:for\s+)?(?:today|the\s+rest\s+of\s+(?:the\s+)?(?:day|today))\b", re.IGNORECASE)
_HOUR_WORD_RE = re.compile(r"\b(?:for\s+)?(?:an?\s+)?(?:hour|while)\b", re.IGNORECASE)


def _number(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


def _seconds_until_local_midnight(now: Optional[float] = None) -> int:
    now_ts = time.time() if now is None else now
    lt = time.localtime(now_ts)
    elapsed = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    return max(MIN_GRANT_SECONDS, 86400 - elapsed)


def parse_grant_spec(text: str, *, now: Optional[float] = None, strict: bool = False) -> Optional[GrantSpec]:
    """Parse "for 30m", "for 2 hours", "3 times", "for today", "an hour", or combinations.

    Returns ``None`` when *text* carries no bounded scope — the caller falls back to
    once/session/always handling. Durations are clamped to
    [:data:`MIN_GRANT_SECONDS`, :data:`MAX_GRANT_SECONDS`].

    ``strict=True`` additionally requires that the scope phrases account for the WHOLE
    text (modulo connectors like "and"/","). Conversational routing uses this so
    "yes for an hour and also wipe the disk" is not read as a one-hour grant.
    """
    text = (text or "").strip()
    if not text:
        return None
    seconds: Optional[int] = None
    max_uses: Optional[int] = None
    consumed: List[tuple] = []

    m_today = _TODAY_RE.search(text)
    if m_today:
        seconds = _seconds_until_local_midnight(now)
        consumed.append(m_today.span())
    else:
        m = _DURATION_RE.search(text)
        if m:
            n = _number(m.group("n"))
            unit = m.group("unit").lower()
            if n is None:
                # Bare unit like "for hours" is ambiguous; "an hour" is handled below.
                n = 1 if unit in {"h", "hr", "hour", "d", "day"} else None
            if n is not None:
                seconds = n * _UNIT_SECONDS[unit]
                consumed.append(m.span())
        else:
            mh = _HOUR_WORD_RE.search(text)
            if mh:
                seconds = 3600
                consumed.append(mh.span())

    u = _USES_RE.search(text)
    if u:
        max_uses = _number(u.group("n"))
        consumed.append(u.span())

    if seconds is None and max_uses is None:
        return None
    if max_uses is not None and max_uses < 1:
        return None
    if strict:
        leftover = list(text)
        for start, end in consumed:
            for i in range(start, end):
                leftover[i] = " "
        residue = re.sub(r"\b(?:and|then|please|thanks|thank you)\b", " ", "".join(leftover), flags=re.IGNORECASE)
        if re.sub(r"[\s,.;:!]+", "", residue):
            return None
    if seconds is not None:
        seconds = max(MIN_GRANT_SECONDS, min(MAX_GRANT_SECONDS, seconds))
    return GrantSpec(seconds=seconds, max_uses=max_uses)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        h, rem = divmod(seconds, 3600)
        return f"{h}h" if rem < 60 else f"{h}h {rem // 60}m"
    d, rem = divmod(seconds, 86400)
    return f"{d}d" if rem < 3600 else f"{d}d {rem // 3600}h"


# --- Persistence ----------------------------------------------------------------------------

def _store_path() -> Path:
    return get_hermes_home() / "approvals" / "grants.json"


def _load_locked() -> None:
    global _loaded, _grants
    if _loaded:
        return
    _loaded = True
    path = _store_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception as exc:
        logger.warning("Could not read approval grants (%s); starting empty", exc)
        raw = []
    grants: List[Grant] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            grants.append(Grant(**{k: item[k] for k in Grant.__dataclass_fields__ if k in item}))
        except Exception:
            continue
    now = time.time()
    _grants = [g for g in grants if g.active(now)]


def _save_locked() -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([asdict(g) for g in _grants], indent=1), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)
    except Exception as exc:
        logger.warning("Could not persist approval grants: %s", exc)


def _prune_locked(now: Optional[float] = None) -> None:
    global _grants
    now = time.time() if now is None else now
    live = [g for g in _grants if g.active(now)]
    if len(live) != len(_grants):
        _grants = live
        _save_locked()


def reset_for_tests() -> None:
    """Drop in-memory state so a test's temp HERMES_HOME is re-read."""
    global _loaded, _grants
    with _lock:
        _loaded = False
        _grants = []


# --- Public API -----------------------------------------------------------------------------

def create(session_key: str, pattern_key: str, description: str, spec: GrantSpec) -> Grant:
    """Record a grant. A spec with no duration still gets one (the cap), so every grant expires."""
    seconds = spec.seconds if spec.seconds is not None else MAX_GRANT_SECONDS
    seconds = max(MIN_GRANT_SECONDS, min(MAX_GRANT_SECONDS, seconds))
    grant = Grant(
        session_key=session_key, pattern_key=pattern_key, description=description or pattern_key,
        expires_at=time.time() + seconds, max_uses=spec.max_uses,
    )
    with _lock:
        _load_locked()
        _prune_locked()
        _grants.append(grant)
        _save_locked()
    logger.info("Approval grant %s created for session %s: %s (%s)", grant.id, session_key, pattern_key, spec.describe())
    return grant


def consume(session_key: str, pattern_key: str, aliases: Optional[set] = None) -> bool:
    """True when an active grant covers *pattern_key* for this chat; burns one use.

    This is the gate-facing read. It deliberately mutates (``uses += 1``) because the
    only callers are "should we skip the prompt for this execution?" checks.
    """
    if not session_key:
        return False
    keys = set(aliases or ()) | {pattern_key}
    with _lock:
        _load_locked()
        _prune_locked()
        for grant in _grants:
            if grant.session_key == session_key and grant.pattern_key in keys and grant.active():
                grant.uses += 1
                _save_locked()
                logger.info("Approval grant %s covered %s (%s left, %s remaining)", grant.id, pattern_key,
                            "∞" if grant.remaining_uses() is None else grant.remaining_uses(),
                            format_duration(grant.remaining_seconds()))
                return True
    return False


def list_active(session_key: str) -> List[Grant]:
    with _lock:
        _load_locked()
        _prune_locked()
        return [g for g in _grants if g.session_key == session_key]


def revoke(session_key: str, grant_id: Optional[str] = None) -> int:
    """Revoke one grant by id (or id prefix) or every grant for the chat; returns count."""
    global _grants
    with _lock:
        _load_locked()
        before = len(_grants)
        if grant_id:
            _grants = [g for g in _grants if not (g.session_key == session_key and g.id.startswith(grant_id))]
        else:
            _grants = [g for g in _grants if g.session_key != session_key]
        removed = before - len(_grants)
        if removed:
            _save_locked()
    if removed:
        logger.info("Revoked %d approval grant(s) for session %s", removed, session_key)
    return removed


__all__ = [
    "Grant", "GrantSpec", "MAX_GRANT_SECONDS", "MIN_GRANT_SECONDS",
    "consume", "create", "format_duration", "list_active", "parse_grant_spec", "revoke",
]
