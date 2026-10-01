"""Authority ranking and freshness resolution.

The spec defines a strict authority order:
  1. latest explicit Owner decision
  2. canonical current-state / register / contract
  3. approved project spec
  4. newer operational evidence
  5. older plans, logs, narrative history

Within the same rank, freshness decides: prefer a non-expired ``review_by``,
then ``last_updated``, then filesystem mtime.

An expired ``review_by``, missing required freshness, or unresolved same-/
higher-rank contradiction returns ``data_hold``. Lower-ranked text NEVER
silently overrides higher-ranked evidence — that's what makes the system
defensible. See acceptance tests #10 and #11.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class AuthorityRank(Enum):
    """Strict order (lower enum value = higher authority)."""

    OWNER_DECISION = 1
    CANONICAL_REGISTER = 2
    APPROVED_SPEC = 3
    NEWER_EVIDENCE = 4
    HISTORY = 5

    def __lt__(self, other: "AuthorityRank") -> bool:
        return self.value < other.value

    def __le__(self, other: "AuthorityRank") -> bool:
        return self.value <= other.value


@dataclass(frozen=True)
class Freshness:
    basis: str  # "review_by" | "last_updated" | "mtime"
    value: str  # ISO-8601 string
    stale: bool = False

    def as_datetime(self) -> datetime:
        # Try several common formats.
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(self.value, fmt)
            except ValueError:
                continue
        return datetime.min.replace(tzinfo=timezone.utc)

    def __lt__(self, other: "Freshness") -> bool:
        return self.as_datetime() < other.as_datetime()

    def __le__(self, other: "Freshness") -> bool:
        return self.as_datetime() <= other.as_datetime()

    def __gt__(self, other: "Freshness") -> bool:
        return self.as_datetime() > other.as_datetime()

    def __ge__(self, other: "Freshness") -> bool:
        return self.as_datetime() >= other.as_datetime()


@dataclass(frozen=True)
class SourceMetadata:
    path: str
    range_text: str  # "lines 1-20"
    characters_read: int
    authority_rank: AuthorityRank
    freshness: Freshness


@dataclass(frozen=True)
class Conflict:
    paths: Tuple[str, ...]
    rank: AuthorityRank
    detail: str


# Frontmatter parsing — deliberately simple and tolerant. We are not trying
# to be a full YAML parser; we just need the few keys we care about.
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(path: Path) -> Dict[str, str]:
    """Parse a YAML-ish frontmatter block from the top of a Markdown file.

    Returns a dict of ``key: value`` strings (whitespace-stripped). Tolerant
    of malformed input: returns whatever it can extract or ``{}`` if the
    block is missing/unparseable.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}
    body = m.group(1)
    out: Dict[str, str] = {}
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        k = key.strip()
        v = val.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def is_expired(review_by: Optional[str], now: Optional[datetime] = None) -> bool:
    """True iff review_by is a parseable ISO date strictly before now."""
    if not review_by:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        # ISO date (no time component)
        exp = datetime.strptime(review_by.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return exp.date() < now.date()


def freshness_value(
    path: Path, fm: Dict[str, str], now: Optional[datetime] = None
) -> Freshness:
    """Resolve the freshness for a candidate source.

    Preference order:
      1. ``review_by`` if present and not expired (basis = review_by, stale=False)
      2. ``last_updated`` (basis = last_updated, stale=False if parseable)
      3. filesystem mtime (basis = mtime)
    """
    review_by = fm.get("review_by")
    if review_by and not is_expired(review_by, now):
        return Freshness(basis="review_by", value=review_by, stale=False)

    last_updated = fm.get("last_updated")
    if last_updated:
        # If review_by exists but is expired, mark stale=True but still prefer
        # last_updated as a fallback.
        stale = bool(review_by) and is_expired(review_by, now)
        return Freshness(basis="last_updated", value=last_updated, stale=stale)

    try:
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        mtime = "1970-01-01T00:00:00+00:00"
    return Freshness(basis="mtime", value=mtime, stale=bool(review_by) and is_expired(review_by, now))


def better_authority(a: SourceMetadata, b: SourceMetadata) -> SourceMetadata:
    """Return the higher-authority source.

    Lower rank value (enum) wins. Ties broken by freshness (newer wins;
    the freshest non-stale evidence is preferred; if both stale, prefer
    the less stale one).
    """
    if a.authority_rank < b.authority_rank:
        return a
    if b.authority_rank < a.authority_rank:
        return b
    # Same rank. Newer wins; among equals, non-stale wins.
    if a.freshness.stale != b.freshness.stale:
        return a if not a.freshness.stale else b
    af = a.freshness.as_datetime()
    bf = b.freshness.as_datetime()
    return a if af >= bf else b


def detect_conflict(a: SourceMetadata, b: SourceMetadata) -> Optional[Conflict]:
    """Return a Conflict when two same-rank sources disagree on the same fact.

    For simplicity we say: any two same-rank sources from DIFFERENT paths
    are a candidate conflict. The caller decides whether the extracted
    content actually contradicts. (Range-level conflict detection belongs
    in the retrieval layer where the content lives.)
    """
    if a.authority_rank != b.authority_rank:
        return None
    if a.path == b.path:
        return None
    return Conflict(
        paths=(a.path, b.path),
        rank=a.authority_rank,
        detail=f"unresolved same-rank conflict between {a.path} and {b.path}",
    )


def rank_candidates(items: Iterable[SourceMetadata]) -> List[SourceMetadata]:
    """Stable sort by (rank ascending, freshness descending, path)."""
    def key(s: SourceMetadata) -> tuple:
        return (
            s.authority_rank.value,
            -s.freshness.as_datetime().timestamp(),
            s.path,
        )
    return sorted(items, key=key)


# Convenience: derive rank from frontmatter hints. This is a conservative
# heuristic — the spec says "canonical_for" / "status" hint at canonical
# registers and approved specs. Owner decision is detected by the
# presence of a `decision_id` line in the body (best effort).
_DECISION_ID_RE = re.compile(r"^\s*decision_id:\s*DEC-[A-Z0-9-]+-\d{8}-\d{3}", re.MULTILINE)
_EVENT_ID_RE = re.compile(r"^\s*event_id:\s*EVT-[A-Z0-9-]+-\d{8}-\d{3}", re.MULTILINE)


def rank_source(path: Path, body: Optional[str] = None) -> AuthorityRank:
    """Heuristic rank from frontmatter + body shape.

    Owner-decision notes contain a ``DEC-…`` id. Canonical registers carry
    a ``canonical_for`` frontmatter key. Approved specs carry an
    ``approval: approved`` or ``status: approved`` key. Newer operational
    evidence carries ``last_updated`` but no canonical_for. Everything
    else falls through to HISTORY.
    """
    fm = parse_frontmatter(path)
    text = body if body is not None else _safe_read(path)
    if _DECISION_ID_RE.search(text or ""):
        return AuthorityRank.OWNER_DECISION
    if fm.get("canonical_for"):
        # Current-state notes & registers are canonical.
        if fm.get("schema") in {"project-current-state/v1",
                                  "decision-register-entry/v1",
                                  "event-register-entry/v1"}:
            return AuthorityRank.CANONICAL_REGISTER
        return AuthorityRank.CANONICAL_REGISTER
    status = fm.get("status", "")
    if status in {"approved", "approved-decision-memo", "canonical", "canonical-log"}:
        return AuthorityRank.APPROVED_SPEC
    if _EVENT_ID_RE.search(text or ""):
        return AuthorityRank.NEWER_EVIDENCE
    if fm.get("last_updated"):
        return AuthorityRank.NEWER_EVIDENCE
    return AuthorityRank.HISTORY


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
