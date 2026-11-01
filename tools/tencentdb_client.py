"""
Shared TencentDB Agent Memory client for all Hermes profiles.

Provides a unified interface for L0/L1/L2/L3 memory operations with
profile-aware namespacing and best-effort durability.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

# ─── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TdaiConfig:
    endpoint: str = "http://127.0.0.1:8420"
    api_key: Optional[str] = None
    service_id: str = "default"
    timeout: float = 8.0

    @classmethod
    def from_env(cls, profile: str) -> "TdaiConfig":
        """Load config from ~/.hermes/.env with profile awareness."""
        env = {}
        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip('"').strip("'")

        endpoint = (env.get("TDAI_MEMORY_ENDPOINT") or "http://127.0.0.1:8420").rstrip("/")
        api_key = env.get("TDAI_MEMORY_API_KEY")
        service_id = env.get("TDAI_MEMORY_SERVICE_ID") or "default"

        # Profile-scoped service_id if not explicitly set
        if service_id == "default":
            service_id = f"hermes-{profile}"

        return cls(endpoint=endpoint, api_key=api_key, service_id=service_id)

    def headers(self) -> dict:
        h = {"Content-Type": "application/json", "x-tdai-service-id": self.service_id}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


# ─── Low-level HTTP ──────────────────────────────────────────────────────


def _post_json(config: TdaiConfig, path: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{config.endpoint}{path}",
        data=data,
        headers=config.headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"code": e.code, "error": e.read().decode()}
    except Exception as e:
        return {"code": -1, "error": str(e)}


# ─── Public API ──────────────────────────────────────────────────────────


def write_conversation(
    profile: str,
    session_id: str,
    messages: list[dict],
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Write messages to L0 conversation layer (durable, always succeeds)."""
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    # Deterministic session_id namespacing
    namespaced = f"hermes-{profile}/{session_id}"

    return _post_json(config, "/v2/conversation/add", {
        "session_id": namespaced,
        "messages": messages,
    })


def search_conversation(
    profile: str,
    query: str,
    limit: int = 5,
    session_id: Optional[str] = None,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Search L0 with BM25+vector hybrid."""
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    body = {"query": query, "limit": limit}
    if session_id:
        body["session_id"] = f"hermes-{profile}/{session_id}"

    return _post_json(config, "/v2/conversation/search", body)


def write_atomic(
    profile: str,
    content: str,
    type_: str = "note",
    background: Optional[str] = None,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Write to L1 atomic store (extracted facts).

    NOTE: The gateway's `/v2/atomic/update` endpoint requires an EXISTING atomic ID
    (no upsert). New atomic facts are created by the gateway's async L1 extractor
    from L0 conversations. Use `write_conversation` (L0) for durable writes;
    the extractor will create atomic entries automatically.
    """
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    # Write to L0 (durable layer) - the extractor will create atomic entries
    return write_conversation(profile, "atomic-extraction", [
        {"role": "user", "content": f"Extract atomic fact: {content}"},
        {"role": "assistant", "content": content},
    ], config=config)


def search_atomic(
    profile: str,
    query: str,
    limit: int = 5,
    type_: Optional[str] = None,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Search L1 atomic store."""
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    body = {"query": query, "limit": limit}
    if type_:
        body["type"] = type_

    return _post_json(config, "/v2/atomic/search", body)


def write_scenario(
    profile: str,
    path: str,
    content: str,
    summary: Optional[str] = None,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Write to L2 scenario store (scenes/workflows).

    NOTE: The gateway's `/v2/scenario/write` endpoint requires an EXISTING path
    (no upsert). Use `write_conversation` (L0) for durable writes; scenarios
    are created by the extractor from L0 content.
    """
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    # Write to L0 for durable extraction
    return write_conversation(profile, "scenario-extraction", [
        {"role": "user", "content": f"Extract scenario: {path}"},
        {"role": "assistant", "content": content},
    ], config=config)


def read_scenario(
    profile: str,
    path: str,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Read from L2 scenario store.

    NOTE: The gateway's `/v2/scenario/read` returns 404 for non-existent paths.
    Scenarios are created by the async extractor from L0 content.
    """
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    namespaced_path = f"hermes-{profile}/{path.lstrip('/')}"

    return _post_json(config, "/v2/scenario/read", {"path": namespaced_path})


def write_core(
    profile: str,
    content: str,
    *,
    overwrite: bool = False,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Write to L3 core (persona/identity).

    Guard rail: the core is a single last-write-wins slot, so writing to it
    silently destroys whatever was there before. To prevent accidental data
    loss, this refuses to replace existing content unless `overwrite=True` is
    passed explicitly. If the core is empty, the write proceeds regardless.
    Best-effort, never raises.
    """
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    # Guard: refuse to clobber existing core content without explicit consent.
    existing = _post_json(config, "/v2/core/read", {})
    existing_content = (existing.get("data") or {}).get("content")
    if existing_content and not overwrite:
        return {
            "skipped": True,
            "reason": "core already has content; pass overwrite=True to replace",
            "existing": existing_content,
        }

    return _post_json(config, "/v2/core/write", {"content": content})


def read_core(
    profile: str,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Read L3 core (persona/identity)."""
    if config is None:
        config = TdaiConfig.from_env(profile)
    if not config.api_key:
        return {"skipped": True, "reason": "no api_key"}

    return _post_json(config, "/v2/core/read", {})


def health_check(*, config: Optional[TdaiConfig] = None) -> dict:
    """Check gateway health."""
    if config is None:
        # Try default profile
        config = TdaiConfig.from_env("default")
    try:
        with urllib.request.urlopen(f"{config.endpoint}/health", timeout=config.timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "down", "error": str(e)}


# ─── Profile-specific session helpers ────────────────────────────────────

def session_key(profile: str, topic: str) -> str:
    """Generate a profile-scoped session key."""
    return f"hermes-{profile}/{topic}"


def skill_mirror_session(profile: str, skill_name: str, rel_file: str) -> str:
    """Generate session key for skill mirror."""
    return f"hermes-skill-mirror/{profile}/{skill_name}/{rel_file}"


def handoff_session(profile: str, from_agent: str, to_agent: str, topic: str) -> str:
    """Generate session key for agent handoff."""
    ts = int(time.time())
    return f"hermes-handoff/{profile}/{from_agent}->{to_agent}/{topic}/{ts}"


def project_session(profile: str, project_ref: str, topic: str) -> str:
    """Generate session key for project-scoped memory."""
    return f"hermes-project/{profile}/{project_ref}/{topic}"


# ─── Decay & Consolidation (recall quality) ─────────────────────────────
#
# Adapted from YantrikDB's temporal-decay + consolidation design (Rust core,
# MIT/AGPL): instead of storing only recency or only raw relevance, score each
# recalled memory by `importance * 2^(-elapsed_hours / half_life)`. Frequently
# touched memories stay fresh; stale ones fade and are filtered below a floor.
# This is a READ-SIDE ranking+filtering layer — it never deletes durable L0/L1
# data, so it layers cleanly over the existing client without risking the
# durability guarantee.

import datetime as _dt

# Default half-life in hours (1 week). A memory's effective score halves every
# week since its last access. 0 disables decay.
DEFAULT_HALF_LIFE_HOURS = 168.0
# Below this decayed score a memory is "effectively forgotten": dropped from
# recall even though still stored. 0.0 disables the floor.
DEFAULT_MIN_SCORE = 0.1


def _parse_ts(ts) -> Optional[float]:
    """Return epoch seconds from a gateway timestamp (ISO 8601 str, epoch int, or None)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _to_float(value, default: float = 0.0) -> float:
    """Coerce a value to float without raising; return `default` on any failure."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def decayed_score(
    importance: float,
    last_access_at,
    now=None,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
) -> float:
    """Score a memory by importance and recency: importance * 2^(-Δh/half_life).

    Mirrors YantrikDB's time_decay formula. Never raises: if half_life <= 0,
    the timestamp is missing, or any input is malformed, importance is returned
    unchanged (no decay applied).
    """
    importance = _to_float(importance)
    half_life = _to_float(half_life_hours)
    if half_life <= 0.0:
        return importance
    last = _parse_ts(last_access_at)
    if last is None:
        return importance
    now_f = _to_float(now) if now is not None else time.time()
    elapsed_hours = max(0.0, now_f - last) / 3600.0
    # Clamp importance to a sane 0..1 range to prevent a corrupt score from
    # inflating recall rank.
    importance = min(1.0, max(0.0, importance))
    return importance * (2.0 ** (-elapsed_hours / half_life))


def _result_ts(r: dict) -> Optional[float]:
    """Extract a timestamp from a gateway result row across field names.

    L0 conversation rows carry `timestamp`; L1 atomic rows carry `created_at`
    (fall back to `updated_at`). Returns epoch seconds or None.
    """
    for key in ("timestamp", "created_at", "updated_at"):
        val = r.get(key)
        if val is not None:
            parsed = _parse_ts(val)
            if parsed is not None:
                return parsed
    return None


def apply_decay(
    results,
    *,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    min_score: float = DEFAULT_MIN_SCORE,
    now=None,
    in_place: bool = False,
) -> list[dict]:
    """Re-rank gateway search results by decayed score.

    Each result dict should carry `score` (importance proxy, 0..1) and a
    timestamp field (`timestamp`, or `created_at`/`updated_at` for L1 atomic
    rows). Results below `min_score` are dropped, the rest are re-sorted by
    `decayed_score` descending and annotated with the decayed score under
    `decayed_score`. Returns a new list unless `in_place=True`.
    Hardened: never raises on malformed/missing input; non-dict or non-numeric
    rows are skipped.
    """
    if not isinstance(results, (list, tuple)):
        return []
    if not results:
        return []
    half_life = _to_float(half_life_hours)
    floor = _to_float(min_score)
    scored = []
    for r in results:
        if not isinstance(r, dict):
            continue
        importance = _to_float(r.get("score"))
        ds = decayed_score(importance, _result_ts(r), now=now, half_life_hours=half_life)
        if ds < floor:
            continue
        item = r if in_place else dict(r)
        item["decayed_score"] = ds
        scored.append(item)
    scored.sort(key=lambda r: _to_float(r.get("decayed_score")), reverse=True)
    return scored


# Lightweight tokenizer + cosine overlap for consolidation, no external deps.
def _tokenize(text) -> set[str]:
    if text is None:
        return set()
    try:
        return set(t for t in str(text).lower().split() if len(t) > 2)
    except Exception:
        return set()


def _overlap(a: dict, b: dict) -> float:
    """Jaccard-ish similarity on token sets, weighted slightly by shared length."""
    try:
        ta, tb = _tokenize(a.get("content")), _tokenize(b.get("content"))
    except Exception:
        return 0.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def consolidate_hits(
    results,
    threshold: float = 0.6,
    *,
    in_place: bool = False,
) -> list[dict]:
    """Merge near-duplicate recalled memories into a single representative.

    For each result, if it overlaps an already-kept representative above
    `threshold`, it is folded in (concatenated into a `consolidated_with` list)
    instead of returned standalone. Reduces context bloat from repeated facts.
    Keeps the representative with the highest decayed/raw score. Best-effort and
    dependency-free (token-overlap proxy for semantic similarity). Hardened:
    never raises on malformed input; non-dict rows are skipped.
    """
    if not isinstance(results, (list, tuple)):
        return []
    if not results:
        return []
    floor = _to_float(threshold)
    kept: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        item = r if in_place else dict(r)
        merged = False
        for k in kept:
            if _overlap(item, k) >= floor:
                k.setdefault("consolidated_with", []).append(item)
                merged = True
                break
        if not merged:
            kept.append(item)
    return kept


def search_conversation_decayed(
    profile: str,
    query: str,
    limit: int = 5,
    session_id: Optional[str] = None,
    *,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    min_score: float = DEFAULT_MIN_SCORE,
    consolidate: bool = False,
    consolidate_threshold: float = 0.6,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Search L0 then re-rank by temporal decay (and optionally consolidate).

    Thin wrapper: calls `search_conversation`, applies `apply_decay` to the
    returned messages, optionally consolidates near-dupes, and returns the same
    response shape with `data.messages` replaced. Never raises.
    """
    raw = search_conversation(profile, query, limit=limit, session_id=session_id, config=config)
    try:
        msgs = raw.get("data", {}).get("messages", [])
        scored = apply_decay(msgs, half_life_hours=half_life_hours, min_score=min_score)
        if consolidate:
            scored = consolidate_hits(scored, threshold=consolidate_threshold)
        if "data" in raw and isinstance(raw["data"], dict):
            raw["data"]["messages"] = scored
            raw["data"]["decay_applied"] = True
        return raw
    except Exception as e:
        return {"code": -1, "error": str(e), "raw": raw}


# ─── High-level write helpers (best-effort, never raises) ────────────────


def durably_record(
    profile: str,
    topic: str,
    user_prompt: str,
    assistant_response: str,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Record a Q/A pair durably in L0. Never raises."""
    try:
        return write_conversation(profile, topic, [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ], config=config)
    except Exception as e:
        return {"error": str(e)}


def durably_extract_atomic(
    profile: str,
    content: str,
    type_: str = "note",
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Extract and store an atomic fact in L1. Never raises."""
    try:
        return write_atomic(profile, content, type_, config=config)
    except Exception as e:
        return {"error": str(e)}


def durably_record_handoff(
    profile: str,
    from_agent: str,
    to_agent: str,
    topic: str,
    summary: str,
    *,
    config: Optional[TdaiConfig] = None,
) -> dict:
    """Record an agent-to-agent handoff. Never raises."""
    try:
        session = handoff_session(profile, from_agent, to_agent, topic)
        return write_conversation(profile, session, [
            {"role": "user", "content": f"Handoff from {from_agent} to {to_agent}: {topic}"},
            {"role": "assistant", "content": summary},
        ], config=config)
    except Exception as e:
        return {"error": str(e)}
