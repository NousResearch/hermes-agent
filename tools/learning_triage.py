#!/usr/bin/env python3
"""Learning triage tool — safe MVP proposal engine for memory/skill hygiene.

The MVP is intentionally deterministic and proposal-only. It inspects provided
or loaded transcript text, classifies candidate learnings with conservative
heuristics, reports current memory pressure, and never mutates memory or skills.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from hermes_constants import get_hermes_home
from tools.memory_tool import ENTRY_DELIMITER, MemoryStore
from tools.registry import registry
from tools.session_search_tool import _HIDDEN_SESSION_SOURCES

_VALID_SCOPES = {"current", "recent", "session"}
_VALID_TARGETS = {"user", "memory", "skills", "all"}
_VALID_MODES = {"propose", "apply"}
_VALID_RISKS = {"low", "medium", "high"}
_VALID_CONFIDENCE = {"low", "medium", "high"}

_MAX_SNIPPET_CHARS = 360
_MAX_TRANSCRIPT_CHARS = 24_000

_TRANSIENT_PATTERNS = re.compile(
    r"\b(done|completed|implemented|fixed|merged|opened pr|pr #?\d+|ticket|jira|kanban|"
    r"tests? pass(?:ed)?|next step|tomorrow|today|branch|commit|deploy(?:ed)?|status update)\b",
    re.IGNORECASE,
)
_USER_PATTERNS = re.compile(
    r"\b(user correction|user prefers?|user wants?|user expects?|please remember|remember this|"
    r"don't do that again|do not .*again|communication style|terminal-readable|concise responses?|"
    r"address me|call me|prefers? .{0,80}(responses?|style|format|tone|workflow))\b",
    re.IGNORECASE,
)
_MEMORY_PATTERNS = re.compile(
    r"\b(environment fact|machine|host|os|repo(?:sitory)?|project convention|client convention|"
    r"tool quirk|api quirk|uses? scripts/run_tests\.sh|must use scripts/run_tests\.sh|"
    r"hermes_home|h?ermes home|profile-safe|path|installed|version|ci parity|"
    r"direct pytest.*diverge|actual source of truth)\b",
    re.IGNORECASE,
)
_SKILL_PATTERNS = re.compile(
    r"\b(recurring workflow|repeated workflow|procedure|playbook|workflow:|steps?:|"
    r"when .* then |run .* then |gotcha|verification steps?|patch(?:ing)? umbrella skills?|"
    r"create a skill|skill candidate|multi-step|adversarial review|brainstorm first)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "direct", "do", "does", "fact", "for", "from", "in", "into", "is",
    "it", "must", "of", "on", "or", "repo", "the", "this", "to", "use",
    "uses", "using", "with",
}


def _empty_candidate(text: str, reason: str) -> Dict[str, Any]:
    return {
        "candidate": _compact_text(text),
        "target": "ignore",
        "action": "ignore",
        "reason": reason,
        "risk": "low",
        "confidence": "medium",
        "old_text": None,
        "skill_name": None,
    }


def _candidate(
    text: str,
    target: str,
    action: str,
    reason: str,
    risk: str,
    confidence: str,
    *,
    old_text: str | None = None,
    skill_name: str | None = None,
) -> Dict[str, Any]:
    risk = risk if risk in _VALID_RISKS else "medium"
    confidence = confidence if confidence in _VALID_CONFIDENCE else "medium"
    return {
        "candidate": _compact_text(text),
        "target": target,
        "action": action,
        "reason": reason,
        "risk": risk,
        "confidence": confidence,
        "old_text": old_text,
        "skill_name": skill_name,
    }


def _compact_text(text: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _iter_snippets(snippets: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in snippets or []:
        if isinstance(item, dict):
            role = item.get("role") or "unknown"
            content = item.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            text = f"{role}: {content}"
        else:
            text = str(item or "")
        text = _compact_text(text, 900)
        if text:
            out.append(text)
    return out


def _looks_transient(text: str) -> bool:
    return bool(_TRANSIENT_PATTERNS.search(text))


def _suggest_skill_name(text: str) -> str:
    lowered = text.lower()
    if "memory" in lowered or "skill" in lowered or "hermes" in lowered:
        return "hermes-agent"
    if "test" in lowered or "tdd" in lowered:
        return "test-driven-development"
    if "debug" in lowered or "root cause" in lowered:
        return "systematic-debugging"
    if "git" in lowered or "pr" in lowered or "ci" in lowered:
        return "github-pr-workflow"
    if "brainstorm" in lowered or "adversarial review" in lowered or "plan" in lowered:
        return "gsd-hermes-adapter"
    return "hermes-agent"


def _classify_one(text: str) -> Dict[str, Any]:
    # Skill-worthy workflow lessons should win over generic transient language:
    # "when X, run Y, then verify Z" may contain "run" or "tests pass".
    if _SKILL_PATTERNS.search(text):
        skill_name = _suggest_skill_name(text)
        return _candidate(
            text,
            "skills",
            "patch_skill",
            "Reusable or recurring procedure; prefer patching an umbrella skill over creating a narrow one-off skill.",
            "medium",
            "high" if "recurring" in text.lower() or "repeated" in text.lower() else "medium",
            skill_name=skill_name,
        )

    if _USER_PATTERNS.search(text):
        return _candidate(
            text,
            "user",
            "add",
            "Durable user preference/correction likely to reduce future steering.",
            "low",
            "high" if re.search(r"\b(user correction|please remember|remember this|prefers?)\b", text, re.I) else "medium",
        )

    if _MEMORY_PATTERNS.search(text):
        return _candidate(
            text,
            "memory",
            "add",
            "Stable environment, repo, tool, or project convention likely useful across sessions.",
            "medium",
            "high" if re.search(r"\b(environment fact|must use|project convention|tool quirk)\b", text, re.I) else "medium",
        )

    if _looks_transient(text):
        return _empty_candidate(text, "Ignored as transient task progress/status rather than durable learning.")

    return _empty_candidate(text, "No durable preference, environment fact, or reusable workflow detected by MVP heuristics.")


def classify_learning_candidates(
    snippets: Iterable[Any],
    target: str = "all",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Classify transcript snippets into proposal candidates.

    Deterministic MVP: conservative regex heuristics, no LLM calls, no mutation.
    """
    target = target if target in _VALID_TARGETS else "all"
    limit = _coerce_limit(limit)
    candidates: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for text in _iter_snippets(snippets):
        cand = _classify_one(text)
        if target != "all" and cand["target"] not in {target, "ignore"}:
            continue
        key = (cand["candidate"].lower(), cand["target"], cand["action"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)
        if len(candidates) >= limit:
            break

    return candidates


def _coerce_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 10
    return max(1, min(value, 25))

def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_.-]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }

def _similar_existing_entry(candidate_text: str, entries: List[str]) -> str | None:
    candidate_tokens = _tokens(candidate_text)
    if not candidate_tokens:
        return None
    for entry in entries:
        entry_tokens = _tokens(entry)
        if not entry_tokens:
            continue
        overlap = candidate_tokens & entry_tokens
        # Prefer conservative replacement: only suggest when the existing entry
        # shares a strong anchor or at least three meaningful terms.
        if "scripts/run_tests.sh" in candidate_text and {"hermes", "tests"} <= entry_tokens:
            return entry
        if len(overlap) >= 3:
            return entry
    return None


def _enrich_memory_replace_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mem_dir = get_hermes_home() / "memories"
    entries_by_target = {
        "user": MemoryStore._read_file(mem_dir / "USER.md"),
        "memory": MemoryStore._read_file(mem_dir / "MEMORY.md"),
    }
    enriched: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        target = item.get("target")
        if item.get("action") == "add" and target in entries_by_target:
            existing = _similar_existing_entry(item.get("candidate", ""), entries_by_target[target])
            if existing:
                item["action"] = "replace"
                item["old_text"] = existing
                item["reason"] = item["reason"] + " Similar existing memory found; replace is safer than duplicating."
                item["risk"] = "medium"
        enriched.append(item)
    return enriched


def _memory_usage() -> Dict[str, Dict[str, int]]:
    store = MemoryStore()
    mem_dir = get_hermes_home() / "memories"
    usage: Dict[str, Dict[str, int]] = {}
    for target, filename in (("user", "USER.md"), ("memory", "MEMORY.md")):
        entries = MemoryStore._read_file(mem_dir / filename)
        current = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        limit = store.user_char_limit if target == "user" else store.memory_char_limit
        usage[target] = {"current_chars": current, "limit_chars": limit}
    return usage


def _message_snippets(messages: List[Dict[str, Any]], max_chars: int = _MAX_TRANSCRIPT_CHARS) -> List[str]:
    snippets: List[str] = []
    total = 0
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "tool":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        text = f"{role}: {content}"
        total += len(text)
        if total > max_chars:
            snippets.append("[transcript truncated for compact learning triage]")
            break
        snippets.append(text)
    return snippets


def _load_source(
    *,
    scope: str,
    session_id: str | None,
    db: Any = None,
    current_session_id: str | None = None,
    limit: int = 10,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    notes: List[str] = []
    source: Dict[str, Any] = {"scope": scope}

    if db is None:
        notes.append(
            "Session database not available to learning_triage in this context; returning memory usage and guidance only."
        )
        return source, [], notes

    try:
        if scope == "current":
            session_id = session_id or current_session_id
            if not session_id:
                notes.append("No current session_id was provided; use scope='session' with session_id or run from the agent loop.")
                return source, [], notes
            messages = db.get_messages_as_conversation(session_id, include_ancestors=True)
            meta = db.get_session(session_id) or {}
            source.update({
                "session_id": session_id,
                "source": meta.get("source"),
                "message_count": len(messages),
            })
            return source, _message_snippets(messages), notes

        if scope == "session":
            if not session_id:
                notes.append("scope='session' requires a session_id.")
                return source, [], notes
            resolved = getattr(db, "resolve_session_id", lambda x: x)(session_id) or session_id
            messages = db.get_messages_as_conversation(resolved, include_ancestors=True)
            meta = db.get_session(resolved) or {}
            source.update({
                "session_id": resolved,
                "source": meta.get("source"),
                "message_count": len(messages),
            })
            if not messages:
                notes.append(f"No messages found for session_id={resolved!r}.")
            return source, _message_snippets(messages), notes

        # recent
        sessions = db.list_sessions_rich(
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=min(max(limit, 1), 5),
            order_by_last_active=True,
        )
        source.update({"session_count": len(sessions), "sessions": [s.get("id") for s in sessions]})
        snippets: List[str] = []
        for session in sessions:
            sid = session.get("id")
            if not sid:
                continue
            messages = db.get_messages_as_conversation(sid, include_ancestors=True)
            snippets.extend(_message_snippets(messages, max_chars=max(1000, _MAX_TRANSCRIPT_CHARS // max(1, len(sessions)))))
        if not snippets:
            notes.append("No recent session transcript text found.")
        return source, snippets, notes
    except Exception as exc:
        notes.append(f"Failed to load session transcript: {type(exc).__name__}: {exc}")
        return source, [], notes


def learning_triage(
    session_id: str | None = None,
    scope: str = "recent",
    target: str = "all",
    mode: str = "propose",
    limit: int = 10,
    db: Any = None,
    current_session_id: str | None = None,
) -> str:
    """Return a JSON proposal for memory/skill learning capture.

    ``mode='apply'`` is intentionally unsupported in the safe MVP. Passing it
    returns the same proposal plus a clear note; memory and skills are never
    mutated by this tool.
    """
    scope = scope if scope in _VALID_SCOPES else "recent"
    target = target if target in _VALID_TARGETS else "all"
    mode = mode if mode in _VALID_MODES else "propose"
    limit = _coerce_limit(limit)

    source, snippets, notes = _load_source(
        scope=scope,
        session_id=session_id,
        db=db,
        current_session_id=current_session_id,
        limit=limit,
    )

    candidates = classify_learning_candidates(snippets, target=target, limit=limit) if snippets else []
    candidates = _enrich_memory_replace_candidates(candidates)

    if mode == "apply":
        notes.append(
            "mode='apply' is not implemented yet; returning proposal-only triage and making no memory or skill mutations."
        )

    if not candidates and snippets:
        notes.append("No durable learning candidates detected by deterministic MVP heuristics.")
    if not snippets:
        notes.append("Provide session_id/scope with session DB access, or use future LLM-assisted triage for current in-memory context.")

    result = {
        "success": True,
        "mode": mode,
        "source": source,
        "candidates": candidates,
        "memory_usage": _memory_usage(),
        "notes": notes,
    }
    return json.dumps(result, ensure_ascii=False)


def check_learning_triage_requirements() -> bool:
    """Learning triage has no external requirements."""
    return True


LEARNING_TRIAGE_SCHEMA = {
    "name": "learning_triage",
    "description": (
        "Inspect recent/current session content and propose what should become user memory, "
        "agent/environment memory, a skill patch/create candidate, or ignored transient progress. "
        "Safe by default: mode='propose'; mode='apply' is intentionally not implemented yet "
        "and returns proposal-only output without mutating memory or skills. "
        "Use after substantial work, corrections, or repeated workflow discoveries to avoid polluting durable memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Optional specific session to inspect. Required for scope='session'.",
            },
            "scope": {
                "type": "string",
                "enum": ["current", "recent", "session"],
                "description": "Content scope to inspect. Defaults to recent sessions.",
                "default": "recent",
            },
            "target": {
                "type": "string",
                "enum": ["user", "memory", "skills", "all"],
                "description": "Candidate target filter. Defaults to all.",
                "default": "all",
            },
            "mode": {
                "type": "string",
                "enum": ["propose", "apply"],
                "description": "Safe default is proposal-only. mode='apply' is not implemented in the MVP and never mutates memory or skills.",
                "default": "propose",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of candidates to return (default 10, max 25).",
                "default": 10,
            },
        },
        "required": [],
    },
}


registry.register(
    name="learning_triage",
    toolset="learning",
    schema=LEARNING_TRIAGE_SCHEMA,
    handler=lambda args, **kw: learning_triage(
        session_id=args.get("session_id"),
        scope=args.get("scope", "recent"),
        target=args.get("target", "all"),
        mode=args.get("mode", "propose"),
        limit=args.get("limit", 10),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_learning_triage_requirements,
    emoji="🧭",
)
