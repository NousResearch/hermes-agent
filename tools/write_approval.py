#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

Background
----------
The agent writes to two persistent stores that survive across sessions:

  * **memory** — MEMORY.md / USER.md, small (~200 char) declarative entries
  * **skills** — SKILL.md + supporting files, potentially huge (10-100 KB)

Both stores are written from two origins:

  * **foreground** — a normal agent turn (user is present / chatting)
  * **background_review** — the self-improvement review fork that runs after a
    turn and autonomously decides what to save (the source of the
    "wrong assumptions" users complained about)

This module lets the user gate those writes per-subsystem with a boolean
``write_approval``:

  * ``false`` (default) — write freely (the pre-gate behaviour)
  * ``true``            — require approval: do not commit the write; either
    prompt inline (memory, interactive CLI only) or **stage** it to a pending
    store and surface it for the user to approve or reject out-of-band

The size asymmetry between memory and skills is real and unavoidable: a memory
entry can be reviewed inline in a chat bubble; a 100 KB SKILL.md cannot. So
the gate stages BOTH to disk, but review affordances differ by subsystem
(see ``hermes_cli`` slash handlers): memory shows full content, skills show
metadata + a one-line gist + a ``diff`` escape hatch (CLI/dashboard/file).

Staging is mandatory for background-origin writes (a daemon thread cannot
block on an interactive prompt) and for gateway sessions (no inline prompt
channel — review happens via ``/memory pending``). Foreground CLI memory
writes prompt inline via the dangerous-command approval callback; skill
writes always stage (too big to eyeball mid-loop).

Pending records live under ``<HERMES_HOME>/pending/{memory,skills}/<id>.json``
so they survive process restarts and can be reviewed from CLI, gateway, or the
web dashboard.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import re
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is process-local
    fcntl = None

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)

# Config key (per subsystem). A single boolean: the approval gate is OFF by
# default (writes flow freely, the pre-gate behaviour), and ON means stage /
# prompt every write for the user's approval. There is intentionally no third
# "block all writes" state — to disable a subsystem entirely use its own
# enable flag (e.g. ``memory.memory_enabled: false``).
CONFIG_KEY = "write_approval"
TIERED_KEY = "write_approval_tiered"
MEMORY_SECRET_REJECT = "MEMORY_SECRET_REJECT"
MEMORY_FLEET_DRAIN_ACTIVE = "MEMORY_FLEET_DRAIN_ACTIVE"
_MEMORY_FLEET_DRAIN_MARKER = b'{"schema":"hermes-memory-fleet-drain/v1"}\n'


class PendingWriteError(RuntimeError):
    """Raised when a pending write cannot be durably persisted."""


class MemoryFleetDrainActiveError(RuntimeError):
    """Raised when tiered memory writes are blocked by a fleet drain."""


class MemoryWriteConfigError(RuntimeError):
    """Raised when the tiered-memory config cannot be read or parsed.

    Callers MUST fail closed (refuse / hold the write); an unreadable config is
    not evidence that the gate is off.
    """


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def write_approval_enabled(subsystem: str) -> bool:
    """Return whether the approval gate is enabled for ``subsystem``.

    Reads ``<subsystem>.write_approval`` from config.yaml. Defaults to
    ``False`` (gate off — writes flow freely) for any unset / invalid value so
    existing installs keep their current behaviour until the user opts in.
    """
    if subsystem not in _SUBSYSTEMS:
        return False
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        raw = cfg_get(cfg, subsystem, CONFIG_KEY, default=False)
    except Exception:
        return False
    return _normalize_enabled(raw)


def memory_write_tiered_enabled() -> bool:
    """Return whether the opt-in tiered memory gate is enabled.

    A config that cannot be read or parsed (malformed YAML, unreadable
    permissions) does NOT resolve to 'disabled' — that would fail OPEN and let
    an ungated direct write through. Such failures raise MemoryWriteConfigError
    so callers fail CLOSED.
    """
    try:
        from hermes_cli.config import load_config, cfg_get
    except Exception as exc:
        raise MemoryWriteConfigError("tiered memory config module unavailable") from exc
    try:
        cfg = load_config()
        raw = cfg_get(cfg, MEMORY, TIERED_KEY, default=False)
    except Exception as exc:
        raise MemoryWriteConfigError("tiered memory config unreadable") from exc
    return _normalize_enabled(raw)


def memory_write_gate_active() -> bool:
    """Return whether the tiered memory gate should run for a memory write.

    Evaluates the tiered flag FIRST so an unreadable/malformed config raises
    MemoryWriteConfigError (callers fail CLOSED) instead of being masked by a
    short-circuited ``write_approval_enabled`` that fails open to False.
    """
    tiered = memory_write_tiered_enabled()
    return tiered and write_approval_enabled(MEMORY)


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to a bool. Default (unknown) is False (gate off).

    Accepts real bools and the usual truthy/falsey strings. YAML 1.1 parses
    bare ``on``/``off``/``yes``/``no`` as bools already, so the string branch
    is mostly for hand-edited configs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "approve", "enabled"}
    return False


# ---------------------------------------------------------------------------
# Tiered memory decisions and redacted audit
# ---------------------------------------------------------------------------

class MemoryWriteTier(Enum):
    TIER0 = "Tier0"
    TIER1 = "Tier1"
    TIER2 = "Tier2"

    @property
    def risk(self) -> int:
        return {
            MemoryWriteTier.TIER0: 0,
            MemoryWriteTier.TIER1: 1,
            MemoryWriteTier.TIER2: 2,
        }[self]


class _MemoryWriteReasonCode(str, Enum):
    OPERATIONAL_FACT = "operational_fact"
    USER_PROFILE = "user_profile"
    PROPER_NAME = "proper_name"
    PII = "pii"
    CUSTOMER_FINANCIAL_DATA = "customer_financial_data"
    SENSITIVE_IDENTIFIER = "sensitive_identifier"
    HIGH_ENTROPY_VALUE = "high_entropy_value"
    IMPERATIVE_FUTURE_INSTRUCTION = "imperative_future_instruction"
    UNCERTAIN = "uncertain"
    PRIVATE_KEY = "private_key"
    AWS_ACCESS_KEY = "aws_access_key"
    GITHUB_TOKEN = "github_token"
    SLACK_TOKEN = "slack_token"
    JWT = "jwt"
    BEARER_TOKEN = "bearer_token"
    CREDENTIAL_ASSIGNMENT = "credential_assignment"
    DETERMINISTIC_TOKEN = "deterministic_token"


MEMORY_REASON_CODES = frozenset(reason.value for reason in _MemoryWriteReasonCode)


@dataclass(frozen=True)
class MemoryWriteDecision:
    tier: MemoryWriteTier
    reason_codes: tuple[str, ...]
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        for reason_code in self.reason_codes:
            if reason_code not in MEMORY_REASON_CODES:
                raise ValueError(f"Unknown memory write reason code: {reason_code}")


_TIER2_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (_MemoryWriteReasonCode.PRIVATE_KEY.value, re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.I)),
    (_MemoryWriteReasonCode.AWS_ACCESS_KEY.value, re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (_MemoryWriteReasonCode.GITHUB_TOKEN.value, re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    (_MemoryWriteReasonCode.SLACK_TOKEN.value, re.compile(r"\bxox[aboprs]-[A-Za-z0-9-]{20,}\b", re.I)),
    (_MemoryWriteReasonCode.JWT.value, re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (_MemoryWriteReasonCode.BEARER_TOKEN.value, re.compile(r"\b(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I)),
    (_MemoryWriteReasonCode.CREDENTIAL_ASSIGNMENT.value, re.compile(
        r"\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
        r"client[_-]?secret|credential)\s*[:=]\s*[^\s,;]{4,}",
        re.I,
    )),
    (_MemoryWriteReasonCode.DETERMINISTIC_TOKEN.value, re.compile(
        r"\b(?:AIza[0-9A-Za-z_-]{30,}|sk_(?:live|test)_[0-9A-Za-z]{16,}|npm_[A-Za-z0-9]{20,})\b"
    )),
)

_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\bpassport\s+(?:id|number|no\.?)\b", re.I),
)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FINANCIAL_RE = re.compile(
    r"\b(?:customer|client)\b.{0,48}\b(?:account|balance|bank|budget|card|cost|credit|"
    r"debit|invoice|iban|routing|payment|transaction|mrr|arr|revenue|gross margin)\b|"
    r"\b(?:account balance|bank account|bank balance|credit card|debit card|payment card|"
    r"routing number|transaction amount)\b|"
    r"\b(?:budget|cost|credit|debit|invoice|iban|payment|mrr|arr|revenue|gross margin)\b|"
    r"\b(?:USD|EUR|GBP|JPY|CAD|AUD|CHF)\s*\d[\d,.]*|"
    r"(?<!\w)[$€£]\s*\d[\d,.]*|"
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    re.I,
)
_SENSITIVE_IDENTIFIER_PATTERNS = (
    re.compile(r"\b[a-z][a-z0-9+.-]*://", re.I),
    re.compile(r"\barn:aws:[^\s]+", re.I),
    re.compile(r"\b\d{12}\b"),
    re.compile(r"\b[UCDGBWT](?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{8,}\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:database|db)\s+username\b|\busername\s*(?:is|[:=])", re.I),
)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:always|never|remember|ensure|configure|set|enable|disable|update|"
    r"create|install|start|stop|deploy|run|use|send|contact|notify|delete|remove|rotate|restart|"
    r"clear|do not|make sure)\b|"
    r"\b(?:should|must|next time|in the future|will need to|after every)\b",
    re.I,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|likely|might|could|uncertain|unknown|unsure|"
    r"appears?|seems?)\b|\?",
    re.I,
)
_PROFILE_RE = re.compile(
    r"\b(?:user|customer|client|they|he|she|i|my)\b.{0,40}"
    r"\b(?:prefer|like|live|name|email|phone|address|role|work|salary|birthday)\w*\b",
    re.I,
)
_OPERATIONAL_RE = re.compile(
    r"`[^`]+`|(?:^|\s)(?:\.?\.?/|~/)[^\s]+|\b[\w.-]+/[\w./~-]+|"
    r"\b[A-Za-z0-9_-]+\.(?:md|ya?ml|json|toml|ini|cfg|conf|py|tsx?|jsx?|go|rs|sh)\b|"
    r"#\d+|(?:^|\s)--?[A-Za-z][\w-]*|\b\w+\(\)|"
    r"\b(?:retry (?:count|limit)|timeout|port|worker count|replica count)\s+is\s+\d+(?:\.\d+)?\b|"
    r"\b(?:listens? on|binds? to)\s+port\s+\d{2,5}\b",
    re.I,
)
_PROPER_NAME_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|"
    r"\b[A-Z][a-z]{2,}\s+(?:owns?|leads?|manages?|maintains?|works?|uses?|prefers?)\b"
)
_ENTROPY_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_+=/-]{24,}(?![A-Za-z0-9])")
_AUDIT_LOCK = threading.Lock()


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _has_high_entropy_value(text: str) -> bool:
    for match in _ENTROPY_TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(set(token)) >= 12 and _entropy(token) >= 3.7:
            return True
    return False


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def classify_memory_write(
    *,
    target: str,
    content: Optional[str],
    old_text: Optional[str] = None,
) -> MemoryWriteDecision:
    """Classify both new and selector text, returning the highest-risk tier."""
    text = "\n".join(part for part in (content or "", old_text or "") if part).strip()

    tier2_reasons = [code for code, pattern in _TIER2_SIGNATURES if pattern.search(text)]
    if tier2_reasons:
        return MemoryWriteDecision(MemoryWriteTier.TIER2, _dedupe(tier2_reasons))

    tier1_reasons: list[str] = []
    if target == "user":
        tier1_reasons.append(_MemoryWriteReasonCode.USER_PROFILE.value)
    if _PROPER_NAME_RE.search(text):
        tier1_reasons.append(_MemoryWriteReasonCode.PROPER_NAME.value)
    phone_text = _ISO_DATE_RE.sub("", text)
    if (
        any(pattern.search(text) for pattern in _PII_PATTERNS)
        or _PHONE_RE.search(phone_text)
        or _PROFILE_RE.search(text)
    ):
        tier1_reasons.append(_MemoryWriteReasonCode.PII.value)
    if _FINANCIAL_RE.search(text):
        tier1_reasons.append(_MemoryWriteReasonCode.CUSTOMER_FINANCIAL_DATA.value)
    if any(pattern.search(text) for pattern in _SENSITIVE_IDENTIFIER_PATTERNS):
        tier1_reasons.append(_MemoryWriteReasonCode.SENSITIVE_IDENTIFIER.value)
    if _has_high_entropy_value(text):
        tier1_reasons.append(_MemoryWriteReasonCode.HIGH_ENTROPY_VALUE.value)
    if _IMPERATIVE_RE.search(text):
        tier1_reasons.append(_MemoryWriteReasonCode.IMPERATIVE_FUTURE_INSTRUCTION.value)
    if not text or len(text) > 1000 or _UNCERTAIN_RE.search(text):
        tier1_reasons.append(_MemoryWriteReasonCode.UNCERTAIN.value)
    if not _OPERATIONAL_RE.search(text):
        tier1_reasons.append(_MemoryWriteReasonCode.UNCERTAIN.value)
    if tier1_reasons:
        return MemoryWriteDecision(MemoryWriteTier.TIER1, _dedupe(tier1_reasons))

    return MemoryWriteDecision(
        MemoryWriteTier.TIER0,
        (_MemoryWriteReasonCode.OPERATIONAL_FACT.value,),
    )


def classify_memory_batch(
    *, target: str, operations: Sequence[Dict[str, Any]],
) -> MemoryWriteDecision:
    """Classify an atomic batch at the highest risk of any operation."""
    decisions = [
        classify_memory_write(
            target=target,
            content=(op or {}).get("content"),
            old_text=(op or {}).get("old_text"),
        )
        for op in operations
    ]
    if not decisions:
        return MemoryWriteDecision(
            MemoryWriteTier.TIER1,
            (_MemoryWriteReasonCode.UNCERTAIN.value,),
        )
    highest = max(decision.tier.risk for decision in decisions)
    tier = next(decision.tier for decision in decisions if decision.tier.risk == highest)
    reasons = _dedupe(
        reason
        for decision in decisions
        if decision.tier is tier
        for reason in decision.reason_codes
    )
    return MemoryWriteDecision(tier, reasons)


def _audit_provenance() -> Dict[str, str]:
    def session_value(name: str) -> str:
        try:
            from gateway.session_context import get_session_env
            return str(get_session_env(name, "") or "")
        except Exception:
            return str(os.environ.get(name, "") or "")

    return {
        "profile": session_value("HERMES_SESSION_PROFILE") or os.environ.get("HERMES_PROFILE", ""),
        "session": session_value("HERMES_SESSION_ID"),
        "task": os.environ.get("HERMES_KANBAN_TASK", "") or os.environ.get("HERMES_TASK_ID", ""),
        "origin": current_origin(),
    }


def memory_content_sha256(content: Any, old_text: Any = None) -> str:
    canonical = json.dumps(
        {"content": content, "old_text": old_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_memory_decision(
    decision: MemoryWriteDecision,
    *,
    action: str,
    target: str,
    store: str,
    content: Any,
    old_text: Any = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a redacted structured decision record and return it."""
    record = _build_memory_audit_record(
        decision,
        event_type="decision",
        action=action,
        target=target,
        store=store,
        content=content,
        old_text=old_text,
        provenance=provenance,
    )
    _append_memory_audit_record(record)
    return record


_MEMORY_AUDIT_COMMON_FIELDS = (
    "decision_id",
    "profile",
    "session",
    "task",
    "origin",
    "action",
    "target",
    "store",
    "tier",
    "reason_codes",
    "content_sha256",
)


def memory_audit_context(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the redacted fields needed to correlate later lifecycle events."""
    return {key: record.get(key) for key in _MEMORY_AUDIT_COMMON_FIELDS}


def _append_memory_audit_record(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    audit_path = get_hermes_home() / "logs" / "memory-write-audit.jsonl"
    with _AUDIT_LOCK:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            # The mutation-audit trail is only durable once the appended bytes
            # AND the directory entry survive a crash. fsync the file, then the
            # parent directory, before any caller may treat the record as
            # durably recorded.
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(audit_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if record.get("tier") == MemoryWriteTier.TIER2.value:
        logger.warning("%s %s", MEMORY_SECRET_REJECT, line)
    else:
        logger.info("MEMORY_WRITE_AUDIT %s", line)


def audit_memory_lifecycle(
    event_type: str,
    audit_context: Dict[str, Any],
    *,
    pending_id: Optional[str] = None,
    failure_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a redacted outcome record correlated to an earlier decision."""
    if event_type not in {
        "staged", "applied", "replayed", "failed", "rejected", "discarded",
    }:
        raise ValueError(f"Unknown memory audit event_type: {event_type}")
    record = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **memory_audit_context(audit_context),
    }
    if pending_id:
        record["pending_id_sha256"] = hashlib.sha256(
            str(pending_id).encode("utf-8")
        ).hexdigest()
    if failure_code:
        record["failure_code"] = str(failure_code)
    if record.get("tier") == MemoryWriteTier.TIER2.value:
        record["marker"] = MEMORY_SECRET_REJECT
    _append_memory_audit_record(record)
    return record


def _audit_memory_lifecycle_fail_safe(
    event_type: str,
    audit_context: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        record = audit_memory_lifecycle(event_type, audit_context, **kwargs)
        record["_durable"] = True
        return record
    except Exception as exc:
        record = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **memory_audit_context(audit_context),
            "failure_code": kwargs.get("failure_code") or "audit_sink_unavailable",
        }
        pending_id = kwargs.get("pending_id")
        if pending_id:
            record["pending_id_sha256"] = hashlib.sha256(
                str(pending_id).encode("utf-8")
            ).hexdigest()
        if record.get("tier") == MemoryWriteTier.TIER2.value:
            record["marker"] = MEMORY_SECRET_REJECT
        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        logger.error(
            "MEMORY_AUDIT_SINK_FAILURE %s (%s)", line, type(exc).__name__,
        )
        record["_durable"] = False
        return record


def _build_memory_audit_record(
    decision: MemoryWriteDecision,
    *,
    event_type: str,
    action: str,
    target: str,
    store: str,
    content: Any,
    old_text: Any = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_id": decision.decision_id,
        **_audit_provenance(),
        "action": action or "unknown",
        "target": target or "memory",
        "store": store or "builtin",
        "tier": decision.tier.value,
        "reason_codes": list(decision.reason_codes),
        "content_sha256": memory_content_sha256(content, old_text),
    }
    if provenance:
        for key in ("profile", "session", "task", "origin"):
            if provenance.get(key):
                record[key] = str(provenance[key])
    if decision.tier is MemoryWriteTier.TIER2:
        record["marker"] = MEMORY_SECRET_REJECT
    return record


def _audit_memory_decision_fail_safe(
    decision: MemoryWriteDecision,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Keep a redacted logger audit when the durable JSONL sink is unavailable."""
    try:
        record = audit_memory_decision(decision, **kwargs)
        record["_durable"] = True
        return record
    except Exception as exc:
        record = _build_memory_audit_record(decision, event_type="decision", **kwargs)
        record["failure_code"] = "audit_sink_unavailable"
        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        marker = f" {MEMORY_SECRET_REJECT}" if decision.tier is MemoryWriteTier.TIER2 else ""
        logger.error("MEMORY_AUDIT_SINK_FAILURE%s %s (%s)", marker, line, type(exc).__name__)
        record["_durable"] = False
        return record


# ---------------------------------------------------------------------------
# Pending store (file-backed)
# ---------------------------------------------------------------------------

def _pending_dir(subsystem: str) -> Path:
    return get_hermes_home() / "pending" / subsystem


def _atomic_write(path: Path, record: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    descriptor: Optional[int] = None
    directory_descriptor: Optional[int] = None
    temp_created = False
    replaced = False
    try:
        descriptor = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        temp_created = True
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("pending write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(tmp, path)
        replaced = True
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
    except Exception:
        for open_descriptor in (descriptor, directory_descriptor):
            if open_descriptor is not None:
                try:
                    os.close(open_descriptor)
                except Exception:
                    logger.warning(
                        "Failed to close incomplete pending write descriptor",
                    )
        if temp_created:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove incomplete pending write %s", tmp)
        if replaced:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove non-durable pending write %s", path)
        raise


_MEMORY_PENDING_THREAD_LOCK = threading.RLock()
_MEMORY_PENDING_LOCK_STATE: Dict[str, Any] = {"depth": 0, "fd": None, "path": None}


def memory_fleet_drain_marker_path() -> Path:
    """Return the fleet-wide marker path for the active Hermes profile."""
    return get_hermes_home().parent / "audit" / "memory-drain" / "fleet-drain-active.json"


def _memory_fleet_drain_marker_path(profiles_root: Path) -> Path:
    return Path(profiles_root) / "audit" / "memory-drain" / "fleet-drain-active.json"


def _inspect_memory_fleet_drain_marker(path: Path) -> bool:
    """Return whether the marker exists, rejecting every unsafe representation."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except Exception as exc:
        raise MemoryFleetDrainActiveError(MEMORY_FLEET_DRAIN_ACTIVE) from exc

    descriptor = None
    try:
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("invalid fleet drain marker metadata")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ValueError("fleet drain marker changed during inspection")
        content = os.read(descriptor, len(_MEMORY_FLEET_DRAIN_MARKER) + 1)
        if content != _MEMORY_FLEET_DRAIN_MARKER:
            raise ValueError("invalid fleet drain marker content")
    except Exception as exc:
        raise MemoryFleetDrainActiveError(MEMORY_FLEET_DRAIN_ACTIVE) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return True


def memory_fleet_drain_active(profiles_root: Path | None = None) -> bool:
    """Return whether a valid durable fleet drain marker is active."""
    path = (
        _memory_fleet_drain_marker_path(profiles_root)
        if profiles_root is not None
        else memory_fleet_drain_marker_path()
    )
    return _inspect_memory_fleet_drain_marker(path)


def _fsync_memory_fleet_drain_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    for directory_path in (path.parent, path.parent.parent, path.parent.parent.parent):
        directory = os.open(
            directory_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def begin_memory_fleet_drain(profiles_root: Path) -> None:
    """Durably publish the fleet marker before any profile lock is acquired."""
    path = _memory_fleet_drain_marker_path(profiles_root)
    if _inspect_memory_fleet_drain_marker(path):
        _fsync_memory_fleet_drain_path(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(_MEMORY_FLEET_DRAIN_MARKER):
                written = os.write(descriptor, _MEMORY_FLEET_DRAIN_MARKER[offset:])
                if written <= 0:
                    raise OSError("fleet drain marker write made no forward progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_memory_fleet_drain_path(path)
    finally:
        temporary.unlink(missing_ok=True)


def clear_memory_fleet_drain(profiles_root: Path) -> None:
    """Durably clear the fleet marker after successful finalization."""
    path = _memory_fleet_drain_marker_path(profiles_root)
    if not _inspect_memory_fleet_drain_marker(path):
        return
    path.unlink()
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def memory_pending_lock():
    """Hold the reentrant process/file lock used by memory pending drains."""
    with _MEMORY_PENDING_THREAD_LOCK:
        path = _pending_dir(MEMORY) / ".drain.lock"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = _MEMORY_PENDING_LOCK_STATE
        if state["depth"] == 0:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
            except Exception:
                os.close(descriptor)
                raise
            state.update({"fd": descriptor, "path": path, "depth": 1})
        else:
            if state["path"] != path:
                raise RuntimeError("memory pending lock home changed while held")
            state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1
            if state["depth"] == 0:
                descriptor = state["fd"]
                try:
                    if fcntl is not None:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
                    state.update({"fd": None, "path": None})


@contextmanager
def memory_write_coordination():
    """Block tiered memory writes while a durable fleet drain is active."""
    try:
        with memory_pending_lock():
            if not memory_fleet_drain_active():
                yield
                return
            raise MemoryFleetDrainActiveError(MEMORY_FLEET_DRAIN_ACTIVE)
    finally:
        pass


def _pending_locked(function):
    def wrapped(subsystem, *args, **kwargs):
        if subsystem == MEMORY:
            with memory_pending_lock():
                return function(subsystem, *args, **kwargs)
        return function(subsystem, *args, **kwargs)

    return wrapped


@_pending_locked
def stage_write(subsystem: str, payload: Dict[str, Any],
                *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return a short record describing it.

    Args:
        subsystem: ``memory`` or ``skills``.
        payload: the exact kwargs needed to replay the write when approved
            (e.g. ``{"action": "add", "target": "user", "content": "..."}``
            for memory, or the full ``skill_manage`` kwargs for skills).
        summary: a one-line human-readable description shown in pending lists.
            For skills this is the LLM/heuristic gist; for memory it can be the
            entry text itself.
        origin: ``foreground`` or ``background_review`` — recorded for audit.

    Returns a dict with ``id`` and metadata. Raises ``PendingWriteError``
    when the record cannot be durably persisted.
    """
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid,
        "subsystem": subsystem,
        "action": payload.get("action", ""),
        "summary": (summary or "").strip(),
        "origin": origin or "foreground",
        "created_at": time.time(),
        "payload": payload,
    }
    try:
        d = _pending_dir(subsystem)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{pid}.json"
        _atomic_write(path, record)
    except Exception as e:
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
        audit_context = payload.get("_memory_audit") if subsystem == MEMORY else None
        if isinstance(audit_context, dict):
            _audit_memory_lifecycle_fail_safe(
                "failed",
                audit_context,
                pending_id=pid,
                failure_code="pending_store_unavailable",
            )
        raise PendingWriteError(
            f"Failed to persist pending {subsystem} write"
        ) from e
    audit_context = payload.get("_memory_audit") if subsystem == MEMORY else None
    if isinstance(audit_context, dict):
        _audit_memory_lifecycle_fail_safe("staged", audit_context, pending_id=pid)
    return record


@_pending_locked
def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    d = _pending_dir(subsystem)
    if not d.exists():
        return []
    records: List[Dict[str, Any]] = []
    for p in d.glob("*.json"):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_dir(subsystem) / f"{pending_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@_pending_locked
def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    path = _pending_dir(subsystem) / f"{pending_id}.json"
    record = get_pending(subsystem, pending_id) if subsystem == MEMORY else None
    try:
        if path.exists():
            path.unlink()
            audit_context = ((record or {}).get("payload") or {}).get("_memory_audit")
            if isinstance(audit_context, dict):
                _audit_memory_lifecycle_fail_safe(
                    "discarded", audit_context, pending_id=pending_id,
                )
            return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    d = _pending_dir(subsystem)
    if not d.exists():
        return 0
    try:
        return sum(1 for _ in d.glob("*.json"))
    except Exception:
        return 0


def _replay_receipt(subsystem: str, pending_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return {
        "schema": "hermes-pending-replay-receipt/v1",
        "subsystem": subsystem,
        "pending_id_sha256": hashlib.sha256(str(pending_id).encode("utf-8")).hexdigest(),
        "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _replay_receipt_path(receipt: Dict[str, str]) -> Path:
    return (
        get_hermes_home()
        / "audit"
        / "pending-replay"
        / f"{receipt['pending_id_sha256']}.json"
    )


def _verify_replay_receipt(path: Path, expected: Dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except Exception:
        return False


def _persist_replay_receipt(expected: Dict[str, str]) -> None:
    path = _replay_receipt_path(expected)
    if path.exists():
        if not _verify_replay_receipt(path, expected):
            raise RuntimeError("pending replay receipt mismatch")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        data = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_prepared_pending(
    subsystem: str, pending_id: str, record: Dict[str, Any],
) -> None:
    path = _pending_dir(subsystem) / f"{pending_id}.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        data = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@_pending_locked
def replay_pending(subsystem: str, pending_id: str) -> bool:
    """Replay once, persist a redacted receipt, then discard pending state."""
    record = get_pending(subsystem, pending_id)
    if record is None:
        return False
    payload = record.get("payload") or {}
    audit_context = payload.get("_memory_audit")
    if subsystem == MEMORY and payload.get("action") != "fact_store":
        from tools.memory_tool import load_on_disk_store, prepare_memory_pending_replay
        try:
            prepared = prepare_memory_pending_replay(payload, load_on_disk_store())
        except Exception as exc:
            logger.error(
                "Pending memory replay preparation failed for id sha256:%s (%s)",
                hashlib.sha256(str(pending_id).encode("utf-8")).hexdigest(),
                type(exc).__name__,
            )
            if isinstance(audit_context, dict):
                _audit_memory_lifecycle_fail_safe(
                    "failed",
                    audit_context,
                    pending_id=pending_id,
                    failure_code="replay_preparation_failed",
                )
            return False
        if prepared != payload:
            record = dict(record)
            record["payload"] = prepared
            _persist_prepared_pending(subsystem, pending_id, record)
            payload = prepared
            audit_context = payload.get("_memory_audit")
    receipt = _replay_receipt(subsystem, pending_id, payload)
    receipt_path = _replay_receipt_path(receipt)
    if receipt_path.exists():
        if not _verify_replay_receipt(receipt_path, receipt):
            return False
        return discard_pending(subsystem, pending_id)
    try:
        if subsystem == MEMORY:
            from tools.memory_tool import apply_memory_pending, load_on_disk_store
            if payload.get("action") == "fact_store":
                result = apply_memory_pending(
                    payload,
                    load_on_disk_store(),
                    replay_key=receipt["pending_id_sha256"],
                    replay_payload_sha256=receipt["payload_sha256"],
                )
            else:
                result = apply_memory_pending(payload, load_on_disk_store())
        elif subsystem == SKILLS:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        else:
            return False
    except Exception as exc:
        logger.error(
            "Pending %s replay failed for id sha256:%s (%s)",
            subsystem,
            hashlib.sha256(str(pending_id).encode("utf-8")).hexdigest(),
            type(exc).__name__,
        )
        if isinstance(audit_context, dict):
            _audit_memory_lifecycle_fail_safe(
                "failed",
                audit_context,
                pending_id=pending_id,
                failure_code="replay_exception",
            )
        return False
    if not result.get("success"):
        return False
    _persist_replay_receipt(receipt)
    if isinstance(audit_context, dict):
        _audit_memory_lifecycle_fail_safe(
            "replayed", audit_context, pending_id=pending_id,
        )
    return discard_pending(subsystem, pending_id)


# ---------------------------------------------------------------------------
# Write origin
# ---------------------------------------------------------------------------

def current_origin() -> str:
    """Return the active write origin: ``foreground`` or ``background_review``.

    Reuses the skill-provenance ContextVar, which the background review fork
    already sets (see ``agent.background_review`` /
    ``AIAgent._spawn_background_review``). Foreground agent turns leave it at
    the default ``foreground``.
    """
    try:
        from tools.skill_provenance import get_current_write_origin
        return get_current_write_origin()
    except Exception:
        return "foreground"


def is_background() -> bool:
    return current_origin() == "background_review"


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

class GateDecision:
    """Result of evaluating the write gate for a single write attempt.

    Exactly one of the boolean flags is True:
      * ``allow``  — proceed with the real write (gate off, or an inline
        approval was granted).
      * ``blocked`` — refuse the write (the user denied an inline approval
        prompt). ``message`` explains why; surface it to the agent.
      * ``stage``  — do not write; the caller should stage the payload via
        ``stage_write`` (gate on, and no inline prompt is available — gateway,
        background review, script, or any skill write). ``message`` is the
        user-facing "staged for approval" note.
    """

    __slots__ = ("allow", "blocked", "stage", "message")

    def __init__(self, *, allow=False, blocked=False, stage=False, message=""):
        self.allow = allow
        self.blocked = blocked
        self.stage = stage
        self.message = message


def evaluate_gate(subsystem: str, *, inline_summary: str = "",
                  inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write for ``subsystem``.

    Args:
        subsystem: ``memory`` or ``skills``.
        inline_summary: short description used as the inline approval prompt
            header (memory foreground path only).
        inline_detail: full content shown in the inline prompt (memory entries
            are small; skills never take the inline path).

    Decision matrix:
        gate off (default)                    → allow (writes flow freely)
        gate on, memory + interactive CLI     → inline approve/deny prompt
        gate on, memory + gateway/script/bg   → stage
        gate on, skills (any origin)          → stage (too big to review inline)

    Note: there is no config-driven "blocked" outcome — the gate only ever
    delays a write for approval, never silently refuses it. ``blocked`` is
    still produced when the user *actively denies* an inline prompt.
    """
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)

    background = is_background()

    # Skills always stage — a SKILL.md is too large to review inline, and a
    # background skill write happens in a daemon thread with no user present.
    if subsystem == SKILLS or background:
        where = "/skills pending" if subsystem == SKILLS else "/memory pending"
        return GateDecision(
            stage=True,
            message=(
                f"Staged for approval ({subsystem}.write_approval is on). "
                f"Not yet saved — review with {where}."
            ),
        )

    # Memory + foreground: if an interactive approval channel exists (a CLI
    # approval callback registered on this thread), prompt inline — entries
    # are small enough to show in full. Otherwise (gateway, script, batch,
    # no listener) stage instead of forcing a blind deny.
    if _interactive_approval_available():
        granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
        if granted is True:
            return GateDecision(allow=True)
        if granted is False:
            return GateDecision(
                blocked=True,
                message="Memory write denied by user. The change was not saved.",
            )
        # granted is None → prompt failed; fall through to staging.

    return GateDecision(
        stage=True,
        message=(
            "Staged for approval (memory.write_approval is on). "
            "Not yet saved — review with /memory pending."
        ),
    )


def _interactive_approval_available() -> bool:
    """True when a foreground memory write can be approved inline.

    Inline prompting requires a per-thread approval callback registered by the
    interactive CLI (``tools.terminal_tool.set_approval_callback``). Every
    other surface stages instead:

    * **Gateway/API sessions** — the dangerous-command ``/approve`` round-trip
      lives in the pending-approval queue (``submit_pending`` +
      ``_await_gateway_decision``), which ``prompt_dangerous_approval`` never
      reaches; trying to prompt from a gateway session would hit the
      ``input()`` fallback and silently deny. Staging gives the user a real
      review affordance (``/memory pending``) instead.
    * Scripts, cron, and background threads — no user present.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback() is not None
    except Exception:
        return False


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt the user inline to approve a memory write.

    Returns True (approved), False (denied), or None (no interactive prompt
    available / prompt failed → caller should stage instead).

    Reuses the per-thread CLI approval callback registered for dangerous
    commands (``tools.terminal_tool.set_approval_callback``). The callback is
    invoked directly — NOT via ``prompt_dangerous_approval`` — because that
    wrapper falls back to ``input()`` (deadlock-prone under prompt_toolkit,
    see #15216) and converts callback errors into a silent deny; here a
    failed prompt must stage the write instead.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None

    callback = _get_approval_callback()
    if callback is None:
        # No interactive channel on this thread — stage rather than risk the
        # input() fallback (deadlock under prompt_toolkit, EOF-deny in tests).
        return None

    header = summary.strip() or "Save to memory?"
    body = detail.strip()
    description = f"Save to memory: {header}"
    command = body if body else header
    # Invoke the callback directly instead of via prompt_dangerous_approval:
    # that wrapper swallows callback exceptions into "deny", which would
    # silently refuse the write. Direct invocation lets a crashed prompt fall
    # back to staging (the gate only ever delays a write, never drops it).
    try:
        choice = callback(command, description, allow_permanent=False)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None

    if choice in {"once", "session"}:
        return True
    if choice == "deny":
        return False
    # Any other outcome (e.g. timeout that returns "deny" already handled) →
    # treat unknown as no-decision so we stage rather than silently drop.
    return None


# ---------------------------------------------------------------------------
# Skill-specific helpers (gist + diff for the review affordances)
# ---------------------------------------------------------------------------

def skill_gist(action: str, name: str, *, content: str = "",
               file_path: str = "", old_string: str = "",
               new_string: str = "") -> str:
    """Build a one-line human gist for a pending skill write.

    Heuristic, no model call — the gist surfaces enough to decide approve/reject
    in a chat bubble, while the full diff stays behind /skills diff (CLI/
    dashboard/file). For create/edit it pulls the frontmatter ``description:``;
    for patch/write_file it describes the size of the change.
    """
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        verb = "create" if action == "create" else "rewrite"
        if desc:
            return f"{verb} '{name}' — {desc} ({size})"
        return f"{verb} '{name}' ({size})"
    if action == "patch":
        target = file_path or "SKILL.md"
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {target} (+{added}/-{removed} lines)"
    if action == "write_file":
        return f"write {file_path} in '{name}'"
    if action == "remove_file":
        return f"remove {file_path} from '{name}'"
    if action == "delete":
        return f"delete skill '{name}'"
    return f"{action} '{name}'"


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter."""
    import re
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    if not m:
        return ""
    desc = m.group(1).strip().strip("'\"")
    return desc[:140]


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Build a full unified diff (or full content) for a staged skill write.

    Used by /skills diff <id> on a surface that can render it (CLI pager, web
    dashboard, or by opening the pending JSON file). For create this is the new
    file content; for edit/patch it is a unified diff against the current
    on-disk skill.
    """
    import difflib
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")

    if action == "create":
        return (payload.get("content") or "")

    # Resolve current on-disk content for diffable actions.
    try:
        from tools.skill_manager_tool import _find_skill
    except Exception:
        _find_skill = None  # type: ignore

    current = ""
    target_label = "SKILL.md"
    if _find_skill is not None:
        found = _find_skill(name)
        if found:
            base = found["path"]
            if action == "edit":
                p = base / "SKILL.md"
            elif action in {"patch", "write_file"}:
                rel = payload.get("file_path") or "SKILL.md"
                p = base / rel
                target_label = rel
            else:
                p = base / "SKILL.md"
            try:
                if p.exists():
                    current = p.read_text(encoding="utf-8")
            except Exception:
                current = ""

    if action == "edit":
        new = payload.get("content") or ""
    elif action == "patch":
        old_s = payload.get("old_string") or ""
        new_s = payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    elif action == "write_file":
        new = payload.get("file_content") or ""
    elif action == "remove_file":
        return f"remove file: {payload.get('file_path')} from skill '{name}'"
    elif action == "delete":
        return f"delete skill '{name}'"
    else:
        return f"({action} on '{name}')"

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{target_label}",
        tofile=f"b/{target_label}",
    )
    text = "".join(diff)
    return text or "(no textual change)"
