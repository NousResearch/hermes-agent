"""Reversibility-tier preflight helpers for tool dispatch.

Receipts intentionally carry only semantic metadata plus a payload hash.  The
trusted approval decision must be supplied by the runtime boundary separately
from model-provided tool arguments.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ReversibilityTier(str, Enum):
    """Coarse approval tier for a prospective tool action."""

    READ_ONLY = "read_only"
    STAGED_LOCAL_CHANGE = "staged_local_change"
    EXTERNALLY_VISIBLE_PUBLISH = "externally_visible_publish"
    SECRET_CREDENTIAL_HANDLING = "secret_credential_handling"
    DESTRUCTIVE_DELETE = "destructive_delete"
    UNKNOWN_RISKY = "unknown_risky"


_SECRET_KEY_RE = re.compile(
    r"(secret|token|credential|password|passwd|api[_-]?key|access[_-]?key|auth)",
    re.IGNORECASE,
)
_SECRET_PATH_RE = re.compile(
    r"(^|[/\\.])(?:\.env(?:\.|$)|auth(?:[/\\.]|$)|credentials?(?:[/\\.]|$)|"
    r"secrets?(?:[/\\.]|$)|token(?:[/\\.]|$)|keychain|credential_store)",
    re.IGNORECASE,
)
_RISKY_WRITE_RE = re.compile(r"(write|patch|edit|update|create|upload|publish|post|send)", re.IGNORECASE)
_DESTRUCTIVE_RE = re.compile(r"(delete|remove|destroy|drop|purge|wipe|truncate)", re.IGNORECASE)
_PUBLISH_RE = re.compile(r"(publish|post|send|message|tweet|email|notify|webhook)", re.IGNORECASE)

_READ_ONLY_TOOLS = frozenset({"read_file", "search_files"})
_STAGED_WRITE_TOOLS = frozenset({"write_file", "patch"})
_DESTRUCTIVE_TOOLS = frozenset({"delete_file", "remove_file", "rm", "unlink"})


@dataclass(frozen=True)
class ActionPreflight:
    """Pure classification output for one tool name + argument payload."""

    tool_name: str
    tier: ReversibilityTier
    payload_hash: str
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class TrustedActionDecision:
    """Runtime-owned approval decision, never parsed from model tool args."""

    receipt: "SemanticReceipt"
    source: str
    trusted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _redact_text(self.source) or "")


@dataclass(frozen=True)
class SemanticReceipt:
    """Approval receipt without raw args or secret values.

    Only payload_hash binds the approved payload to the later execution attempt;
    raw tool arguments are intentionally absent from the receipt.
    """

    tool_name: str
    tier: ReversibilityTier
    payload_hash: str
    approved: bool = False
    pre_state_ref: str | None = None
    expected_delta: str | None = None
    summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _redact_text(self.summary))
        object.__setattr__(self, "pre_state_ref", _redact_text(self.pre_state_ref))
        object.__setattr__(self, "expected_delta", _redact_text(self.expected_delta))

    @classmethod
    def for_preflight(
        cls,
        preflight: ActionPreflight,
        *,
        approved: bool,
        pre_state_ref: str | None = None,
        expected_delta: str | None = None,
        summary: str | None = None,
    ) -> "SemanticReceipt":
        return cls(
            tool_name=preflight.tool_name,
            tier=preflight.tier,
            payload_hash=preflight.payload_hash,
            approved=approved,
            pre_state_ref=pre_state_ref,
            expected_delta=expected_delta,
            summary=summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tier": self.tier.value,
            "payload_hash": self.payload_hash,
            "approved": self.approved,
            "pre_state_ref": self.pre_state_ref,
            "expected_delta": self.expected_delta,
            "summary": self.summary,
        }


def classify_tool_action(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    workspace_root: str | Path | None = None,
) -> ActionPreflight:
    """Classify a prospective tool call, failing closed for risky unknowns."""

    normalized_name = (tool_name or "").strip()
    payload_hash = _payload_hash(normalized_name, args or {})
    reason = "default"

    if _contains_secret_material(args or {}):
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.SECRET_CREDENTIAL_HANDLING,
            payload_hash,
            "secret-like argument key or path",
        )

    path_state = _path_state(args or {}, workspace_root)
    if path_state == "secret":
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.SECRET_CREDENTIAL_HANDLING,
            payload_hash,
            "secret-like destination path",
        )
    if path_state == "outside_workspace":
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.UNKNOWN_RISKY,
            payload_hash,
            "path resolves outside workspace",
        )

    if normalized_name in _READ_ONLY_TOOLS:
        return ActionPreflight(normalized_name, ReversibilityTier.READ_ONLY, payload_hash, "read-only tool")

    if normalized_name == "send_message":
        action = str((args or {}).get("action", "send")).lower()
        tier = ReversibilityTier.READ_ONLY if action == "list" else ReversibilityTier.EXTERNALLY_VISIBLE_PUBLISH
        reason = "send_message list" if tier is ReversibilityTier.READ_ONLY else "externally visible send"
        return ActionPreflight(normalized_name, tier, payload_hash, reason)

    if normalized_name in _DESTRUCTIVE_TOOLS or _DESTRUCTIVE_RE.search(normalized_name):
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.DESTRUCTIVE_DELETE,
            payload_hash,
            "destructive tool name",
        )

    if normalized_name in _STAGED_WRITE_TOOLS:
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.STAGED_LOCAL_CHANGE,
            payload_hash,
            "workspace-local staged change",
        )

    if _PUBLISH_RE.search(normalized_name) or _RISKY_WRITE_RE.search(normalized_name):
        return ActionPreflight(
            normalized_name,
            ReversibilityTier.UNKNOWN_RISKY,
            payload_hash,
            "unknown risky write/publish tool",
        )

    return ActionPreflight(normalized_name, ReversibilityTier.UNKNOWN_RISKY, payload_hash, reason)


def validate_receipt(
    preflight: ActionPreflight,
    receipt: SemanticReceipt | None,
    trusted_decision: bool,
) -> ValidationResult:
    """Validate receipt binding and trusted approval status for a preflight."""

    if preflight.tier is ReversibilityTier.READ_ONLY:
        return ValidationResult(True, "read-only passthrough")

    if receipt is None:
        return ValidationResult(False, "approval receipt required")
    if not trusted_decision:
        return ValidationResult(False, "trusted approval decision required")
    if not receipt.approved:
        return ValidationResult(False, "receipt is not approved")
    if receipt.tool_name != preflight.tool_name:
        return ValidationResult(False, "receipt tool mismatch")
    if receipt.tier is not preflight.tier:
        return ValidationResult(False, "receipt tier mismatch")
    if not hmac.compare_digest(receipt.payload_hash, preflight.payload_hash):
        return ValidationResult(False, "payload hash mismatch")

    if preflight.tier is ReversibilityTier.STAGED_LOCAL_CHANGE:
        if not receipt.pre_state_ref or not receipt.expected_delta:
            return ValidationResult(False, "staged changes require pre_state_ref and expected_delta")

    if preflight.tier in {
        ReversibilityTier.EXTERNALLY_VISIBLE_PUBLISH,
        ReversibilityTier.SECRET_CREDENTIAL_HANDLING,
        ReversibilityTier.DESTRUCTIVE_DELETE,
        ReversibilityTier.STAGED_LOCAL_CHANGE,
    }:
        return ValidationResult(True, "trusted receipt accepted")

    return ValidationResult(False, "fail closed for unknown risky action")


def _payload_hash(tool_name: str, args: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"tool_name": tool_name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text):
                return True
            if key_text.lower() in {"path", "filepath", "file_path", "dest", "destination"} and _SECRET_PATH_RE.search(str(child)):
                return True
            if _contains_secret_material(child):
                return True
    elif isinstance(value, (list, tuple, set)):
        return any(_contains_secret_material(item) for item in value)
    return False


def _path_state(args: Mapping[str, Any], workspace_root: str | Path | None) -> str | None:
    raw_path = _extract_path(args)
    if not raw_path:
        return None
    if _SECRET_PATH_RE.search(raw_path):
        return "secret"
    if workspace_root is None:
        return None

    root = Path(workspace_root).expanduser().resolve()
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return "outside_workspace"
    return None


def _extract_path(args: Mapping[str, Any]) -> str | None:
    for key in ("path", "filepath", "file_path", "target_path", "dest", "destination"):
        value = args.get(key)
        if value:
            return str(value)
    return None


def _redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|credential)\s*[:=]\s*\S+", r"\1=***", text)
    text = re.sub(r"(?i)\b(sk-[A-Za-z0-9._-]{6,}|[A-Za-z0-9._-]*token[A-Za-z0-9._-]*)\b", "***", text)
    return text
