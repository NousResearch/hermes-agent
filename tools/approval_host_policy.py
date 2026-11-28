"""Host approval constraints, independent of profile snapshots and prompt state.

The owner-only host policy authorizes exceptions; ordinary session state cannot
mint one. Absent policy preserves the existing unmanaged contract.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import math
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

_MODES = {"manual": 0, "smart": 1, "off": 2}
_BINARY_KEYS = ("cron_mode", "single_query_mode", "unattended_mode")
_BOOL_KEYS = ("subagent_auto_approve", "allow_standing_grants")
_EXCEPTION_PERMISSIONS = frozenset({
    "yolo", "profile_mode", "hosted_room", "standing_grants",
    "subagent_auto_approve", "unattended_approve",
})
_EXCEPTION_FIELDS = frozenset({
    "id", "profile", "session_key", "reason", "issued_at", "expires_at", "allow",
})
_CURRENT: contextvars.ContextVar[ResolvedPolicy | None] = contextvars.ContextVar(
    "host_approval_policy", default=None,
)
_GRANTS_LOCK = threading.Lock()
_SESSION_GRANTS: dict[tuple[str, str, str], tuple[str, str]] = {}


def _invalid(message: str):
    from tools.host_policy import HostPolicyError
    raise HostPolicyError(message)


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_timestamp(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _validate_exception(value) -> dict:
    if not isinstance(value, dict) or set(value) != _EXCEPTION_FIELDS:
        _invalid("approval override has unsupported or missing fields")
    if not all(_nonempty(value[key]) for key in ("id", "profile", "session_key", "reason")):
        _invalid("approval override identity and scope must be nonempty strings")
    profile = Path(value["profile"])
    if not profile.is_absolute() or str(profile.resolve()) != value["profile"]:
        _invalid("approval override profile must be an absolute canonical home")
    for key in ("issued_at", "expires_at"):
        number = value[key]
        if not _finite_timestamp(number):
            _invalid("approval override timestamps must be finite numbers")
    if not 0 < value["expires_at"] - value["issued_at"] <= 86400:
        _invalid("approval overrides must expire within 24 hours of issue")
    permissions = value["allow"]
    if (not isinstance(permissions, list) or not permissions
            or any(not isinstance(item, str) or item not in _EXCEPTION_PERMISSIONS for item in permissions)
            or len(set(permissions)) != len(permissions)):
        _invalid("approval override permissions must be unique supported names")
    return dict(value)


def validate_approvals_policy(value) -> dict:
    """Validate the approvals section without loading policy or profile config."""
    if not isinstance(value, dict) or set(value) - {"mode", *_BINARY_KEYS, *_BOOL_KEYS, "overrides"}:
        _invalid("host approvals must be an object with supported fields")
    mode = value.get("mode", "manual")
    if not isinstance(mode, str) or mode not in _MODES:
        _invalid("host approval mode must be manual, smart, or off")
    result = {"mode": mode}
    for key in _BINARY_KEYS:
        item = value.get(key, "deny")
        if not isinstance(item, str) or item not in ("deny", "approve"):
            _invalid("host unattended approval modes must be deny or approve")
        result[key] = item
    for key in _BOOL_KEYS:
        item = value.get(key, False)
        if type(item) is not bool:
            _invalid("host approval permissions must be booleans")
        result[key] = item
    overrides = value.get("overrides", [])
    if not isinstance(overrides, list):
        _invalid("host approval overrides must be a list")
    result["overrides"] = [_validate_exception(item) for item in overrides]
    identifiers = [item["id"] for item in result["overrides"]]
    if len(set(identifiers)) != len(identifiers):
        _invalid("host approval override ids must be unique")
    return result


@dataclass
class ResolvedPolicy:
    revision: str
    rules: dict
    profile: str
    session_key: str
    mode: str
    permissions: set[str] = field(default_factory=set)
    override_ids: list[str] = field(default_factory=list)
    rejected: set[str] = field(default_factory=set)
    error: bool = False
    callback_present: bool = False

    def fingerprint(self) -> tuple[str, str]:
        encoded = json.dumps(self.rules, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return self.revision, hashlib.sha256(encoded.encode()).hexdigest()

    def status(self) -> dict:
        """Allowlisted metadata: never includes reasons, commands, or config."""
        return {
            "source": "host-policy", "revision": self.revision,
            "policy_digest": self.fingerprint()[1],
            "profile": self.profile, "session_key": self.session_key,
            "mode": self.mode, "override_ids": list(self.override_ids),
            "rejected_overrides": sorted(self.rejected), "error": self.error,
        }


def _profile_and_session(session_key: str | None = None) -> tuple[str, str]:
    from hermes_constants import get_hermes_home
    from tools.approval_context import get_current_session_key
    return str(get_hermes_home().resolve()), (
        get_current_session_key(default="") if session_key is None else session_key
    )


def _restrict_mode(policy: ResolvedPolicy, requested: str, source: str) -> None:
    if _MODES[requested] <= _MODES[policy.mode] or source in policy.permissions:
        policy.mode = requested
    else:
        policy.rejected.add(source)


def _resolve(session_key: str | None = None) -> ResolvedPolicy | None:
    from tools.host_policy import HostPolicyError, load_host_policy
    from tools import approval_context

    profile, session = _profile_and_session(session_key)
    try:
        host = load_host_policy()
    except HostPolicyError:
        return ResolvedPolicy("", {}, profile, session, "manual", error=True)
    if host is None:
        return None
    try:
        rules = validate_approvals_policy(host.get("approvals", {}))
    except HostPolicyError:
        return ResolvedPolicy(host["revision"], {}, profile, session, "manual", error=True)
    policy = ResolvedPolicy(host["revision"], rules, profile, session, rules["mode"])
    now = time.time()
    for override in rules["overrides"]:
        if (override["profile"] == profile and override["session_key"] == session
                and override["issued_at"] <= now < override["expires_at"]):
            policy.permissions.update(override["allow"])
            policy.override_ids.append(override["id"])
    config = approval_context._get_approval_config()
    if "mode" in config:
        _restrict_mode(policy, approval_context._normalize_approval_mode(config["mode"]), "profile_mode")
    from gateway.hosted_room_execution_policy import current_room_execution_policy
    room = current_room_execution_policy()
    if room is not None:
        _restrict_mode(policy, approval_context._normalize_approval_mode(room.approval_mode), "hosted_room")
    return policy


def current_policy(session_key: str | None = None) -> ResolvedPolicy | None:
    active = _CURRENT.get()
    profile, session = _profile_and_session(session_key)
    if active is not None and (active.profile, active.session_key) == (profile, session):
        return active
    return _resolve(session_key)


def approval_policy_status() -> dict:
    """Read-only effective-policy diagnostic for the bound profile/session."""
    policy = current_policy()
    if policy is not None:
        return policy.status()
    return {"source": "profile", "revision": None, "managed": False}


def managed_approval_gate(function):
    """Freeze a decision's policy read, fail closed, and attach provenance."""
    signature = inspect.signature(function)
    @wraps(function)
    def guarded(*args, **kwargs):
        policy = current_policy()
        if policy is None:
            return function(*args, **kwargs)
        if policy.error:
            return {
                "approved": False, "outcome": "policy_error", "user_consent": False,
                "message": "BLOCKED: host approval policy is unavailable or invalid. "
                           "Ask the operator to repair it. Do NOT retry or rephrase.",
                "approval_policy": policy.status(),
            }
        arguments = signature.bind_partial(*args, **kwargs).arguments
        policy.callback_present |= callable(arguments.get("approval_callback"))
        token = _CURRENT.set(policy)
        try:
            result = function(*args, **kwargs)
            return {**result, "approval_policy": policy.status()}
        finally:
            _CURRENT.reset(token)
    return guarded


def managed_mode() -> str | None:
    policy = current_policy()
    return policy.mode if policy is not None else None


def bypass_allowed(requested: bool, session_key: str | None = None) -> bool:
    policy = current_policy(session_key)
    if policy is None:
        return requested
    if policy.error:
        return False
    if requested and "yolo" not in policy.permissions:
        policy.rejected.add("yolo")
        return False
    return requested


def binary_mode(key: str, requested: str) -> str:
    policy = current_policy()
    if policy is None:
        return requested
    if policy.error:
        return "deny"
    if requested == "deny":
        return "deny"
    if policy.rules[key] == "approve" or "unattended_approve" in policy.permissions:
        return "approve"
    policy.rejected.add(key)
    return "deny"


def standing_grants_allowed() -> bool:
    policy = current_policy()
    if policy is None:
        return True
    if policy.error:
        return False
    from tools.approval import _unattended_contexts
    if any(ctx.mode() == "deny" for ctx in _unattended_contexts()):
        return False
    return policy.rules["allow_standing_grants"] or "standing_grants" in policy.permissions


def permanent_grants_allowed() -> bool:
    policy = current_policy()
    return policy is None or (
        standing_grants_allowed() and "standing_grants" in policy.permissions
    )


def record_session_grant(session_key: str, pattern_key: str) -> None:
    policy = current_policy(session_key)
    if policy is not None and not policy.error:
        with _GRANTS_LOCK:
            _SESSION_GRANTS[(policy.profile, session_key, pattern_key)] = policy.fingerprint()


def session_grant_matches(session_key: str, pattern_key: str) -> bool:
    policy = current_policy(session_key)
    if policy is None:
        return True
    with _GRANTS_LOCK:
        return _SESSION_GRANTS.get((policy.profile, session_key, pattern_key)) == policy.fingerprint()


def clear_session_grants(session_key: str) -> None:
    with _GRANTS_LOCK:
        for key in [key for key in _SESSION_GRANTS if key[1] == session_key]:
            del _SESSION_GRANTS[key]


def prepared_guard_allowed(result: dict) -> bool:
    """A denial stays denied; prepared consent cannot outlive its policy."""
    if not result.get("approved"):
        return True
    policy = current_policy()
    if policy is None:
        return True
    previous = result.get("approval_policy", {})
    status = policy.status()
    return not policy.error and all(previous.get(key) == status[key] for key in (
        "revision", "policy_digest", "profile", "session_key", "mode", "override_ids",
    ))


def subagent_auto_approve_allowed(requested: bool) -> bool:
    """Called at callback invocation, not only when a worker is constructed."""
    policy = current_policy()
    if policy is None:
        return requested
    return bool(not policy.error and requested and (
        policy.rules["subagent_auto_approve"] or "subagent_auto_approve" in policy.permissions
    ))


def managed_child_mode() -> str:
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly().get("delegation", {}) or {}
    requested = config.get("subagent_auto_approve") is True
    return "approve" if subagent_auto_approve_allowed(requested) else "deny"


def delegated_child() -> bool:
    from agent.delegation_context import is_delegated_child_process_context
    return is_delegated_child_process_context()


def constrain_presence(approval_callback, is_cli: bool, is_gateway: bool, is_ask: bool) -> tuple:
    policy = current_policy()
    if policy is None:
        return approval_callback, is_cli, is_gateway, is_ask
    from tools.approval import _gateway_notify_cb
    from tools.approval_context import get_current_session_key
    notify = _gateway_notify_cb(get_current_session_key())
    child = delegated_child()
    return (
        approval_callback,
        is_cli and (approval_callback is not None or policy.callback_present) and not child,
        is_gateway and notify is not None and not child,
        is_ask and notify is not None and not child,
    )


def managed_unattended_contexts(contexts: list) -> list:
    policy = current_policy()
    if policy is None:
        return contexts
    from tools.approval import _presence, _Unattended
    present = any(_presence()[1:])
    if present:
        return [ctx for ctx in contexts if ctx.name != "unattended"]
    if contexts:
        return contexts
    if delegated_child():
        return [_Unattended(
            "subagent", "subagent_auto_approve", "this delegated worker has no human approver",
            "in delegated workers", "the owner-authorized child policy permits it",
        )]
    return [_Unattended(
        "unattended", "unattended_mode", "no eligible human approval surface is present",
        "without an approval surface", "the owner-authorized unattended policy permits it",
    )]


def persistence_permissions(smart_denied: bool) -> tuple[bool, bool]:
    return (
        not smart_denied and standing_grants_allowed(),
        not smart_denied and permanent_grants_allowed(),
    )


def managed_prompt_kwargs() -> dict:
    if current_policy() is not None and not standing_grants_allowed():
        return {"allow_session": False}
    return {}
