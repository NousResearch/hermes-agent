"""Exact-action approval: a human decision bound to the precise (tool_name, final
args) that will dispatch, for consequential tool calls where a generic ``approve``
is not enough.

Why this exists: a ``pre_tool_call`` hook's ``approve`` directive is resolved from a
plugin-authored message computed from the ORIGINAL args, while dispatch may use
``modify``-transformed FINAL args merged in from any hook (same plugin or another
one) registered anywhere in the hook list. Those two are computed independently, so
a human can approve "delete draft A" and have Hermes dispatch "send email B" — a
confused-deputy gap, not a hypothetical one, once two or more ``pre_tool_call``
plugins are enabled (see the ``pre_tool_call`` section of
``website/docs/user-guide/features/hooks.md`` for the user-facing contract and a
minimal safe plugin example).

``require_exact_action`` (the new ``pre_tool_call`` directive, resolved in
``hermes_cli/plugins.py``) closes it generically:

1. ``hermes_cli.plugins._get_pre_tool_call_directive_details`` finishes collecting
   every ``modify`` directive (from every hook, regardless of order) before this
   module is ever called, so ``final_args`` here is always the true dispatch args.
2. :func:`request_exact_action_approval` (host-only; called from
   ``hermes_cli.plugins._resolve_block_from_details``) computes a canonical digest of
   ``(tool_name, final_args)``, shows a redacted rendering of the exact final action to
   a human, and — only on a fresh, explicit approval — mints a one-time receipt bound
   to the current session key and tool-call id. It intentionally does NOT consult
   ``--yolo``, ``approvals.mode: off``, or any allowlist: this path is designed to be
   the one kind of approval those settings cannot satisfy.
3. :func:`consume_exact_action_approval` (handler-facing; called BY a plugin's tool
   handler, not by the host) verifies the receipt against the args the handler is
   about to act on and pops it — single use. It fails closed (raises
   :class:`ExactActionApprovalError`) on a missing, expired, already-consumed,
   wrong-tool, wrong-argument, wrong-session, or wrong-tool-call token. A handler
   MUST call this itself before mutating anything; dispatch reaching the handler is
   not, by itself, consent.

Scope, honestly stated: receipts live in this process's memory only. A gateway
restart between approval and consumption invalidates every pending receipt (the
handler's consume call then fails closed, which is the correct behavior — it is
not a durable, cross-restart signed capability, and does not attempt to be one for
this first version). This module also does not solve a plugin's own domain-specific
transactional safety (e.g. "did the email actually send exactly once"); that
remains the plugin's responsibility, same as with a normal ``approve``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tools.approval import _ACTION_GATE, _approved, _blocked, _human_decision, _presence
from tools.approval_context import _approval_tool_call_id, get_current_session_key

logger = logging.getLogger(__name__)

# Independent of ``approvals.timeout``: that config bounds how long we wait for a
# human to answer, which already happened by the time a receipt is minted. This
# bounds only the (normally sub-second) gap between "hook resolved" and "handler
# dispatched", so a receipt cannot be left around indefinitely if a handler never
# claims it.
_DEFAULT_TTL_SECONDS = 120
_MAX_DISPLAY_CHARS = 4000


class ExactActionApprovalError(RuntimeError):
    """Raised by :func:`consume_exact_action_approval` on any non-match. Safe to
    surface to the model as a tool-result error string — never includes raw args."""


@dataclass(frozen=True)
class _Receipt:
    tool_name: str
    digest: str
    expires_at: float


_lock = threading.Lock()
# Keyed by (session_key, tool_call_id): both are host-bound context, never
# model-controlled input, and both are stable across the hook-resolution call and
# the later handler-dispatch call for the same tool call (see module docstring).
_receipts: dict[tuple[str, str], _Receipt] = {}


def _to_canonical_json_value(value: Any) -> Any:
    """Recursively validate that ``value`` contains only unambiguous JSON-native types
    (``None``, ``str``, ``bool``, ``int``, finite ``float``, mapping, list/tuple).

    Deliberately does NOT fall back to ``str(value)`` for anything else: ``pre_tool_call``
    hooks are Python code and may merge arbitrary objects into the final dispatch args, and a
    ``default=str`` fallback would let two semantically/type-distinct values (``Path("x")`` vs
    the string ``"x"``, ``Decimal("1")`` vs ``"1"``, or any plugin object whose ``__str__``
    happens to collide with another value's canonical form) hash to the same digest. That would
    let an approval minted for one value be silently consumed for a different one — the exact
    wrong-argument confusion this module exists to prevent. Unrecognized types and non-finite
    floats are rejected outright (fail closed) rather than coerced."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not a canonical action value: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_canonical_json_value(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_canonical_json_value(v) for v in value]
    raise ValueError(f"value of type {type(value).__name__!r} cannot be canonicalized for an "
                     "exact-action approval")


def canonicalize_action(tool_name: str, args: Mapping[str, Any]) -> str:
    """Digest binding an approval to one exact dispatch action. Raises ``ValueError``
    on malformed or non-canonical input (fail closed at the caller) — see
    :func:`_to_canonical_json_value` for why non-JSON-native values are rejected rather
    than stringified."""
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(args, Mapping):
        raise ValueError("args must be a mapping")
    safe_args = _to_canonical_json_value(dict(args))
    canonical = json.dumps({"tool_name": tool_name, "args": safe_args},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _render_final_action(tool_name: str, args: Mapping[str, Any]) -> str:
    """Best-effort, redacted, length-bounded rendering of the exact action for a
    human to review — generated by the HOST from the final args, independent of
    whatever text the requesting plugin wrote, so a stale/wrong plugin message
    cannot hide what will actually execute."""
    from agent.redact import redact_sensitive_text
    try:
        raw = json.dumps(dict(args), sort_keys=True, indent=2, default=str, ensure_ascii=False)
    except Exception:
        raw = str(args)
    # force=True: this is a safety boundary that must never leak a raw secret
    # forward into a CLI transcript, gateway message, or log line.
    redacted = redact_sensitive_text(f"{tool_name}({raw})", force=True)
    if len(redacted) > _MAX_DISPLAY_CHARS:
        redacted = redacted[:_MAX_DISPLAY_CHARS] + "\n... (truncated)"
    return redacted


def _purge_expired_locked(now: float) -> None:
    for key in [key for key, receipt in _receipts.items() if receipt.expires_at <= now]:
        del _receipts[key]


def request_exact_action_approval(
    tool_name: str, message: str, final_args: Mapping[str, Any], *,
    rule_key: str = "", tool_call_id: str = "", approval_callback=None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> dict:
    """Host-only entry point: escalate to the non-bypassable exact-action gate and,
    on approval, mint a one-time receipt for the exact ``(tool_name, final_args)``.

    Called from ``hermes_cli.plugins._resolve_block_from_details`` for a
    ``require_exact_action`` directive — a plugin never calls this directly (it
    returns the directive from its ``pre_tool_call`` hook instead). Never consults
    ``--yolo``, ``approvals.mode: off``, or the session/permanent allowlist: this
    is the one approval kind those cannot satisfy. Always requires a live human
    surface (CLI, gateway, or ask-mode) — cron/single-query/unattended/no-human
    contexts fail closed unconditionally, with no config escape hatch, unlike the
    generic ``approve`` gate.
    """
    description = message or f"Exact-action approval required for {tool_name}"
    if not tool_call_id:
        # Without a tool-call id we cannot bind the receipt to "this exact call", so
        # there is nothing safe to mint — fail closed rather than approve unscoped.
        return _blocked("BLOCKED: exact-action approval requires a tool-call id to bind to; "
                        "none was provided by the dispatcher.",
                        pattern_key=f"exact_action:{tool_name}:no-call-id", description=description)
    try:
        digest = canonicalize_action(tool_name, final_args)
    except ValueError as exc:
        return _blocked(f"BLOCKED: exact-action approval could not canonicalize the final action: {exc}",
                        pattern_key=f"exact_action:{tool_name}:invalid-args", description=description)

    session_key = get_current_session_key()
    pattern_key = f"exact_action:{rule_key}" if rule_key else f"exact_action:{tool_name}:{digest[:16]}"
    rendered = _render_final_action(tool_name, final_args)
    display = f"{description}\n\nExact final action that will execute:\n{rendered}"

    approval_callback, is_cli, is_gateway, is_ask = _presence(approval_callback)
    if not (is_cli or is_gateway or is_ask):
        return _blocked("BLOCKED: exact-action approval is required but no interactive user or "
                        "gateway session is present to approve it. This action never runs "
                        "unattended, regardless of cron/single-query/unattended approval config.",
                        pattern_key=pattern_key, description=description)

    try:
        # warnings=[] so a "session"/"always" choice on the prompt persists nothing:
        # every exact action gets a fresh human decision, no matter what a human
        # picks on this one, because the digest is (by design) different every time.
        # permanent_capable=False additionally hides "[a]lways" on the CLI prompt.
        result = _human_decision(
            _ACTION_GATE, command=display, description=description, pattern_key=pattern_key,
            pattern_keys=[pattern_key], warnings=[], session_key=session_key,
            approval_callback=approval_callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
            smart=False, permanent_capable=False,
        )
    except Exception:
        logger.exception("Exact-action approval gate failed for %s", tool_name)
        return _blocked(f"BLOCKED: exact-action approval gate failed for {tool_name}",
                        pattern_key=pattern_key, description=description)

    if not result.get("approved"):
        return result

    now = time.time()
    with _lock:
        _purge_expired_locked(now)
        _receipts[(session_key, tool_call_id)] = _Receipt(
            tool_name=tool_name, digest=digest, expires_at=now + max(1, int(ttl_seconds)))
    return _approved()


def consume_exact_action_approval(tool_name: str, args: Mapping[str, Any]) -> None:
    """Handler-facing: verify and single-use-consume the receipt for the CURRENT
    tool call, or raise :class:`ExactActionApprovalError`.

    Call this from inside a mutation-capable tool handler while it is running
    under normal dispatch — the session key and tool-call id it binds to are read
    from the SAME host-owned context (`tools.approval_context`) that
    ``model_tools._execute_tool`` binds around every handler call, never from
    caller-supplied arguments, so a plugin cannot spoof which call it is
    consuming for. Pass the EXACT args the handler is about to act on: a mismatch
    (including a value changed after minting) fails closed like everything else
    here.
    """
    session_key = get_current_session_key()
    tool_call_id = _approval_tool_call_id.get()
    if not tool_call_id:
        raise ExactActionApprovalError(
            "no tool-call context is bound; exact-action approval cannot be verified here")
    try:
        digest = canonicalize_action(tool_name, args)
    except ValueError as exc:
        raise ExactActionApprovalError(f"cannot verify exact-action approval: {exc}") from exc

    now = time.time()
    key = (session_key, tool_call_id)
    with _lock:
        _purge_expired_locked(now)
        receipt = _receipts.get(key)
        matches = (receipt is not None and receipt.tool_name == tool_name
                  and receipt.digest == digest and receipt.expires_at > now)
        if matches:
            del _receipts[key]  # single use: pop under the same lock that validated it
    if not matches:
        raise ExactActionApprovalError(
            f"no valid exact-action approval for '{tool_name}' with these exact arguments "
            "in this session/tool-call — it may be missing, expired, already used, or the "
            "arguments changed after approval")
