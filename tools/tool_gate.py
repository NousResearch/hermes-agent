#!/usr/bin/env python3
"""Generic, config-driven, human-in-the-loop approval gate for tool calls.

This module is the **deferred (staged)** half of the tool-approval gate plus
its shared config/replay plumbing. The **inline (blocking)** half — and the
top-level :func:`check_tool_approval` decision ladder — live in
``tools/approval.py`` next to the dangerous-command guard engine they reuse.

Design (see docs/hermes-tool-approval-gate.fork.md):

* A designated tool (listed in ``approvals.tool_gate.require_approval``) must
  get human approval before it executes. **Default OFF** — with no
  ``tool_gate`` config the behaviour is unchanged everywhere.
* **Deferred mode** (cron / background / no live approval channel, or config
  ``force_deferred``): do not block. Serialize the call to the file-backed
  pending store (``pending/actions/<id>.json``, reusing
  ``tools.write_approval``), open a Kanban approval card for a human via the
  in-process ``hermes_cli.kanban_db`` engine, and return a non-error
  ``staged`` result so the agent continues.
* **Execution-on-approval (Path A — Hermes-native):** approving the card via
  :func:`approve_action` mints an *agent-assigned* execution card. The kanban
  dispatcher wakes that agent; its worker calls :func:`replay_pending_action`,
  which re-invokes the tool with a **one-shot per-pending-id token** so the
  gate (step 4 of the ladder) lets it through exactly once. TTL/staleness and
  a ``status`` state machine (``pending`` → ``executing`` → ``done``) guard
  against double execution.
"""

from __future__ import annotations

import contextvars
import fnmatch
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Pending-store subsystem for staged tool calls.
SUBSYSTEM = "actions"

# Body marker that flags a Kanban card as a deferred tool-approval artifact so
# a UI / the worker-side replay hook can recognise it without parsing prose.
APPROVAL_MARKER = "hermes-action-approval"
REPLAY_MARKER = "hermes-action-replay"

_DEFAULT_TTL_HOURS = 72
_SUMMARY_CAP = 600

# Sentinel assignee for approval cards when no human ``board_assignee`` is
# configured. It is deliberately NOT a real Hermes profile, so the kanban
# dispatcher buckets the card ``skipped_nonspawnable`` and never auto-runs it —
# even when ``kanban.default_assignee`` is set (which would otherwise claim and
# spawn an *unassigned* ready card). Approval cards are review-only by design.
REVIEW_ASSIGNEE = "human-review"


# ---------------------------------------------------------------------------
# One-shot replay token (consumed by the gate ladder in approval.py)
# ---------------------------------------------------------------------------
# When an approved action is replayed, the worker sets this context value to
# ``{"pending_id": ..., "tool_name": ..., "token": ...}`` before invoking the
# tool. ``check_tool_approval`` consumes it on first use for the matching tool
# so the replay is allowed through once instead of re-staging. It is NEVER a
# blanket allowlist entry — it authorises exactly one pending id.
_replay_token: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "hermes_tool_replay_token", default=None
)


def set_replay_token(info: Optional[Dict[str, Any]]) -> contextvars.Token:
    """Bind a one-shot replay token to the current context. Returns the token
    handle for :func:`reset_replay_token`."""
    return _replay_token.set(info)


def reset_replay_token(token: contextvars.Token) -> None:
    """Restore the prior replay-token context."""
    try:
        _replay_token.reset(token)
    except Exception:
        pass


def consume_replay_token(tool_name: str) -> Optional[str]:
    """Return and clear the replay token's pending id iff it authorises
    ``tool_name``; otherwise return ``None`` and leave any token in place.

    One-shot: a matching token is cleared so a second call to the same tool in
    the same run re-stages instead of silently replaying twice.
    """
    info = _replay_token.get()
    if not info:
        return None
    if info.get("tool_name") != tool_name:
        return None
    _replay_token.set(None)
    return info.get("pending_id")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_tool_gate_config() -> dict:
    """Return the ``approvals.tool_gate`` config block (``{}`` when unset).

    Reads via ``load_config`` (mtime-cached) + ``cfg_get`` so a config edit
    takes effect mid-session, matching ``_get_approval_config``.
    """
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        block = cfg_get(cfg, "approvals", "tool_gate", default={})
        return block if isinstance(block, dict) else {}
    except Exception as e:  # pragma: no cover - config load failure
        logger.debug("tool_gate config load failed: %s", e)
        return {}


def gate_enabled(config: Optional[dict] = None) -> bool:
    cfg = config if config is not None else get_tool_gate_config()
    return _truthy(cfg.get("enabled", False))


def requires_approval(tool_name: str, config: Optional[dict] = None) -> bool:
    """True when ``tool_name`` matches an entry in ``require_approval``.

    Match by exact name first, then ``fnmatch.fnmatchcase`` glob (so
    ``send_*`` works). Entries are plain strings.
    """
    cfg = config if config is not None else get_tool_gate_config()
    patterns = cfg.get("require_approval") or []
    if not isinstance(patterns, (list, tuple)):
        return False
    for pat in patterns:
        if not isinstance(pat, str):
            continue
        if pat == tool_name or fnmatch.fnmatchcase(tool_name, pat):
            return True
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "enabled"}
    return False


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Display + replay serialization
# ---------------------------------------------------------------------------

def summarize_tool_call(tool_name: str, args: Optional[dict]) -> str:
    """Build a short, **display-only**, length-capped summary of a tool call.

    Argument values may be agent/LLM-generated from untrusted input; this
    string is shown to a human (chat prompt, Kanban card) and is NEVER
    interpreted as a command. Capped at ~600 chars.
    """
    name = tool_name or "tool"
    if not isinstance(args, dict) or not args:
        return name
    parts = []
    for key, val in args.items():
        try:
            sval = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)
        except Exception:
            sval = str(val)
        sval = " ".join(sval.split())  # collapse whitespace/newlines
        if len(sval) > 120:
            sval = sval[:117] + "…"
        parts.append(f"{key}={sval}")
    summary = f"{name}(" + ", ".join(parts) + ")"
    if len(summary) > _SUMMARY_CAP:
        summary = summary[: _SUMMARY_CAP - 1] + "…"
    return summary


def serialize_tool_call(tool_name: str, args: Optional[dict]) -> dict:
    """Return the full replayable args payload for a deferred call.

    JSON-round-trips ``args`` so only serializable kwargs survive (the staged
    record must persist to disk and be replayed in a fresh process). Values
    that don't serialize are coerced via ``default=str``.
    """
    if not isinstance(args, dict):
        return {}
    try:
        return json.loads(json.dumps(args, ensure_ascii=False, default=str))
    except Exception:
        # Fall back to a shallow string coercion so we never lose the call.
        return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in args.items()}


# ---------------------------------------------------------------------------
# Context resolution helpers
# ---------------------------------------------------------------------------

def _current_profile() -> str:
    """Best-effort name of the profile this process is running as."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, default) or default
    except Exception:
        import os
        return os.getenv(name, default) or default


def _resolve_tenant(deferred: dict) -> str:
    if _truthy(deferred.get("tenant_from_session", True)):
        return _session_env("HERMES_TENANT", "")
    return str(deferred.get("fixed_tenant") or "").strip()


# ---------------------------------------------------------------------------
# Deferred staging + Kanban card
# ---------------------------------------------------------------------------

def _post_mattermost_approval(pending_id: str, summary: str) -> None:
    """Post approve/deny buttons to the Mattermost approvals channel for a
    staged action (best-effort). No-op unless ``MATTERMOST_URL`` + ``_TOKEN`` +
    ``MATTERMOST_APPROVALS_CHANNEL_ID`` + ``MATTERMOST_APPROVAL_CALLBACK_URL``
    are all set.

    Runs in the staging process — which for the cron sweep is an *unattended
    worker* with no live Mattermost adapter — so it talks to the Mattermost REST
    API directly. The per-action shared secret is persisted to the pending
    record (``mm_secret``) so the gateway's always-on callback site can validate
    the button click cross-process. Never raises: a Mattermost hiccup must not
    fail staging (the Kanban card + ``hermes action`` CLI still work).
    """
    import os
    # Ensure the PROFILE .env is loaded into os.environ. Not every staging
    # context has it: the TUI worker (tui_gateway.slash_worker) and some
    # entrypoints load the *default* ~/.hermes/.env at import time and only
    # switch to the profile's config.yaml on demand — so the gate fires
    # (config-driven) but os.environ lacks the profile's MATTERMOST_* secrets.
    # HERMES_HOME points at the profile here (the pending store is profile-
    # scoped), so this idempotently fills in the profile .env (override=True).
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
    except Exception:
        pass
    url = os.getenv("MATTERMOST_URL", "").rstrip("/")
    token = os.getenv("MATTERMOST_TOKEN", "")
    channel = os.getenv("MATTERMOST_APPROVALS_CHANNEL_ID", "")
    callback = os.getenv("MATTERMOST_APPROVAL_CALLBACK_URL", "")
    logger.info("MMAPPROVAL[%s]: entry url=%s token=%s channel=%s callback=%s",
                pending_id, bool(url), bool(token), bool(channel), bool(callback))
    if not (url and token and channel and callback):
        logger.info("MMAPPROVAL[%s]: skipped (missing config)", pending_id)
        return
    try:
        import json as _json
        import urllib.request
        from tools import write_approval as wa
        from plugins.platforms.mattermost import approval as mm_approval

        secret = uuid.uuid4().hex
        post_ref = uuid.uuid4().hex
        wa.update_pending(SUBSYSTEM, pending_id,
                          {"mm_secret": secret, "mm_post_ref": post_ref})
        text = (f"🛂 **Outbound action awaiting approval**\n"
                f"```\n{summary[:1500]}\n```\n(pending `{pending_id}`)")
        attachment = mm_approval.build_approval_attachment(
            text=text, callback_url=callback, kind="card",
            token=secret, post_ref=post_ref, pending_id=pending_id,
            include_scopes=False)
        payload = _json.dumps({
            "channel_id": channel,
            "message": "",
            "props": {"attachments": [attachment]},
        }).encode()
        req = urllib.request.Request(
            f"{url}/api/v4/posts", data=payload, method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
        wa.update_pending(SUBSYSTEM, pending_id,
                          {"mm_post_id": data.get("id"), "mm_channel_id": channel})
        logger.info("MMAPPROVAL[%s]: posted ok, post_id=%s", pending_id, data.get("id"))
    except Exception as e:  # pragma: no cover - network/runtime best-effort
        logger.warning("MMAPPROVAL[%s]: post FAILED: %s", pending_id, e)


def stage_deferred(tool_name: str, args: Optional[dict], *,
                   summary: str, config: dict) -> Dict[str, Any]:
    """Stage a tool call for out-of-band approval and open a Kanban card.

    Returns ``{"approved": False, "status": "staged", "message": str,
    "pending_id": str, "card_id": str|None}``. Never blocks; never raises for
    expected failures (a staging miss returns a ``staged`` result anyway so
    the agent does not loop — the action is simply not executed).
    """
    from tools import write_approval as wa

    deferred = config.get("deferred") or {}
    if not isinstance(deferred, dict):
        deferred = {}
    ttl_hours = _int(deferred.get("pending_ttl_hours"), _DEFAULT_TTL_HOURS)
    now = time.time()
    expires_at = now + max(ttl_hours, 0) * 3600
    token = uuid.uuid4().hex
    tenant = _resolve_tenant(deferred)
    profile = _current_profile()

    # Resolve the session key without importing approval at module load
    # (approval.py imports this module lazily; keep the dependency one-way).
    try:
        from tools.approval import get_current_session_key
        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""

    payload = {
        "tool_name": tool_name,
        "args": serialize_tool_call(tool_name, args),
        "tenant": tenant,
        "session_key": session_key,
        "expires_at": expires_at,
    }
    record = wa.stage_write(SUBSYSTEM, payload, summary=summary,
                            origin=wa.current_origin())
    pending_id = record.get("id", "")

    # Annotate top-level orchestration fields (mutable across the lifecycle).
    wa.update_pending(SUBSYSTEM, pending_id, {
        "token": token,
        "expires_at": expires_at,
        "tenant": tenant,
        "profile": profile,
        "session_key": session_key,
        "status": "pending",
        "tool_name": tool_name,
    })

    card_id = _open_approval_card(
        pending_id=pending_id,
        tool_name=tool_name,
        summary=summary,
        expires_at=expires_at,
        # Empty board_assignee → a non-profile sentinel (NOT None) so a
        # configured kanban.default_assignee can't auto-spawn this card.
        assignee=(str(deferred.get("board_assignee") or "").strip() or REVIEW_ASSIGNEE),
        tenant=tenant or None,
        created_by=profile or None,
    )
    if card_id:
        wa.update_pending(SUBSYSTEM, pending_id, {"card_id": card_id})

    # Surface the action in the Mattermost approvals channel with approve/deny
    # buttons (best-effort; no-op unless configured). Runs here in the staging
    # process — possibly an unattended worker — so it posts via REST directly.
    _post_mattermost_approval(pending_id, summary)

    card_note = f" as card {card_id}" if card_id else ""
    return {
        "approved": False,
        "status": "staged",
        "message": (
            f"Queued for human approval{card_note}; continuing without running "
            f"'{tool_name}'. This is expected — do NOT retry or re-issue the "
            f"call. The action will run automatically once approved "
            f"(pending id {pending_id})."
        ),
        "pending_id": pending_id,
        "card_id": card_id,
    }


def _approval_card_body(pending_id: str, tool_name: str, summary: str,
                        expires_at: float) -> str:
    expiry = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expires_at))
    return (
        f"{APPROVAL_MARKER}: {pending_id}\n"
        f"tool: {tool_name}\n"
        f"expires: {expiry}\n\n"
        f"A tool call is awaiting human approval before it runs:\n\n"
        f"    {summary}\n\n"
        f"Approve it with `hermes action approve {pending_id}` (or the "
        f"approve button if posted to chat). Approval spawns a one-shot "
        f"execution worker; rejecting / letting it expire drops the action."
    )


def _open_approval_card(*, pending_id: str, tool_name: str, summary: str,
                        expires_at: float, assignee: Optional[str],
                        tenant: Optional[str], created_by: Optional[str]) -> Optional[str]:
    """Create the human-facing approval card via the in-process kanban engine.

    Assignee is a free-text human or the :data:`REVIEW_ASSIGNEE` sentinel —
    either way it is not a Hermes profile, so the dispatcher buckets it
    ``skipped_nonspawnable`` (review-only) and never auto-spawns it.
    ``idempotency_key`` guards against a double-stage racing two cards for the
    same pending id.
    """
    try:
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        title = f"Approve: {summary}"
        if len(title) > 200:
            title = title[:199] + "…"
        return kb.create_task(
            conn,
            title=title,
            body=_approval_card_body(pending_id, tool_name, summary, expires_at),
            assignee=assignee,
            created_by=created_by,
            tenant=tenant,
            idempotency_key=f"approval:{pending_id}",
        )
    except Exception as e:
        logger.warning("Failed to open approval card for %s: %s", pending_id, e)
        return None


# ---------------------------------------------------------------------------
# Execution-on-approval (Path A): mint an agent-assigned execution card
# ---------------------------------------------------------------------------

def approve_action(pending_id: str) -> Dict[str, Any]:
    """Approve a staged action: spawn a one-shot agent execution card.

    The execution card is assigned to the **profile that staged the action**
    (a real Hermes profile), so the kanban dispatcher wakes that agent; its
    worker runs :func:`replay_pending_action`. ``parents`` links it under the
    approval card; ``idempotency_key=exec:<id>`` is the double-spawn guard.

    Returns ``{"ok": bool, "message": str, "exec_card_id"?: str}``. Refuses
    expired / already-resolved actions.
    """
    from tools import write_approval as wa

    rec = wa.get_pending(SUBSYSTEM, pending_id)
    if rec is None:
        return {"ok": False, "message": f"No pending action {pending_id}."}

    status = rec.get("status", "pending")
    if status in {"approved", "executing", "done"}:
        return {"ok": False,
                "message": f"Action {pending_id} already {status}; not re-running."}

    expires_at = rec.get("expires_at")
    if isinstance(expires_at, (int, float)) and time.time() > expires_at:
        wa.update_pending(SUBSYSTEM, pending_id, {"status": "expired"})
        return {"ok": False,
                "message": f"Action {pending_id} expired — re-stage to run it."}

    payload = rec.get("payload") or {}
    tool_name = rec.get("tool_name") or payload.get("tool_name") or "tool"
    profile = rec.get("profile") or _current_profile()
    tenant = rec.get("tenant") or payload.get("tenant") or None
    summary = rec.get("summary") or summarize_tool_call(tool_name, payload.get("args"))
    card_id = rec.get("card_id")

    try:
        from hermes_cli import kanban_db as kb
        conn = kb.connect()
        exec_body = (
            f"{REPLAY_MARKER}: {pending_id}\n"
            f"tool: {tool_name}\n"
            f"approval_card: {card_id or '-'}\n\n"
            f"Approved action — replay the staged tool call:\n\n    {summary}\n"
        )
        # NOTE: do NOT link the exec card under the approval card via ``parents``.
        # A parent dependency keeps a child in ``todo`` until every parent is
        # ``done`` (see kb.create_task), but the approval card is human-review-only
        # and is archived (below), never ``done`` — so a parent link would strand
        # the exec card in ``todo`` and the dispatcher would never claim it.
        # Approval *already happened* (it's what created this card), so execution
        # must not be re-gated by the approval card's lifecycle. The link is kept
        # for traceability via the ``approval_card:`` body line + idempotency_key.
        exec_card_id = kb.create_task(
            conn,
            title=f"Run approved: {summary}"[:200],
            body=exec_body,
            assignee=profile,
            created_by=profile,
            tenant=tenant,
            idempotency_key=f"exec:{pending_id}",
        )
        # Archive the human approval card: it has served its purpose, so it
        # leaves the active board (mirrors reject, which also archives).
        if card_id:
            try:
                kb.archive_task(conn, card_id)
            except Exception as arch_err:
                logger.warning("approve_action: could not archive approval card %s: %s",
                               card_id, arch_err)
    except Exception as e:
        logger.error("Failed to create execution card for %s: %s", pending_id, e)
        return {"ok": False, "message": f"Could not spawn execution card: {e}"}

    wa.update_pending(SUBSYSTEM, pending_id, {
        "status": "approved",
        "exec_card_id": exec_card_id,
    })
    return {
        "ok": True,
        "message": f"Approved {pending_id}; execution queued as card {exec_card_id}.",
        "exec_card_id": exec_card_id,
    }


# ---------------------------------------------------------------------------
# Worker-side deterministic replay
# ---------------------------------------------------------------------------

def _ensure_tool_registered(tool_name: str) -> None:
    """Best-effort: make sure ``tool_name`` is in the tool registry.

    Built-in tools self-register at import; MCP tools only appear after
    ``discover_mcp_tools()`` connects to their servers. When a gated tool is
    not yet registered (the worker's background MCP discovery hasn't completed
    at replay time), run discovery synchronously once. Idempotent and never
    raises — a still-missing tool just produces the normal "unknown tool"
    error from the dispatcher.
    """
    try:
        from tools.registry import registry
        if registry.get_entry(tool_name) is not None:
            return
    except Exception:
        return
    try:
        from tools.mcp_tool import discover_mcp_tools
        discover_mcp_tools()
    except Exception as e:
        logger.debug("MCP discovery during replay failed: %s", e)


def parse_replay_marker(body: str) -> Optional[str]:
    """Return the pending id from an execution card body, or ``None``."""
    if not body:
        return None
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith(REPLAY_MARKER + ":"):
            return line.split(":", 1)[1].strip() or None
    return None


def replay_pending_action(pending_id: str) -> Dict[str, Any]:
    """Execute an approved, staged tool call exactly once, deterministically.

    Invoked by the kanban worker (no LLM turn). Enforces TTL/staleness and an
    atomic ``pending`` → ``executing`` → ``done`` transition so a button click
    racing the dispatcher cannot double-execute. Sets a one-shot replay token
    so the re-invoked tool passes the gate instead of re-staging, then routes
    through the normal tool dispatcher (``handle_function_call``).

    Returns ``{"ok": bool, "message": str, "result"?: str}``.
    """
    from tools import write_approval as wa

    rec = wa.get_pending(SUBSYSTEM, pending_id)
    if rec is None:
        return {"ok": False, "message": f"No pending action {pending_id} (already executed?)."}

    status = rec.get("status", "pending")
    if status == "done":
        return {"ok": True, "message": f"Action {pending_id} already done; nothing to do."}
    if status == "executing":
        return {"ok": False, "message": f"Action {pending_id} is already executing; refusing to double-run."}

    expires_at = rec.get("expires_at")
    if isinstance(expires_at, (int, float)) and time.time() > expires_at:
        wa.update_pending(SUBSYSTEM, pending_id, {"status": "expired"})
        return {"ok": False, "message": f"Action {pending_id} expired — re-stage to run it."}

    # Atomic-ish claim: flip to executing immediately so a concurrent replay
    # bails on the status check above. (File store is single-writer per id;
    # the kanban exec card's idempotency_key is the cross-process guard.)
    claimed = wa.update_pending(SUBSYSTEM, pending_id, {"status": "executing"})
    if claimed is None:
        return {"ok": False, "message": f"Could not claim pending action {pending_id}."}

    payload = rec.get("payload") or {}
    tool_name = rec.get("tool_name") or payload.get("tool_name")
    args = payload.get("args") or {}
    token = rec.get("token") or ""

    if not tool_name:
        wa.update_pending(SUBSYSTEM, pending_id, {"status": "error"})
        return {"ok": False, "message": f"Pending action {pending_id} has no tool_name."}

    # Ensure the tool is actually registered before dispatching. The kanban
    # worker (``hermes chat -q "work kanban task …"``) discovers MCP tools in a
    # background thread whose bounded join happens at the agent's first tool
    # snapshot — but this deterministic replay short-circuits *before* that
    # point, so an MCP-backed gated tool may not be in the registry yet. When
    # it's missing, run MCP discovery synchronously (idempotent) once so the
    # replay doesn't spuriously fail closed on a connection race.
    _ensure_tool_registered(tool_name)

    tok = set_replay_token({"pending_id": pending_id, "tool_name": tool_name, "token": token})
    try:
        from model_tools import handle_function_call
        result = handle_function_call(tool_name, args)
    except Exception as e:
        logger.error("Replay of %s (%s) failed: %s", pending_id, tool_name, e, exc_info=True)
        wa.update_pending(SUBSYSTEM, pending_id, {"status": "error", "error": str(e)})
        return {"ok": False, "message": f"Execution of '{tool_name}' failed: {e}"}
    finally:
        reset_replay_token(tok)

    # Success: mark done then discard the pending record so it cannot replay.
    wa.update_pending(SUBSYSTEM, pending_id, {"status": "done"})
    wa.discard_pending(SUBSYSTEM, pending_id)
    return {
        "ok": True,
        "message": f"Executed '{tool_name}' for approved action {pending_id}.",
        "result": result if isinstance(result, str) else str(result),
    }
