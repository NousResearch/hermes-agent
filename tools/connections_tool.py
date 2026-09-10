#!/usr/bin/env python3
"""Manage remote connector accounts served through the tool gateway.

``manage_connections`` is the never-deferred surface for connection
lifecycle:

- ``status`` — which connectors exist for this account and whether each is
  connected (read-only).
- ``connect`` / ``reconnect`` — start (or restart) an authorization flow.
  The gateway returns a connect link, passed through UN-redacted: the model
  shows it to the user, who opens it in a browser. Each connector's
  ``instruction`` text is surfaced once per session, not on every call.
- ``wait`` — block inside the call until the named connectors report
  connected, or the budget runs out. A model has no clock: told to wait it
  says "I'll check back in a minute" and its next action lands immediately,
  so guidance produced a burst of polls rather than a paced one. Waiting
  inside the call cannot be skipped and works the same on every platform.

Scope: gateway connectors ONLY. Local MCP servers stay with ``setup_mcp``,
which still exists and still works. An earlier draft folded ``install`` /
``enable`` / ``authorize`` in here, but the desktop consent card arrives
through a per-tool interception branch keyed on the name ``setup_mcp``
(agent/tool_executor.py, agent/agent_runtime_helpers.py) and
``registry.dispatch`` never forwards a ``callback``. So the fold could only
ever return the "use the terminal" fallback while its schema advertised the
consent flow — a promise with no delivery path.

De-authentication is deliberately NOT exposed to the model: disconnecting
an account is a user decision, made in the portal dashboard.

Availability: gated by the portal sign-in the managed tools already use
(``check_fn``), so signed-out sessions see exactly today's behavior.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_CONNECTOR_ACTIONS = ("status", "connect", "reconnect", "wait")

# (session_id, connector) pairs whose `instruction` text has already been
# shown. Keyed per session, not per process: the gateway multiplexes many
# sessions through one process, and guidance suppressed for session A must
# still reach session B. Module-level dict + lock is the house idiom
# (browser_use `_pending_create_keys` precedent); an unknown session keys
# on "" and degrades to per-process, never crashes.
_seen_instructions: set = set()
_seen_instructions_lock = threading.Lock()

# session_id -> {connector slug: monotonic time the connect call addressed
# it}. Same keying and same idiom as `_seen_instructions` above, for the same
# reason. Honest scope: this records the moment THIS TOOL handed the model a
# link (or confirmed the connector already active) — it cannot prove the model
# relayed the link to the user. Two guards ride on it: a wait for a connector
# this session never addressed at all is refused (nothing to wait for), and a
# wait arriving within seconds of the mint — the connect-and-wait-in-one-batch
# shape, where no message with the links can have reached the user yet — is
# bounced as a never-error nudge instead of a silent multi-minute block.
_rendered_links: Dict[str, Dict[str, float]] = {}
_rendered_links_lock = threading.Lock()

# The just-minted window: a wait that begins this soon after its links were
# minted can only come from the same tool batch (a real turn boundary costs a
# model round trip). Bounced, not served — the user has not seen the links.
_LINKS_JUST_MINTED_SECONDS = 2.0

# How long a `wait` runs by default, and the bounds it is clamped into. The
# floor keeps a wait long enough to be worth the round trip; the ceiling keeps
# one call inside a span a user reads as "a moment" — the model can always ask
# and wait again. Honest scope: the budget bounds the loop's own decisions
# (sleeps, and whether another poll starts); a poll that has already begun
# runs to the transport's own per-request timeout, so a hung gateway can
# overrun the ceiling by one poll's worth.
_WAIT_DEFAULT_SECONDS = 120.0
_WAIT_MIN_SECONDS = 5.0
_WAIT_MAX_SECONDS = 180.0
# The gap between polls, and so the notice delay on a connector coming live.
# Every poll is a live `v1/connectors` read — nothing is cached, because the
# whole point of the loop is to see a change made outside this process.
_POLL_GAP_SECONDS = 5.0
# The budget is counted as it is spent — the waits plus the time each poll
# actually takes — rather than read off a wall clock. A slow gateway therefore
# costs polls instead of overrunning the call, and the loop stays testable
# without a fake clock.
#
# Waits are taken in slices so they stay answerable. Nothing outside a tool can
# end a call that has already started — the executor only checks for an
# interrupt between tools — so a tool that blocks this long watches the flag
# itself, and touches the activity heartbeat so the gateway's inactivity
# timeout does not kill the session underneath it.
_POLL_WAIT_SLICE_SECONDS = 1.0

_WAIT_UNFINISHED_NOTE = (
    "This is NOT an error: the user simply has not finished connecting yet. "
    "ASK THE USER what they want to do — keep waiting (call wait again), "
    "continue without the pending apps, or get fresh connect links (action "
    "'connect'). Do not retry silently and do not treat the pending apps as "
    "broken."
)


def _connectors_available() -> bool:
    try:
        from tools.tool_gateway.config import connectors_available

        return connectors_available()
    except Exception:
        return False


def _default_client():
    from tools.tool_gateway.client import ConnectorClient

    return ConnectorClient()


def _clamp_timeout(raw: Any) -> tuple:
    """Return (seconds, note). *note* is None unless the ask was clamped."""
    try:
        asked = _WAIT_DEFAULT_SECONDS if raw is None else float(raw)
    except (TypeError, ValueError):
        return _WAIT_DEFAULT_SECONDS, None
    if asked > _WAIT_MAX_SECONDS:
        return _WAIT_MAX_SECONDS, (
            f"timeout_seconds was capped at {int(_WAIT_MAX_SECONDS)}s "
            f"(asked for {asked:g}s). Call wait again to keep waiting."
        )
    if asked < _WAIT_MIN_SECONDS:
        return _WAIT_MIN_SECONDS, (
            f"timeout_seconds was raised to the {int(_WAIT_MIN_SECONDS)}s "
            f"minimum (asked for {asked:g}s)."
        )
    return asked, None


def _rendered_for(
    session_id: Optional[str], rendered: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    with _rendered_links_lock:
        return dict(rendered.get(str(session_id or ""), {}))


def _record_rendered(
    session_id: Optional[str],
    connector: str,
    rendered: Dict[str, Dict[str, float]],
    *,
    never_fresh: bool = False,
) -> None:
    # Lowercased on the way in: the wait matcher and the input path both
    # lowercase, and a case drift here would turn into a permanent refusal.
    # `never_fresh` records a connector with no link to show (already active):
    # membership holds, but the just-minted bounce can never fire for it.
    stamp = float("-inf") if never_fresh else time.monotonic()
    with _rendered_links_lock:
        rendered.setdefault(str(session_id or ""), {})[connector.lower()] = stamp


def _wait_between_polls(seconds: float, activity_state: Dict[str, Any]) -> bool:
    """Hold the call open until the next poll; False if the user interrupted."""
    from tools.interrupt import is_interrupted

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:
        touch_activity_if_due = None

    remaining = seconds
    while remaining > 0:
        if is_interrupted():
            return False
        if touch_activity_if_due is not None:
            try:
                touch_activity_if_due(activity_state, "waiting for connections")
            except Exception:
                pass
        this_slice = min(_POLL_WAIT_SLICE_SECONDS, remaining)
        time.sleep(this_slice)
        remaining -= this_slice
    return True


def _wait_for_connections(
    client: Any,
    connectors: List[str],
    *,
    timeout_seconds: float,
    timeout_note: Optional[str],
) -> str:
    """Poll the gateway until the named connectors are live, or time runs out.

    Never reports a wait outcome as an error. A connector the user has not
    finished authorizing is an ordinary, expected state — the model's next move
    is a question to the user, not a repair.
    """
    wanted = set(connectors)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    spent = 0.0
    live_entries: List[Dict[str, Any]] = []
    pending: List[str] = list(connectors)

    def result(status: str, note: str) -> str:
        payload: Dict[str, Any] = {
            "status": status,
            "connectors": live_entries,
            "pending": pending,
            "note": note,
        }
        if timeout_note:
            payload["timeout_note"] = timeout_note
        return json.dumps(payload, ensure_ascii=False)

    consecutive_errors = 0

    while True:
        poll_started = time.monotonic()
        try:
            items = client.list_connectors()
        except Exception:
            # A transient gateway blip costs one poll, never the whole wait.
            # Three in a row means the gateway is genuinely down mid-wait —
            # still not the model's error: report what the last good poll saw
            # and hand the decision back, same as a timeout.
            spent += time.monotonic() - poll_started
            consecutive_errors += 1
            if consecutive_errors >= 3:
                return result(
                    "timeout",
                    "The connector gateway stopped answering while waiting; "
                    "still not confirmed: " + ", ".join(pending) + ". "
                    + _WAIT_UNFINISHED_NOTE,
                )
            if spent >= timeout_seconds:
                return result(
                    "timeout",
                    f"Waited about {int(spent)}s; still not connected: "
                    f"{', '.join(pending)}. " + _WAIT_UNFINISHED_NOTE,
                )
            gap = min(_POLL_GAP_SECONDS, timeout_seconds - spent)
            if not _wait_between_polls(gap, activity_state):
                return result(
                    "interrupted",
                    "The user interrupted the wait; still not connected: "
                    f"{', '.join(pending)}. " + _WAIT_UNFINISHED_NOTE,
                )
            spent += gap
            continue
        consecutive_errors = 0
        spent += time.monotonic() - poll_started

        live_entries = []
        live_slugs = set()
        for item in items or ():
            if not isinstance(item, dict):
                continue
            slug = str(item.get("connector", "")).lower()
            if slug in wanted and item.get("connected"):
                live_entries.append(item)
                live_slugs.add(slug)
        pending = [c for c in connectors if c not in live_slugs]

        if not pending:
            return result(
                "connected",
                "All requested apps are connected. Go ahead and use them.",
            )
        if spent >= timeout_seconds:
            return result(
                "timeout",
                f"Waited about {int(spent)}s; still not connected: "
                f"{', '.join(pending)}. " + _WAIT_UNFINISHED_NOTE,
            )
        # The last gap before the budget line is the REMAINDER, not a full
        # gap: requiring room for a whole gap silently halved short asks (a 6s
        # ask returned after one poll and zero sleep) and undershot every
        # budget by one gap (120s asks exited near 115s).
        gap = min(_POLL_GAP_SECONDS, timeout_seconds - spent)
        if not _wait_between_polls(gap, activity_state):
            # Interrupted mid-wait: answer with what the last poll saw rather
            # than spending a round trip the user has just asked us to stop for.
            return result(
                "interrupted",
                "The user interrupted the wait; still not connected: "
                f"{', '.join(pending)}. " + _WAIT_UNFINISHED_NOTE,
            )
        spent += gap


def manage_connections(
    args: Dict[str, Any],
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    seen_instructions: Optional[set] = None,
    rendered_links: Optional[Dict[str, Dict[str, float]]] = None,
    session_id: Optional[str] = None,
) -> str:
    """Dispatch one ``manage_connections`` action. Returns a JSON string."""
    action = str(args.get("action") or "status").strip().lower()

    if action not in _CONNECTOR_ACTIONS:
        return tool_error(
            f"action must be one of {', '.join(_CONNECTOR_ACTIONS)}. "
            "Local MCP servers are set up with setup_mcp, not here. "
            "Disconnecting an account is done by the user in the Nous Portal "
            "dashboard, not through this tool."
        )

    raw_connectors = args.get("connectors")
    if isinstance(raw_connectors, str):
        raw_connectors = [raw_connectors]
    connectors: List[str] = []
    if isinstance(raw_connectors, list):
        for c in raw_connectors:
            c = str(c or "").strip().lower()
            if c and c not in connectors:
                connectors.append(c)

    try:
        client = (client_factory or _default_client)()
        if action == "status":
            items = client.list_connectors()
            if connectors:
                wanted = set(connectors)
                items = [i for i in items if str(i.get("connector", "")).lower() in wanted]
            return json.dumps(
                {
                    "connectors": items,
                    "hint": (
                        "connected=false means calls to that connector will return "
                        "CONNECTION_REQUIRED. Use action 'connect' to get an "
                        "authorization link for the user."
                    ),
                },
                ensure_ascii=False,
            )

        if not connectors:
            return tool_error(
                f"'{action}' requires 'connectors': the connector slugs to authorize "
                "(e.g. [\"gmail\"]). Use action 'status' to list them."
            )

        rendered = rendered_links if rendered_links is not None else _rendered_links
        if action == "wait":
            shown = _rendered_for(session_id, rendered)
            never_shown = [c for c in connectors if c not in shown]
            if never_shown:
                return tool_error(
                    "wait refused: this session never obtained a connect link "
                    f"for {', '.join(never_shown)}, so there is nothing to "
                    "wait for. Call action 'connect' for those connectors first "
                    "and put the links in front of the user, then wait."
                )
            just_minted = [
                c
                for c in connectors
                if time.monotonic() - shown[c] < _LINKS_JUST_MINTED_SECONDS
            ]
            if just_minted:
                # The connect that minted these links ran moments ago — same
                # tool batch, so no message carrying them has reached the user
                # yet. Bounce (never an error) instead of blocking a spinner.
                return json.dumps(
                    {
                        "status": "pending",
                        "connectors": [],
                        "pending": list(connectors),
                        "note": (
                            "Not waiting yet: the connect links for "
                            f"{', '.join(just_minted)} were minted moments ago, "
                            "in this same turn — the user has not seen them. "
                            "Send your message showing the links FIRST, then "
                            "call wait on your next turn."
                        ),
                    },
                    ensure_ascii=False,
                )
            timeout_seconds, timeout_note = _clamp_timeout(args.get("timeout_seconds"))
            return _wait_for_connections(
                client,
                connectors,
                timeout_seconds=timeout_seconds,
                timeout_note=timeout_note,
            )

        response = client.connections(connectors, reinitiate=(action == "reconnect"))
        seen = seen_instructions if seen_instructions is not None else _seen_instructions
        results = []
        for entry in response.get("results", []):
            connector = str(entry.get("connector") or "")
            out: Dict[str, Any] = {
                "connector": connector,
                "status": entry.get("status"),
            }
            if entry.get("connect_url"):
                out["connect_url"] = entry["connect_url"]
                out["note"] = (
                    "Show this link to the user; they open it in a browser to "
                    "authorize. Then use action 'wait' to hold for the "
                    "connection instead of guessing when they are done."
                )
                _record_rendered(session_id, connector, rendered)
            elif entry.get("status") == "active":
                # Already authorized: the gateway mints no link for a live
                # connection. This connector is still ADDRESSED by this call —
                # record it, or the documented connect-then-wait sequence
                # refuses on the success case and loops the model through
                # fresh mints that can never fill the record.
                out["note"] = "Already connected — no link needed."
                # never_fresh: there is no link the user must see before a
                # wait, so the just-minted bounce must not fire for this one —
                # an immediate wait legitimately returns connected on poll one.
                _record_rendered(session_id, connector, rendered, never_fresh=True)
            instruction = entry.get("instruction")
            if instruction:
                seen_key = (str(session_id or ""), connector)
                with _seen_instructions_lock:
                    if seen_key not in seen:
                        seen.add(seen_key)
                        out["instruction"] = instruction
            results.append(out)
        return json.dumps(
            {"results": results, "summary": response.get("summary", {})},
            ensure_ascii=False,
        )
    except Exception as exc:
        # Registered tools go through the registry's catch-wrap, but keep the
        # message model-actionable rather than a raw traceback.
        logger.debug("manage_connections %s failed: %s", action, exc)
        return tool_error(
            f"The connector gateway request failed: {exc}. "
            "If this persists, the user can manage connections in the Nous Portal."
        )


MANAGE_CONNECTIONS_SCHEMA = {
    "name": "manage_connections",
    "description": (
        "Manage remote connector accounts (Gmail, Linear, Notion, ...) served "
        "through the tool gateway. Actions: "
        "'status' lists connectors and whether each is connected; 'connect' "
        "starts an authorization for the given connectors and returns a link "
        "for the USER to open in a browser (never open it yourself); "
        "'reconnect' restarts a broken authorization; "
        "'wait' blocks until the given connectors report connected. Pass "
        "SEVERAL slugs in one call to get all authorization links at once. "
        "When a connector tool "
        "call returns CONNECTION_REQUIRED, use 'connect' and show the link. "
        "Send the message that shows the user the links FIRST; on your NEXT "
        "turn call 'wait' with those same slugs instead of guessing when the "
        "user is done — it polls for you (a wait in the same turn as the "
        "connect is bounced, because the user cannot have seen the links "
        "yet). 'wait' requires 'connectors', and only accepts connectors this "
        "session already addressed with 'connect' (already-connected apps "
        "count). A 'timeout' or 'interrupted' result is NOT an "
        "error: the user has not finished connecting, so ask them whether to "
        "keep waiting, continue without those apps, or get fresh links. "
        "Local MCP servers are configured separately. "
        "This tool can NOT disconnect, delete, or revoke an account — that is "
        "deliberately user-only. When asked, say so and direct the user to "
        "the Nous Portal (their org's Connectors page) or the desktop app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_CONNECTOR_ACTIONS),
                "description": "Defaults to status.",
            },
            "connectors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Connector slugs. REQUIRED for connect, reconnect and wait "
                    "(e.g. [\"gmail\", \"linear\"]); optional filter for status."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": (
                    "For action 'wait' only: how long to hold the call open. "
                    f"Defaults to {int(_WAIT_DEFAULT_SECONDS)}, clamped to "
                    f"{int(_WAIT_MIN_SECONDS)}-{int(_WAIT_MAX_SECONDS)}. Ask for "
                    "more and the result carries a 'timeout_note' saying the cap "
                    "was applied; call wait again to keep waiting."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="manage_connections",
    toolset="connections",
    schema=MANAGE_CONNECTIONS_SCHEMA,
    # Registry dispatch does not re-run check_fn: enforce the off switch for
    # stale schemas without rebuilding a conversation's cached tool list.
    handler=lambda args, **kw: (
        manage_connections(args, session_id=kw.get("session_id"))
        if _connectors_available()
        else tool_error("Connectors are not available in this session.")
    ),
    check_fn=_connectors_available,
    emoji="🔗",
)
