"""Human prompt surfaces for :mod:`tools.approval`.

The interactive CLI prompt (callback panel or ``input()`` fallback) and the
operator-selected plugin approval transport. Detection, allowed scopes,
persistence, timeout policy and the final authorization stay host-owned in
``tools.approval``; this module only asks and reports the answer.
"""

import logging
import os
import sys
import threading
from tools import approval_context as _ctx, approval_gateway_wait as _gw
from tools.approval_human_wait import activity_heartbeat, human_wait_window
from tools.interrupt import is_interrupted

logger = logging.getLogger("tools.approval")


def prompt_dangerous_approval(command: str, description: str, timeout_seconds: int | None = None,
                              allow_permanent: bool = True, approval_callback=None,
                              *, allow_session: bool = True, smart_denied: bool = False) -> str:
    """Prompt the user to approve a dangerous command (CLI only).

    allow_permanent=False hides [a]lways (tirith warnings present: broad permanent
    allowlisting is wrong for content-level findings). allow_session=False hides
    [s]ession too — the caller grants one operation and re-asks next time (the
    protected agent-instruction gate in ``tools/file_tools.py``); offering a scope
    the caller discards makes every later write re-prompt and reads as broken.
    smart_denied: owner override of a Smart DENY, offer only once/deny.
    approval_callback: CLI prompt_toolkit callback ``(command, description, *,
    allow_permanent=True, allow_session=True, smart_denied=False) -> str``; legacy
    signatures keep working while both keywords hold their defaults.

    Returns 'once', 'session', 'always', 'deny', or 'timeout'. 'timeout' means no
    user response — still blocked (fail-closed), but callers report "no response"
    rather than an explicit denial.

    See #81887.
    """
    if timeout_seconds is None:
        timeout_seconds = _ctx._get_approval_timeout()
    # Everything below is a human prompt (callback panel or input() fallback, both bounded by the approval deadline):
    # record it as human-wait time so the concurrent batch deadline excludes it.
    # See #79719.
    with human_wait_window():
        return _ask_human(command, description, timeout_seconds, allow_permanent,
                          approval_callback, allow_session, smart_denied)


_CLI_CHOICE_ALIASES = {
    "o": "once", "once": "once",
    "s": "session", "session": "session",
    "a": "always", "always": "always",
}

_CLI_CHOICE_I18N = {
    "once": "approval.allowed_once",
    "session": "approval.allowed_session",
    "always": "approval.allowed_always",
    "deny": "approval.denied",
}


def _read_choice(prompt: str, timeout_seconds: int) -> str | None:
    """Read one answer on a daemon thread; None when the user never answered."""
    result = {"choice": ""}

    def get_input():
        try:
            result["choice"] = input(prompt).strip().lower()
        except (EOFError, OSError):
            result["choice"] = ""

    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    return None if thread.is_alive() else result["choice"]


def _ask_human(command: str, description: str, timeout_seconds: int, allow_permanent: bool,
               approval_callback, allow_session: bool, smart_denied: bool) -> str:
    # Redact before any user-visible rendering; the original `command` still executes after approval. Same redactor as
    # memory/log sanitization so tokens mask consistently across surfaces.
    from agent.redact import redact_sensitive_text
    display_command = redact_sensitive_text(command)
    display_description = redact_sensitive_text(description)
    # Smart DENY and a session-less gate both reduce the menu to once/deny.
    once_only = smart_denied or not allow_session

    if approval_callback is not None:
        try:
            # Non-default scopes only: legacy callbacks lack the newer keywords.
            callback_kwargs = {"allow_permanent": allow_permanent,
                               **({"allow_session": False} if not allow_session else {}),
                               **({"smart_denied": True} if smart_denied else {})}
            return approval_callback(display_command, display_description, **callback_kwargs)
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # Fail-closed guard: when prompt_toolkit owns the terminal and no callback is registered on this thread, the
    # input() fallback would spawn a daemon thread whose read never sees Enter (keystrokes go to prompt_toolkit) — an
    # invisible deadlock. Deny loudly instead; threads needing interactive approval must install a callback via
    # tools.terminal_tool.set_approval_callback() first.
    try:
        # Deny fast and log loudly instead so the caller can surface a real error to the agent. Any thread
        # that needs interactive approval must install a callback via
        # tools.terminal_tool.set_approval_callback() before reaching this point (see delegate_tool.py,
        # run_agent.py _execute_tool_calls_concurrent / _spawn_background_review for the established
        # pattern). See #15216.
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning("Dangerous-command approval requested on a thread with no "
                           "approval callback while prompt_toolkit is active; denying "
                           "to avoid stdin deadlock. command=%r description=%r", command, description)
            return "deny"
    except Exception:
        pass  # prompt_toolkit absent or detection failed: legacy input() path is safe

    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    try:
        from agent.i18n import t
        # (prompt key, menu key) by menu shape: once/deny, full, or no [a]lways.
        shape = "smart_deny" if once_only else "long" if allow_permanent else "short"
        prompt_key, menu_key = f"approval.prompt_{shape}", f"approval.choose_{shape}"
        print(f"\n  {t('approval.dangerous_header', description=display_description)}"
              f"\n      {display_command}\n\n{t(menu_key)}\n")
        sys.stdout.flush()
        choice = _read_choice(t(prompt_key), timeout_seconds)
        if choice is None:
            print("\n" + t("approval.timeout"))
            return "timeout"  # distinct from deny: the user never answered
        if once_only:
            decision = {**dict.fromkeys(t("approval.smart_deny_once_inputs").split(","), "once"),
                        **dict.fromkeys(t("approval.smart_deny_deny_inputs").split(","), "deny"),
                        }.get(choice, "deny")
        else:
            decision = _CLI_CHOICE_ALIASES.get(choice, "deny")
            if decision == "always" and not allow_permanent:
                decision = "session"
        print(t(_CLI_CHOICE_I18N[decision]))
        return decision
    except (EOFError, KeyboardInterrupt):
        print("\n" + t("approval.cancelled"))
        return "deny"
    finally:
        os.environ.pop("HERMES_SPINNER_PAUSE", None)
        print()
        sys.stdout.flush()


def get_plugin_manager():
    """Lazy plugin-manager seam used by tests and early tool-only imports."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager as _get_manager
    # Approval can be imported before model_tools (which triggers discovery); make an explicitly selected transport
    # available on the first approval instead of treating the undiscovered registry as unavailable.
    discover_plugins()
    return _get_manager()


def _attempt(name: str, choice, failure, fallback) -> dict:
    """Result shape of :func:`_present_with_selected_transport` once a transport is selected."""
    return {"selected": True, "choice": choice, "failure": failure, "fallback": fallback, "name": name}


def _present_with_selected_transport(*, command: str, description: str, pattern_key: str,
                                     pattern_keys: list[str], session_key: str, surface: str,
                                     allow_session: bool, allow_permanent: bool) -> dict:
    """Present through an explicitly selected plugin transport, if any. A selected
    transport replaces every built-in prompt surface; detection, allowed scopes,
    persistence, timeout, and final authorization stay host-owned. A failed
    transport reaches a built-in surface only under the explicit
    ``transport_fallback: builtin`` opt-in."""
    name, fallback = _ctx._get_approval_transport_config()
    if name == "builtin":
        return {"selected": False}

    try:
        registered = get_plugin_manager().get_approval_transport(name)
    except Exception:
        # Plugin/discovery exception text may contain plugin-owned secrets.
        logger.warning("Could not resolve selected approval transport %r", name)
        registered = None
    if registered is None:
        logger.warning("Selected approval transport %r is unavailable", name)
        return _attempt(name, "deny", "unavailable", fallback)

    try:
        from agent.redact import redact_sensitive_text
        from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport

        timeout_seconds = _ctx._get_approval_timeout()
        request = ApprovalRequest.create(
            command=redact_sensitive_text(command, force=True),
            description=redact_sensitive_text(description, force=True), pattern_key=pattern_key,
            pattern_keys=tuple(pattern_keys), session_key=session_key, surface=surface, allow_session=allow_session,
            allow_permanent=allow_permanent, timeout_seconds=timeout_seconds,
        )
    except Exception:
        # Never fall back to raw text if redaction or request construction fails:
        # fail closed without calling the plugin or leaking the unredacted payload.
        logger.warning("Could not build redacted plugin approval request")
        return _attempt(name, "deny", "error", None)
    hook_kwargs = dict(
        command=request.command, description=request.description, pattern_key=pattern_key,
        pattern_keys=list(pattern_keys), session_key=session_key, surface=f"transport:{name}",
        request_id=request.request_id, request_digest=request.digest,
    )
    _ctx._fire_approval_hook("pre_approval_request", **hook_kwargs)
    with human_wait_window(session_key):
        result = invoke_approval_transport(
            registered.present, request, timeout_seconds=timeout_seconds,
            on_poll=activity_heartbeat("waiting for plugin approval transport"),
            is_interrupted=is_interrupted,
        )
    hook_choice = result.choice if result.failure is None else f"transport_{result.failure}"
    _ctx._fire_approval_hook("post_approval_response", **hook_kwargs, choice=hook_choice)
    return _attempt(name, result.choice, result.failure, fallback)


def _transport_choice(attempt: dict, *, pattern_key: str, description: str):
    """Interpret a ``_present_with_selected_transport`` attempt into
    ``(choice, denied_result)``: both None when the built-in surfaces should run
    (no transport selected, or a failure with the explicit builtin fallback); a
    denied result for any other failure; else the user's choice."""
    if not attempt.get("selected"):
        return None, None
    failure = attempt.get("failure")
    if not failure:
        return attempt.get("choice"), None
    if attempt.get("fallback") == "builtin":
        logger.warning("Approval transport %r failed (%s); using explicit builtin fallback",
                       attempt.get("name"), failure)
        return None, None
    from tools import approval as _a
    breaker_addendum = _a._denial_breaker_addendum(_ctx.get_current_session_key())
    return None, _a._denied(
        f"BLOCKED: Selected approval transport failed ({failure}); the user "
        "has NOT consented to this action. Do NOT retry this command or "
        f"attempt the same outcome through another route.{breaker_addendum}",
        pattern_key=pattern_key, description=description, outcome=f"transport_{failure}",
    )


def _consent(choice, unresolved: str) -> str:
    """Map an approval choice to an elicitation verdict; *unresolved* is the no-answer outcome."""
    if choice in ("once", "session", "always"):
        return "accept"
    return unresolved if choice == "timeout" else "decline"


def request_elicitation_consent(message: str, description: str, *,
                                timeout_seconds: int | None = None,
                                surface: str = "mcp-elicitation") -> str:
    """Route an MCP elicitation request to the surface owning the active session:
    gateway sessions through ``_await_gateway_decision``, CLI/TUI through
    ``prompt_dangerous_approval``. Always fails closed: a missing notify_cb in a
    gateway session, timeouts, and exceptions map to ``"decline"`` so a server
    treats them as "user did not approve" rather than retrying or hanging.
    Returns ``"accept" | "decline" | "cancel"``."""
    from tools import approval as _a
    try:
        session_key = _ctx.get_current_session_key()
    except Exception as exc:  # pragma: no cover -- defensive
        logger.warning("Elicitation consent: session lookup failed: %s", exc)
        return "decline"

    if _ctx._is_gateway_approval_context():
        notify_cb = _a._gateway_notify_cb(session_key)
        if notify_cb is None:
            logger.warning("Elicitation requested in gateway session %s but no "
                           "notify_cb is registered — failing closed", session_key)
            return "decline"
        try:
            decision = _gw._await_gateway_decision(
                session_key, notify_cb, {"command": message, "description": description,
                                         "pattern_key": "mcp_elicitation",
                                         "pattern_keys": ["mcp_elicitation"]}, surface=surface)
        except Exception as exc:
            logger.error("Elicitation gateway dispatch failed: %s", exc, exc_info=True)
            return "decline"
        if decision.get("notify_failed"):
            return "decline"
        if not decision.get("resolved"):
            return "cancel"
        return _consent(decision.get("choice"), "decline")

    # allow_permanent=False: elicitation is a per-call confirmation — no pattern to remember.
    try:
        choice = prompt_dangerous_approval(message, description, timeout_seconds=timeout_seconds,
                                           allow_permanent=False)
    except Exception as exc:
        logger.error("Elicitation CLI prompt failed: %s", exc, exc_info=True)
        return "decline"
    return _consent(choice, "cancel")  # timeout mirrors the gateway's unresolved outcome
