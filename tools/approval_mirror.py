"""Optional parallel presentation of a queued approval through a plugin transport."""

import asyncio
import inspect
import logging
import threading

from agent.memory_provider import ctx_bound, spawn_context_thread
from hermes_cli.approval_transport import _MAX_ACTIVE_TRANSPORT_WORKERS
from tools import approval_context as _ctx

logger = logging.getLogger("tools.approval")
_mirror_worker_slots = threading.BoundedSemaphore(_MAX_ACTIVE_TRANSPORT_WORKERS)


async def _await_value(value):
    return await value


def _present(present, request):
    """Call sync plugins without a loop; await async bodies in the same bound scope."""
    value = present(request)
    return asyncio.run(_await_value(value)) if inspect.isawaitable(value) else value


def _run_mirror(session_key, entry, registered, request):
    from hermes_cli.approval_transport import invoke_approval_transport
    from tools.approval import resolve_gateway_approval

    try:
        # invoke_approval_transport starts another worker: bind that hop too, including the
        # coroutine body (binding only the call that creates a coroutine would lose its scope).
        present = ctx_bound(lambda req: _present(registered.present, req))
        result = invoke_approval_transport(
            present, request, timeout_seconds=request.timeout_seconds,
            is_interrupted=entry.event.is_set,
            worker_slots=_mirror_worker_slots,
        )
        if result.failure is not None:
            logger.debug("Mirrored approval transport did not answer (%s)", result.failure)
        elif result.choice in request.allowed_choices:
            resolve_gateway_approval(session_key, result.choice, request_id=request.request_id)
    except Exception:  # health: allow BLE001 -- plugin exceptions may contain secrets; never log their tracebacks
        # Plugin exceptions may contain secrets; failures never answer for the human.
        logger.debug("Mirrored approval transport failed")


# Sessions whose built-in prompt is a cancellable card answered by request id. Classic CLI
# ("cli") is left out: its prompt blocks on terminal input that a phone answer cannot withdraw.
_MIRROR_SOURCES = frozenset({"tui", "gui", "acp"})


def _mirror_session() -> bool:
    try:
        from gateway.session_policy import active_policy
    except ImportError:  # install without the gateway package: no local session can be running
        return False
    policy = active_policy()
    return policy is not None and policy.source in _MIRROR_SOURCES


def start_mirror(session_key, entry):
    """Start only for a published leader; all answers use the queue's atomic resolver."""
    if _ctx._get_approval_transport_mode() != "mirror" or entry.event.is_set():
        return
    if not _mirror_session() or not entry.data.get("transport_eligible", False):
        return
    try:
        from agent.redact import redact_sensitive_text
        from hermes_cli.approval_transport import ApprovalRequest
        from tools.approval_prompt import get_plugin_manager

        name, _ = _ctx._get_approval_transport_config()
        if name == "builtin":
            return
        registered = get_plugin_manager().get_approval_transport(name)
        if registered is None:
            return
        data = entry.data
        key = data.get("pattern_key", "")
        request = ApprovalRequest.create(
            command=redact_sensitive_text(data.get("command", ""), force=True),
            description=redact_sensitive_text(data.get("description", ""), force=True),
            pattern_key=key, pattern_keys=tuple(data.get("pattern_keys", [key])),
            session_key=session_key, surface=entry.surface,
            allow_session=data.get("allow_session", False),
            allow_permanent=data.get("allow_permanent", False),
            # The timeout the built-in wait captured, so a config edit mid-request cannot split the two.
            timeout_seconds=data["timeout_s"] if "timeout_s" in data else _ctx._get_approval_timeout(),
            request_id=data["request_id"],
        )
        spawn_context_thread(_run_mirror, name=f"approval-mirror-{request.request_id[:8]}",
                             args=(session_key, entry, registered, request)).start()
    except Exception:  # health: allow BLE001 -- discovery/redaction failures must leave the built-in prompt live without logging secrets
        logger.debug("Could not start mirrored approval transport")
