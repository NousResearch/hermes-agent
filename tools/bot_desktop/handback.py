"""Opt-in, process-local continuation for the conversation that last drove the screen."""

import contextvars
import logging
import sys
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from gateway.session_context import get_session_env
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from tools.bot_desktop import lease

logger = logging.getLogger(__name__)
MAX_AGE_SECONDS = 60 * 60
HANDBACK_TEXT = (
    "[Hermes gateway notice, not typed by the user: the person handed the Bot Screen back to you. "
    "Re-capture the screen before acting. If your earlier request in this conversation is not "
    "finished, continue it without asking them to confirm and do not repeat steps already done; "
    "otherwise say briefly that you have control again.]"
)


@dataclass
class _Driver:
    runner: object
    route: dict
    epoch: int
    timestamp: float
    generation: int


_lock = Lock()
_drivers: dict[str, _Driver] = {}
_generations: dict[str, int] = {}
_subscribed = False


def forget(profile_key: str | None = None) -> int:
    """Disarm this profile's driver and any notice that has not passed its pending check."""
    key = profile_key or hermes_home_key()
    with _lock:
        _generations[key] = _generations.get(key, 0) + 1
        _drivers.pop(key, None)
        return _generations[key]


def note_driver() -> None:
    """Every admitted action supersedes; only eligible live conversations arm continuation."""
    key = hermes_home_key()
    generation = forget(key)
    from hermes_cli.config import load_config_readonly

    if not (load_config_readonly().get("bot_desktop") or {}).get("resume_on_handback", False):
        return
    if not get_session_env("HERMES_SESSION_KEY") or get_session_env("HERMES_CRON_SESSION"):
        return
    runner_ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
    runner = runner_ref() if callable(runner_ref) else None
    loop = getattr(runner, "_gateway_loop", None)
    if not getattr(runner, "_running", False) or loop is None or not loop.is_running():
        return
    route = {field: get_session_env("HERMES_SESSION_" + field.upper()) for field in (
        "platform", "chat_id", "chat_type", "thread_id", "user_id", "user_name", "scope_id")}
    route.update(session_key=get_session_env("HERMES_SESSION_KEY"),
                 parent_session_id=get_session_env("HERMES_SESSION_ID"))
    if route["platform"] != "local" and (not route["platform"] or route["chat_type"] != "dm"):
        return
    if route["platform"] == "slack" and not (route["chat_id"] or "").startswith("D"):
        return
    global _subscribed
    with _lock:
        if _generations[key] != generation:
            return
        if not _subscribed:
            lease.on_change(_on_lease_change)
            _subscribed = True
        _drivers[key] = _Driver(runner, route, lease.get(key).epoch, monotonic(), generation)


def _on_lease_change(profile_key: str, current: lease.Lease) -> None:
    try:
        with _lock:
            record = _drivers.get(profile_key)
            if record is None or current.holder != lease.AGENT or current.epoch <= record.epoch:
                return
            _drivers.pop(profile_key)
            if monotonic() - record.timestamp >= MAX_AGE_SECONDS:
                return
        from hermes_cli.config import load_config_readonly

        token = set_hermes_home_override(profile_key)
        try:
            if not (load_config_readonly().get("bot_desktop") or {}).get("resume_on_handback", False):
                return
        finally:
            reset_hermes_home_override(token)

        def still_pending():
            with _lock:
                return (_generations[profile_key] == record.generation
                        and lease.get(profile_key).holder == lease.AGENT)

        from agent.async_utils import safe_schedule_threadsafe
        from gateway.wake import wake_route

        contextvars.Context().run(
            safe_schedule_threadsafe,
            wake_route(record.runner, record.route, HANDBACK_TEXT, still_pending, producer="bot-screen-handback"),
            record.runner._gateway_loop, logger=logger, log_message="Bot Screen handback wake scheduling failed",
        )
    except Exception:  # health: allow BLE001 -- a failed notice must never break the lease writer
        logger.debug("Bot Screen handback wake failed", exc_info=True)
