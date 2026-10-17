"""Bounded, fail-open account limits shared by desktop slash and TUI usage."""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import threading
from typing import Any

from tools.daemon_pool import DaemonThreadPoolExecutor

logger = logging.getLogger(__name__)

# Quota endpoints are remote and cosmetic. Timed-out calls must not occupy the
# RPC reader or the general long-handler pool indefinitely.
_account_usage_pool = DaemonThreadPoolExecutor(
    max_workers=2, thread_name_prefix="hermes-account-usage"
)
_account_usage_slots = threading.BoundedSemaphore(2)
atexit.register(lambda: _account_usage_pool.shutdown(wait=False, cancel_futures=True))
_ACCOUNT_USAGE_TIMEOUT_SECONDS = 10.0


def _usage_provider_identity(session: dict) -> tuple[str, str, Any]:
    """Resolve quota inputs without persisting or returning credentials."""
    from tui_gateway.server import _load_cfg
    from tui_gateway.compute_host_bridge import _metadata_mirror

    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    live_provider = str(getattr(agent, "provider", "") or "").strip()
    provider = live_provider or str(mirror.get("provider") or "").strip()
    base_url = str(getattr(agent, "base_url", "") or "").strip() if live_provider else ""
    api_key = getattr(agent, "api_key", None) if live_provider else None
    if not base_url and provider == str(mirror.get("provider") or "").strip():
        base_url = str(mirror.get("base_url") or "").strip()
    if not provider or not base_url:
        cfg = _load_cfg()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
        configured_provider = str(model_cfg.get("provider") or "").strip()
        if not provider and configured_provider.lower() != "auto":
            provider = configured_provider
        # Never combine the selected provider with another provider's URL.
        if not base_url and provider == configured_provider:
            base_url = str(model_cfg.get("base_url") or "").strip()
    return provider, base_url, api_key


def _scoped_account_lines(session: dict) -> list[str]:
    """Resolve the owning profile, including cold external secrets, in the worker."""
    from tui_gateway.server import _launch_home, _session_profile_runtime_scope
    from tui_gateway.launch_profile_policy import launch_profile_runtime_scope
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    # Hydration can run a secret helper subprocess. Keep it, config resolution,
    # and the quota request inside the same timeout and reset scope on exit.
    scope = (_session_profile_runtime_scope(session) if session.get("profile_home")
             else launch_profile_runtime_scope(_launch_home()))
    with scope:
        provider, base_url, api_key = _usage_provider_identity(session)
        if not provider:
            return []
        snapshot = fetch_account_usage(provider, base_url=base_url or None, api_key=api_key)
        return list(render_account_usage_lines(snapshot) or [])


def _reset_live_usage(session: dict, *, force: bool) -> str:
    """Redeem only the account whose quota the live session displays."""
    from agent.account_usage import redeem_codex_reset_credit
    from agent.i18n import t
    from tui_gateway.launch_profile_policy import launch_profile_runtime_scope
    from tui_gateway.server import _launch_home, _session_profile_runtime_scope

    scope = (_session_profile_runtime_scope(session) if session.get("profile_home")
             else launch_profile_runtime_scope(_launch_home()))
    with scope:
        provider, base_url, api_key = _usage_provider_identity(session)
        if provider.lower() != "openai-codex":
            return t("gateway.usage.reset_wrong_provider")
        # This is an explicit mutation, not a cosmetic quota read: do not put it
        # in the fail-open display pool, where it could finish after a timeout.
        return redeem_codex_reset_credit(base_url=base_url or None, api_key=api_key, force=force).message


def _usage_provider_lines(session: dict) -> tuple[list[str], list[str]]:
    """Return bounded, fail-open provider account and rate-limit lines."""
    account_lines: list[str] = []
    rate_limit_lines: list[str] = []
    try:
        slots = _account_usage_slots
        # A timed-out request still owns its slot until the worker exits. Never
        # accumulate queued profile/credential work behind wedged endpoints.
        if slots.acquire(blocking=False):
            try:
                future = _account_usage_pool.submit(_scoped_account_lines, session)
            except BaseException:
                slots.release()
                raise
            future.add_done_callback(lambda _: slots.release())
            try:
                account_lines = future.result(timeout=_ACCOUNT_USAGE_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
    except Exception:
        logger.debug("Provider quota unavailable", exc_info=True)
        account_lines = []
    agent = session.get("agent")
    if agent is not None:
        try:
            state = agent.get_rate_limit_state()
            if state and getattr(state, "has_data", False):
                from agent.rate_limit_tracker import format_rate_limit_display
                rate_limit_lines = format_rate_limit_display(state).splitlines()
        except Exception:
            logger.debug("Provider rate limits unavailable", exc_info=True)
            rate_limit_lines = []
    return account_lines, rate_limit_lines
