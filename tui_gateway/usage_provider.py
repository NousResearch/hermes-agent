"""Bounded, fail-open account limits shared by desktop slash and TUI usage."""

from __future__ import annotations

import atexit
import concurrent.futures
from typing import Any

from tools.daemon_pool import DaemonThreadPoolExecutor

# Quota endpoints are remote and cosmetic. Timed-out calls must not occupy the
# RPC reader or the general long-handler pool indefinitely.
_account_usage_pool = DaemonThreadPoolExecutor(
    max_workers=2, thread_name_prefix="hermes-account-usage"
)
atexit.register(lambda: _account_usage_pool.shutdown(wait=False, cancel_futures=True))
_ACCOUNT_USAGE_TIMEOUT_SECONDS = 10.0


def _usage_provider_identity(session: dict) -> tuple[str, str, Any]:
    """Resolve quota inputs without persisting or returning credentials."""
    from tui_gateway.server import _load_cfg
    from tui_gateway.compute_host_bridge import _metadata_mirror

    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    provider = str(getattr(agent, "provider", "") if agent else "").strip() or str(mirror.get("provider") or "").strip()
    base_url = str(getattr(agent, "base_url", "") if agent else "").strip() or str(mirror.get("base_url") or "").strip()
    api_key = getattr(agent, "api_key", None) if agent else None
    if not provider or not base_url:
        try:
            cfg = _load_cfg()
            model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
            model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
            configured_provider = str(model_cfg.get("provider") or "").strip()
            if not provider:
                provider = configured_provider
            # Never combine a mirrored provider with another provider's URL.
            if not base_url and provider == configured_provider:
                base_url = str(model_cfg.get("base_url") or "").strip()
        except Exception:
            pass
    return provider, base_url, api_key


def _scoped_account_lines(session: dict) -> list[str]:
    """Resolve the owning profile, including cold external secrets, in the worker."""
    from tui_gateway.server import _session_profile_runtime_scope
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    # Hydration can run a secret helper subprocess. Keep it, config resolution,
    # and the quota request inside the same timeout and reset scope on exit.
    with _session_profile_runtime_scope(session):
        provider, base_url, api_key = _usage_provider_identity(session)
        if not provider:
            return []
        snapshot = fetch_account_usage(provider, base_url=base_url or None, api_key=api_key)
        return list(render_account_usage_lines(snapshot) or [])


def _usage_provider_lines(session: dict) -> tuple[list[str], list[str]]:
    """Return bounded, fail-open provider account and rate-limit lines."""
    account_lines: list[str] = []
    rate_limit_lines: list[str] = []
    try:
        future = _account_usage_pool.submit(_scoped_account_lines, session)
        try:
            account_lines = future.result(timeout=_ACCOUNT_USAGE_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise
    except Exception:
        account_lines = []
    agent = session.get("agent")
    if agent is not None:
        try:
            state = agent.get_rate_limit_state()
            if state and getattr(state, "has_data", False):
                from agent.rate_limit_tracker import format_rate_limit_display
                rate_limit_lines = format_rate_limit_display(state).splitlines()
        except Exception:
            rate_limit_lines = []
    return account_lines, rate_limit_lines
