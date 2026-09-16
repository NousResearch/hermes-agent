"""Bounded, fail-open account limits shared by desktop slash and TUI usage."""

from __future__ import annotations

import atexit
import concurrent.futures
from pathlib import Path
from typing import Any

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

# Quota endpoints are remote and cosmetic. Timed-out calls must not occupy the
# RPC reader or the general long-handler pool indefinitely.
_account_usage_pool = concurrent.futures.ThreadPoolExecutor(
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
        token = None
        try:
            if session.get("profile_home"):
                token = set_hermes_home_override(Path(session["profile_home"]))
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
        finally:
            if token is not None:
                reset_hermes_home_override(token)
    return provider, base_url, api_key


def _usage_provider_lines(session: dict) -> tuple[list[str], list[str]]:
    """Return bounded, fail-open provider account and rate-limit lines."""
    account_lines: list[str] = []
    rate_limit_lines: list[str] = []
    provider, base_url, api_key = _usage_provider_identity(session)
    if provider:
        try:
            from agent.account_usage import fetch_account_usage, render_account_usage_lines

            def fetch():
                from agent.secret_scope import (
                    build_profile_secret_scope, reset_secret_scope, set_secret_scope,
                )
                from hermes_constants import get_hermes_home

                token = None
                secret_token = None
                try:
                    if session.get("profile_home"):
                        token = set_hermes_home_override(Path(session["profile_home"]))
                    secret_token = set_secret_scope(build_profile_secret_scope(get_hermes_home()))
                    return fetch_account_usage(provider, base_url=base_url or None, api_key=api_key)
                finally:
                    if secret_token is not None:
                        reset_secret_scope(secret_token)
                    if token is not None:
                        reset_hermes_home_override(token)

            future = _account_usage_pool.submit(fetch)
            try:
                snapshot = future.result(timeout=_ACCOUNT_USAGE_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise
            account_lines = list(render_account_usage_lines(snapshot) or [])
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
