"""Resolution-time fallback for delegation routes: when a route's primary provider refuses at credential
resolution with an auth, quota or transient network error, walk the route's own fallback chain before the
task is refused (#146645). The first entry that resolves becomes the child's primary; the entries after it
stay the child's mid-run chain. The fall-through predicate is ``cron.scheduler_preflight.primary_resolve_falls_through``,
which cron's ``_resolve_job_runtime`` uses too.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

T = TypeVar("T")

# AuthError codes that are misconfiguration, not a credential state: a typo'd provider, an absent key, a provider
# with no endpoint or a CLI that is not installed must refuse with today's message rather than quietly run the
# task on another provider.
_MISCONFIGURED_AUTH_CODES = frozenset({
    "invalid_provider", "missing_api_key", "no_provider_configured", "missing_base_url",
    "missing_external_process_cli"})


def _fallthrough_cause(exc: ValueError) -> Optional[BaseException]:
    """The resolver failure behind a route refusal when it may walk the chain, else None. The route resolvers
    wrap ``resolve_runtime_provider``'s exception as ``__cause__``. A refusal without one (the resolved runtime
    has no API key or base_url, or its pinned command is not on PATH) and an ``AuthError`` whose code marks
    configuration (an unknown provider, an API-key provider without a key, no endpoint, an external-process CLI
    the resolver could not find) refuse."""
    from cron.scheduler_preflight import primary_resolve_falls_through
    cause = exc.__cause__
    if cause is None or getattr(cause, "code", None) in _MISCONFIGURED_AUTH_CODES:
        return None
    return cause if primary_resolve_falls_through(cause) else None


def _failure_text(exc: ValueError) -> str:
    return str(exc.__cause__ or exc)


def resolve_with_fallback(
    resolve_primary: Callable[[], T], fallback_chain: Callable[[], Optional[list[dict[str, Any]]]],
    resolve_entry: Callable[[dict[str, Any]], T],
) -> tuple[T, Optional[list[dict[str, Any]]]]:
    """``(resolved, remaining_chain)``: the primary's result and None, or the first chain entry that resolves
    and the entries after it. ``fallback_chain`` is read only after the primary refuses, so a resolving
    primary behaves exactly as before. Both resolvers raise ``ValueError`` (with the resolver error as ``__cause__``)
    to refuse. A primary refusal that is not an auth, quota or transient failure, or one with no chain to
    walk, re-raises unchanged. When every entry refuses, the error names the primary's failure and each
    entry's."""
    try:
        return resolve_primary(), None
    except ValueError as primary_exc:
        cause = _fallthrough_cause(primary_exc)
        chain = fallback_chain() if cause is not None else None
        if not chain:
            raise
        failures = []
        for index, entry in enumerate(chain):
            try:
                resolved = resolve_entry(entry)
            except ValueError as entry_exc:
                logger.debug("Delegation fallback %s/%s failed: %s", entry["provider"], entry["model"], entry_exc)
                failures.append(f"{entry['provider']}/{entry['model']}: {_failure_text(entry_exc)}")
                continue
            from hermes_cli.auth import AuthError, primary_failure_wording
            wording = primary_failure_wording(cause)[0] if isinstance(cause, AuthError) else "unreachable"
            logger.warning("Primary provider %s (%s). Falling back to %s/%s",
                           wording, cause, entry["provider"], entry["model"])
            return resolved, chain[index + 1:]
        raise ValueError(f"{primary_exc} Every fallback provider also failed: {'; '.join(failures)}") from primary_exc


def entry_runtime_request(entry: dict[str, Any]) -> dict[str, Any]:
    """``resolve_runtime_provider`` kwargs for one fallback entry, the way cron and the main session resolve
    it: the entry's provider and model, plus its ``base_url`` and key (inline ``api_key`` or ``key_env``)."""
    from hermes_cli.fallback_config import resolve_entry_api_key
    request: dict[str, Any] = {"requested": entry["provider"], "target_model": entry["model"]}
    if entry.get("base_url"):
        request["explicit_base_url"] = entry["base_url"]
    if api_key := resolve_entry_api_key(entry):
        request["explicit_api_key"] = api_key
    return request


def config_route_credentials(route: dict[str, Any], parent_agent: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(creds, routing_cfg)`` for a config-described route (the ``delegation`` section, a backend option,
    an internal caller's block). After a fall-through, ``routing_cfg`` carries the remaining entries as
    ``fallback_providers`` so the child's mid-run chain never retries the entry it now runs on."""
    from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
    from tools.delegate_tool_config import _resolve_child_fallback_chain, _runtime_provider_credentials

    def resolve_entry(entry: dict[str, Any]) -> dict[str, Any]:
        values = {"model": entry["model"], "provider": entry["provider"], "base_url": None, "api_key": None,
                  "api_mode": None}
        explicit = {k: v for k, v in entry_runtime_request(entry).items() if k.startswith("explicit_")}
        overrides = route.get("request_overrides") if isinstance(route.get("request_overrides"), dict) else None
        return _runtime_provider_credentials(values, overrides, explicit)

    # Only the provider branch raises a fall-through refusal, and a route naming a provider builds a pinned
    # child, so the walk uses the chain that child would get: the route's own, never the parent's.
    creds, remaining = resolve_with_fallback(
        lambda: _resolve_delegation_credentials(route, parent_agent),
        lambda: _resolve_child_fallback_chain(parent_agent, route, pinned=True), resolve_entry)
    return creds, route if remaining is None else {**route, "fallback_providers": remaining}
