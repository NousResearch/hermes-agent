"""Delegation config knobs (delegation.* keys) and child credential/provider resolution."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from utils import base_url_hostname, is_truthy_value

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

# Runtime-provider sentinel for providers that are not natively known; must
# match hermes_cli.runtime_provider.RUNTIME_PROVIDER_TYPE_CUSTOM.
_RUNTIME_PROVIDER_CUSTOM = "custom"

_DEFAULT_MAX_CONCURRENT_CHILDREN = 10
# One-shot guard: _get_max_concurrent_children() runs on every get_definitions()
# schema rebuild, so the >10 cost advisory would otherwise log on every turn.
_HIGH_CONCURRENCY_WARNED = False
MAX_DEPTH = 1  # flat by default: parent (0) -> child (1); deeper needs max_spawn_depth
_MIN_SPAWN_DEPTH = 1  # floor for the configurable cap; MAX_DEPTH stays the default
_LEGACY_MAX_ASYNC_WARNED = False
# No default wall-clock cap on children: legitimate heavy work (deep reviews, research fan-outs, slow reasoning
# models) was being killed mid-task. Stuck-child detection is the heartbeat staleness monitor;
# delegation.child_timeout_seconds opts back in.
DEFAULT_CHILD_TIMEOUT: Optional[float] = None

def _cfg() -> dict:
    """The ``delegation`` section, read through the origin so tests can patch it."""
    from tools.delegate_tool import _load_config
    return _load_config()


# ── Subagent approval callbacks ─────────────────────────────────────────────
# Subagent worker threads don't inherit the CLI's threading.local approval
# callback, so prompt_dangerous_approval() would fall back to input() and
# deadlock the parent's prompt_toolkit TUI. Every worker gets a non-interactive
# callback via ThreadPoolExecutor(initializer=...): deny by default, approve when
# delegation.subagent_auto_approve is true. Both warn for audit. Gateway sessions
# are unaffected (they resolve approvals via tools/approval.py's per-session queue).
def _subagent_auto_deny(command: str, description: str, **kwargs) -> str:
    """Auto-deny (safe default): returns 'deny' so the child sees a recoverable refusal."""
    logger.warning(
        "Subagent auto-denied dangerous command: %s (%s). Set delegation.subagent_auto_approve: true to allow.",
        command, description,
    )
    return "deny"

def _subagent_auto_approve(command: str, description: str, **kwargs) -> str:
    """Auto-approve (opt-in YOLO via delegation.subagent_auto_approve): returns 'once'."""
    logger.warning("Subagent auto-approved dangerous command: %s (%s)", command, description)
    return "once"

def _get_subagent_approval_callback():
    """Callback for subagent worker threads per delegation.subagent_auto_approve (default False)."""
    if is_truthy_value(_cfg().get("subagent_auto_approve", False)):
        return _subagent_auto_approve
    return _subagent_auto_deny

def _knob(key: str, env_var: Optional[str], parse, default, invalid_msg: str):
    """delegation.<key> > <env_var> > default. A config value that fails ``parse`` logs ``invalid_msg`` (``%r`` = the
    value) and yields the default; an env value that fails is silently ignored."""
    val = _cfg().get(key)
    if val is not None:
        try:
            return parse(val)
        except (TypeError, ValueError):
            logger.warning(invalid_msg, val)
            return default
    env_val = os.getenv(env_var) if env_var else None
    if env_val:
        try:
            return parse(env_val)
        except (TypeError, ValueError):
            pass
    return default

def _warn_once(flag_name: str, message: str, *args: Any) -> None:
    """Module-global one-shot warning (``flag_name`` is the ``_*_WARNED`` global)."""
    if not globals()[flag_name]:
        globals()[flag_name] = True
        logger.warning(message, *args)

def _get_max_concurrent_children() -> int:
    """delegation.max_concurrent_children > DELEGATION_MAX_CONCURRENT_CHILDREN env > 10.

    Floor of 1 is the only bound enforced; there is no ceiling.
    """
    result = _knob(
        "max_concurrent_children", "DELEGATION_MAX_CONCURRENT_CHILDREN", lambda v: max(1, int(v)),
        _DEFAULT_MAX_CONCURRENT_CHILDREN,
        f"delegation.max_concurrent_children=%r is not a valid integer; using default {_DEFAULT_MAX_CONCURRENT_CHILDREN}",
    )
    if result > 10 and _cfg().get("max_concurrent_children") is not None:
        _warn_once(
            "_HIGH_CONCURRENCY_WARNED", "delegation.max_concurrent_children=%d: each child consumes API tokens "
            "independently. High values multiply cost linearly.", result,
        )
    return result

def _get_worktree_isolation() -> bool:
    """delegation.worktree_isolation (bool, default False): each child gets its own
    git worktree off the parent's HEAD so parallel children never contend for one
    working copy. Git-only and local-backend-only; otherwise silently ignored."""
    return bool(_cfg().get("worktree_isolation", False))

def _get_max_async_children() -> int:
    """Concurrency cap for background delegations == delegation.max_concurrent_children. At capacity a new async
    dispatch is REJECTED (not queued) so a runaway model can't pile up unbounded background work; the caller then
    runs synchronously. A leftover ``delegation.max_async_children`` key is ignored with a one-time warning."""
    from tools.delegate_tool import _get_max_concurrent_children
    if _cfg().get("max_async_children") is not None:
        _warn_once(
            "_LEGACY_MAX_ASYNC_WARNED", "delegation.max_async_children is deprecated and ignored; "
            "delegation.max_concurrent_children now caps background "
            "delegations too. Remove the stale key from config.yaml.",
        )
    return _get_max_concurrent_children()

def _parse_timeout(raw: Any) -> Optional[float]:
    """Seconds → None (<= 0 disables) or max(30, value). Raises on non-numeric."""
    parsed = float(raw)
    return None if parsed <= 0 else max(30.0, parsed)

def _get_child_timeout() -> Optional[float]:
    """Hard wall-clock cap for one child, or None (default: no timeout). Failures should come from what the child does
    (API/tool errors, iteration budget), not a stopwatch; stuck children are caught by the heartbeat staleness
    monitor. delegation.child_timeout_seconds > 0 opts in (floor 30 s); 0 or negative disables. Env fallback:
    DELEGATION_CHILD_TIMEOUT_SECONDS."""
    return _knob(
        "child_timeout_seconds", "DELEGATION_CHILD_TIMEOUT_SECONDS", _parse_timeout, DEFAULT_CHILD_TIMEOUT,
        "delegation.child_timeout_seconds=%r is not a valid number; using default (no timeout)",
    )

def _get_max_spawn_depth() -> int:
    """delegation.max_spawn_depth floored at 1 (no ceiling). Depth 0 is the parent; agents at depths 0..N-1 may spawn,
    depth N is the leaf floor. Default 1 is flat. Each extra level multiplies API cost."""
    def _floored(v):
        ival = int(v)
        if ival < _MIN_SPAWN_DEPTH:
            logger.warning("delegation.max_spawn_depth=%d below floor %d; using %d", ival, _MIN_SPAWN_DEPTH, _MIN_SPAWN_DEPTH)
        return max(_MIN_SPAWN_DEPTH, ival)

    return _knob(
        "max_spawn_depth", None, _floored, MAX_DEPTH,
        f"delegation.max_spawn_depth=%r is not a valid integer; using default {MAX_DEPTH}",
    )

def _get_orchestrator_enabled() -> bool:
    """delegation.orchestrator_enabled kill switch (default True): False forces every child to leaf."""
    val = _cfg().get("orchestrator_enabled", True)
    if isinstance(val, bool):
        return val
    # Accept "true"/"false" strings from YAML that doesn't auto-coerce.
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "on"}
    return True

def _get_inherit_mcp_toolsets() -> bool:
    """Whether narrowed child toolsets should keep the parent's MCP toolsets."""
    return is_truthy_value(_cfg().get("inherit_mcp_toolsets"), default=True)

def _normalized_runtime_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")

def _inherit_parent_capabilities(parent_agent, override_provider, override_base_url) -> Optional[dict]:
    """Parent's endpoint-trust capability map for a child, or None. ``agent.capabilities`` is a trust decision scoped
    to one provider+endpoint: inherited ONLY when the child runs the parent's exact route; any provider or base_url
    override stays DEFAULT-DENY (matches the /model switch posture).

    See #94036, #97292.
    """
    if override_provider or override_base_url:
        return None
    parent_caps = getattr(parent_agent, "capabilities", None)
    if not isinstance(parent_caps, dict):
        return None
    return {key: value for key, value in parent_caps.items() if isinstance(key, str) and isinstance(value, bool)}

def _inherit_parent_base_url(parent_agent, fallback_base_url: Optional[str]) -> Optional[str]:
    """Base URL the parent is actually calling (live client), not a stale attribute: ``parent_agent.base_url`` can lag
    the live client (old OpenRouter URL vs local Ollama) and inheriting the stale one 401s with a dummy/local key."""
    surface_url = _normalized_runtime_url(fallback_base_url)
    client_kwargs = getattr(parent_agent, "_client_kwargs", None)
    client = getattr(parent_agent, "client", None)
    live_candidates = (
        client_kwargs.get("base_url") if isinstance(client_kwargs, dict) else None,
        # OpenAI SDK exposes base_url as httpx.URL — coerce before comparing.
        getattr(client, "base_url", "") if client is not None else None,
    )
    for raw in live_candidates:
        url = _normalized_runtime_url(raw)
        if url and url != surface_url and url.startswith(("http://", "https://")):
            return url
    return fallback_base_url or None

def _loaded_pool(key: Any):
    """``load_pool(key)`` when it holds credentials, else None."""
    from agent.credential_pool import load_pool
    pool = load_pool(key)
    return pool if pool is not None and pool.has_credentials() else None

def _resolve_child_credential_pool(
    effective_provider: Optional[str], parent_agent, effective_base_url: Optional[str] = None,
):
    """Credential pool for the child: parent's pool (same provider), that provider's own pool, or None (child keeps
    its fixed credential). Custom endpoints all collapse to ``provider="custom"``, so they are matched by endpoint
    identity (the ``custom:<name>`` pool key) — sharing the parent's pool across different custom endpoints would
    overwrite the child's delegated base_url on lease; an unregistered custom endpoint (no custom_providers entry)
    keeps the child's fixed credential rather than inherit the parent's.

    Custom endpoints are a special case: every direct ``delegation.base_url`` runtime collapses to
    ``provider="custom"``, so bare provider equality would treat two *different* custom endpoints as
    interchangeable and let the child inherit the parent's pool. We therefore resolve custom runtimes by
    endpoint identity (the ``custom:<name>`` pool key derived from the base_url) and only share the parent's
    pool when both resolve to the *same* custom endpoint. See #7833.
    """
    parent_pool = getattr(parent_agent, "_credential_pool", None)
    if not effective_provider:
        return parent_pool
    parent_provider = getattr(parent_agent, "provider", None) or ""
    try:
        if effective_provider == "custom":
            from agent.credential_pool import get_custom_provider_pool_key
            child_key = get_custom_provider_pool_key(effective_base_url)
            if child_key is None:
                return None
            parent_key = get_custom_provider_pool_key(getattr(parent_agent, "base_url", None))
            if parent_pool is not None and parent_provider == "custom" and parent_key is not None and parent_key == child_key:
                return parent_pool
            return _loaded_pool(child_key)
        if parent_pool is not None and effective_provider == parent_provider:
            return parent_pool
        return _loaded_pool(effective_provider)
    except Exception as exc:
        if effective_provider == "custom":
            logger.debug("Could not resolve custom credential pool for child endpoint '%s': %s", effective_base_url, exc)
        else:
            logger.debug("Could not load credential pool for child provider '%s': %s", effective_provider, exc)
    return None

def _merge_request_overrides(runtime_overrides, explicit_overrides):
    """Merge explicit ``delegation.request_overrides`` OVER runtime-derived ones. Explicit top-level keys win;
    ``extra_body`` is deep-merged ONE level so provider personality (e.g. ``thinking: {type: disabled}``) survives
    unless the explicit dict redefines that exact key. Both sides are deep-copied so transport-side mutation can't
    leak into the config/runtime cache. None when both are empty."""
    import copy as _copy
    runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else None
    explicit_overrides = explicit_overrides if isinstance(explicit_overrides, dict) else None
    if not runtime_overrides and not explicit_overrides:
        return None
    merged = _copy.deepcopy(runtime_overrides) if runtime_overrides else {}
    explicit = _copy.deepcopy(explicit_overrides) if explicit_overrides else {}
    runtime_extra = merged.get("extra_body")
    explicit_extra = explicit.pop("extra_body", None)
    merged.update(explicit)
    if isinstance(runtime_extra, dict) and isinstance(explicit_extra, dict):
        runtime_extra.update(explicit_extra)
        merged["extra_body"] = runtime_extra
    elif explicit_extra is not None:
        merged["extra_body"] = explicit_extra
    return merged or None

# Native-SDK providers speak their own wire protocol and can't be reached via chat_completions against a base_url:
# always take the runtime-provider path (a configured base_url still flows through it, e.g. a Bedrock region).
_NATIVE_SDK_PROVIDERS = frozenset({"bedrock", "vertex", "google", "google-genai"})
_EXPLICIT_API_MODES = frozenset({"chat_completions", "codex_responses", "anthropic_messages"})

def _require_pinned_command(command: Optional[str], message: str) -> None:
    """A pinned ACP transport command must exist on PATH — refuse loudly rather
    than let the child silently fall back to another transport."""
    import shutil as _shutil
    if command and not _shutil.which(command):
        raise ValueError(message)

def _credential_bundle(model, provider, base_url, api_key, api_mode, request_overrides, max_output_tokens, **extra) -> dict:
    """The child credential dict every branch of ``_resolve_delegation_credentials`` returns."""
    return {
        "model": model, "provider": provider, "base_url": base_url, "api_key": api_key, "api_mode": api_mode,
        "request_overrides": request_overrides, "max_output_tokens": max_output_tokens, **extra,
    }

def _direct_endpoint_credentials(v: dict, explicit_request_overrides) -> dict:
    """``delegation.base_url`` branch: provider/api_mode from URL heuristics."""
    # Shared URL-based api_mode detector so Anthropic-compatible direct endpoints (/anthropic suffix: Azure AI
    # Foundry, MiniMax, Zhipu, LiteLLM) get the Messages transport instead of 404ing on chat_completions.
    # Without this, subagents would default to chat_completions and hit 404s on endpoints that only speak
    # the Anthropic Messages protocol. Fixes #10213.
    from hermes_cli.runtime_provider import _detect_api_mode_for_url
    base_lower = v["base_url"].lower()
    host = base_url_hostname(v["base_url"])
    provider = "custom"
    api_mode = _detect_api_mode_for_url(v["base_url"]) or "chat_completions"
    if host == "chatgpt.com" and "/backend-api/codex" in base_lower:
        provider, api_mode = "openai-codex", "codex_responses"
    elif host == "api.anthropic.com":
        provider, api_mode = "anthropic", "anthropic_messages"
    elif "api.kimi.com/coding" in base_lower:
        api_mode = "anthropic_messages"
    # Explicit delegation.api_mode always wins over the URL heuristic.
    if v["api_mode"] in _EXPLICIT_API_MODES:
        api_mode = v["api_mode"]

    # provider configured ALONGSIDE base_url: pull that provider's request personality (request_overrides /
    # max_output_tokens) onto the explicit endpoint. Best-effort — a resolution failure only skips the overrides.
    request_overrides = max_output_tokens = None
    if v["provider"]:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(requested=v["provider"], target_model=v["model"])
            request_overrides = dict(runtime.get("request_overrides") or {}) or None
            max_output_tokens = runtime.get("max_output_tokens")
        except Exception as exc:
            logger.debug(
                "delegation.base_url: runtime resolution for provider '%s' failed; proceeding without request_overrides: %s",
                v["provider"], exc,
            )
    # api_key None → inherited from parent in _build_child_agent
    return _credential_bundle(
        v["model"], provider, v["base_url"], v["api_key"], api_mode,
        _merge_request_overrides(request_overrides, explicit_request_overrides), max_output_tokens,
    )

def _runtime_provider_credentials(v: dict, explicit_request_overrides) -> dict:
    """``delegation.provider`` branch: full bundle via the runtime provider system."""
    configured_provider = v["provider"]
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested=configured_provider, target_model=v["model"])
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes auth'."
        )
    # A pinned ACP transport command must exist — refuse the spawn loudly rather than letting the child
    # silently fall back to another transport (#80450).
    pinned_command = runtime.get("command")
    _require_pinned_command(
        pinned_command, f"Delegation provider '{configured_provider}' is pinned to the "
        f"'{pinned_command}' command, which was not found on PATH. "
        f"Install it or choose a different delegation provider.",
    )
    return _credential_bundle(
        v["model"] or runtime.get("model") or None,
        configured_provider if runtime.get("provider") == _RUNTIME_PROVIDER_CUSTOM else runtime.get("provider"),
        runtime.get("base_url"), api_key, runtime.get("api_mode"),
        _merge_request_overrides(runtime.get("request_overrides"), explicit_request_overrides) or {},
        runtime.get("max_output_tokens"), command=pinned_command, args=list(runtime.get("args") or []),
    )

def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Child credential bundle from the ``delegation`` config section. Three branches: ``base_url`` set → direct
    endpoint (``api_key`` None means inherit the parent's key, so providers keyed outside OPENAI_API_KEY work);
    ``provider`` set → full bundle via the runtime provider system (same path as CLI/gateway startup); neither →
    None values, child inherits everything. ``request_overrides`` is honored on every branch. Raises ValueError
    with a user-facing message."""
    values = {k: str(cfg.get(k) or "").strip() or None for k in ("model", "provider", "base_url", "api_key")}
    values["api_mode"] = str(cfg.get("api_mode") or "").strip().lower() or None
    explicit_request_overrides = cfg.get("request_overrides") if isinstance(cfg.get("request_overrides"), dict) else None
    is_native_sdk_provider = (values["provider"] or "").strip().lower() in _NATIVE_SDK_PROVIDERS

    if values["base_url"] and not is_native_sdk_provider:
        return _direct_endpoint_credentials(values, explicit_request_overrides)
    if not values["provider"]:
        # Pure inherit; explicit request_overrides still merge OVER the parent's.
        return _credential_bundle(
            values["model"], None, None, None, None,
            _merge_request_overrides(getattr(parent_agent, "request_overrides", None), explicit_request_overrides), None,
        )
    return _runtime_provider_credentials(values, explicit_request_overrides)

def _load_config() -> dict:
    """The ``delegation`` config section (read-only — do NOT mutate). Prefers the shared ``load_config_readonly()``
    (follows HERMES_HOME/profile; no deepcopy, since this runs on every get_definitions() rebuild) over the legacy
    ``cli.CLI_CONFIG``, which can hide user-set keys — except that ``HERMES_IGNORE_USER_CONFIG=1`` is only honored
    by the legacy loader, so it stays authoritative when that flag is set."""
    if os.environ.get("HERMES_IGNORE_USER_CONFIG") != "1":
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly().get("delegation") or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    try:
        from cli import CLI_CONFIG
        cfg = CLI_CONFIG.get("delegation") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}

# OpenRouter routing filters: inherited from the parent, but reset to these defaults under a pinned provider — parent
# filters (e.g. only=["Anthropic"]) would silently force the child back onto the parent's provider.
# openrouter_min_coding_score stays inherited: model-gated, no-op elsewhere.
_ROUTING_FILTER_DEFAULTS = (
    ("providers_allowed", None), ("providers_ignored", None), ("providers_order", None), ("provider_sort", None),
    ("provider_require_parameters", False), ("provider_data_collection", ""),
)

_NOUS_PROVIDERS = frozenset({"nous", "nous-portal", "nousresearch"})

def _resolve_child_runtime(
    parent_agent, delegation_cfg: dict, parent_api_key: Any, *, model: Optional[str], override_provider: Optional[str],
    override_base_url: Optional[str], override_api_key: Optional[str], override_api_mode: Optional[str],
    override_max_tokens: Optional[int], override_acp_command: Optional[str], override_acp_args: Optional[List[str]],
) -> Dict[str, Any]:
    """Child credentials, transport and routing (config override > parent inherit) as ``AIAgent`` kwargs. Rules that
    are easy to break: api_mode is re-derived (not inherited) when the child's provider differs from the parent's
    or is Nous Portal (dual-wire); a pinned ``delegation.command`` must exist on PATH or the spawn fails loudly;
    ``override_provider`` clears the parent's ACP transport, fallback chain and OpenRouter routing filters so the
    pinned provider is actually honoured."""
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or _inherit_parent_base_url(parent_agent, parent_agent.base_url)
    # api_mode: each provider has its own wire, so a different provider re-derives (None) instead of inheriting (404s
    # otherwise). Nous Portal is dual-wire within one provider (anthropic/* → Messages, else chat_completions), so
    # same-provider inheritance would pin the child on the wrong wire — re-derive.
    # Bug #20558 / PR #20563: api_mode must NOT be inherited when the child uses a different provider than
    # the parent — each provider has its own API surface (e.g. MiniMax uses anthropic_messages, DeepSeek
    # uses chat_completions). Inheriting the parent's mode causes 404 errors when the child routes to the
    # wrong endpoint. Derive the mode from the target provider when it differs. Same-provider inheritance
    # would pin a child Hermes/Qwen subagent onto the parent's Claude Messages wire (or the reverse).
    # agent_init honors an explicit api_mode above its nous branch, so re-derive here before construction.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif (effective_provider or "").strip().lower() in _NOUS_PROVIDERS:
        from hermes_cli.providers import nous_api_mode
        effective_api_mode = nous_api_mode(effective_model)
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # A pinned transport that cannot run must fail the spawn loudly, never fall
    # back silently (delegate_task pre-validates; this covers direct callers).
    _require_pinned_command(
        override_acp_command, f"Pinned delegation command '{override_acp_command}' was not "
        f"found on PATH. Install it or remove delegation.command from config.yaml.",
    )
    effective_acp_command = override_acp_command or getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(
        override_acp_args if override_acp_args is not None else (getattr(parent_agent, "acp_args", []) or [])
    )
    # A pinned provider must use direct API calls; inheriting the parent's ACP
    # transport would bypass the override credentials entirely.
    # Inheriting acp_command unconditionally causes run_agent.py to initialize CopilotACPClient, bypassing
    # override credentials entirely (issue #16816).
    if override_provider and not override_acp_command:
        effective_acp_command, effective_acp_args = None, []
    # Defensive: validate trusted delegation.command exists on PATH before honoring it. An explicitly pinned
    # transport that cannot run must fail the spawn loudly (#80450) — silently falling back to the default
    # transport would run the child somewhere the user explicitly routed it away from. Normally unreachable
    # via delegate_task, which pre-validates the command in _resolve_delegation_credentials.
    if override_acp_command:
        # Forced ACP transport requires provider copilot-acp for run_agent to init the client.
        effective_provider, effective_api_mode = "copilot-acp", "chat_completions"

    # Reasoning: delegation.reasoning_effort > parent. Keep the raw value — a
    # YAML ``false`` must disable thinking, not coerce to "" and inherit.
    child_reasoning = getattr(parent_agent, "reasoning_config", None)
    try:
        delegation_effort = delegation_cfg.get("reasoning_effort")
        if delegation_effort or delegation_effort is False:
            from hermes_constants import parse_reasoning_effort
            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is None:
                logger.warning("Unknown delegation.reasoning_effort '%s', inheriting parent level", delegation_effort)
            else:
                child_reasoning = parsed
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

    kwargs: Dict[str, Any] = {
        "base_url": effective_base_url, "api_key": override_api_key or parent_api_key, "model": effective_model,
        "provider": effective_provider,
        "capabilities": _inherit_parent_capabilities(parent_agent, override_provider, override_base_url),
        "api_mode": effective_api_mode, "acp_command": effective_acp_command, "acp_args": effective_acp_args,
        "reasoning_config": child_reasoning,
        # Inherit the parent's fallback chain EXCEPT under a pinned provider: a mid-run 429/auth failure must not
        # silently reroute the quiet child onto the parent's fallbacks. Predictability > liveness for explicit pins.
        "fallback_model": None if override_provider else (getattr(parent_agent, "_fallback_chain", None) or None),
        "openrouter_min_coding_score": getattr(parent_agent, "openrouter_min_coding_score", None),
        # Routing filters reset to their defaults under a pinned provider (see _ROUTING_FILTER_DEFAULTS).
        **{a: d if override_provider else getattr(parent_agent, a, d) for a, d in _ROUTING_FILTER_DEFAULTS},
    }
    if not override_provider:
        kwargs["provider_data_collection"] = kwargs["provider_data_collection"] or ""
    child_max_tokens = override_max_tokens if override_max_tokens is not None else getattr(parent_agent, "max_tokens", None)
    if isinstance(child_max_tokens, int):
        kwargs["max_tokens"] = child_max_tokens
    return kwargs
