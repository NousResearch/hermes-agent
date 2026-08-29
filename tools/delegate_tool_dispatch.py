"""Delegation validation, credentials, schema, and dispatch support."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tools.delegate_tool")

from tools.delegation_outcome import DELEGATION_OUTCOME_TOOL_GUIDANCE
from tools.delegate_tool_control import (
    MAX_DEPTH,
    _DEFAULT_MAX_CONCURRENT_CHILDREN,
    _RUNTIME_PROVIDER_CUSTOM,
    base_url_hostname,
)
from utils import is_truthy_value


def _facade():
    from tools import delegate_tool as facade

    return facade


def _get_max_concurrent_children():
    return _facade()._get_max_concurrent_children()


def _get_max_spawn_depth():
    return _facade()._get_max_spawn_depth()


def _get_orchestrator_enabled():
    return _facade()._get_orchestrator_enabled()


# Placeholder shapes for batch goal validation: bare 'TODO', bare 'task N'
# labels, or goals still carrying unexpanded template markers.
#
# The marker regex is deliberately NARROW: it only fires on snake_case /
# space-separated placeholder identifiers (`<feature_name>`, `{file path}`,
# `<FEATURE-NAME>`) — the shape LLM templates actually leave behind. Bare
# single-word brackets are left alone because legitimate coding goals are
# full of them: generics (`Vec<T>`, `Result<String>`), HTML tags (`<div>`),
# JSON/dict snippets (`{"key": 1}`), glob braces (`{a,b}`), and f-string
# style (`{i}`) must never be rejected (post-merge audit of #81141).
_PLACEHOLDER_GOAL_RE = re.compile(r"^(todo|task\s*\d+)$", re.IGNORECASE)
_TEMPLATE_MARKER_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+>"
    r"|\{[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+\}"
)
_MIN_BATCH_GOAL_LEN = 10


def _validate_batch_tasks(task_list: List[Dict[str, Any]]) -> Optional[str]:
    """Validate a tasks=[...] batch beyond per-task goal presence.

    Returns an actionable error string, or None when the batch is valid.

    A one-entry array is the canonical single-task shape (the advertised
    interface is tasks-only; legacy top-level `goal` is wrapped into a
    one-entry batch), so no minimum count is enforced. The placeholder/
    template checks below still run on every entry.

    Duplicate goals are deliberately NOT rejected: identical-goal fan-outs
    are a legitimate pattern (best-of-N / ensemble sampling), and blocking
    them broke real workflows (post-merge audit of #81141).
    """

    for i, task in enumerate(task_list):
        goal = str(task.get("goal", "")).strip()
        normalized = " ".join(goal.lower().split())

        if _PLACEHOLDER_GOAL_RE.match(normalized):
            return (
                f"Task {i} has a placeholder goal ({goal!r}). Replace it "
                "with a specific, self-contained description of what the "
                "subagent should accomplish."
            )
        marker = _TEMPLATE_MARKER_RE.search(goal)
        if marker:
            return (
                f"Task {i} goal contains an unexpanded template marker "
                f"({marker.group(0)!r}). Substitute the real value before "
                "calling delegate_task — subagents cannot resolve "
                "placeholders."
            )
        if len(goal) < _MIN_BATCH_GOAL_LEN and len(task_list) >= 2:
            # Multi-task fan-outs with terse goals are usually unexpanded
            # templates; a SINGLE task legitimately uses short goals
            # ("Fix the tests"), so one-entry arrays keep the historical
            # single-`goal` exemption.
            return (
                f"Task {i} goal is too short ({goal!r}). Write a specific, "
                "self-contained goal of at least "
                f"{_MIN_BATCH_GOAL_LEN} characters so the subagent knows "
                "exactly what to do."
            )
    return None

def _resolve_child_credential_pool(
    effective_provider: Optional[str],
    parent_agent,
    effective_base_url: Optional[str] = None,
):
    """Resolve a credential pool for the child agent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the child keep the inherited
       fixed credential behavior.

    Custom endpoints are a special case: every direct ``delegation.base_url``
    runtime collapses to ``provider="custom"``, so bare provider equality would
    treat two *different* custom endpoints as interchangeable and let the child
    inherit the parent's pool. Leasing from that pool then overwrites the
    child's delegated ``base_url`` with the parent's endpoint (issue #7833).
    We therefore resolve custom runtimes by endpoint identity (the
    ``custom:<name>`` pool key derived from the base_url) and only share the
    parent's pool when both resolve to the *same* custom endpoint.
    """
    if not effective_provider:
        return getattr(parent_agent, "_credential_pool", None)

    parent_provider = getattr(parent_agent, "provider", None) or ""
    parent_pool = getattr(parent_agent, "_credential_pool", None)

    # Custom endpoints: distinguish by endpoint identity, not the bare "custom"
    # provider string. Two custom runtimes are only interchangeable when they
    # resolve to the same custom:<name> pool key.
    if effective_provider == "custom":
        try:
            from agent.credential_pool import get_custom_provider_pool_key, load_pool

            child_key = get_custom_provider_pool_key(effective_base_url)
            if child_key is None:
                # Unregistered endpoint (raw delegation.base_url with no
                # matching custom_providers entry) -> no shared pool exists.
                # Keep the child's fixed delegated credential rather than
                # risk inheriting the parent's custom endpoint.
                return None

            # Reuse the parent's pool only when it is the same custom endpoint.
            parent_key = get_custom_provider_pool_key(
                getattr(parent_agent, "base_url", None)
            )
            if (
                parent_pool is not None
                and parent_provider == "custom"
                and parent_key is not None
                and parent_key == child_key
            ):
                return parent_pool

            pool = load_pool(child_key)
            if pool is not None and pool.has_credentials():
                return pool
        except Exception as exc:
            logger.debug(
                "Could not resolve custom credential pool for child endpoint '%s': %s",
                effective_base_url,
                exc,
            )
        return None

    if parent_pool is not None and effective_provider == parent_provider:
        return parent_pool

    try:
        from agent.credential_pool import load_pool

        pool = load_pool(effective_provider)
        if pool is not None and pool.has_credentials():
            return pool
    except Exception as exc:
        logger.debug(
            "Could not load credential pool for child provider '%s': %s",
            effective_provider,
            exc,
        )
    return None


def _merge_request_overrides(runtime_overrides, explicit_overrides):
    """Merge explicit ``delegation.request_overrides`` over runtime-derived ones.

    Precedence contract: the explicit config key WINS over runtime-derived
    (provider-catalog or parent-inherited) overrides. Top-level keys from the
    explicit dict replace same-named runtime keys; the ``extra_body`` sub-dict
    is deep-merged ONE level — runtime ``extra_body`` keys survive unless the
    explicit dict redefines that exact key. This keeps provider personality
    (e.g. ``thinking: {type: disabled}``) intact while letting users layer
    routing hints (e.g. ``extra_body.provider = {"sort": "throughput"}``) on
    top.

    Both inputs are deep-copied (``copy.deepcopy``) so transport-side mutation
    of the child's request kwargs can never leak back into the loaded config
    dict or the provider runtime cache.

    Returns ``None`` when both sides are empty/non-dict.
    """
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


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.base_url`` is configured, subagents use that direct
    OpenAI-compatible endpoint. ``delegation.api_key`` overrides the key; when
    omitted, ``api_key`` is returned as ``None`` so ``_build_child_agent``
    inherits the parent agent's key (``effective_api_key = override_api_key or
    parent_api_key``). This lets providers that store their key outside
    ``OPENAI_API_KEY`` (e.g. ``MINIMAX_API_KEY``, ``DASHSCOPE_API_KEY``) work
    without a duplicate config entry.

    Otherwise, if ``delegation.provider`` is configured, the full credential
    bundle (base_url, api_key, api_mode, provider) is resolved via the runtime
    provider system — the same path used by CLI/gateway startup. This lets
    subagents run on a completely different provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None
    configured_api_mode = str(cfg.get("api_mode") or "").strip().lower() or None

    # delegation.request_overrides: explicit per-child request settings from
    # config. Honored on EVERY resolution branch (direct base_url, named
    # provider, and parent-inherit) so the key never silently no-ops.
    # Precedence: explicit merges OVER runtime/parent-derived overrides via
    # _merge_request_overrides (top-level explicit keys win; extra_body is
    # deep-merged one level). Non-dict values are ignored.
    explicit_request_overrides = (
        cfg.get("request_overrides")
        if isinstance(cfg.get("request_overrides"), dict)
        else None
    )

    # Native-SDK providers (Bedrock, Vertex, Google GenAI) speak their own
    # wire protocol — they cannot be reached via OpenAI chat_completions against
    # a base_url. For these, always fall through to resolve_runtime_provider()
    # so the proper SDK path is taken. The configured base_url is still
    # forwarded through runtime-provider resolution when applicable (e.g. a
    # custom Bedrock regional endpoint).
    _NATIVE_SDK_PROVIDERS = {"bedrock", "vertex", "google", "google-genai"}
    _provider_lower = (configured_provider or "").strip().lower()
    _is_native_sdk_provider = _provider_lower in _NATIVE_SDK_PROVIDERS

    if configured_base_url and not _is_native_sdk_provider:
        # delegation.request_overrides: an explicit dict of per-child request
        # settings merged into the child's API kwargs by the transport's
        # profile path. Keys are top-level kwargs (e.g. service_tier); an
        # "extra_body" sub-dict is merged into extra_body. This is how a
        # direct-endpoint delegation (provider=custom) forwards OpenRouter
        # routing hints such as extra_body.provider = {"sort": "throughput"}
        # to its children — the child's CustomProfile does not emit provider
        # preferences, and the parent-inheritance path is deliberately cleared
        # when delegation.provider/base_url overrides the parent (see the
        # provider-preference clearing in _build_child_agent).
        #
        # Precedence: explicit delegation.request_overrides MERGES OVER any
        # runtime-derived overrides (see _merge_request_overrides) — top-level
        # explicit keys win; extra_body is deep-merged one level so runtime
        # extra_body keys survive unless the explicit key redefines them.
        # (explicit_request_overrides is parsed once at the top of this
        # function and applied to every branch.)

        # When delegation.api_key is not set, return None so _build_child_agent
        # falls back to the parent agent's API key via the credential inheritance
        # path (effective_api_key = override_api_key or parent_api_key). This
        # lets providers that store their key in a non-OPENAI_API_KEY env var
        # (e.g. MINIMAX_API_KEY, DASHSCOPE_API_KEY) work without requiring
        # callers to duplicate the key under delegation.api_key.
        api_key = configured_api_key  # None → inherited from parent in _build_child_agent

        # Use the shared URL-based api_mode detector (same path the main agent's
        # runtime resolver uses) so Anthropic-compatible direct endpoints with a
        # /anthropic suffix — Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM
        # proxies — pick the right transport automatically. Without this,
        # subagents would default to chat_completions and hit 404s on endpoints
        # that only speak the Anthropic Messages protocol. Fixes #10213.
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = _detect_api_mode_for_url(configured_base_url) or "chat_completions"
        if (
            base_url_hostname(configured_base_url) == "chatgpt.com"
            and "/backend-api/codex" in base_lower
        ):
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif base_url_hostname(configured_base_url) == "api.anthropic.com":
            provider = "anthropic"
            api_mode = "anthropic_messages"
        elif "api.kimi.com/coding" in base_lower:
            provider = "custom"
            api_mode = "anthropic_messages"

        # Explicit delegation.api_mode in config always wins. Lets users force
        # a transport for non-standard endpoints the URL heuristic can't detect.
        if configured_api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            api_mode = configured_api_mode

        # A provider configured ALONGSIDE base_url means the user wants that
        # provider's request personality on an explicit endpoint. This
        # short-circuit runs before the resolve_runtime_provider() call below,
        # so without this block the runtime-carried request_overrides
        # (extra_body / extra_headers, e.g. `thinking: {type: disabled}`) and
        # max_output_tokens are silently dropped for subagents (#65035).
        # Best-effort: the explicit endpoint worked before this change even
        # when the provider can't resolve, so a resolution failure only skips
        # the overrides — it must not fail the dispatch.
        request_overrides = None
        max_output_tokens = None
        if configured_provider:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider

                runtime = resolve_runtime_provider(
                    requested=configured_provider, target_model=configured_model
                )
                request_overrides = dict(runtime.get("request_overrides") or {}) or None
                max_output_tokens = runtime.get("max_output_tokens")
            except Exception as exc:
                logger.debug(
                    "delegation.base_url: runtime resolution for provider '%s' "
                    "failed; proceeding without request_overrides: %s",
                    configured_provider,
                    exc,
                )

        # Explicit delegation.request_overrides merges OVER the runtime-derived
        # overrides (explicit wins; extra_body deep-merged one level).
        request_overrides = _merge_request_overrides(
            request_overrides, explicit_request_overrides
        )

        return {
            "model": configured_model,
            "provider": provider,
            "base_url": configured_base_url,
            "api_key": api_key,
            "api_mode": api_mode,
            "request_overrides": request_overrides,
            "max_output_tokens": max_output_tokens,
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent.
        # delegation.request_overrides still applies: merge the explicit key
        # OVER the parent's own request_overrides so the config key works even
        # in pure-inherit setups (never a silent no-op). None when neither
        # side has values → _build_child_agent falls back to the parent's
        # request_overrides unchanged.
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": _merge_request_overrides(
                getattr(parent_agent, "request_overrides", None),
                explicit_request_overrides,
            ),
            "max_output_tokens": None,
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=configured_provider, target_model=configured_model)
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

    # A pinned ACP transport command must exist — refuse the spawn loudly
    # rather than letting the child silently fall back to another transport
    # (#80450).
    pinned_command = runtime.get("command")
    if pinned_command:
        import shutil as _shutil

        if not _shutil.which(pinned_command):
            raise ValueError(
                f"Delegation provider '{configured_provider}' is pinned to the "
                f"'{pinned_command}' command, which was not found on PATH. "
                f"Install it or choose a different delegation provider."
            )

    return {
        "model": configured_model or runtime.get("model") or None,
        "provider": configured_provider if runtime.get("provider") == _RUNTIME_PROVIDER_CUSTOM else runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        # Explicit delegation.request_overrides merges OVER the named
        # provider's runtime overrides (explicit wins; extra_body deep-merged
        # one level) — same precedence as the direct-base_url branch above.
        "request_overrides": _merge_request_overrides(
            runtime.get("request_overrides"), explicit_request_overrides
        )
        or {},
        "max_output_tokens": runtime.get("max_output_tokens"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _load_config() -> dict:
    """Load delegation config from the active Hermes config.

    Prefer the shared persistent loader because it follows the active
    HERMES_HOME/profile. ``cli.CLI_CONFIG`` is a legacy fallback for entry
    points that cannot import the shared loader; importing it first can return
    an old default ``delegation`` block and hide user-set keys such as
    ``max_concurrent_children``.

    Uses ``load_config_readonly()``: every consumer of this dict is read-only
    (``.get()`` lookups), and this runs on each ``get_definitions()`` schema
    rebuild via ``_get_max_concurrent_children``, so skipping the defensive
    deepcopy matters. Do NOT mutate the returned dict.

    ``HERMES_IGNORE_USER_CONFIG=1`` (``hermes chat --ignore-user-config``) is
    only honored by the legacy ``cli`` loader, not the shared one, so when the
    flag is set we keep ``cli.CLI_CONFIG`` authoritative to preserve the
    flag's contract of suppressing user config.yaml settings.
    """
    prefer_legacy = os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"
    if not prefer_legacy:
        try:
            from hermes_cli.config import load_config_readonly

            full = load_config_readonly()
            cfg = full.get("delegation") or {}
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


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------


def _build_top_level_description() -> str:
    """Compose the delegate_task tool description.

    Deliberately carries ONLY guidance that exists nowhere else in the
    schema. Batch/concurrency limits live in the 'tasks' parameter
    description and the nesting clause lives in the 'role' parameter
    description (both rebuilt per get_definitions() call with the user's
    actual delegation.max_concurrent_children / max_spawn_depth), so the
    top-level text stays static and duplication-free. If you add text
    here, check it is not already stated in a parameter description.
    """
    try:
        orchestration_available = _get_max_spawn_depth() >= 2 and _get_orchestrator_enabled()
    except Exception:
        orchestration_available = False

    # The child-restrictions rule renders per config: on nesting-enabled
    # installs the orchestrator clause is load-bearing; on depth-1/disabled
    # installs (the default) it would describe an unreachable state — the
    # role param already explains that 'orchestrator' is inert there.
    # send_message is deliberately not named: it's gateway-internal
    # vocabulary most sessions never see. The list below is the fail-safe
    # superset; model_tools session-filters it to the tools the session
    # actually has, dropping the whole line when none apply.
    # Delegation capability is depth-derived (no role param): mention
    # recursion only where it's actually available.
    if orchestration_available:
        restrictions_rule = (
            "- Children cannot call clarify, memory, or cronjob.\n"
            "- Children can themselves delegate while depth remains "
            f"(max_spawn_depth={_get_max_spawn_depth()}); the runtime "
            "derives this from depth automatically.\n"
        )
    else:
        restrictions_rule = (
            "- Children cannot call delegate_task, clarify, memory, or "
            "cronjob.\n"
        )

    return (
        "Spawn subagents in isolated contexts; each gets its own conversation, "
        "terminal session, and toolset, and only its final summary returns to "
        "you. Pass every task in `tasks` — one entry spawns one subagent, "
        "several run in parallel (limit in the tasks description).\n\n"
        "Runs in the background: dispatch returns immediately with live "
        "transcript paths, and the completed result (one consolidated message, "
        "results in task order) re-enters the conversation on its own. Do NOT "
        "wait or poll; continue other work. While children run, `action` "
        "(list/steer/stop) controls them live — steer when a transcript shows "
        "a child drifting.\n\n"
        "USE FOR: reasoning-heavy subtasks, work that would flood your context "
        "with intermediate data, or independent parallel workstreams.\n"
        "DO NOT USE FOR (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
        "- A single tool call -> call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot ask questions\n"
        "- Durable work that must survive this session -> cronjob or "
        "terminal(background=True, notify=True); /stop, /new, or "
        "process exit discards running subagents.\n\n"
        "RULES:\n"
        "- Children know nothing of this conversation: pass everything needed "
        "via 'context', including any required output language, tone, or "
        "style (e.g. \"respond in Chinese\").\n"
        + DELEGATION_OUTCOME_TOOL_GUIDANCE
        + "- Child summaries are SELF-REPORTS, not verified facts: a child "
        "claiming \"uploaded successfully\" or \"file written\" may be wrong. "
        "For external side effects (uploads, remote writes, publishing), "
        "require a verifiable handle (URL, ID, absolute path) and verify it "
        "yourself before telling the user the operation succeeded.\n"
        + restrictions_rule +
        "- Children inherit the parent model unless pinned via "
        "delegation.provider / delegation.model in config.yaml."
    )


def _build_tasks_param_description() -> str:
    """Compose the 'tasks' parameter description with current concurrency limit."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    return (
        f"The task(s), up to {max_children} in parallel for this user (set "
        "via delegation.max_concurrent_children). Each entry spawns one "
        "subagent with isolated context and terminal session; a single task "
        "is a one-entry array. Required when spawning."
    )


def _build_role_param_description() -> str:
    """Legacy helper — the `role` param is no longer advertised.

    Delegation capability is depth-derived (see the role-resolution block in
    _build_child_agent): a child may itself delegate iff
    delegation.orchestrator_enabled and its depth < max_spawn_depth. The
    handler still accepts role for wire compat (old transcripts, kanban
    dispatcher) but ignores it. Kept because external callers import this
    symbol; returns the depth story for any such use.
    """
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    return (
        "Legacy parameter, ignored: whether a child can delegate is derived "
        f"from delegation config (max_spawn_depth={max_depth}), not declared "
        "by the caller."
    )


def _build_dynamic_schema_overrides() -> dict:
    """Return per-call schema overrides reflecting current config.

    Plugged into ToolEntry.dynamic_schema_overrides so every
    get_definitions() pass rewrites the description fields to the user's
    actual limits.
    """
    overrides_params = {
        **DELEGATE_TASK_SCHEMA["parameters"],
    }
    # Deep-copy properties so we don't mutate the static schema dict.
    overrides_params["properties"] = {
        k: dict(v) for k, v in DELEGATE_TASK_SCHEMA["parameters"]["properties"].items()
    }
    overrides_params["properties"]["tasks"]["description"] = _build_tasks_param_description()

    return {
        "description": _build_top_level_description(),
        "parameters": overrides_params,
    }


DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    # NOTE: description / tasks.description / role.description are placeholder
    # values. The real text is generated per get_definitions() call by
    # _build_dynamic_schema_overrides() (registered via
    # dynamic_schema_overrides below) so the model sees the user's actual
    # delegation.max_concurrent_children / max_spawn_depth, not the framework
    # defaults. Building these lazily (instead of at module import) also
    # avoids forcing cli.CLI_CONFIG to load before the test conftest can
    # redirect HERMES_HOME.
    "description": (
        "Spawn one or more subagents in isolated contexts. "
        "Description is rebuilt at every get_definitions() call to reflect "
        "the user's current delegation limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            # NOTE: the handler also accepts the legacy single-goal shape —
            # top-level `goal` (string), `context` (string), `output_schema`
            # (object) — wrapped into a one-entry batch at dispatch. Legacy,
            # unadvertised (old transcripts/callers only); tasks=[...] is the
            # only advertised shape. Do not re-add these to the schema.
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": (
                                "What this subagent should accomplish. Be "
                                "specific and self-contained — it knows "
                                "nothing about your conversation history."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Background THIS child needs: file paths, "
                                "error messages, constraints. Each child "
                                "sees only its own context — repeat shared "
                                "background in every task that needs it."
                            ),
                        },
                        "output_schema": {
                            "type": "object",
                            "description": (
                                "Optional JSON Schema this child's final "
                                "answer must validate against (told to the "
                                "child up front; parent validates with one "
                                "bounded correction retry; result gains "
                                "schema_valid, plus schema_errors on "
                                "failure). Keep it forgiving — require only "
                                "fields you will read."
                            ),
                        },
                    },
                    "required": ["goal"],
                },
                # No maxItems — the runtime limit is configurable via
                # delegation.max_concurrent_children (default 3) and
                # enforced with a clear error in delegate_task().
                # NOTE: the handler also accepts a per-task `role` — legacy,
                # ignored: delegation capability is depth-derived, not
                # caller-declared. Unadvertised on purpose; do not re-add.
                "description": "(rebuilt at get_definitions() time)",
            },
            # NOTE: the handler also accepts `background` (bool) — DEPRECATED,
            # ignored: top-level delegations always run in the background.
            # Deliberately unadvertised (old transcripts/callers only); do not
            # re-add to the schema.
            "action": {
                "type": "string",
                "enum": ["spawn", "list", "steer", "stop"],
                "description": (
                    "Default 'spawn'. Live control of running children: "
                    "'list' = ids/goals/status/transcripts; 'steer' = queue "
                    "course-correction text into one child (subagent_id + "
                    "message) without stopping it; 'stop' = end one child "
                    "early (subagent_id; partial result still returns). "
                    "Control actions return immediately; goal/tasks are "
                    "ignored unless spawning."
                ),
            },
            "subagent_id": {
                "type": "string",
                "description": (
                    "Target for action='steer'/'stop' (ids from the spawn "
                    "response or action='list')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "For action='steer': the course correction, appended to "
                    "the child's next tool result mid-run. Be directive and "
                    "specific."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error


def _model_background_value(args: dict, parent_agent=None) -> bool:
    """Background flag for the MODEL-facing dispatch path (registry fallback).

    Delegations from the top-level agent always run in the background — the
    model does not choose. This applies to both a single task and a fan-out
    batch (the whole batch is one async unit that joins on all children and
    returns one consolidated result). The one
    exception is a delegation from an orchestrator subagent (depth > 0), which
    needs its workers' results within its own turn. The live path is
    ``run_agent._dispatch_delegate_task``; this lambda mirrors it for the rare
    case the intercept is bypassed. Direct Python callers of ``delegate_task``
    keep the historical synchronous default.
    """
    is_subagent = getattr(parent_agent, "_delegate_depth", 0) > 0
    return not is_subagent


_MODEL_HIDDEN_TASK_FIELDS = {"acp_command", "acp_args"}


def _strip_model_hidden_task_fields(tasks: Any) -> Any:
    if not isinstance(tasks, list):
        return tasks
    stripped_tasks = []
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            stripped_tasks.append(task)
            continue
        stripped = {
            key: value
            for key, value in task.items()
            if key not in _MODEL_HIDDEN_TASK_FIELDS
        }
        changed = changed or len(stripped) != len(task)
        stripped_tasks.append(stripped)
    return stripped_tasks if changed else tasks
