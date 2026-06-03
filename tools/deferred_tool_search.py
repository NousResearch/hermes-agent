#!/usr/bin/env python3
"""Feature-flagged deferred tool_search / schema promotion (prototype).

Background
----------
Large tool surfaces cost tokens on *every* request: each tool's JSON schema is
re-sent in the ``tools`` array of every model call. The pattern documented by
bytedance/deer-flow (MIT, rev e683ed6a7683ed298ed2fea470a1a76ceb8b9203)
addresses this by *deferring* most tool schemas: only a small eager core plus a
single ``tool_search`` tool is advertised up front. The model calls
``tool_search`` to "promote" the schemas it actually needs into the live tool
surface, and only then can call them.

This module is a Hermes-native reimplementation of that *pattern* (no upstream
code is copied). It is **disabled by default** and gated behind a feature flag.
When disabled, every public entry point is an explicit no-op so the agent's
tool surface is byte-for-byte identical to today.

Design constraints (from the implementation plan / card t_9b28d3c2)
-------------------------------------------------------------------
1. No module-global promotion state. All per-session state lives in a
   caller-owned :class:`DeferredToolState` instance (held on the agent).
   ``model_tools.get_tool_definitions`` and its memo cache are never mutated.
2. ``enabled_toolsets`` / MCP include-exclude remain the *coarse* permission
   gate. Deferred search only hides schemas the session already has access to;
   it is **not** a security boundary. A deferred tool is still dispatchable by
   the runtime — promotion only controls model-facing advertisement.
3. Kanban lifecycle tools and agent-loop tools are always eager and can never
   be deferred.
4. A direct call to an un-promoted deferred tool is blocked with an actionable
   JSON error telling the model to call ``tool_search`` first.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Feature flag
# =============================================================================

#: Environment override. When set to a truthy value it wins over config; when
#: set to a falsy value it force-disables. Unset → fall back to config.
FEATURE_FLAG_ENV = "HERMES_DEFERRED_TOOL_SEARCH"

#: Config location: ``tools.deferred_tool_search.enabled`` (bool, default False)
_CONFIG_SECTION = ("tools", "deferred_tool_search")

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"0", "false", "no", "off", "disabled", ""}


def _env_flag() -> Optional[bool]:
    """Return the env-var override as a tri-state (True / False / None)."""
    raw = os.environ.get(FEATURE_FLAG_ENV)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    # Unrecognised value — treat as unset rather than guessing.
    logger.debug("Unrecognised %s value %r; ignoring", FEATURE_FLAG_ENV, raw)
    return None


def _load_config_section(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve the ``tools.deferred_tool_search`` config sub-tree.

    Accepts an already-loaded config (tests / hot callers pass one to avoid a
    disk read) and otherwise reads the user config lazily. Import of the config
    module is deferred to call time to avoid an import cycle at module load.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("deferred_tool_search: config load failed: %s", exc)
            return {}
    node: Any = config
    for key in _CONFIG_SECTION:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when deferred tool search is turned on.

    Resolution order: env override (``HERMES_DEFERRED_TOOL_SEARCH``) → config
    (``tools.deferred_tool_search.enabled``) → default ``False``.
    """
    env = _env_flag()
    if env is not None:
        return env
    section = _load_config_section(config)
    return bool(section.get("enabled", False))


# =============================================================================
# Eager-keep policy
# =============================================================================

#: Tools intercepted by the agent loop (run_agent / tool_executor). These need
#: agent-level state and must always be advertised eagerly.
AGENT_LOOP_EAGER_TOOLS: Set[str] = {
    "todo",
    "memory",
    "session_search",
    "delegate_task",
    "clarify",
}

#: Kanban multi-agent lifecycle tools — a dispatcher-spawned worker must always
#: see its completion/block/heartbeat surface, so these are never deferred.
KANBAN_EAGER_TOOLS: Set[str] = {
    "kanban_show",
    "kanban_list",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
    "kanban_create",
    "kanban_link",
    "kanban_unblock",
}

#: The search tool itself is always eager (it is the promotion entry point).
TOOL_SEARCH_NAME = "tool_search"

#: Tools that may never be deferred regardless of config. The agent loop relies
#: on these existing on the live surface.
MANDATORY_EAGER: Set[str] = AGENT_LOOP_EAGER_TOOLS | KANBAN_EAGER_TOOLS | {TOOL_SEARCH_NAME}


def _configured_extra_eager(config: Optional[Dict[str, Any]]) -> Set[str]:
    """Return operator-configured ``always_eager`` tool names (optional)."""
    section = _load_config_section(config)
    extra = section.get("always_eager")
    if isinstance(extra, (list, tuple, set)):
        return {str(x) for x in extra}
    return set()


def eager_tool_names(
    config: Optional[Dict[str, Any]] = None,
    *,
    extra: Optional[Set[str]] = None,
) -> Set[str]:
    """Compute the full set of tool names that stay eager (never deferred)."""
    return set(MANDATORY_EAGER) | _configured_extra_eager(config) | set(extra or set())


def _agent_dynamic_eager_names(agent) -> Set[str]:
    """Return session-local agent-loop tool names that must stay eager.

    Memory-provider and context-engine schemas are injected at agent init and
    dispatched by the agent loop rather than by the global registry. They are
    still subject to the coarse enabled_toolsets gate before injection, but once
    injected they must remain visible instead of being hidden behind deferred
    search.
    """
    names: Set[str] = set()
    context_names = getattr(agent, "_context_engine_tool_names", set()) or set()
    if isinstance(context_names, (set, list, tuple)):
        names.update(str(n) for n in context_names if n)
    manager = getattr(agent, "_memory_manager", None)
    get_memory_names = getattr(manager, "get_all_tool_names", None)
    if callable(get_memory_names):
        try:
            names.update(get_memory_names() or set())
        except Exception:  # pragma: no cover - defensive
            logger.debug("deferred_tool_search: failed to read memory tool names", exc_info=True)
    return {str(n) for n in names if n}


# =============================================================================
# tool_search schema
# =============================================================================

TOOL_SEARCH_SCHEMA: Dict[str, Any] = {
    "name": TOOL_SEARCH_NAME,
    "description": (
        "Search for and promote additional tools into your available tool set. "
        "Most tools are deferred to save context: their full schemas are not "
        "shown until you promote them. Call this with a short natural-language "
        "query describing the capability you need (e.g. 'search the web', "
        "'edit a file', 'send a message'). Matching tools are promoted and "
        "their schemas become available on the next turn so you can call them "
        "directly. Calling a deferred tool before promoting it is blocked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the capability you need. "
                    "Keywords are matched against tool names and descriptions."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of tools to promote (default 5). The "
                    "highest-scoring matches are promoted."
                ),
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": ["query"],
    },
}


def tool_search_definition() -> Dict[str, Any]:
    """Return the OpenAI-format tool definition for ``tool_search``."""
    return {"type": "function", "function": dict(TOOL_SEARCH_SCHEMA)}


# =============================================================================
# Per-session state
# =============================================================================

def _def_name(tool_def: Dict[str, Any]) -> Optional[str]:
    """Extract a tool name from an OpenAI-format tool definition."""
    fn = tool_def.get("function") if isinstance(tool_def, dict) else None
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    return None


class DeferredToolState:
    """Per-session deferred-tool bookkeeping.

    Holds the schemas of currently-deferred tools and the set the model has
    promoted via ``tool_search``. Owned by a single agent/session — there is no
    cross-session sharing and no module-global registry of these.
    """

    def __init__(
        self,
        deferred_defs: Dict[str, Dict[str, Any]],
        eager_names: Set[str],
        promoted: Optional[Set[str]] = None,
    ) -> None:
        # name -> full {"type": "function", "function": {...}} definition
        self._deferred: Dict[str, Dict[str, Any]] = dict(deferred_defs)
        self._eager_names: Set[str] = set(eager_names)
        # Only keep promotions that still correspond to a known deferred tool.
        self._promoted: Set[str] = {
            n for n in (promoted or set()) if n in self._deferred
        }

    # -- queries ----------------------------------------------------------
    @property
    def deferred_names(self) -> Set[str]:
        return set(self._deferred)

    @property
    def promoted_names(self) -> Set[str]:
        return set(self._promoted)

    @property
    def eager_names(self) -> Set[str]:
        return set(self._eager_names)

    def is_deferred(self, name: str) -> bool:
        return name in self._deferred

    def is_promoted(self, name: str) -> bool:
        return name in self._promoted

    def is_blocked(self, name: str) -> bool:
        """A call is blocked iff the tool is deferred and not yet promoted."""
        return name in self._deferred and name not in self._promoted

    def deferred_definition(self, name: str) -> Optional[Dict[str, Any]]:
        return self._deferred.get(name)

    def promoted_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions for every currently-promoted tool."""
        return [self._deferred[n] for n in self._deferred if n in self._promoted]

    # -- mutation ---------------------------------------------------------
    def promote(self, names) -> List[str]:
        """Mark *names* promoted; return the names newly promoted this call."""
        newly: List[str] = []
        for name in names:
            if name in self._deferred and name not in self._promoted:
                self._promoted.add(name)
                newly.append(name)
        return newly


# =============================================================================
# Partition / reapply
# =============================================================================

def partition(
    tool_defs: List[Dict[str, Any]],
    *,
    config: Optional[Dict[str, Any]] = None,
    prior_state: Optional[DeferredToolState] = None,
    extra_eager: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], DeferredToolState]:
    """Split *tool_defs* into the model-facing surface + deferred state.

    Returns ``(visible_defs, state)`` where ``visible_defs`` is the eager core
    (plus any already-promoted tools and the ``tool_search`` tool) and *state*
    captures the deferred remainder.

    This is a pure function over *tool_defs*: the input list is not mutated and
    the returned ``visible_defs`` is a fresh list of the original dict
    references (schemas are treated read-only by all callers, matching
    ``get_tool_definitions``' contract).

    ``prior_state`` lets a refresh (MCP/ACP server change rebuilding the tool
    list) carry promotions forward: any tool the model already promoted stays
    visible if it still exists after the refresh.
    """
    eager = eager_tool_names(config, extra=extra_eager)
    carried_promotions: Set[str] = (
        set(prior_state.promoted_names) if prior_state is not None else set()
    )

    visible: List[Dict[str, Any]] = []
    deferred_defs: Dict[str, Dict[str, Any]] = {}
    seen_tool_search = False

    for td in tool_defs:
        name = _def_name(td)
        if name is None:
            # Malformed/unnamed definition — keep it visible (fail-open); it
            # cannot be searched for anyway.
            visible.append(td)
            continue
        if name == TOOL_SEARCH_NAME:
            seen_tool_search = True
            visible.append(td)
            continue
        if name in eager:
            visible.append(td)
        else:
            deferred_defs[name] = td

    state = DeferredToolState(deferred_defs, eager, promoted=carried_promotions)

    # Re-add promoted tools to the visible surface, preserving input order.
    if state.promoted_names:
        for td in tool_defs:
            name = _def_name(td)
            if name in state.promoted_names:
                visible.append(td)

    # Ensure tool_search is always advertised.
    if not seen_tool_search:
        visible.append(tool_search_definition())

    return visible, state


def visible_after_promotion(
    base_visible: List[Dict[str, Any]],
    newly_promoted_defs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append newly-promoted tool defs to a live surface, de-duplicating names.

    Used by the agent loop after a ``tool_search`` call to extend
    ``agent.tools`` in place-safe fashion (returns a new list).
    """
    existing = {_def_name(td) for td in base_visible}
    out = list(base_visible)
    for td in newly_promoted_defs:
        if _def_name(td) not in existing:
            out.append(td)
            existing.add(_def_name(td))
    return out


# =============================================================================
# Search / promotion
# =============================================================================

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _score(query_tokens: List[str], name: str, description: str) -> float:
    """Deterministic keyword overlap score between a query and a tool.

    Name matches are weighted higher than description matches. The score is a
    plain sum so results are stable and explainable (no RNG, no embeddings).
    """
    if not query_tokens:
        return 0.0
    name_tokens = set(_tokenize(name))
    name_compact = name.lower().replace("_", "")
    desc_tokens = set(_tokenize(description))
    score = 0.0
    for qt in query_tokens:
        if qt in name_tokens:
            score += 3.0
        elif qt in name_compact:
            score += 2.0
        if qt in desc_tokens:
            score += 1.0
    return score


def search(
    query: str,
    state: DeferredToolState,
    limit: int = 5,
) -> Dict[str, Any]:
    """Rank deferred tools against *query* and promote the top matches.

    Returns a JSON-serialisable result dict describing what was promoted. This
    mutates *state* (the caller's per-session object) but nothing global.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "error": "tool_search requires a non-empty 'query' string.",
            "promoted": [],
        }
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 25))

    query_tokens = _tokenize(query)
    ranked: List[Tuple[float, str, str]] = []
    for name in sorted(state.deferred_names):
        td = state.deferred_definition(name) or {}
        fn = td.get("function", {}) if isinstance(td, dict) else {}
        description = fn.get("description", "") if isinstance(fn, dict) else ""
        s = _score(query_tokens, name, description)
        if s > 0:
            ranked.append((s, name, description))

    # Highest score first; stable tiebreak on name for determinism.
    ranked.sort(key=lambda r: (-r[0], r[1]))
    top = ranked[:limit]

    already = sorted(n for _, n, _ in top if state.is_promoted(n))
    newly = state.promote([n for _, n, _ in top])

    matches = [
        {
            "name": name,
            "score": round(score, 2),
            "description": (description[:200] if description else ""),
        }
        for score, name, description in top
    ]

    if not top:
        message = (
            f"No deferred tools matched '{query}'. "
            f"{len(state.deferred_names)} tools are deferred; try different "
            "keywords describing the capability you need."
        )
    else:
        message = (
            f"Promoted {len(newly)} tool(s): {', '.join(newly) if newly else '(none new)'}. "
            "Their schemas are now available — you can call them directly."
        )

    return {
        "query": query,
        "promoted": newly,
        "already_promoted": already,
        "matches": matches,
        "deferred_remaining": len(state.deferred_names) - len(state.promoted_names),
        "message": message,
    }


# =============================================================================
# Blocking direct calls to un-promoted deferred tools
# =============================================================================

def block_message(name: str, state: Optional[DeferredToolState]) -> Optional[str]:
    """Return an actionable JSON error string if *name* is blocked, else None.

    A tool is blocked when deferred search is active for the session and the
    tool is deferred-but-not-yet-promoted. The returned string is ready to be
    used as the ``tool`` message content.
    """
    if state is None:
        return None
    if not state.is_blocked(name):
        return None
    return json.dumps(
        {
            "error": "deferred_tool_not_promoted",
            "tool": name,
            "message": (
                f"The tool '{name}' is deferred and has not been promoted in "
                "this session. Call 'tool_search' with a query describing the "
                f"capability (e.g. matching '{name}') to promote it, then call "
                f"'{name}' again."
            ),
            "next_action": {
                "tool": TOOL_SEARCH_NAME,
                "arguments": {"query": name.replace("_", " ")},
            },
        },
        ensure_ascii=False,
    )


# =============================================================================
# Agent integration helpers
# =============================================================================

def apply_to_agent(agent, *, config: Optional[Dict[str, Any]] = None) -> None:
    """Partition ``agent.tools`` in place when the feature is enabled.

    Idempotent and safe to call on every (re)build of the agent's tool surface
    (init, MCP/ACP refresh). When the flag is off it clears any prior state and
    leaves ``agent.tools`` untouched. When on, it carries forward promotions
    from a prior state so a mid-session MCP refresh doesn't un-promote tools.
    """
    if not is_enabled(config):
        # Disabled: restore a previously-partitioned live surface before
        # clearing state so runtime feature-flag flips cannot strand an agent
        # with only the reduced deferred-visible tool list.
        full_surface = getattr(agent, "_deferred_tool_full_surface", None)
        if getattr(agent, "_deferred_tool_surface_partitioned", False) and full_surface:
            agent.tools = list(full_surface)
            try:
                agent.valid_tool_names = {
                    _def_name(td) for td in agent.tools if _def_name(td) is not None
                }
            except Exception:  # pragma: no cover - defensive
                pass
        if getattr(agent, "_deferred_tool_state", None) is not None:
            agent._deferred_tool_state = None
        agent._deferred_tool_full_surface = None
        agent._deferred_tool_surface_partitioned = False
        return

    prior = getattr(agent, "_deferred_tool_state", None)
    already_partitioned = bool(getattr(agent, "_deferred_tool_surface_partitioned", False))
    if already_partitioned and prior is not None:
        tools = list(getattr(agent, "_deferred_tool_full_surface", None) or getattr(agent, "tools", None) or [])
    else:
        tools = list(getattr(agent, "tools", None) or [])
    extra_eager = _agent_dynamic_eager_names(agent)

    visible, state = partition(
        tools,
        config=config,
        prior_state=prior,
        extra_eager=extra_eager,
    )
    agent.tools = visible
    agent._deferred_tool_state = state
    agent._deferred_tool_full_surface = list(tools)
    agent._deferred_tool_surface_partitioned = True
    try:
        agent.valid_tool_names = {
            _def_name(td) for td in visible if _def_name(td) is not None
        }
    except Exception:  # pragma: no cover - defensive
        pass


def handle_tool_search_call(agent, function_args: Dict[str, Any]) -> str:
    """Execute a ``tool_search`` call against the agent's per-session state.

    Promotes matching tools, extends ``agent.tools`` / ``agent.valid_tool_names``
    so the promoted schemas are advertised on the next turn, and returns the
    JSON result string. Intended to be wired into the agent loop's tool
    dispatch (alongside ``todo`` / ``memory`` / ``session_search``).
    """
    state: Optional[DeferredToolState] = getattr(agent, "_deferred_tool_state", None)
    if state is None:
        return json.dumps(
            {
                "error": "tool_search_unavailable",
                "message": (
                    "Deferred tool search is not active in this session; all "
                    "tools are already available."
                ),
            }
        )
    query = function_args.get("query", "")
    limit = function_args.get("limit", 5)
    result = search(query, state, limit=limit)

    newly = result.get("promoted") or []
    if newly:
        newly_defs = [
            state.deferred_definition(n) for n in newly if state.deferred_definition(n)
        ]
        agent.tools = visible_after_promotion(list(agent.tools or []), newly_defs)
        try:
            agent.valid_tool_names = {
                _def_name(td) for td in agent.tools if _def_name(td) is not None
            }
        except Exception:  # pragma: no cover - defensive
            pass
        invalidate = getattr(agent, "_invalidate_system_prompt", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:  # pragma: no cover - defensive
                pass

    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Registry registration
# =============================================================================
#
# ``tool_search`` is registered in its own toolset (NOT part of any default
# toolset) and gated on the feature flag via check_fn, so it never appears on
# the tool surface unless the feature is enabled. The agent loop intercepts
# the call (see handle_tool_search_call); this registry handler is a safe
# fallback for any path that dispatches straight through model_tools.
#
# NOTE: the ``registry.register(...)`` below must remain a *top-level*
# statement (not nested in a try/if) so that ``discover_builtin_tools``' AST
# scan recognises this file as a self-registering tool module and imports it.
def _registry_tool_search_handler(args: Dict[str, Any], **_kw) -> str:
    return json.dumps(
        {
            "error": "tool_search_not_routed",
            "message": (
                "tool_search must be handled by the agent loop (it mutates "
                "per-session promotion state)."
            ),
        }
    )


from tools.registry import registry

registry.register(
    name=TOOL_SEARCH_NAME,
    toolset=TOOL_SEARCH_NAME,
    schema=dict(TOOL_SEARCH_SCHEMA),
    handler=_registry_tool_search_handler,
    check_fn=is_enabled,
    emoji="🔎",
    description=TOOL_SEARCH_SCHEMA["description"],
)
