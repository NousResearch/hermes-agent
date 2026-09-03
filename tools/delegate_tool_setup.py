"""Child construction, progress, diagnostics, and summary helpers."""

import json
import logging
import os
import threading
import weakref
from typing import Any, Dict, List, Optional

from tools.delegate_tool_control import (
    DEFAULT_MAX_SUMMARY_CHARS,
    DEFAULT_TOOLSETS,
    DELEGATE_BLOCKED_TOOLS,
    DelegateEvent,
    SUBAGENT_FAILURE_STATUSES,
    TOOLSETS,
    _LEGACY_EVENT_MAP,
    _MIN_SUMMARY_CHARS,
    _SUMMARY_HEADROOM_FRACTION,
    _active_subagents,
    _active_subagents_lock,
    _expand_parent_toolsets,
    _get_inherit_mcp_toolsets,
    _preserve_parent_mcp_toolsets,
    format_subagent_failure_line,
)

logger = logging.getLogger("tools.delegate_tool")


def _facade():
    from tools import delegate_tool as facade

    return facade


def _load_config():
    return _facade()._load_config()


def _resolve_child_credential_pool(*args, **kwargs):
    return _facade()._resolve_child_credential_pool(*args, **kwargs)


def _get_max_spawn_depth():
    return _facade()._get_max_spawn_depth()


def _get_orchestrator_enabled():
    return _facade()._get_orchestrator_enabled()


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    role: str = "leaf",
    max_spawn_depth: int = 2,
    child_depth: int = 1,
) -> str:
    """Build a focused system prompt for a child agent.

    When role='orchestrator', appends a delegation-capability block
    modeled on OpenClaw's buildSubagentSystemPrompt (canSpawn branch at
    inspiration/openclaw/src/agents/subagent-system-prompt.ts:63-95).
    The depth note is literal truth (grounded in the passed config) so
    the LLM doesn't confabulate nesting capabilities that don't exist.
    """
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
        # Project context files (AGENTS.md / CLAUDE.md / .cursorrules ...)
        # from the workspace, via the SAME discovery/priority/cap logic the
        # main agent's system prompt uses. Children are constructed with
        # skip_context_files=True (their prompt is this focused one), so
        # without this a subagent works in a repo without the repo's own
        # conventions unless it thinks to go read them. SOUL.md is skipped —
        # identity belongs to the parent. workspace_path comes only from
        # explicit sources (_resolve_workspace_hint: TERMINAL_CWD / agent cwd
        # hints, never bare getcwd), so the #64590 install-tree-fallback leak
        # doesn't apply here. Best-effort: on any failure the child prompt is
        # simply built without the block.
        try:
            from agent.prompt_builder import build_context_files_prompt

            _ctx_files = build_context_files_prompt(
                cwd=str(workspace_path), skip_soul=True
            )
        except Exception:
            logger.debug(
                "subagent: workspace context-files load failed", exc_info=True
            )
            _ctx_files = ""
        if _ctx_files.strip():
            parts.append(
                "\nThe workspace's project context files are reproduced "
                "below. Their conventions and invariants are binding for "
                "your work in this workspace.\n\n" + _ctx_files.strip()
            )
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Keep your final summary tight: lead with outcomes, prefer bullet "
        "points over paragraphs, and don't replay your whole process. Your "
        "response is returned to the parent agent as a summary, and overlong "
        "summaries crowd out the parent's context window."
    )
    if role == "orchestrator":
        child_note = (
            "Your own children MUST be leaves (cannot delegate further) "
            "because they would be at the depth floor — you cannot pass "
            "role='orchestrator' to your own delegate_task calls."
            if child_depth + 1 >= max_spawn_depth
            else "Your own children can themselves be orchestrators or leaves, "
            "depending on the `role` you pass to delegate_task. Default is "
            "'leaf'; pass role='orchestrator' explicitly when a child "
            "needs to further decompose its work."
        )
        parts.append(
            "\n## Subagent Spawning (Orchestrator Role)\n"
            "You have access to the `delegate_task` tool and CAN spawn "
            "your own subagents to parallelize independent work.\n\n"
            "WHEN to delegate:\n"
            "- The goal decomposes into 2+ independent subtasks that can "
            "run in parallel (e.g. research A and B simultaneously).\n"
            "- A subtask is reasoning-heavy and would flood your context "
            "with intermediate data.\n\n"
            "WHEN NOT to delegate:\n"
            "- Single-step mechanical work — do it directly.\n"
            "- Trivial tasks you can execute in one or two tool calls.\n"
            "- Re-delegating your entire assigned goal to one worker "
            "(that's just pass-through with no value added).\n\n"
            "Coordinate your workers' results and synthesize them before "
            "reporting back to your parent. You are responsible for the "
            "final summary, not your workers.\n\n"
            f"NOTE: You are at depth {child_depth}. The delegation tree "
            f"is capped at max_spawn_depth={max_spawn_depth}. {child_note}"
        )
    return "\n".join(parts)


def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts.

    We only inject a path when we have a concrete absolute directory. This avoids
    teaching subagents a fake container path while still helping them avoid
    guessing `/workspace/...` for local repo tasks.
    """
    candidates = [
        os.getenv("TERMINAL_CWD"),
        getattr(
            getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None
        ),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            continue
        if os.path.isabs(text) and os.path.isdir(text):
            return text
    return None


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools.

    The strip set is derived from DELEGATE_BLOCKED_TOOLS plus the explicit
    composite/scenario toolsets (delegation, code_execution) that have no
    one-to-one tool. This keeps the blocklist and the strip set in lockstep
    so new blocked tools can't silently leak through as toolset names.
    """
    # Composite toolsets that should never pass through to children, even
    # though their individual tools aren't all in DELEGATE_BLOCKED_TOOLS.
    _COMPOSITE_BLOCKED_TOOLSETS = frozenset({"delegation"})
    blocked_toolset_names = {
        name
        for name, defn in TOOLSETS.items()
        if name in _COMPOSITE_BLOCKED_TOOLSETS
        or all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    blocked_toolset_names.add("kanban")
    return [t for t in toolsets if t not in blocked_toolset_names]


def _blocked_toolsets_for_role(role: str) -> List[str]:
    """Return one-tool deny toolsets for a delegated child role.

    ``_strip_blocked_tools`` can remove fully blocked toolsets, but it must keep
    mixed platform bundles such as ``hermes-cli`` because those also contain
    useful tools. Passing these exact deny toolsets to AIAgent lets
    ``model_tools`` subtract blocked names *after* composite expansion, and the
    restriction survives later registry/MCP refreshes through the agent's
    stored ``disabled_toolsets``.
    """
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    if role == "orchestrator":
        blocked_names.discard("delegate_task")
    return sorted(
        name
        for name, defn in TOOLSETS.items()
        if defn.get("tools")
        and set(defn.get("tools", ())).issubset(blocked_names)
    )


_BATCH_ORDINALS: Dict[str, int] = {}
_BATCH_ORDINALS_LOCK = threading.Lock()


def format_batch_tag(delegation_id: Optional[str]) -> str:
    """Return a stable, human-readable ordinal for a delegation batch."""
    if not isinstance(delegation_id, str) or not delegation_id:
        return ""
    # Keep the facade attribute as the compatibility/monkeypatch seam that
    # existed before these helpers were extracted from delegate_tool.py.
    facade = _facade()
    ordinals = getattr(facade, "_BATCH_ORDINALS", _BATCH_ORDINALS)
    with _BATCH_ORDINALS_LOCK:
        n = ordinals.get(delegation_id)
        if n is None:
            n = len(ordinals) + 1
            ordinals[delegation_id] = n
    return f"set {n}"


def _batch_prefix(
    delegation_id: Optional[str], task_index: int, task_count: int
) -> str:
    """Build progress prefix with optional delegation and task-position tags."""
    tag = format_batch_tag(delegation_id)
    if task_count > 1:
        inner = (
            f"{tag} · {task_index + 1}/{task_count}"
            if tag
            else f"{task_index + 1}/{task_count}"
        )
        return f"[{inner}] "
    return f"[{tag}] " if tag else ""


def _emit_parent_console(parent_agent, line: str) -> None:
    """Emit a human-readable progress line to the parent's console.

    Routes through ``parent_agent._safe_print`` when available so headless
    stdio hosts (ACP, gateway API) can redirect non-protocol output to
    stderr via their configured ``_print_fn``. A bare ``print()`` would
    otherwise land on stdout and corrupt JSON-RPC framing.
    """
    printer = getattr(parent_agent, "_safe_print", None)
    if callable(printer):
        try:
            printer(line)
            return
        except Exception:
            pass
    print(line)


def _build_child_progress_callback(
    task_index: int,
    goal: str,
    parent_agent,
    task_count: int = 1,
    *,
    subagent_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    depth: Optional[int] = None,
    model: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    session_ref: Optional[Dict[str, Any]] = None,
) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    The identity kwargs (``subagent_id``, ``parent_id``, ``depth``, ``model``,
    ``toolsets``) are threaded into every relayed event so the TUI can
    reconstruct the live spawn tree and route per-branch controls (kill,
    pause) back by ``subagent_id``.  All are optional for backward compat —
    older callers that ignore them still produce a flat list on the TUI.

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Resolve the short delegation tag lazily. The callback is built before
    # delegate_task stamps the id into this shared reference.
    def _prefix() -> str:
        delegation_id = session_ref.get("delegation_id") if session_ref else None
        return _batch_prefix(delegation_id, task_index, task_count)

    goal_label = (goal or "").strip()

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []
    _tool_count = [0]  # per-subagent running counter (list for closure mutation)

    def _identity_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "task_index": task_index,
            "task_count": task_count,
            "goal": goal_label,
        }
        if subagent_id is not None:
            kw["subagent_id"] = subagent_id
        if parent_id is not None:
            kw["parent_id"] = parent_id
        if depth is not None:
            kw["depth"] = depth
        if model is not None:
            kw["model"] = model
        if toolsets is not None:
            kw["toolsets"] = list(toolsets)
        # The child's own session id — filled into the shared ref once the
        # child agent exists (the callback is built first), so every relayed
        # event lets UIs open/inspect the subagent's session directly.
        if session_ref and session_ref.get("session_id"):
            kw["child_session_id"] = str(session_ref["session_id"])
        if session_ref and session_ref.get("delegation_id"):
            kw["delegation_id"] = str(session_ref["delegation_id"])
        kw["tool_count"] = _tool_count[0]
        return kw

    def _relay(
        event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        if not parent_cb:
            return
        payload = _identity_kwargs()
        payload.update(kwargs)  # caller overrides (e.g. status, duration_seconds)
        try:
            parent_cb(event_type, tool_name, preview, args, **payload)
        except Exception as e:
            logger.debug("Parent callback failed: %s", e)

    def _callback(
        event_type, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        # Lifecycle events emitted by the orchestrator itself — handled
        # before enum normalisation since they are not part of DelegateEvent.
        if event_type == "subagent.start":
            if spinner and goal_label:
                short = (
                    (goal_label[:55] + "...") if len(goal_label) > 55 else goal_label
                )
                try:
                    spinner.print_above(f" {_prefix()}├─ 🔀 {short}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.start", preview=preview or goal_label or "", **kwargs)
            return

        if event_type == "subagent.complete":
            # Failed child: echo one clean reason line into the CLI tree so
            # the human sees WHY, not just a vanished branch. Gateway-side
            # rendering happens in TurnRunner.progress_callback off the
            # relayed event below.
            if spinner and kwargs.get("status") in SUBAGENT_FAILURE_STATUSES:
                _fail_line = format_subagent_failure_line(
                    goal_label,
                    kwargs.get("status"),
                    error=kwargs.get("summary") or preview,
                    duration_seconds=kwargs.get("duration_seconds"),
                )
                try:
                    spinner.print_above(f" {_prefix()}├─ {_fail_line}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.complete", preview=preview, **kwargs)
            return

        if event_type == "subagent.text":
            # Streamed assistant reply text from the child. Relay verbatim so a
            # gateway watch window can mirror the child "talking" as it streams.
            # No spinner echo — the CLI shows the child via the tree, and the
            # CLI/TUI progress handlers ignore non-tool event types, so this is
            # inert there; only a gateway watch window consumes it.
            _relay("subagent.text", preview=preview)
            return

        # Normalise legacy strings, new-style "delegate.*" strings, and
        # DelegateEvent enum values all to a single DelegateEvent.  The
        # original implementation only accepted the five legacy strings;
        # enum-typed callers were silently dropped.
        if isinstance(event_type, DelegateEvent):
            event = event_type
        else:
            event = _LEGACY_EVENT_MAP.get(event_type)
            if event is None:
                try:
                    event = DelegateEvent(event_type)
                except (ValueError, TypeError):
                    return  # Unknown event — ignore

        if event == DelegateEvent.TASK_THINKING:
            text = preview or tool_name or ""
            if spinner:
                short = (text[:55] + "...") if len(text) > 55 else text
                try:
                    spinner.print_above(f' {_prefix()}├─ 💭 "{short}"')
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.thinking", preview=text)
            return

        if event == DelegateEvent.TASK_TOOL_COMPLETED:
            return

        if event == DelegateEvent.TASK_PROGRESS:
            # Pre-batched progress summary relayed from a nested
            # orchestrator's grandchild (upstream emits as
            # parent_cb("subagent_progress", summary_string) where the
            # summary lands in the tool_name positional slot).  Treat as
            # a pass-through: render distinctly (not via the tool-start
            # emoji lookup, which would mistake the summary string for a
            # tool name) and relay upward without re-batching.
            summary_text = tool_name or preview or ""
            if spinner and summary_text:
                try:
                    spinner.print_above(f" {_prefix()}├─ 🔀 {summary_text}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            if parent_cb:
                try:
                    parent_cb("subagent_progress", f"{_prefix()}{summary_text}")
                except Exception as e:
                    logger.debug("Parent callback relay failed: %s", e)
            return

        # TASK_TOOL_STARTED — display and batch for parent relay
        _tool_count[0] += 1
        if subagent_id is not None:
            with _active_subagents_lock:
                rec = _active_subagents.get(subagent_id)
                if rec is not None:
                    rec["tool_count"] = _tool_count[0]
                    rec["last_tool"] = tool_name or ""
        if spinner:
            short = (
                (preview[:35] + "...")
                if preview and len(preview) > 35
                else (preview or "")
            )
            from agent.display import get_tool_emoji

            emoji = get_tool_emoji(tool_name or "")
            line = f" {_prefix()}├─ {emoji} {tool_name}"
            if short:
                line += f'  "{short}"'
            try:
                spinner.print_above(line)
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _relay("subagent.tool", tool_name, preview, args)
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                _relay("subagent.progress", preview=f"🔀 {_prefix()}{summary}")
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            _relay("subagent.progress", preview=f"🔀 {_prefix()}{summary}")
            _batch.clear()

    _callback._flush = _flush
    return _callback


def _normalized_runtime_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _inherit_parent_capabilities(
    parent_agent, override_provider, override_base_url
) -> Optional[dict]:
    """Return the parent's endpoint-trust capability map for a child, or None.

    The trusted-proxy capability map (``agent.capabilities``, e.g.
    ``openai_native_compaction`` from a custom_providers entry) is a trust
    decision scoped to one provider+endpoint. A child inherits it ONLY when
    it runs against the parent's exact route — any delegation override that
    changes provider or base_url stays DEFAULT-DENY, matching the /model
    switch posture (#94036/#97292).
    """
    if override_provider or override_base_url:
        return None
    parent_caps = getattr(parent_agent, "capabilities", None)
    if not isinstance(parent_caps, dict):
        return None
    return {
        key: value
        for key, value in parent_caps.items()
        if isinstance(key, str) and isinstance(value, bool)
    }


def _inherit_parent_base_url(parent_agent, fallback_base_url: Optional[str]) -> Optional[str]:
    """Return the base URL the parent is actually calling, not a stale attribute.

    ``parent_agent.base_url`` can still carry a leftover OpenRouter URL from an
    old config while the live OpenAI client in ``_client_kwargs`` already points
    at local Ollama. Subagents must inherit the active endpoint or they 401
    against OpenRouter with a dummy/local key.
    """
    surface_url = _normalized_runtime_url(fallback_base_url)
    client_kwargs = getattr(parent_agent, "_client_kwargs", None)
    if isinstance(client_kwargs, dict):
        kwargs_url = _normalized_runtime_url(client_kwargs.get("base_url"))
        if (
            kwargs_url
            and kwargs_url != surface_url
            and kwargs_url.startswith(("http://", "https://"))
        ):
            return kwargs_url

    client = getattr(parent_agent, "client", None)
    if client is not None:
        # OpenAI SDK exposes ``base_url`` as an ``httpx.URL``, not ``str`` —
        # coerce so the comparison works regardless of the client's type.
        live_url = _normalized_runtime_url(getattr(client, "base_url", ""))
        if (
            live_url
            and live_url != surface_url
            and live_url.startswith(("http://", "https://"))
        ):
            return live_url

    return fallback_base_url or None


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    task_count: int,
    parent_agent,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    override_request_overrides: Optional[Dict[str, Any]] = None,
    override_max_tokens: Optional[int] = None,
    # ACP transport overrides from trusted delegation config.
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Per-call role controlling whether the child can further delegate.
    # 'leaf' (default) cannot; 'orchestrator' retains the delegation
    # toolset subject to depth/kill-switch bounds applied below.
    role: str = "leaf",
):
    """
    Build a child AIAgent on the main thread (thread-safe construction).
    Returns the constructed child agent without running it.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent
    import uuid as _uuid

    # ── Role resolution ─────────────────────────────────────────────────
    # Depth-derived, not caller-declared: a child may delegate iff the
    # kill switch is on and depth budget remains below max_spawn_depth.
    # The legacy `role` arg no longer participates (it asked the caller
    # to guess a fact the config already knows); it is still accepted and
    # normalised for wire compat, but capability comes from depth alone.
    child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    max_spawn = _get_max_spawn_depth()
    orchestrator_ok = _get_orchestrator_enabled() and child_depth < max_spawn
    effective_role = "orchestrator" if orchestrator_ok else "leaf"

    # ── Subagent identity (stable across events, 0-indexed for TUI) ─────
    # subagent_id is generated here so the progress callback, the
    # spawn_requested event, and the _active_subagents registry all share
    # one key.  parent_id is non-None when THIS parent is itself a subagent
    # (nested orchestrator -> worker chain).
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
    tui_depth = max(0, child_depth - 1)  # 0 = first-level child for the UI

    delegation_cfg = _load_config()

    # When no explicit toolsets given, inherit from parent's enabled toolsets
    # so disabled tools (e.g. web) don't leak to subagents.
    # Note: enabled_toolsets=None means "all tools enabled" (the default),
    # so we must derive effective toolsets from the parent's loaded tools.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        # enabled_toolsets is None (all tools) — derive from loaded tool names
        import model_tools

        parent_toolsets = {
            ts
            for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks.
        # Expand composite toolsets (e.g. hermes-cli) so that individual
        # toolset names (e.g. web, terminal) are recognised during intersection.
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        child_toolsets = [t for t in toolsets if t in expanded_parent]
        if _get_inherit_mcp_toolsets():
            child_toolsets = _preserve_parent_mcp_toolsets(
                child_toolsets, parent_toolsets
            )
        child_toolsets = _strip_blocked_tools(child_toolsets)
    elif parent_agent and parent_enabled is not None:
        child_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        child_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    # Blocked tools also live inside mixed platform bundles (hermes-cli,
    # hermes-telegram, etc.) that _strip_blocked_tools must keep because they
    # carry useful tools too. Pass exact one-tool deny toolsets through to the
    # child so model_tools subtracts the blocked names AFTER composite
    # expansion, and the restriction survives later registry/MCP refreshes.
    raw_parent_disabled = getattr(parent_agent, "disabled_toolsets", None)
    if isinstance(raw_parent_disabled, (list, tuple, set)):
        inherited_disabled = [str(name) for name in raw_parent_disabled]
    else:
        inherited_disabled = []
    if effective_role == "orchestrator":
        # Role grants delegate_task explicitly, matching the unconditional
        # delegation toolset re-add below.
        inherited_disabled = [
            name for name in inherited_disabled if name != "delegation"
        ]
    child_disabled_toolsets = list(
        dict.fromkeys(
            inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"]
        )
    )

    # Orchestrators retain the 'delegation' toolset that _strip_blocked_tools
    # removed.  The re-add is unconditional on parent-toolset membership because
    # orchestrator capability is granted by role, not inherited — see the
    # test_intersection_preserves_delegation_bound test for the design rationale.
    if effective_role == "orchestrator" and "delegation" not in child_toolsets:
        child_toolsets.append("delegation")

    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        workspace_path=workspace_hint,
        role=effective_role,
        max_spawn_depth=max_spawn,
        child_depth=child_depth,
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Resolve the child's effective model early so it can ride on every event.
    effective_model_for_cb = model or getattr(parent_agent, "model", None)

    # Build progress callback to relay tool calls to parent display.
    # Identity kwargs thread the subagent_id through every emitted event so the
    # TUI can reconstruct the spawn tree and route per-branch controls.
    child_session_ref: Dict[str, Any] = {}
    child_progress_cb = _build_child_progress_callback(
        task_index,
        goal,
        parent_agent,
        task_count,
        subagent_id=subagent_id,
        parent_id=parent_subagent_id,
        depth=tui_depth,
        model=effective_model_for_cb,
        toolsets=child_toolsets,
        session_ref=child_session_ref,
    )

    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    child_thinking_cb = None
    if child_progress_cb:

        def _child_thinking(text: str) -> None:
            if not text:
                return
            try:
                child_progress_cb("_thinking", text)
            except Exception as e:
                logger.debug("Child thinking callback relay failed: %s", e)

        child_thinking_cb = _child_thinking

    # Resolve effective credentials: config override > parent inherit
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    if not override_base_url:
        effective_base_url = _inherit_parent_base_url(parent_agent, effective_base_url)
    effective_api_key = override_api_key or parent_api_key
    # Same-class follow-up to #94036/#97292: the trusted-proxy capability map
    # (`agent.capabilities`, e.g. ``openai_native_compaction`` from a
    # custom_providers entry) is an endpoint-scoped trust decision. Children
    # inherit it ONLY when they run against the parent's exact provider and
    # base_url — a provider- or endpoint-changing delegation override stays
    # DEFAULT-DENY, matching the /model switch posture. Without this, a child
    # on the same trusted proxy silently falls back to local summarization.
    child_capabilities = _inherit_parent_capabilities(
        parent_agent, override_provider, override_base_url
    )
    # Bug #20558 / PR #20563: api_mode must NOT be inherited when the child uses a
    # different provider than the parent — each provider has its own API surface
    # (e.g. MiniMax uses anthropic_messages, DeepSeek uses chat_completions).
    # Inheriting the parent's mode causes 404 errors when the child routes to the
    # wrong endpoint.  Derive the mode from the target provider when it differs.
    #
    # Nous Portal is dual-wire within a single provider: anthropic/* → Messages,
    # everything else → chat_completions. Same-provider inheritance would pin a
    # child Hermes/Qwen subagent onto the parent's Claude Messages wire (or the
    # reverse). agent_init honors an explicit api_mode above its nous branch, so
    # re-derive here before construction.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    _effective_provider_norm = (effective_provider or "").strip().lower()
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif _effective_provider_norm in {"nous", "nous-portal", "nousresearch"}:
        from hermes_cli.providers import nous_api_mode

        effective_api_mode = nous_api_mode(effective_model)
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # Defensive: validate trusted delegation.command exists on PATH before
    # honoring it. An explicitly pinned transport that cannot run must fail
    # the spawn loudly (#80450) — silently falling back to the default
    # transport would run the child somewhere the user explicitly routed it
    # away from. Normally unreachable via delegate_task, which pre-validates
    # the command in _resolve_delegation_credentials.
    if override_acp_command:
        import shutil as _shutil

        if not _shutil.which(override_acp_command):
            raise ValueError(
                f"Pinned delegation command '{override_acp_command}' was not "
                f"found on PATH. Install it or remove delegation.command from "
                f"config.yaml."
            )
    effective_acp_command = override_acp_command or getattr(
        parent_agent, "acp_command", None
    )
    effective_acp_args = list(
        override_acp_args
        if override_acp_args is not None
        else (getattr(parent_agent, "acp_args", []) or [])
    )

    # When override_provider is set (e.g. delegation.provider: minimax-cn),
    # the subagent must use direct API calls — not the parent's ACP transport.
    # Inheriting acp_command unconditionally causes run_agent.py to initialize
    # CopilotACPClient, bypassing override credentials entirely (issue #16816).
    if override_provider and not override_acp_command:
        effective_acp_command = None
        effective_acp_args = []

    if override_acp_command:
        # If explicitly forcing an ACP transport override, the provider MUST be copilot-acp
        # so run_agent.py initializes the CopilotACPClient.
        effective_provider = "copilot-acp"
        effective_api_mode = "chat_completions"

    # Resolve reasoning config: delegation override > parent inherit
    parent_reasoning = getattr(parent_agent, "reasoning_config", None)
    child_reasoning = parent_reasoning
    try:
        # Keep the raw value — ``str(x or "")`` would coerce a YAML boolean
        # False (``reasoning_effort: false``) to "" and inherit the parent
        # instead of disabling thinking for children.
        delegation_effort = delegation_cfg.get("reasoning_effort")
        if delegation_effort or delegation_effort is False:
            from hermes_constants import parse_reasoning_effort

            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is not None:
                child_reasoning = parsed
            else:
                logger.warning(
                    "Unknown delegation.reasoning_effort '%s', inheriting parent level",
                    delegation_effort,
                )
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

    # Inherit the parent's fallback provider chain so subagents can recover
    # from rate-limits and credential exhaustion exactly like the top-level
    # agent does.  _fallback_chain is a list accepted by AIAgent's
    # fallback_model parameter (which handles both list and dict forms).
    #
    # EXCEPT when the user pinned delegation.provider: an explicit pin means
    # "children run on THIS provider".  Inheriting the parent chain would let
    # a mid-run auth/429 failure silently reroute the quiet-mode child onto
    # the parent's fallback models with no surfaced signal (#80450) — the
    # same class of silent-drag the override_provider filter-clearing below
    # already prevents for OpenRouter routing preferences.  Predictability >
    # liveness for explicit pins: the pinned child fails loudly instead.
    parent_fallback = (
        None
        if override_provider
        else (getattr(parent_agent, "_fallback_chain", None) or None)
    )

    # Inherit the parent's OpenRouter provider-preference filters by default
    # (so subagents routed to the same provider honour the same routing
    # constraints).  BUT: when `delegation.provider` is set the user is
    # explicitly asking the child to run on a different provider, and
    # parent-level OpenRouter filters (e.g. `only=["Anthropic"]`) would
    # silently force the child back onto the parent's provider. Clear the
    # filters in that case so the delegated provider is honoured.
    child_providers_allowed = getattr(parent_agent, "providers_allowed", None)
    child_providers_ignored = getattr(parent_agent, "providers_ignored", None)
    child_providers_order = getattr(parent_agent, "providers_order", None)
    child_provider_sort = getattr(parent_agent, "provider_sort", None)
    child_provider_require_parameters = getattr(
        parent_agent, "provider_require_parameters", False
    )
    child_provider_data_collection = getattr(
        parent_agent, "provider_data_collection", None
    ) or ""
    child_openrouter_min_coding_score = getattr(parent_agent, "openrouter_min_coding_score", None)
    if override_provider:
        child_providers_allowed = None
        child_providers_ignored = None
        child_providers_order = None
        child_provider_sort = None
        child_provider_require_parameters = False
        child_provider_data_collection = ""
        # Note: openrouter_min_coding_score is model-gated (only emitted on
        # openrouter/pareto-code), so we keep it inherited even when the
        # provider is overridden — it's a no-op on any other model.

    child_max_tokens = (
        override_max_tokens
        if override_max_tokens is not None
        else getattr(parent_agent, "max_tokens", None)
    )
    child_optional_kwargs: Dict[str, Any] = {}
    if isinstance(child_max_tokens, int):
        child_optional_kwargs["max_tokens"] = child_max_tokens

    # Each child gets a DEDICATED SessionDB connection instead of the parent's
    # live object. The parent's handle is owned by the parent's lifecycle
    # (cron run_job's finally block, gateway session end, /new) and can be
    # closed while a fire-and-forget background child is still flushing on a
    # daemon thread — every subsequent flush then hits the closed handle and
    # the child's transcript is silently dropped (#81267). A dedicated handle
    # can't be closed out from under the child; it is released by the child's
    # own close() via the owned flag set below. It MUST point at the same
    # database FILE as the parent's handle: parents can hold non-default
    # per-profile handles (tui_gateway opens SessionDB(db_path=<profile>/
    # state.db) for non-launch profiles), and a bare SessionDB() would write
    # the child's transcript into the launch profile's db, breaking
    # parent_session_id lineage and session_search. AsyncSessionDB wrappers
    # (gateway) forward .db_path via __getattr__, so this works through them.
    child_session_db = None
    parent_session_db = getattr(parent_agent, "_session_db", None)
    if parent_session_db is not None:
        try:
            from hermes_state import get_shared_session_db

            _parent_db_path = getattr(parent_session_db, "db_path", None)
            child_session_db = (
                get_shared_session_db(_parent_db_path)
                if _parent_db_path is not None
                else get_shared_session_db()
            )
        except Exception:
            logger.debug(
                "subagent: failed to open dedicated SessionDB; child persistence disabled",
                exc_info=True,
            )
            child_session_db = None

    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        try:
            child = AIAgent(
                base_url=effective_base_url,
                api_key=effective_api_key,
                model=effective_model,
                provider=effective_provider,
                capabilities=child_capabilities,
                api_mode=effective_api_mode,
                acp_command=effective_acp_command,
                acp_args=effective_acp_args,
                max_iterations=max_iterations,

                reasoning_config=child_reasoning,
                prefill_messages=getattr(parent_agent, "prefill_messages", None),
                fallback_model=parent_fallback,
                enabled_toolsets=child_toolsets,
                disabled_toolsets=child_disabled_toolsets,
                quiet_mode=True,
                ephemeral_system_prompt=child_prompt,
                log_prefix=f"[subagent-{task_index}]",
                platform="subagent",
                skip_context_files=True,
                skip_memory=True,
                clarify_callback=None,
                thinking_callback=child_thinking_cb,
                session_db=child_session_db,
                parent_session_id=getattr(parent_agent, "session_id", None),
                providers_allowed=child_providers_allowed,
                providers_ignored=child_providers_ignored,
                providers_order=child_providers_order,
                provider_sort=child_provider_sort,
                provider_require_parameters=child_provider_require_parameters,
                provider_data_collection=child_provider_data_collection,
                request_overrides=(
                    # override_request_overrides is honored whenever set —
                    # including the inherit branch (override_provider=None),
                    # where _resolve_delegation_credentials already merged
                    # delegation.request_overrides OVER the parent's values.
                    dict(override_request_overrides)
                    if override_request_overrides is not None
                    else (
                        {}
                        if override_provider
                        else dict(getattr(parent_agent, "request_overrides", {}) or {})
                    )
                ),
                openrouter_min_coding_score=child_openrouter_min_coding_score,
                tool_progress_callback=child_progress_cb,
                iteration_budget=None,  # fresh budget per subagent
                **child_optional_kwargs,
            )
        except BaseException:
            # Construction failed: the dedicated handle has no owner and no
            # child close() will ever run — release it here so the sqlite fds
            # don't outlive the failed spawn.
            if child_session_db is not None:
                try:
                    from hermes_state import release_or_close

                    release_or_close(child_session_db)
                except Exception:
                    pass
            raise
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    # Ownership transfer for the dedicated handle: the child's close() must
    # release it (nothing else holds a reference), and no parent teardown can
    # close it out from under a background child (#81267).
    if child_session_db is not None:
        child._owns_session_db = True
    # Now the child exists, its session id can ride on every relayed event
    # (including the spawn_requested below — first emit happens after this).
    child_session_ref["session_id"] = getattr(child, "session_id", "") or ""
    # delegate_task fills the batch id after construction; retain the same
    # shared ref so existing callbacks pick it up without being rebuilt.
    setattr(child, "_progress_identity_ref", child_session_ref)
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = child_depth
    # Stash the post-degrade role for introspection (leaf if the
    # kill switch or depth bounded the caller's requested role).
    child._delegate_role = effective_role
    # Stash subagent identity for nested-delegation event propagation and
    # for _run_single_child / interrupt_subagent to look up by id.
    child._subagent_id = subagent_id
    child._parent_subagent_id = parent_subagent_id
    child._subagent_goal = goal
    child._parent_turn_id = getattr(parent_agent, "_current_turn_id", "") or ""
    # Ownership chain for the model-facing control plane (action=list/steer/
    # stop): a parent may only control agents whose weakref chain reaches it.
    # Weakref so a finished parent can be collected while a detached child
    # record briefly lingers in the registry.
    try:
        child._delegate_parent_ref = weakref.ref(parent_agent)
    except TypeError:
        # Test doubles (MagicMock et al.) may not be weakref-able; control
        # actions then simply don't resolve ownership for this child.
        child._delegate_parent_ref = None
    # Stable sidebar marker: delegate subagent sessions must stay out of
    # session pickers even when a parent delete orphans them (parent_session_id
    # → NULL). Mirrors /branch's ``_branched_from`` pattern — see
    # ``list_sessions_rich`` child-exclusion clause.
    parent_sid = getattr(parent_agent, "session_id", None)
    if parent_sid and getattr(child, "_session_init_model_config", None) is not None:
        child._session_init_model_config["_delegate_from"] = parent_sid

    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(
        effective_provider, parent_agent, effective_base_url
    )
    if child_pool is not None:
        child._credential_pool = child_pool

    # Register child for interrupt propagation
    if hasattr(parent_agent, "_active_children"):
        lock = getattr(parent_agent, "_active_children_lock", None)
        if lock:
            with lock:
                parent_agent._active_children.append(child)
        else:
            parent_agent._active_children.append(child)

    # Announce the spawn immediately — the child may sit in a queue
    # for seconds if max_concurrent_children is saturated, so the TUI
    # wants a node in the tree before run starts.
    if child_progress_cb:
        try:
            child_progress_cb("subagent.spawn_requested", preview=goal)
        except Exception as exc:
            logger.debug("spawn_requested relay failed: %s", exc)

    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "subagent_start",
            parent_session_id=getattr(parent_agent, "session_id", None),
            parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
            parent_subagent_id=parent_subagent_id,
            child_session_id=getattr(child, "session_id", None),
            child_subagent_id=subagent_id,
            child_role=effective_role,
            child_goal=goal,
        )
    except Exception:
        logger.debug("subagent_start hook invocation failed", exc_info=True)

    return child


def _dump_subagent_timeout_diagnostic(
    *,
    child: Any,
    task_index: int,
    timeout_seconds: float,
    duration_seconds: float,
    worker_thread: Optional[threading.Thread],
    goal: str,
) -> Optional[str]:
    """Write a structured diagnostic dump for a subagent that timed out
    before making any API call.

    See issue #14726: users hit "subagent timed out after 300s with no response"
    with zero API calls and no way to inspect what happened. This helper
    writes a dedicated log under ``~/.hermes/logs/subagent-<sid>-<ts>.log``
    capturing the child's config, system-prompt / tool-schema sizes, activity
    tracker snapshot, and the worker thread's Python stack at timeout.

    Returns the absolute path to the diagnostic file, or None on failure.
    """
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        import sys as _sys
        import traceback as _traceback
        import threading as _threading

        hermes_home = get_hermes_home()
        logs_dir = hermes_home / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{ts}.log"

        lines: List[str] = []
        def _w(line: str = "") -> None:
            lines.append(line)

        _w("# Subagent timeout diagnostic — issue #14726")
        _w(f"# Generated: {_dt.datetime.now().isoformat()}")
        _w("")
        _w("## Timeout")
        _w(f"  task_index:        {task_index}")
        _w(f"  subagent_id:       {subagent_id}")
        _w(f"  configured_timeout: {timeout_seconds}s")
        _w(f"  actual_duration:   {duration_seconds:.2f}s")
        _w("")

        _w("## Goal")
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        _w(_goal_preview or "(empty)")
        _w("")

        _w("## Child config")
        for attr in (
            "model", "provider", "api_mode", "base_url", "max_iterations",
            "quiet_mode", "skip_memory", "skip_context_files", "platform",
            "_delegate_role", "_delegate_depth",
        ):
            try:
                val = getattr(child, attr, None)
                # Redact api_key-shaped values defensively
                if isinstance(val, str) and attr == "base_url":
                    pass
                _w(f"  {attr}: {val!r}")
            except Exception:
                _w(f"  {attr}: <unreadable>")
        _w("")

        _w("## Toolsets")
        enabled = getattr(child, "enabled_toolsets", None)
        _w(f"  enabled_toolsets:  {enabled!r}")
        tool_names = getattr(child, "valid_tool_names", None)
        if tool_names:
            _w(f"  loaded tool count: {len(tool_names)}")
            try:
                _w(f"  loaded tools:      {sorted(tool_names)}")
            except Exception:
                pass
        _w("")

        _w("## Prompt / schema sizes")
        try:
            sys_prompt = getattr(child, "ephemeral_system_prompt", None) \
                or getattr(child, "system_prompt", None) \
                or ""
            _w(f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if isinstance(sys_prompt, str) else 'n/a'}")
            _w(f"  system_prompt_chars: {len(sys_prompt) if isinstance(sys_prompt, str) else 'n/a'}")
        except Exception as exc:
            _w(f"  system_prompt: <error: {exc}>")
        try:
            tools_schema = getattr(child, "tools", None)
            if tools_schema is not None:
                _schema_json = json.dumps(tools_schema, default=str)
                _w(f"  tool_schema_count: {len(tools_schema)}")
                _w(f"  tool_schema_bytes: {len(_schema_json.encode('utf-8'))}")
        except Exception as exc:
            _w(f"  tool_schema: <error: {exc}>")
        _w("")

        _w("## Activity summary")
        try:
            summary = child.get_activity_summary()
            for k, v in summary.items():
                _w(f"  {k}: {v!r}")
        except Exception as exc:
            _w(f"  <get_activity_summary failed: {exc}>")
        _w("")

        _w("## Worker thread stack at timeout")
        if worker_thread is not None and worker_thread.is_alive():
            frames = _sys._current_frames()
            worker_frame = frames.get(worker_thread.ident)
            if worker_frame is not None:
                stack = _traceback.format_stack(worker_frame)
                for frame_line in stack:
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"  {sub}")
            else:
                _w("  <worker frame not available>")
        elif worker_thread is None:
            _w("  <no worker thread handle>")
        else:
            _w("  <worker thread already exited>")
        _w("")

        # All other live threads. The conversation worker's own stack often
        # shows it parked waiting on a nested helper thread (interrupt worker,
        # daemon-pool sibling) — without the full picture, a pre-HTTP wedge
        # (#60203/#62151) is indistinguishable from a slow provider. Best
        # effort and bounded: names + stacks for up to 40 threads.
        _w("## All thread stacks at timeout")
        try:
            frames = _sys._current_frames()
            by_ident = {
                th.ident: th for th in _threading.enumerate() if th.ident
            }
            worker_ident = worker_thread.ident if worker_thread else None
            dumped = 0
            for ident, frame in frames.items():
                if ident == worker_ident:
                    continue  # already dumped above
                if dumped >= 40:
                    _w(f"  <{len(frames) - dumped - 1} more threads omitted>")
                    break
                th = by_ident.get(ident)
                name = th.name if th else f"ident={ident}"
                daemon = " daemon" if (th and th.daemon) else ""
                _w(f"  --- {name}{daemon} ---")
                for frame_line in _traceback.format_stack(frame):
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"    {sub}")
                dumped += 1
        except Exception as exc:
            _w(f"  <all-thread dump failed: {exc}>")
        _w("")

        _w("## Notes")
        _w("  This file is written ONLY when a subagent times out with 0 API calls.")
        _w("  0-API-call timeouts mean the child never reached its first LLM request.")
        _w("  Common causes: oversized prompt rejected by provider, transport hang,")
        _w("  credential resolution stuck. See issue #14726 for context.")

        dump_path.write_text("\n".join(lines), encoding="utf-8")
        return str(dump_path)
    except Exception as exc:
        logger.warning("Subagent timeout diagnostic dump failed: %s", exc)
        return None


def _spill_summary_to_file(task_index: int, summary: str) -> Optional[str]:
    """Write a subagent's full summary to the delegation cache and return path.

    Mirrors web_extract's ``_store_full_text``: the file lands in
    ``cache/delegation`` which is mounted read-only into remote backends
    (Docker/Modal/SSH) via ``credential_files._CACHE_DIRS``, so the parent's
    terminal/``read_file`` tools can page through the complete text on any
    backend. Returns the absolute path, or None on failure (best-effort:
    the trimmed head+tail is still returned to the parent regardless).
    """
    try:
        from hermes_constants import get_hermes_dir
        import datetime as _dt

        cache_dir = get_hermes_dir("cache/delegation", "delegation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = cache_dir / f"subagent-summary-{task_index}-{ts}.txt"
        from tools.spill_safety import write_text_exclusive

        # Exclusive symlink-refusing create; not private because
        # cache/delegation is bind-mounted read-only into remote backends
        # whose container UID must be able to read it.
        write_text_exclusive(path, summary, private=False)
        return str(path)
    except Exception as exc:
        logger.debug("Failed to spill subagent summary to file: %s", exc)
        return None


def _trim_summary_with_footer(
    summary: str, cap: int, task_index: int
) -> tuple[str, Optional[str]]:
    """Return (model_text, spill_path) for one over-budget summary.

    Mirrors web_extract's ``_truncate_with_footer``: keep a head+tail window
    (~75% head / ~25% tail, snapped to line boundaries) so the subagent's
    opening AND its closing (outcomes / files-changed / issues, which live at
    the end) both survive, spill the full text to disk, and append a footer
    telling the parent exactly how much it's seeing and the precise
    ``read_file offset=`` to page into the omitted middle. Deterministic.
    """
    original_len = len(summary)
    head_budget = int(cap * 0.75)
    tail_budget = cap - head_budget

    head = summary[:head_budget]
    tail = summary[-tail_budget:]
    # Snap the head cut back to the last newline so we don't slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # Snap the tail cut forward to the next newline for the same reason.
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    spill_path = _spill_summary_to_file(task_index, summary)

    footer_lines = [
        "",
        "─" * 8 + " [SUMMARY TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {original_len:,} total — trimmed to protect the parent's context window.",
    ]
    if spill_path:
        # read_file is 1-indexed; +2 moves past the last head line shown.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full subagent output saved to: {spill_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{spill_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete "
            f"summary; raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full output could not be stored to disk; the head+tail above is "
            "all that was preserved."
        )
    footer_lines.append("─" * 37)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail + "\n".join(footer_lines)
    return model_text, spill_path


def _parent_summary_char_budget(parent_agent, n_summaries: int) -> Optional[int]:
    """Per-summary character budget sized against the parent's *remaining*
    context headroom, split across the batch.

    The overflow this guards against is N summaries entering the parent
    context at once (batch fan-out), not any single summary being large.  We
    take a fraction of the headroom the parent has left (resolved context
    length minus what's already in its prompt) and divide it across the batch,
    converting tokens→chars at the standard ~4 chars/token estimate.

    Returns the per-summary char budget, or None when the parent's context
    state is unknown (no compressor / no token count) — in which case the
    caller falls back to the static char ceiling only.
    """
    try:
        compressor = getattr(parent_agent, "context_compressor", None)
        context_length = getattr(compressor, "context_length", None)
        if not isinstance(context_length, int) or context_length <= 0:
            return None

        used_tokens = getattr(parent_agent, "session_prompt_tokens", 0)
        if not isinstance(used_tokens, (int, float)) or used_tokens < 0:
            used_tokens = 0

        # Reserve the compressor's output budget so we measure INPUT headroom.
        reserved = getattr(compressor, "max_tokens", 0) or 0
        headroom_tokens = context_length - int(used_tokens) - int(reserved)
        if headroom_tokens <= 0:
            # Parent is already over budget — give each summary only the floor.
            return _MIN_SUMMARY_CHARS

        batch_token_budget = int(headroom_tokens * _SUMMARY_HEADROOM_FRACTION)
        per_summary_tokens = batch_token_budget // max(1, n_summaries)
        per_summary_chars = per_summary_tokens * 4  # ~4 chars/token
        return max(_MIN_SUMMARY_CHARS, per_summary_chars)
    except Exception:
        logger.debug("Summary budget computation failed", exc_info=True)
        return None


def _apply_summary_budget(results: List[Dict[str, Any]], parent_agent) -> None:
    """Trim subagent summaries in-place so the batch can't overflow the
    parent's context window, spilling full text to disk so nothing is lost.

    The effective per-summary cap is the MIN of:
      - the dynamic headroom budget (remaining parent context ÷ batch size), and
      - the static ``delegation.max_summary_chars`` ceiling (0 = disabled).

    When a summary exceeds the cap, its full text is written to a file and the
    in-context summary becomes a head slice plus a pointer to that file. This
    addresses issue/PR #9126: batch fan-out returned N full summaries verbatim,
    blowing the parent context and (on rate-limited providers) triggering a
    compression/429 death spiral.
    """
    summaries = [
        r for r in results if isinstance(r, dict) and isinstance(r.get("summary"), str) and r["summary"]
    ]
    if not summaries:
        return

    cfg = _load_config()
    try:
        static_ceiling = int(cfg.get("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS))
    except (TypeError, ValueError):
        static_ceiling = DEFAULT_MAX_SUMMARY_CHARS

    dynamic_budget = _parent_summary_char_budget(parent_agent, len(summaries))

    # Combine the two caps. Either can be absent/disabled.
    candidates = [c for c in (static_ceiling, dynamic_budget) if c and c > 0]
    if not candidates:
        return  # both disabled / unknown → leave summaries untouched
    cap = min(candidates)

    for entry in summaries:
        summary = entry["summary"]
        if len(summary) <= cap:
            continue
        original_len = len(summary)
        model_text, spill_path = _trim_summary_with_footer(
            summary, cap, entry.get("task_index", -1)
        )
        entry["summary"] = model_text
        entry["summary_truncated"] = True
        if spill_path:
            entry["summary_full_path"] = spill_path
        logger.debug(
            "[subagent-%s] summary trimmed %d → ~%d chars (spill=%s)",
            entry.get("task_index", "?"),
            original_len,
            cap,
            spill_path or "none",
        )
