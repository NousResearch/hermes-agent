"""Plugin registration: tool + bounded prompt section + pre-tool hook.

This module is the bridge between ``vault_retrieval.tool`` (pure logic) and
the Hermes plugin surface (``PluginContext``). It is what ``register(ctx)``
loads into the runtime.

Per the locked architecture memo:
  - register a read-only ``vault_context`` tool;
  - register a bounded system-prompt section telling agents to use it;
  - register a ``pre_tool_call`` hook that, in ``enforce`` mode, blocks
    direct ``read_file`` / ``search_files`` calls whose target resolves
    inside the configured Vault and points the caller at ``vault_context``;
    in ``audit`` mode records would-block events (no text, no args payload)
    but still passes through; in ``off`` mode the hook is a no-op.

The hook is an operational guard, NOT a security boundary. Arbitrary
terminal/Python can still read files. If hostile-tool bypass prevention
becomes required, open a separate upstream core/sandbox change.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import (
    VaultConfigError,
    default_config,
    resolve_vault_root,
    validate_config,
)
from .budgets import reset_turn, reset_session_turns
from .tool import (
    TOOL_NAME,
    TOOLSET,
    VaultContextHandler,
    vault_context_tool_schema,
)

logger = logging.getLogger(__name__)


# Hooks are NOT advertised by name in the architecture memo — they are
# observable side effects of the plugin's ``mode`` setting. We only ever
# listen to the read-only file tools (``read_file``, ``search_files``).
_PROTECTED_TOOLS = frozenset({"read_file", "search_files"})


# Bounded system-prompt section. The locked contract says this must be
# small (max_chars caps it via the plugin surface) and that it tells the
# agent to use ``vault_context`` for any Vault evidence. We deliberately
# keep it well under Hermes's per-section budget.
_PROMPT_SECTION_ID = "vault-retrieval-usage"
_PROMPT_SECTION_TEXT = """\
# Vault retrieval — token-efficient Obsidian Vault access

For any evidence from the Obsidian Vault (``/root/Documents/Obsidian Vault``
by default), use the ``vault_context`` tool instead of ``read_file`` or
``search_files``. It is the only read path that respects the per-turn
character budgets (default 12,000, hard ceiling 24,000).

Rules:
  - Pick the smallest range that contains the cited fact. Pass
    ``selectors=[{"path": "...", "line_start": N, "line_end": M}]``.
  - Files over 20,000 chars require an explicit selector (heading, ID,
    date, owner/status key, or line range) — full reads return
    ``large_file_selector_required`` without one.
  - If the budget cannot cover the answer, set
    ``allow_bounded_expansion=true`` and supply an ``expansion_reason``.
  - The returned envelope carries ``status``, ``extracts``, ``conflicts``,
    and ``redactions``; treat ``status: data_hold`` as the answer.
  - Never cite search snippets as evidence. Every claim needs the
    extract's path/range/freshness.

Direct ``read_file`` and ``search_files`` calls inside the Vault are
blocked by the runtime while this plugin is enabled."""


@dataclass(frozen=True)
class _ResolvedConfig:
    """Validation result of the operator-supplied plugin config."""

    cfg: Dict[str, Any]
    vault_root: Path


def _resolve_config(ctx) -> _ResolvedConfig:
    """Read + validate the plugin config from the operator's config.yaml.

    The Hermes config keys live at
    ``plugins.entries.vault-retrieval.settings.<key>`` (see
    ``PluginContext.get_config``).
    """
    raw: Dict[str, Any] = {
        "vault_root": ctx.get_config("vault_root")
        or os.environ.get("VAULT_RETRIEVAL_ROOT")
        or "/root/Documents/Obsidian Vault",
        "mode": ctx.get_config("mode", "enforce"),
        "default_budget_chars": ctx.get_config("default_budget_chars", 12_000),
        "hard_ceiling_chars": ctx.get_config("hard_ceiling_chars", 24_000),
        "large_file_chars": ctx.get_config("large_file_chars", 20_000),
        "candidate_limit": ctx.get_config("candidate_limit", 20),
        "max_primary_extracts": ctx.get_config("max_primary_extracts", 3),
        "max_expansion_extracts": ctx.get_config("max_expansion_extracts", 2),
        "max_range_lines": ctx.get_config("max_range_lines", 120),
        "max_range_chars": ctx.get_config("max_range_chars", 8_000),
        "query_log_enabled": ctx.get_config("query_log_enabled", True),
        "log_raw_query_terms": ctx.get_config("log_raw_query_terms", False),
        "snapshots_enabled": ctx.get_config("snapshots_enabled", False),
        "fts5_enabled": ctx.get_config("fts5_enabled", False),
        "block_direct_file_reads": ctx.get_config("block_direct_file_reads", True),
    }
    # Profile-scoped state dir; ``HERMES_HOME`` is set by Hermes core at
    # profile activation. We resolve via env so tests can mock it. If
    # unset, fall back to ``~/.hermes/state`` — the active profile's
    # root.
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        raw["state_dir"] = str(Path(hermes_home) / "state")
    else:
        raw["state_dir"] = str(Path.home() / ".hermes" / "state")
    cfg = validate_config(raw)
    return _ResolvedConfig(cfg=cfg, vault_root=Path(cfg["vault_root"]))


def _resolve_target_in_vault(
    args: Dict[str, Any], vault_root: Path
) -> Optional[Path]:
    """Best-effort: does this tool call resolve to a path under vault_root?

    Returns the absolute target path if so, else ``None``.

    Reads inside the Vault (block): a tool call whose ``path`` (or
    equivalent key) resolves to a real file under ``vault_root``, or a
    ``search_files`` call whose ``path`` is the search root and resolves
    inside the Vault.

    Outside the Vault (pass through): any call where no resolvable ``path``
    points under ``vault_root``, including ``search_files`` calls that
    only pass a glob ``pattern`` (a glob without a ``path`` anchor is
    ambiguous; we deliberately do NOT block it because blocking every
    arbitrary pattern would over-block unrelated repository searches and
    break legitimate workflows outside the Vault).
    """
    if not isinstance(args, dict):
        return None
    # read_file / search_files: ``path`` arg is the (or) file or search
    # root. If it resolves under vault_root, return the absolute path;
    # the hook treats that as a Vault hit.
    target = args.get("path") or args.get("file_path") or args.get("file")
    if isinstance(target, str) and target:
        try:
            abs_target = Path(target).expanduser().resolve()
            try:
                abs_target.relative_to(vault_root.resolve())
                return abs_target
            except ValueError:
                return None  # explicit path, outside the root
        except (ValueError, OSError):
            return None
    # No path`` argument. If ``pattern`` is a literal glob that names the
    # Vault root (rare; treat as out-of-scope), we would block; otherwise
    # we pass through. Glob patterns without an anchored path are
    # deliberately NOT blocked here — over-blocking random globs would
    # disrupt unrelated repository searches. Searches that genuinely
    # target the Vault must pass ``path=vault_root`` (or a directory
    # inside it) and are caught there.
    return None


def make_pre_tool_call_hook(ctx):
    """Build the ``pre_tool_call`` hook closure.

    The hook accepts the full dispatch kwargs (``task_id``, ``session_id``,
    ``tool_call_id``, ``turn_id``, ``api_request_id``, ``middleware_trace``,
    ``user_task``) that :func:`hermes_cli.plugins._dispatch_pre_tool_call_hooks`
    forwards from :func:`model_tools.handle_function_call`; we keep the
    wider signature so the hook does not silently lose any new runtime
    fields the agent loop adds later. ``**kwargs`` makes the closure
    forward-compatible with whatever the registry sends.
    """
    state: Dict[str, Any] = {"resolved": None}

    def _ensure():
        if state["resolved"] is None:
            try:
                state["resolved"] = _resolve_config(ctx)
            except VaultConfigError as exc:
                logger.error("vault-retrieval config invalid; disabling hook: %s", exc)
                state["resolved"] = False
        return state["resolved"] if state["resolved"] else None

    def pre_tool_call(
        tool_name: str = "",
        args: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        if tool_name not in _PROTECTED_TOOLS:
            return None
        resolved = _ensure()
        if not resolved:
            return None
        if resolved.cfg.get("mode") == "off":
            return None
        if not resolved.cfg.get("block_direct_file_reads", True):
            return None
        target = _resolve_target_in_vault(args or {}, resolved.vault_root)
        if target is None:
            return None  # Outside Vault — don't interfere.

        if resolved.cfg.get("mode") == "audit":
            logger.info(
                "vault-retrieval: would-block tool=%s target=%s",
                tool_name, target,
            )
            return None

        # enforce
        return {
            "action": "block",
            "message": (
                f"[vault-retrieval] Direct {tool_name} inside the Obsidian Vault "
                f"is blocked. Use the ``vault_context`` tool with explicit "
                f"selectors (path + line_start/line_end or heading) and "
                f"respect the per-turn character budgets "
                f"(default 12,000, hard ceiling 24,000). "
                f"Target: {target}"
            ),
        }

    return pre_tool_call


def _build_tool_handler(resolved: _ResolvedConfig):
    """Build the registered tool's handler closure.

    We construct the ``VaultContextHandler`` once and let the closure
    delegate each call into ``handler.handle_runtime()``.

    The closure signature is ``(args, **kwargs)`` because
    :meth:`tools.registry.dispatch` invokes the handler as
    ``handler(args, **kwargs)``; the kwargs are the dispatch metadata
    (e.g. ``task_id``, ``session_id``, ``user_task``,
    ``turn_id``/``turn_key``, ``enabled_tools``) that
    ``model_tools._execute_tool`` forwards from the agent loop. Accepting
    the runtime kwargs makes the handler compatible with the real
    registry dispatch contract; rejecting them was the type-error that
    blocked the previous candidate.
    """
    handler = VaultContextHandler(cfg=resolved.cfg)

    def _tool_handler(args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        return handler.handle_runtime(args or {}, **kwargs)

    return _tool_handler


def make_session_lifecycle_hooks():
    """Build the ``on_session_end`` and ``on_session_finalize`` closures.

    These reset the process-global ``vault-retrieval`` per-turn budget
    counters when a session finishes, so the next session starts from
    zero. Hermes forwards ``session_id`` to ``on_session_end`` and
    ``on_session_finalize``; we use it to drop the matching
    ``runtime:<task>:<session>`` counters. We also accept ``turn_id``
    and drop that key directly if the runtime supplies one.
    """
    def on_session_end(*, session_id: str = "", task_id: str = "",
                       turn_id: str = "", **_: Any) -> None:
        # Drop the matching runtime counter(s).
        prefix = f"runtime:{task_id or ''}:{session_id or ''}"
        reset_session_turns(prefix)
        if turn_id:
            reset_turn(turn_id)

    def on_session_finalize(*, session_id: str = "", task_id: str = "",
                            **_: Any) -> None:
        prefix = f"runtime:{task_id or ''}:{session_id or ''}"
        reset_session_turns(prefix)

    def on_session_reset(*, session_id: str = "", task_id: str = "",
                          **_: Any) -> None:
        # ``on_session_reset`` runs before the next session begins — drop
        # any leftover counter to bound the process-global state.
        prefix = f"runtime:{task_id or ''}:{session_id or ''}"
        reset_session_turns(prefix)

    return on_session_end, on_session_finalize, on_session_reset


def register(ctx) -> None:
    """Hermes plugin entrypoint.

    Called by ``PluginManager`` after discovery. Idempotent across
    repeated calls.
    """
    # 1. Validate config now — fail-closed at registration if invalid.
    try:
        resolved = _resolve_config(ctx)
    except VaultConfigError as exc:
        logger.error(
            "vault-retrieval: refusing to register — config invalid: %s. "
            "Fix plugins.entries.vault-retrieval.settings in config.yaml.",
            exc,
        )
        return

    # 2. Register the read-only tool.
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=vault_context_tool_schema(),
        handler=_build_tool_handler(resolved),
        description=(
            "Token-efficient read-only retrieval against the configured Obsidian "
            "Vault. Filename-first, range-only reads, per-turn character budgets "
            "(default 12,000; hard ceiling 24,000), large-file refusal. Returns "
            "a JSON envelope with status, candidates, extracts, conflicts, and "
            "redaction counts. Use this instead of read_file/search_files for "
            "any Vault evidence."
        ),
        emoji="",
    )

    # 3. Register the bounded system-prompt section.
    ctx.register_system_prompt_section(
        id=_PROMPT_SECTION_ID,
        content=_PROMPT_SECTION_TEXT,
        position="after_memory",
        max_chars=2000,
    )

    # 4. Register the pre_tool_call hook (enforce/audit/off via config).
    ctx.register_hook("pre_tool_call", make_pre_tool_call_hook(ctx))

    # 5. Register lifecycle hooks so per-turn budget counters are
    #    released at session boundaries. Without these, counters
    #    accumulate across sessions and a 12k-default session would see
    #    a previous session's residual budget eaten.
    on_end, on_finalize, on_reset = make_session_lifecycle_hooks()
    ctx.register_hook("on_session_end", on_end)
    ctx.register_hook("on_session_finalize", on_finalize)
    ctx.register_hook("on_session_reset", on_reset)

    logger.info(
        "vault-retrieval registered: mode=%s vault_root=%s",
        resolved.cfg.get("mode"),
        resolved.cfg.get("vault_root"),
    )
