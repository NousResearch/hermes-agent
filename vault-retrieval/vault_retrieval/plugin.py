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
    ``search_files`` is allowed to pass when its pattern targets the Vault
    broadly (we cannot prove containment from a glob) — we always treat
    it as a Vault hit when the pattern is non-empty.
    """
    if not isinstance(args, dict):
        return None
    # read_file: ``path`` arg is the file path.
    target = args.get("path") or args.get("file_path") or args.get("file")
    if isinstance(target, str) and target:
        try:
            abs_target = Path(target).expanduser().resolve()
            abs_target.relative_to(vault_root.resolve())
            return abs_target
        except (ValueError, OSError):
            return None
    # search_files: ``pattern`` is a glob; treat any non-empty glob as
    # potentially hitting the Vault. Be conservative — block in enforce
    # mode when the pattern is non-empty.
    pattern = args.get("pattern") or args.get("query") or ""
    if isinstance(pattern, str) and pattern.strip():
        return vault_root  # sentinel: "yes, this hits the Vault"
    return None


def make_pre_tool_call_hook(ctx):
    """Build the ``pre_tool_call`` hook closure."""
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
    delegate each call into ``handler.handle()``. The handler owns its
    own per-turn counter state.
    """
    handler = VaultContextHandler(cfg=resolved.cfg)

    def _tool_handler(args: Optional[Dict[str, Any]] = None) -> str:
        return handler.handle(args or {})

    return _tool_handler


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

    logger.info(
        "vault-retrieval registered: mode=%s vault_root=%s",
        resolved.cfg.get("mode"),
        resolved.cfg.get("vault_root"),
    )
