"""The ``vault_context`` tool handler.

Implements the locked tool API from the architecture memo:

  query: required string
  project, topic, decision_horizon: optional strings
  selectors[]: {path, heading?, line_start?, line_end?}
  budget_chars: default 12,000, max 24,000
  allow_bounded_expansion: default false
  expansion_reason: required if allow_bounded_expansion=true

Returns one JSON object:
  {
    status: "ok" | "data_hold" | "unknown",
    query_id, scope, budget, candidates, extracts, conflicts,
    redactions, hold, log_ref
  }

The handler is profile-scoped: state paths come from get_hermes_home()
via the plugin's state_dir. The turn_id is injected by the hook (or by
the test harness in unit tests). Multiple calls in one turn share one
24,000-character ceiling via the per-turn registry.

Acceptance contract: #2 (default ≤ 12k), #4 (24k; 24,001 rejected),
#5 (turn-shared ceiling), #6 (large-file refusal), #8 (path safety),
#12 (each extract has path/range/freshness), #13 (redaction),
#15 (unwritable log -> data_hold), #16 (no cache files).

Decision order (matches the architecture memo):
  1. Validate args shape.
  2. For each selector: resolve_safe_path (path containment).
  3. For each selector: large-file gate
     (file > 20k + no line_start => large_file_selector_required).
  4. Extract.
  5. Budget check (sum of actual extracted chars vs per-turn ceiling).
  6. Redact extracted content.
  7. Append to query log (fail-closed: log unavailable => data_hold).
  8. Return envelope.
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .budgets import (
    BudgetConfig,
    BudgetDecision,
    TurnBudgets,
    check_large_file,
    decide_budget,
    get_or_create_turn,
    reset_turn,
)
from .query_log import QueryLogEntry, QueryLogWriteError, QueryLogger
from .paths import (
    default_config,
    resolve_vault_root,
    resolve_safe_path,
    validate_config,
    VaultConfigError,
    VaultPathError,
)
from .redaction import Redactor
from .retrieval import (
    Extract,
    ExtractionError,
    LargeFileRefused,
    extract_from_selectors,
)


TOOL_NAME = "vault_context"
TOOLSET = "vault_retrieval"


def vault_context_tool_schema() -> Dict[str, Any]:
    """The JSON schema exposed to the model — read-only."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Token-efficient read-only retrieval against the configured Obsidian "
            "Vault. Filename-first, targeted range reads, strict per-turn budgets "
            "(default 12,000 chars, hard ceiling 24,000 chars), large-file refusal. "
            "Returns a JSON envelope with status, candidates, extracts, conflicts, "
            "and redaction counts. Use this instead of read_file/search_files for "
            "any Vault evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The retrieval question or topic (NOT logged as raw text).",
                },
                "project": {"type": "string"},
                "topic": {"type": "string"},
                "decision_horizon": {"type": "string"},
                "selectors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "heading": {"type": "string"},
                            "line_start": {"type": "integer"},
                            "line_end": {"type": "integer"},
                            "estimated_chars": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
                "budget_chars": {
                    "type": "integer",
                    "description": "Planned character count for this call (capped at hard ceiling).",
                    "default": 12_000,
                    "minimum": 0,
                    "maximum": 24_000,
                },
                "allow_bounded_expansion": {"type": "boolean", "default": False},
                "expansion_reason": {
                    "type": "string",
                    "description": "Required when allow_bounded_expansion=true; non-empty.",
                },
                "turn_id": {
                    "type": "string",
                    "description": "Internal turn key injected by the plugin hook (not advertised).",
                },
            },
            "required": ["query"],
        },
    }


def build_envelope(
    *,
    status: str,
    query_id: str,
    scope: Dict[str, Any],
    budget_used: int,
    budget_default: int,
    budget_ceiling: int,
    budget_expanded: bool,
    candidates: List[Dict[str, Any]],
    extracts: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    redactions: Dict[str, Any],
    hold: Optional[Dict[str, Any]],
    log_ref: str,
) -> Dict[str, Any]:
    """Build the deterministic envelope."""
    return {
        "status": status,
        "query_id": query_id,
        "scope": scope,
        "budget": {
            "default_chars": budget_default,
            "hard_ceiling_chars": budget_ceiling,
            "used_chars": budget_used,
            "remaining_chars": max(0, budget_ceiling - budget_used),
            "expanded": budget_expanded,
        },
        "candidates": candidates,
        "extracts": extracts,
        "conflicts": conflicts,
        "redactions": redactions,
        "hold": hold,
        "log_ref": log_ref,
    }


def _new_query_id() -> str:
    """Opaque query id: RQL-<UTC-timestamp>-<random hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = _secrets.token_hex(8)
    return f"RQL-{ts}-{rand}"


def _extract_to_dict(ex: Extract) -> Dict[str, Any]:
    return {
        "path": ex.path,
        "heading": ex.heading,
        "line_start": ex.line_start,
        "line_end": ex.line_end,
        "chars": ex.chars,
        "content": ex.content,
        "source_mtime": ex.source_mtime,
    }


@dataclass
class VaultContextHandler:
    """Tool handler. One instance per turn (or one per profile per turn)."""

    cfg: Dict[str, Any]
    turn_id: str = "default"
    _root: Path = field(init=False)
    _budget_cfg: BudgetConfig = field(init=False)
    _logger: Optional[QueryLogger] = field(default=None, init=False)
    _redactor: Redactor = field(default_factory=Redactor, init=False)

    def __post_init__(self) -> None:
        # Re-validate at runtime so a stale config can never pass.
        self.cfg = validate_config(self.cfg)
        self._root = Path(self.cfg["vault_root"])
        self._budget_cfg = BudgetConfig(
            default_budget_chars=self.cfg["default_budget_chars"],
            hard_ceiling_chars=self.cfg["hard_ceiling_chars"],
            large_file_chars=self.cfg["large_file_chars"],
            max_primary_extracts=self.cfg["max_primary_extracts"],
            max_expansion_extracts=self.cfg["max_expansion_extracts"],
            max_range_lines=self.cfg["max_range_lines"],
            max_range_chars=self.cfg["max_range_chars"],
        )
        # Init the logger; failure is fatal per spec — fail closed.
        # Per the locked contract, the log path is
        # ``<state_dir>/vault-retrieval/query-log.jsonl`` (state_dir is the
        # profile's ``state/`` root, the plugin owns its own subdir).
        state_root = self.cfg.get("state_dir") or str(
            Path.home() / ".hermes" / "state"
        )
        plugin_state = Path(state_root) / "vault-retrieval"
        try:
            self._logger = QueryLogger(
                plugin_state,
                query_log_enabled=bool(self.cfg.get("query_log_enabled", True)),
            )
        except QueryLogWriteError:
            self._logger = None  # log unavailable — fail-closed at handle time

    def handle(self, args: Dict[str, Any]) -> str:
        """The tool handler entrypoint — return a JSON string."""
        if not isinstance(args, dict):
            return json.dumps(self._fail("malformed_args", "args must be a dict"))
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return json.dumps(self._fail("missing_query", "query is required"))
        selectors = args.get("selectors", []) or []
        if not isinstance(selectors, list):
            return json.dumps(self._fail("malformed_selectors", "selectors must be a list"))

        turn_id = args.get("turn_id") or self.turn_id or "default"
        allow_expansion = bool(args.get("allow_bounded_expansion", False))
        expansion_reason = args.get("expansion_reason", "") or ""

        # ---- Step 1: path safety + large-file gate (no reads yet) ----
        resolved: List[Tuple[Dict[str, Any], Path, bool]] = []  # (sel, abs_path, is_large)
        for sel in selectors:
            requested = sel.get("path", "")
            if not isinstance(requested, str) or not requested:
                return json.dumps(self._hold(
                    "path_or_extract_error", f"selector missing path: {sel!r}",
                    turn_id=turn_id, query=query, selectors=selectors,
                ))
            try:
                abs_path = resolve_safe_path(self._root, requested)
            except VaultPathError as exc:
                return json.dumps(self._hold(
                    "path_or_extract_error", str(exc),
                    turn_id=turn_id, query=query, selectors=selectors,
                ))
            try:
                file_chars = abs_path.stat().st_size
            except OSError as exc:
                return json.dumps(self._hold(
                    "path_or_extract_error", f"cannot stat {requested}: {exc}",
                    turn_id=turn_id, query=query, selectors=selectors,
                ))
            is_large = file_chars > self._budget_cfg.large_file_chars
            if is_large and sel.get("line_start") is None:
                return json.dumps(self._hold(
                    "large_file_selector_required",
                    (
                        f"file_chars {file_chars} exceeds large_file threshold "
                        f"{self._budget_cfg.large_file_chars}; a heading, stable ID, "
                        "date, owner/status key, or explicit line selector is required"
                    ),
                    turn_id=turn_id, query=query, selectors=selectors,
                ))
            resolved.append((sel, abs_path, is_large))

        # ---- Step 2: extract (large-file selectors apply) ----
        extracts: List[Extract] = []
        candidates_meta: List[Dict[str, Any]] = []
        for sel, abs_path, is_large in resolved:
            if sel.get("line_start") is not None:
                try:
                    lfd = check_large_file(
                        cfg=self._budget_cfg,
                        file_chars=abs_path.stat().st_size,
                        selectors=[sel],
                    )
                    if lfd.is_large and lfd.decision != "ok":
                        return json.dumps(self._hold(
                            lfd.decision, lfd.detail or "",
                            turn_id=turn_id, query=query, selectors=selectors,
                        ))
                    exs = extract_from_selectors(
                        abs_path,
                        [sel],
                        is_large=is_large,
                        max_ranges=self._budget_cfg.max_primary_extracts,
                        max_range_lines=self._budget_cfg.max_range_lines,
                        max_range_chars=self._budget_cfg.max_range_chars,
                    )
                    extracts.extend(exs)
                except (ExtractionError, LargeFileRefused) as exc:
                    return json.dumps(self._hold(
                        "path_or_extract_error", str(exc),
                        turn_id=turn_id, query=query, selectors=selectors,
                    ))
            candidates_meta.append({
                "path": sel.get("path"),
                "chars_total": abs_path.stat().st_size,
            })

        # ---- Step 3: budget check on actual extracted chars ----
        # The default budget (12,000) is a CEILING, not a cost. We charge
        # the caller's planned budget_chars only if they explicitly opt
        # in by passing the argument — otherwise we charge the actual
        # extracted content size. If actual exceeds the planned budget,
        # we charge the actual so the ceiling is enforced truthfully.
        total_chars = sum(e.chars for e in extracts)
        if "budget_chars" in args:
            planned = args["budget_chars"]
            if not isinstance(planned, int) or planned < 0:
                planned = 0
            planned = min(planned, self._budget_cfg.hard_ceiling_chars)
            request_chars = max(total_chars, planned) if total_chars > 0 else planned
        else:
            # No explicit budget hint — charge actual extracted chars.
            # Discovery-only calls (no extracts) consume zero evidence.
            request_chars = total_chars
        tb = get_or_create_turn(turn_id, self._budget_cfg)
        decision = decide_budget(
            tb,
            request_chars=request_chars,
            allow_expansion=allow_expansion,
            expansion_reason=expansion_reason,
        )
        if decision.status == "data_hold":
            return json.dumps(self._hold(
                decision.reason_code or "budget_exceeded",
                decision.detail or "",
                turn_id=turn_id, query=query, selectors=selectors,
            ))

        # ---- Step 4: redact ----
        redactions_total = {"secrets": 0, "pii": 0, "classes": []}
        new_extracts: List[Extract] = []
        for ex in extracts:
            scrubbed, counts = self._redactor.redact_with_counts(ex.content)
            redactions_total["secrets"] += counts.secrets
            redactions_total["pii"] += counts.pii
            new_extracts.append(Extract(
                path=ex.path, line_start=ex.line_start, line_end=ex.line_end,
                chars=len(scrubbed), content=scrubbed, source_mtime=ex.source_mtime,
                heading=ex.heading,
            ))
        extracts = new_extracts

        # ---- Step 5: query log (fail-closed) ----
        if self._logger is None:
            return json.dumps(self._hold(
                "query_log_unavailable", "query logger not initialised",
                turn_id=turn_id, query=query, selectors=selectors,
            ))
        qid = _new_query_id()
        try:
            self._logger.append(QueryLogEntry(
                query_id=qid,
                queried_at=datetime.now(timezone.utc).isoformat(),
                task_id=None,
                run_id=None,
                project_topic=(args.get("project") or "") + ":" + (args.get("topic") or ""),
                decision_horizon=args.get("decision_horizon") or "",
                search_terms=re.split(r"\s+", query.strip()),
                consulted=[
                    {
                        "path": sel.get("path", ""),
                        "range": f"lines {sel.get('line_start', '?')}-{sel.get('line_end', sel.get('line_start', '?'))}",
                        "source_mtime": datetime.fromtimestamp(
                            abs_path.stat().st_mtime, tz=timezone.utc
                        ).isoformat() if abs_path.exists() else "",
                        "source_last_updated": None,
                        "characters_read": sum(e.chars for e in extracts if e.path == str(abs_path)),
                    }
                    for sel, abs_path, _is_large in resolved
                ],
                total_characters_read=total_chars,
                budget_status="expanded" if decision.expanded else "default",
                outcome="SUFFICIENT",
                data_hold_reason=None,
                redaction_counts={
                    "secrets": redactions_total["secrets"],
                    "pii": redactions_total["pii"],
                },
            ))
        except QueryLogWriteError as exc:
            return json.dumps(self._hold(
                "query_log_unavailable", str(exc),
                turn_id=turn_id, query=query, selectors=selectors,
            ))

        # ---- Step 6: envelope ----
        envelope = build_envelope(
            status="ok",
            query_id=qid,
            scope={
                "project": args.get("project"),
                "topic": args.get("topic"),
                "decision_horizon": args.get("decision_horizon"),
            },
            budget_used=decision.used_chars,
            budget_default=self._budget_cfg.default_budget_chars,
            budget_ceiling=self._budget_cfg.hard_ceiling_chars,
            budget_expanded=decision.expanded,
            candidates=candidates_meta,
            extracts=[_extract_to_dict(e) for e in extracts],
            conflicts=[],
            redactions=redactions_total,
            hold=None,
            log_ref=f"query-log.jsonl:{qid}",
        )
        return json.dumps(envelope)

    def _fail(self, reason_code: str, detail: str) -> Dict[str, Any]:
        return build_envelope(
            status="data_hold",
            query_id=_new_query_id(),
            scope={},
            budget_used=0,
            budget_default=self._budget_cfg.default_budget_chars,
            budget_ceiling=self._budget_cfg.hard_ceiling_chars,
            budget_expanded=False,
            candidates=[],
            extracts=[],
            conflicts=[],
            redactions={"secrets": 0, "pii": 0, "classes": []},
            hold={"reason_code": reason_code, "detail": detail, "missing_or_conflicting_sources": []},
            log_ref="query-log.jsonl:uninitialised",
        )

    def _hold(
        self, reason_code: str, detail: str, *,
        turn_id: str, query: str, selectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return build_envelope(
            status="data_hold",
            query_id=_new_query_id(),
            scope={},
            budget_used=0,
            budget_default=self._budget_cfg.default_budget_chars,
            budget_ceiling=self._budget_cfg.hard_ceiling_chars,
            budget_expanded=False,
            candidates=[],
            extracts=[],
            conflicts=[],
            redactions={"secrets": 0, "pii": 0, "classes": []},
            hold={
                "reason_code": reason_code,
                "detail": detail,
                "missing_or_conflicting_sources": [s.get("path", "?") for s in selectors],
            },
            log_ref=f"query-log.jsonl:{turn_id}",
        )
