"""Subagent result post-processing: summary budget/spill, tool-trace summaries, lifecycle hooks and cost rollup."""

from __future__ import annotations

import logging
import json
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

def _stringify_tool_content(content: Any) -> str:
    """Stable text for tool-result content. Some OpenAI-compatible paths return
    content-block lists; observability must never crash on them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] if isinstance(item, dict) and isinstance(item.get("text"), str)
            else json.dumps(item, ensure_ascii=False, default=str) if isinstance(item, dict)
            else str(item)
            for item in content
        )
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)

def _looks_like_error_output(content: Any) -> bool:
    """Conservative error detector for tool-result previews: structured JSON with
    an ``error`` key or an error/failed ``status``, or a first line starting with a
    classic error marker. (Substring "error" alone painted normal output red.)"""
    content = _stringify_tool_content(content)
    if not content:
        return False
    if content.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and (
            parsed.get("error") or str(parsed.get("status") or "").strip().lower() in {"error", "failed", "failure", "timeout"}
        ):
            return True
    first = content.splitlines()[0].strip().lower() if content.splitlines() else ""
    return first.startswith(("error:", "failed:", "traceback ", "exception:"))

def _extract_output_tail(result: Dict[str, Any], *, max_entries: int = 12, max_chars: int = 8000) -> List[Dict[str, Any]]:
    """Last N tool-call results ``{tool, preview, is_error}`` from a child's conversation (the overlay's "Output"
    section), chronological order. Content blocks are flattened first so a block-wrapped "Error: ..." is still
    flagged; line structure is preserved (capped at ``max_chars``) so the overlay shows real output rather than a
    whitespace-collapsed blob."""
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return []
    name_by_call_id = {
        tc["id"]: str((tc.get("function") or {}).get("name") or "tool")
        for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"
        for tc in msg.get("tool_calls") or [] if tc.get("id")
    }
    tail: List[Dict[str, Any]] = []
    for msg in reversed(messages):  # newest first, then restore order below
        if len(tail) >= max_entries:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = _stringify_tool_content(msg.get("content") or "")
        tool_name = name_by_call_id.get(msg.get("tool_call_id") or "", "tool")
        tail.append({"tool": tool_name, "preview": content[:max_chars], "is_error": _looks_like_error_output(content)})
    tail.reverse()
    return tail

_TOOL_INPUT_TARGET_KEYS = frozenset({
    "cwd", "destination_path", "directory", "dst", "endpoint", "file_path", "new_path", "old_path", "path",
    "source_path", "src", "target_path", "url", "urls",
})
_TOOL_INPUT_URL_KEYS = frozenset({"endpoint", "url", "urls"})

def _sanitize_tool_target(key: str, value: Any) -> Any:
    """Keep bounded side-effect targets while dropping URL secrets."""
    if isinstance(value, list):
        cleaned = [item for item in (_sanitize_tool_target(key, item) for item in value[:16]) if item is not None]
        return cleaned or None
    if not isinstance(value, str) or not value:
        return None
    bounded = value[:1024]
    if key in _TOOL_INPUT_URL_KEYS:
        try:
            parsed = urlsplit(bounded)
            if parsed.scheme and parsed.netloc:
                hostname = parsed.hostname
                if not hostname:
                    return None
                # ``SplitResult.netloc`` includes ``user:password@``. Rebuild the authority from parsed host/port so
                # hook-visible history cannot carry URL credentials. Bracket IPv6 literals before appending a
                # validated port.
                host = f"[{hostname}]" if ":" in hostname else hostname
                netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
                return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except ValueError:
            return None
    return bounded

def _sanitize_targets(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only known side-effect target keys, each sanitized (URL secrets dropped)."""
    targets: Dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = str(raw_key).lower()
        if key in _TOOL_INPUT_TARGET_KEYS:
            cleaned = _sanitize_tool_target(key, value)
            if cleaned is not None:
                targets[key] = cleaned
    return targets

def _input_summary(keys: Any, targets: Any) -> Dict[str, Any]:
    """``{argument_keys, targets}`` with bounded, sanitized contents (empty on bad shapes)."""
    return {
        "argument_keys": [str(key)[:128] for key in keys[:64]] if isinstance(keys, list) else [],
        "targets": _sanitize_targets(targets) if isinstance(targets, dict) else {},
    }

def _summarize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Summarize argument names and side-effect targets without raw payloads."""
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else None
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        return _input_summary([], {})
    return _input_summary(sorted(str(key)[:128] for key in parsed), parsed)

def _subagent_stop_tool_call_history(tool_trace: Any) -> List[Dict[str, Any]]:
    """Detached, metadata-only tool history for lifecycle hooks (input summaries re-sanitized)."""
    if not isinstance(tool_trace, list):
        return []

    def _byte_count(item, key: str) -> int:
        value = item.get(key, 0)
        return max(0, int(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    history: List[Dict[str, Any]] = []
    for item in tool_trace:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown").lower()
        summary = item.get("input_summary")
        summary = summary if isinstance(summary, dict) else {}
        history.append({
            "tool_name": str(item.get("tool") or "unknown")[:256],
            "tool_input": _input_summary(summary.get("argument_keys"), summary.get("targets")),
            "input_bytes": _byte_count(item, "args_bytes"), "output_bytes": _byte_count(item, "result_bytes"),
            "status": status if status in {"ok", "error"} else "unknown",
        })
    return history

# Hard per-summary character ceiling layered on top of the dynamic headroom budget (see _apply_summary_budget):
# belt-and-suspenders for models that ignore "be concise". 0 disables the ceiling.
DEFAULT_MAX_SUMMARY_CHARS = 24000
# Fraction of the parent's *remaining* context headroom the whole batch of summaries may consume, split per summary,
# so N children can't collectively blow the parent's window (the compression/429 death spiral).
_SUMMARY_HEADROOM_FRACTION = 0.5
# Floor so a single summary always gets a usable slice even when the parent is
# already nearly full — below this we'd be truncating to noise.
_MIN_SUMMARY_CHARS = 2000

def _spill_summary_to_file(task_index: int, summary: str) -> Optional[str]:
    """Write the full summary under ``cache/delegation`` (mounted read-only into remote backends via
    ``credential_files._CACHE_DIRS``, so the parent's terminal/``read_file`` can page it on any backend). Absolute
    path, or None on failure — the trimmed head+tail is still returned regardless."""
    try:
        from hermes_constants import get_hermes_dir
        import datetime as _dt
        cache_dir = get_hermes_dir("cache/delegation", "delegation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"subagent-summary-{task_index}-{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
        from tools.spill_safety import write_text_exclusive
        # Exclusive symlink-refusing create; not private because cache/delegation is bind-mounted read-only into
        # remote backends whose container UID must be able to read it.
        write_text_exclusive(path, summary, private=False)
        return str(path)
    except Exception as exc:
        logger.debug("Failed to spill subagent summary to file: %s", exc)
        return None

def _trim_summary_with_footer(summary: str, cap: int, task_index: int) -> tuple[str, Optional[str]]:
    """``(model_text, spill_path)`` for one over-budget summary: a ~75% head / ~25% tail window snapped to line
    boundaries (so the opening AND the closing outcomes/files-changed/issues both survive), the full text spilled
    to disk, and a footer giving the exact ``read_file offset=`` for the omitted middle."""
    original_len = len(summary)
    head_budget = int(cap * 0.75)
    tail_budget = cap - head_budget
    head, tail = summary[:head_budget], summary[-tail_budget:]
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    spill_path = _spill_summary_to_file(task_index, summary)
    footer_lines = [
        "", "─" * 8 + " [SUMMARY TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {original_len:,} total — trimmed to protect the parent's context window.",
    ]
    if spill_path:
        # read_file is 1-indexed; +2 moves past the last head line shown.
        footer_lines.append(f"Full subagent output saved to: {spill_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{spill_path}" '
            f"offset={head.count(chr(10)) + 2} limit=200  (the file is the complete "
            f"summary; raise/lower offset to page through it)."
        )
    else:
        footer_lines.append("Full output could not be stored to disk; the head+tail above is all that was preserved.")
    footer_lines.append("─" * 37)
    return head + "\n\n[... middle omitted — see footer ...]\n\n" + tail + "\n".join(footer_lines), spill_path

def _parent_summary_char_budget(parent_agent, n_summaries: int) -> Optional[int]:
    """Per-summary char budget from the parent's *remaining* context headroom (context length − prompt tokens − the
    compressor's output reserve), a fraction of it split across the batch at ~4 chars/token. None when the parent's
    context state is unknown — caller then uses the static ceiling only."""
    try:
        compressor = getattr(parent_agent, "context_compressor", None)
        context_length = getattr(compressor, "context_length", None)
        if not isinstance(context_length, int) or context_length <= 0:
            return None
        used_tokens = getattr(parent_agent, "session_prompt_tokens", 0)
        if not isinstance(used_tokens, (int, float)) or used_tokens < 0:
            used_tokens = 0
        headroom_tokens = context_length - int(used_tokens) - int(getattr(compressor, "max_tokens", 0) or 0)
        if headroom_tokens <= 0:
            return _MIN_SUMMARY_CHARS  # parent already over budget: floor only
        per_summary_tokens = int(headroom_tokens * _SUMMARY_HEADROOM_FRACTION) // max(1, n_summaries)
        return max(_MIN_SUMMARY_CHARS, per_summary_tokens * 4)
    except Exception:
        logger.debug("Summary budget computation failed", exc_info=True)
        return None

def _apply_summary_budget(results: List[Dict[str, Any]], parent_agent) -> None:
    """Trim subagent summaries in-place so a batch can't overflow the parent's context window (full text spilled to
    disk). Per-summary cap = MIN(dynamic headroom budget, static ``delegation.max_summary_chars`` ceiling; 0 =
    disabled); over-cap summaries become head+tail plus a pointer to the spill file."""
    from tools.delegate_tool import _load_config
    summaries = [r for r in results if isinstance(r, dict) and isinstance(r.get("summary"), str) and r["summary"]]
    if not summaries:
        return
    try:
        static_ceiling = int(_load_config().get("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS))
    except (TypeError, ValueError):
        static_ceiling = DEFAULT_MAX_SUMMARY_CHARS
    candidates = [c for c in (static_ceiling, _parent_summary_char_budget(parent_agent, len(summaries))) if c and c > 0]
    if not candidates:
        return  # both disabled / unknown → leave summaries untouched
    cap = min(candidates)
    for entry in summaries:
        summary = entry["summary"]
        if len(summary) <= cap:
            continue
        model_text, spill_path = _trim_summary_with_footer(summary, cap, entry.get("task_index", -1))
        entry["summary"] = model_text
        entry["summary_truncated"] = True
        if spill_path:
            entry["summary_full_path"] = spill_path
        logger.debug(
            "[subagent-%s] summary trimmed %d → ~%d chars (spill=%s)", entry.get("task_index", "?"), len(summary), cap,
            spill_path or "none",
        )

_PARENT_FINALIZATION_LOCK_GUARD = threading.Lock()
_PARENT_FINALIZATION_FALLBACK_LOCK = threading.RLock()
_CHILD_CONSTRUCTION_LOCK = threading.RLock()

def _build_child_preserving_parent_tools(**kwargs):
    """Build a child without leaking its resolved toolset into the parent."""
    from tools.delegate_tool import _build_child_agent
    import model_tools
    with _CHILD_CONSTRUCTION_LOCK:
        parent_tool_names = list(model_tools._last_resolved_tool_names)
        try:
            child = _build_child_agent(**kwargs)
        finally:
            model_tools._last_resolved_tool_names = parent_tool_names
    child._delegate_saved_tool_names = parent_tool_names
    return child

def _parent_finalization_lock(parent_agent) -> threading.RLock:
    """Per-parent lock serializing lifecycle side effects (created once under the guard)."""
    if parent_agent is None:
        return _PARENT_FINALIZATION_FALLBACK_LOCK
    lock = getattr(parent_agent, "_subagent_finalization_lock", None)
    if lock is not None:
        return lock
    with _PARENT_FINALIZATION_LOCK_GUARD:
        lock = getattr(parent_agent, "_subagent_finalization_lock", None)
        if lock is None:
            lock = threading.RLock()
            try:
                setattr(parent_agent, "_subagent_finalization_lock", lock)
            except Exception:
                return _PARENT_FINALIZATION_FALLBACK_LOCK
    return lock

def _notify_memory_manager(results, task_list, child_by_index, parent_agent) -> None:
    memory = getattr(parent_agent, "_memory_manager", None) if parent_agent else None
    if not memory:
        return
    for entry in results:
        try:
            task_index = entry.get("task_index", -1)
            in_range = isinstance(task_index, int) and 0 <= task_index < len(task_list)
            memory.on_delegation(
                task=task_list[task_index]["goal"] if in_range else "", result=entry.get("summary", "") or "",
                child_session_id=getattr(child_by_index.get(task_index), "session_id", ""),
            )
        except Exception:
            pass

def _fire_subagent_stop_hooks(results, child_by_index, parent_agent) -> float:
    """Pop the model-hidden ``_child_role`` / ``_child_cost_usd`` fields from every
    entry, fire ``subagent_stop`` per child, and return the summed child cost."""
    try:
        from hermes_cli.plugins import invoke_hook as invoke_hook
    except Exception:
        invoke_hook = None
    children_cost_total = 0.0
    for entry in results:
        child_role = entry.pop("_child_role", None)
        child_cost = entry.pop("_child_cost_usd", 0.0)
        try:
            if child_cost:
                children_cost_total += float(child_cost)
        except (TypeError, ValueError):
            pass
        if invoke_hook is None:
            continue
        try:
            child = child_by_index.get(entry.get("task_index", -1))
            invoke_hook(
                "subagent_stop", parent_session_id=getattr(parent_agent, "session_id", None),
                parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
                child_session_id=getattr(child, "session_id", None), child_role=child_role,
                child_summary=entry.get("summary"), child_status=entry.get("status"),
                tool_call_history=_subagent_stop_tool_call_history(entry.get("tool_trace")),
                duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
            )
        except Exception:
            logger.debug("subagent_stop hook invocation failed", exc_info=True)
    return children_cost_total

def _rollup_children_cost(parent_agent, children_cost_total: float) -> None:
    """Fold the children's spend into the parent's session cost (source/status
    only set when the parent had none of its own)."""
    if children_cost_total <= 0.0:
        return
    try:
        current = float(getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0)
        parent_agent.session_estimated_cost_usd = current + children_cost_total
        if getattr(parent_agent, "session_cost_source", "none") in {None, "", "none"}:
            parent_agent.session_cost_source = "subagent"
        if getattr(parent_agent, "session_cost_status", "unknown") in {None, "", "unknown"}:
            parent_agent.session_cost_status = "estimated"
    except Exception:
        logger.debug("Subagent cost rollup failed", exc_info=True)

def _finalize_child_results(
    results: List[Dict[str, Any]], task_list: List[Dict[str, Any]], children: List[tuple[int, Dict[str, Any], Any]],
    parent_agent,
) -> None:
    """Apply host-owned summary, memory, hook, and cost contracts once."""
    with _parent_finalization_lock(parent_agent):
        _apply_summary_budget(results, parent_agent)
        child_by_index = {index: child for index, _task, child in children}
        _notify_memory_manager(results, task_list, child_by_index, parent_agent)
        _rollup_children_cost(parent_agent, _fire_subagent_stop_hooks(results, child_by_index, parent_agent))

def _run_child_lifecycle(task_index: int, goal: str, child=None, parent_agent=None) -> Dict[str, Any]:
    """Run one child and apply the same host lifecycle used by delegate_task."""
    from tools.delegate_tool import _run_single_child
    result = _run_single_child(task_index, goal, child, parent_agent)
    result.setdefault("task_index", task_index)
    task = {"goal": goal}
    _finalize_child_results([result], [{"goal": ""} for _ in range(task_index)] + [task], [(task_index, task, child)], parent_agent)
    return result
