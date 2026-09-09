"""Human-readable rendering of ``ProcessRegistry.completion_queue`` events (completion,
watch_match, watch_disabled, watch_overflow_*, async_delegation) into the
``[IMPORTANT: ...]`` / ``[ASYNC DELEGATION ...]`` text the CLI drain loop, gateway and
TUI inject into the agent conversation."""

import time
from dataclasses import dataclass
from contextlib import suppress

_DONE = ("completed", "success")


@dataclass(frozen=True, slots=True)
class ProcessNotificationBatch:
    """Keep completion identity until the owning surface starts its turn."""

    notifications: tuple[tuple[dict, str], ...]

    def render(self, registry) -> str | None:
        messages = [text for event, text in self.notifications
                    if not registry.is_completion_consumed(event.get("session_id", ""))]
        if not messages:
            return None
        if len(messages) == 1:
            return messages[0]
        header = (f"[IMPORTANT: {len(messages)} background processes completed. "
                  "Treat these results as one batch and give one consolidated response; "
                  "preserve failures and actionable results.]")
        return "\n\n".join((header, *messages))


def group_process_notifications(notifications):
    """Group consecutive completions only; watches and delegations are barriers."""
    batch = []
    for event, text in notifications:
        if event.get("type", "completion") == "completion":
            batch.append((event, text))
        else:
            if batch:
                yield tuple(batch)
                batch = []
            yield ((event, text),)
    if batch:
        yield tuple(batch)



def _format_age(seconds: float) -> str:
    """Human-friendly elapsed string ('18m', '2h3m', '45s')."""
    try:
        s = int(max(0, seconds))
    except (TypeError, ValueError):
        return "?"
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m" + (f"{s}s" if s else "")
    h, m = divmod(m, 60)
    return f"{h}h" + (f"{m}m" if m else "")


def _model_not_found_patterns() -> "list[str]":
    """Model-not-found phrases from ``agent.error_classifier`` (the failover path's
    own list, so nothing drifts); a minimal built-in set if the import fails.

    Imported from ``agent.error_classifier`` so the batch renderer applies the SAME classification the
    failover path consumes — no hand-copied pattern list to drift. Fails open to a minimal built-in set so a
    classifier import problem never hides the per-task blocks. See #97667.
    """
    try:
        from agent.error_classifier import _MODEL_NOT_FOUND_PATTERNS
        return list(_MODEL_NOT_FOUND_PATTERNS)
    except Exception:
        return ["is not a valid model", "model not found", "model_not_found"]


def _delegation_config() -> dict:
    """Active delegation config; ``{}`` on any error. Lazy: delegate_tool is heavy."""
    try:
        from tools.delegate_tool import _load_config as _cfg
        return _cfg() or {}
    except Exception:
        return {}


def _delegation_model_not_found(results, config) -> bool:
    """True when a result reflects a config-level model_not_found rejection: needs a
    model-not-found phrase AND the currently-configured model name in the same text,
    so a stale task failing on a removed model is not mis-attributed to the config."""
    model = str((config or {}).get("model") or "").lower()
    if not model:
        return False
    patterns = _model_not_found_patterns()
    texts = (" ".join(str(x) for x in (r.get("error"), r.get("summary")) if x).lower() for r in results or [])
    return any(model in text and any(p in text for p in patterns) for text in texts)


def _delegation_model_not_found_notice(results) -> "list[str] | None":
    """Config-level model_not_found notice lines, or None (fail-open) — once per batch."""
    config = _delegation_config()
    if not _delegation_model_not_found(results, config):
        return None
    model = config.get("model") or "?"
    provider = config.get("provider") or "configured provider"
    lines = [
        "⚠ SUBAGENT MODEL REJECTED: the configured Subagent Model "
        f'"{model}" was rejected by provider "{provider}" '
        "(HTTP 400: not a valid model ID).",
        "Every task in this batch failed for this reason before doing any work.",
        "Check Settings → Advanced → Subagent Model (or: hermes config get delegation.model)."]
    with suppress(Exception):
        from hermes_cli.fallback_config import get_fallback_chain
        if not get_fallback_chain(config):
            lines.append("No fallback chain is configured, so no failover was attempted.")
    return lines


_TRUNCATED_SUMMARY_NOTE = (
    "[TRUNCATED — subagent hit its iteration cap; the summary below "
    "may be incomplete. Verify before relying on it, or re-dispatch "
    "the unfinished part.]")


def _is_truncated(entry: dict) -> bool:
    return bool(entry.get("truncated") or entry.get("exit_reason") == "max_iterations")


def _notice_lines(results) -> "list[str]":
    """Blank + model_not_found notice block, or [] when the notice does not apply."""
    notice = _delegation_model_not_found_notice(results)
    return ["", *notice] if notice else []


def _preamble(evt: dict, title: str, intro: str, completed_at: float, *, with_goal: bool) -> "list[str]":
    """Shared preamble: title, intro, blank, dispatch time, [goal], context/toolsets, role+model."""
    lines = [title, intro, ""]
    dispatched_at = evt.get("dispatched_at")
    if isinstance(dispatched_at, (int, float)):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at))
        lines.append(f"Dispatched: {ts} ({_format_age(completed_at - dispatched_at)} ago)")
    if with_goal:
        lines.append(f"Original goal: {evt.get('goal', '') or ''}")
    if evt.get("context"):
        lines.append(f"Context you provided: {evt['context']}")
    if evt.get("toolsets"):
        lines.append(f"Toolsets: {', '.join(evt['toolsets'])}")
    lines.append(f"Role: {evt.get('role') or 'leaf'}   Model: {evt.get('model') or '?'}")
    return lines


def _format_task_failure_notice(evt: dict, deleg_id: str) -> str:
    """One child of a still-running fan-out failed: say which, why, and that the batch goes on."""
    (r,) = (evt.get("results") or [{}])[:1] or [{}]
    goals, idx, n = evt.get("goals") or [], r.get("task_index", 0), evt.get("n_tasks") or len(evt.get("goals") or [])
    goal = goals[idx] if idx < len(goals) else r.get("goal", "")
    err = str(r.get("error") or "").strip().replace("\n", " ")[:400]
    lines = [
        f"[ASYNC DELEGATION TASK FAILED — {deleg_id}, task {idx + 1}/{n}]",
        "One subagent in a background fan-out you dispatched has failed while its siblings are still running. "
        "The batch's consolidated results will still arrive when the last sibling finishes; this is an early "
        "warning so you can re-dispatch or investigate now instead of then.",
        f"Task: {goal}" if goal else "",
        f"Status: {r.get('status', '?')}   Duration: {r.get('duration_seconds', '?')}s" + (f"\nError: {err}" if err else ""),
    ]
    if r.get("live_transcript"):
        lines.append(f"Live transcript: {r['live_transcript']}")
    return "\n".join(line for line in lines if line)


def _format_batch_delegation(evt: dict, deleg_id: str, completed_at: float) -> str:
    """Consolidated block for a delegate_task fan-out that finished as one unit."""
    results, goals = evt.get("results") or [], evt.get("goals") or []
    # ``goals`` is the whole delegate_task call (task_index indexes it); ``results`` is this unit's subset.
    n, n_unit = len(goals) or len(results), len(results) or len(goals)
    group = evt.get("group")
    unit = f"group '{group}' ({n_unit} subagent(s))" if group is not None else f"{n_unit} subagent(s)"
    lines = _preamble(
        evt,
        f"[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
        f"A background fan-out unit you dispatched earlier — {unit} — has finished; its consolidated results are "
        "below. Any other units from the same delegate_task call report separately as they finish. You may have "
        "moved on since dispatching — act on these or re-dispatch if things have changed. If you are still waiting "
        "on siblings, end your turn after acting on this one.",
        completed_at, with_goal=False)
    lines[-1] += f"   Total duration: {evt.get('total_duration_seconds', evt.get('duration_seconds', '?'))}s"
    if evt.get("error") and not results:
        lines += ["--- ERROR ---", f"The batch did not complete successfully: {evt['error']}"]
        return "\n".join(lines)
    # Config-level rejection notice BEFORE the per-task wall — a rejected
    # delegation model fails every task identically and must not stay buried.
    lines += _notice_lines(results)
    for r in sorted(results, key=lambda x: x.get("task_index", 0)):
        idx, r_truncated = r.get("task_index", 0), _is_truncated(r)
        r_status, r_summary, r_error = r.get("status", "?"), r.get("summary"), r.get("error")
        r_goal = goals[idx] if idx < len(goals) else r.get("goal", "")
        icon = "⚠" if r_truncated else ("✓" if r_status in _DONE else "✗")
        header = (f"--- {icon} TASK {idx + 1}/{n}" + (f": {r_goal}" if r_goal else "") + f"  (status={r_status}"
                  + (f", api_calls={r['api_calls']}" if r.get("api_calls") else "")
                  + (f", {r['duration_seconds']}s" if r.get("duration_seconds") is not None else "")
                  + (", TRUNCATED: hit max_iterations — work may be incomplete" if r_truncated else ""))
        lines += ["", header + ") ---"]
        if r_status in _DONE and r_summary:
            if r_truncated:
                lines.append(_TRUNCATED_SUMMARY_NOTE)
            lines.append(r_summary)
        elif r_summary:
            if r_error:
                lines.append(f"({r_status}: {r_error})")
            lines += ["Partial output:", r_summary]
        else:
            lines.append(f"(no summary — status={r_status}" + (f": {r_error}" if r_error else "") + ")")
        if r.get("live_transcript"):
            lines.append(f"Full live transcript (complete tool/assistant trace): {r['live_transcript']}")
        lines += _process_accounting_lines(r)
    return "\n".join(lines)


def _process_accounting_lines(r: dict) -> list:
    """Runtime-truth lines about a child's background processes: what it handed to you (you own it now, its
    completion lands here) and what it left running (terminated at teardown — never trust a child's "watcher running")."""
    lines = []
    for h in r.get("handed_off_processes") or []:
        lines.append(f"Handed off to you: {h.get('session_id')} ({h.get('command', '')[:120]}) — {h.get('note', '')}. "
                     "You own it now; its completion notice will arrive here.")
    orphans = r.get("orphaned_processes") or []
    if orphans:
        lines.append(f"Child left {len(orphans)} background process(es) running that were TERMINATED with it "
                     "(subagent process notices never reach you): "
                     + "; ".join(f"{o.get('session_id')} `{o.get('command', '')[:100]}` ({o.get('runtime_seconds')}s)" for o in orphans)
                     + ". Re-launch in this session anything you still need.")
    for u in r.get("unread_completions") or []:
        lines.append(f"Child's process {u.get('session_id')} `{u.get('command', '')[:100]}` finished (exit code "
                     f"{u.get('exit_code')}) but the child never read its result; output tail:\n{u.get('output_tail', '')}")
    return lines


def _format_async_delegation(evt: dict) -> str:
    """Self-contained re-injection for an async-delegation completion: the FULL
    original task source (goal, context, toolsets, role, model), dispatch time, status
    and result, so an agent deep in unrelated context can act on it or re-dispatch."""
    deleg_id = evt.get("delegation_id", "unknown")
    completed_at = evt.get("completed_at") or time.time()
    if evt.get("task_failure_notice"):
        return _format_task_failure_notice(evt, deleg_id)
    if evt.get("is_batch") or isinstance(evt.get("results"), list):
        return _format_batch_delegation(evt, deleg_id, completed_at)
    status, summary, error = evt.get("status") or "completed", evt.get("summary"), evt.get("error")
    truncated = _is_truncated(evt)
    lines = _preamble(
        evt,
        f"[ASYNC DELEGATION COMPLETE — {deleg_id}]",
        "A background subagent you dispatched earlier has finished. You may "
        "have moved on since dispatching it; the full task source is below so "
        "you can act on the result or re-dispatch if things have changed.",
        completed_at, with_goal=True)
    lines += _notice_lines([evt]) + [
        f"Status: {status}   API calls: {evt.get('api_calls', 0)}   Duration: {evt.get('duration_seconds', '?')}s"
        + (" [TRUNCATED: hit max_iterations — work may be incomplete]" if truncated else ""),
        "--- RESULT ---"]
    if status in _DONE and summary:
        if truncated:
            lines.append(_TRUNCATED_SUMMARY_NOTE)
        lines.append(summary)
    else:
        if status == "interrupted":
            lines.append("The subagent was interrupted before completing" + (f": {error}" if error else "."))
        else:  # error / timeout / failed
            lines.append(
                f"The subagent did not complete successfully (status={status})." + (f"\n{error}" if error else ""))
        if summary:
            lines += ["Partial output:", summary]
    return "\n".join(lines)


def async_delegation_display_text(evt: dict) -> str:
    """Compact UI title; the separate model notification retains all task evidence."""
    raw_results = evt.get("results")
    results = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    results = results or [evt]
    goals = evt.get("goals") or []
    labels, titles = [], []
    status_labels = {"failed": "Failed", "error": "Failed", "cancelled": "Cancelled",
                     "interrupted": "Interrupted", "timeout": "Timed Out", "stalled": "Stalled",
                     "unknown": "Unknown", "rejected": "Failed"}
    for result in results:
        status = result.get("status") or ("failed" if result.get("error") else "completed")
        label = ("Incomplete" if _is_truncated(result) else "Completed") if status in _DONE else (
            status_labels.get(status, "Incomplete"))
        labels.append(label)
        index = result.get("task_index", 0)
        goal = goals[index] if 0 <= index < len(goals) else result.get("goal", "")
        titles.append(" ".join(str(goal or "Background task").split()))
    if len(results) == 1:
        return f"Subagent Task {labels[0]}: {titles[0]}"
    outcome = labels[0] if len(set(labels)) == 1 else "Finished with Issues"
    title = " ".join(str(evt.get("group") or "").split()) or "; ".join(titles)
    return f"Subagent Tasks {outcome}: {title} ({len(results)} tasks)"


class SubagentNotification(str):
    """Keep queued model text string-compatible, with a separate human preview."""

    display_text: str

    def __new__(cls, text: str, event: dict):
        instance = super().__new__(cls, text)
        instance.display_text = async_delegation_display_text(event)
        return instance


def _delegation_attribution_line(evt: dict) -> "str | None":
    """One-line provenance for a subagent-owned process event, else None. Such a process
    outlives the child and lands in the PARENT conversation, which would otherwise see an
    anonymous output wall. Keyed on ``owner_task_id`` — ``task_id`` may be the session key."""
    task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "")
    if not task_id.startswith("sa-"):
        return None
    info = None
    with suppress(Exception):
        from tools.delegate_tool import get_subagent_attribution
        info = get_subagent_attribution(task_id)
    if not info:
        # Registry entry aged out — still attribute generically, not anonymously.
        return f"Started by subagent {task_id} (delegate_task)."
    goal, deleg = str(info.get("goal") or "").strip(), info.get("delegation_id")
    goal = goal[:117] + "..." if len(goal) > 120 else goal
    return (f"Started by subagent {task_id}" + (f" of delegation {deleg}" if deleg else "") + "."
            + (f' Task: "{goal}"' if goal else ""))


_REASON_STATUS = {"lost": "marked lost because the process backend disappeared", "failed_start": "failed to start"}


def _completion_status(evt: dict) -> str:
    reason = evt.get("completion_reason") or "exited"
    if reason == "killed":
        return f"terminated by {evt.get('termination_source') or 'Hermes'}"
    return _REASON_STATUS.get(reason) or ("completed normally" if evt.get("exit_code", "?") == 0 else "exited")


def format_process_notification(evt: dict) -> "str | None":
    """Format a completion_queue event into an ``[IMPORTANT: ...]`` message."""
    evt_type = evt.get("type", "completion")
    # watch_disabled and overflow events carry their own human-readable `message`;
    # otherwise overflow events would fall through to the completion formatter as a
    # phantom "process exited (exit code ?)".
    if evt_type in ("watch_disabled", "watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"
    if evt_type == "async_delegation":
        return _format_async_delegation(evt)
    _sid, _cmd = evt.get("session_id", "unknown"), evt.get("command", "unknown")
    _attribution = _delegation_attribution_line(evt)
    if evt.get("handoff_note"):
        _attribution = f"Handed off to you by a subagent before it finished. Purpose: {evt['handoff_note']}"
    attribution = f"{_attribution}\n" if _attribution else ""
    if evt_type == "watch_match":
        _sup = evt.get("suppressed", 0)
        return (
            f"[IMPORTANT: Background process {_sid} matched watch pattern \"{evt.get('pattern', '?')}\".\n"
            f"{attribution}Command: {_cmd}\nMatched output:\n{evt.get('output', '')}"
            + (f"\n({_sup} earlier matches were suppressed by rate limit)" if _sup else "") + "]")
    _exit = evt.get("exit_code", "?")
    _out = evt.get("output", "")
    _signal = ", SIGTERM" if _exit in {-15, 143, "-15", "143"} else ""
    # A subagent-owned process's full output belongs in the child's transcript, not as
    # a raw wall in the parent — trim hard but keep enough tail to recognise failures.
    if _attribution and isinstance(_out, str) and len(_out) > 600:
        _out = (
            "...(output trimmed — subagent-owned process; see the "
            "delegation's live transcript for full output)\n"
            + _out[-600:])
    return (
        f"[IMPORTANT: Background process {_sid} {_completion_status(evt)} (exit code {_exit}{_signal}).\n"
        f"{attribution}Command: {_cmd}\nOutput:\n{_out}]")
