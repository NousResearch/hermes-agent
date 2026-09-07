"""Tool schemas for tools.kanban_tools (model-facing; strings are byte-frozen)."""
from __future__ import annotations

from typing import Any

_DESC_TASK_ID_DEFAULT = (
    "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env "
    "(the task the dispatcher spawned you to work on)."
)

_DESC_BOARD = (
    "Kanban board slug to target. When omitted, the call resolves the "
    "active board the usual way: HERMES_KANBAN_DB env → "
    "HERMES_KANBAN_BOARD env → the 'current' symlink under the kanban "
    "home → 'default'. Pass an explicit slug only when the caller (e.g. "
    "a Telegram routing layer) needs to override the env-pinned active "
    "board for this one call."
)


def _prop(type_: str, description: str) -> dict[str, str]:
    return {"type": type_, "description": description}


def _board_schema_prop() -> dict[str, str]:
    """Schema fragment for the optional ``board`` parameter (one place to tweak)."""
    return _prop("string", _DESC_BOARD)


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build a tool schema; every kanban tool takes an optional trailing ``board``."""
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {**properties, "board": _board_schema_prop()},
            "required": required,
        },
    }


KANBAN_SHOW_SCHEMA = _schema(
    "kanban_show",
    (
        "Read a task's full state — title, body, assignee, parent task "
        "handoffs, your prior attempts on this task if any, comments, "
        "and recent events. Use this to (re)orient yourself before "
        "starting work, especially on retries. The response includes a "
        "pre-formatted ``worker_context`` string suitable for inclusion "
        "verbatim in your reasoning."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
    },
    [],
)

KANBAN_LIST_SCHEMA = _schema(
    "kanban_list",
    (
        "List Kanban task summaries so an orchestrator profile can discover "
        "work to route. Supports the same core filters as the CLI: assignee, "
        "status, tenant, include_archived, and limit. Returns compact rows "
        "with ids, title, status, assignee, priority, parent/child ids, and "
        "counts. Bounded to 50 rows by default, 200 max, with truncation "
        "metadata. Also recomputes ready tasks before listing, matching the "
        "CLI. Orchestrator-only — dispatcher-spawned task workers never see "
        "this tool."
    ),
    {
        "assignee": _prop("string", "Optional assignee/profile filter."),
        "status": {
            "type": "string",
            "enum": [
                "triage", "todo", "ready", "running",
                "blocked", "done", "archived",
            ],
            "description": "Optional task status filter.",
        },
        "tenant": _prop("string", "Optional tenant/project namespace filter."),
        "include_archived": _prop("boolean", "Include archived tasks. Defaults to false."),
        "limit": _prop("integer", "Optional maximum rows to return (default 50, max 200)."),
    },
    [],
)

KANBAN_COMPLETE_SCHEMA = _schema(
    "kanban_complete",
    (
        "Mark your current task done with a structured handoff for "
        "downstream workers and humans. Prefer ``summary`` for a "
        "human-readable 1-3 sentence description of what you did; put "
        "machine-readable facts in ``metadata`` (changed_files, "
        "tests_run, decisions, findings, etc). At least one of "
        "``summary`` or ``result`` is required. If you created new "
        "tasks via ``kanban_create`` during this run, list their ids "
        "in ``created_cards`` — the kernel verifies them so phantom "
        "references are caught before they leak into downstream "
        "automation. If you produced deliverable files (charts, PDFs, "
        "spreadsheets, generated images), list their absolute paths "
        "in ``artifacts`` — the gateway notifier will upload them as "
        "native attachments to the human who subscribed to the task, "
        "so the deliverable lands in their chat alongside the summary "
        "instead of being a path they have to fetch by hand."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "summary": _prop("string", (
                "Human-readable handoff, 1-3 sentences. Appears in "
                "Run History on the dashboard and in downstream "
                "workers' context."
        )),
        "metadata": _prop("object", (
                "Free-form dict of structured facts about this "
                "attempt — {\"changed_files\": [...], \"tests_run\": 12, "
                "\"findings\": [...]}. Surfaced to downstream "
                "workers alongside ``summary``."
        )),
        "result": _prop("string", (
                "Short result log line (legacy field, maps to "
                "task.result). Use ``summary`` instead when "
                "possible; this exists for compatibility with "
                "callers that still set --result on the CLI."
        )),
        "created_cards": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional structured manifest of task ids you "
                "created via ``kanban_create`` during this run. "
                "The kernel verifies each id exists and was "
                "created by this worker's profile; any phantom "
                "id blocks the completion with an error listing "
                "what went wrong (auditable in the task's events). "
                "Only list ids you got back from a successful "
                "``kanban_create`` call — do not invent or "
                "remember ids from prose. Omit the field if you "
                "did not create any cards."
            ),
        },
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of absolute paths to deliverable "
                "files you produced during this run — generated "
                "charts, PDFs, spreadsheets, images, archives. "
                "Examples: [\"/tmp/q3-revenue.png\", "
                "\"/tmp/report.pdf\"]. The gateway notifier "
                "uploads each path as a native attachment to the "
                "subscribed chat (images embed inline, everything "
                "else uploads as a file) so the deliverable "
                "lands with the completion notification. Skip "
                "intermediate scratch files and references that "
                "are not the deliverable. The path must exist "
                "on disk at completion. Files inside a managed scratch "
                "workspace are copied to durable task attachments before "
                "cleanup; a missing declared scratch artifact keeps the "
                "task in-flight so you can fix the path and retry."
            ),
        },
    },
    [],
)

KANBAN_BLOCK_SCHEMA = _schema(
    "kanban_block",
    (
        "Stop work on this task and route it according to WHY you're stuck. "
        "Set ``kind`` to say which: 'dependency' (waiting on another task — "
        "goes to todo and auto-resumes when that task finishes, no human "
        "needed), 'needs_input' (you need a human decision/answer), "
        "'capability' (a hard wall: no access, missing credentials, an action "
        "no agent can do), or 'transient' (a flaky failure that may clear). "
        "``reason`` is shown to the human on the board. If a task keeps "
        "getting unblocked and re-blocked for the same reason, it is "
        "auto-escalated to triage. Use for genuine blockers only — don't "
        "block on things you can resolve yourself."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "reason": _prop("string", (
                "What you need answered or what stopped you, in one or "
                "two sentences. Don't paste the whole conversation; the "
                "human has the board and can ask follow-ups via comments."
        )),
        "kind": {
            "type": "string",
            "enum": ["dependency", "needs_input", "capability", "transient"],
            "description": (
                "Why you're blocked. 'dependency' waits in todo and "
                "resumes automatically; the others surface to a human. "
                "Omit only if none apply."
            ),
        },
    },
    ["reason"],
)

KANBAN_REQUEST_REVIEW_SCHEMA = _schema(
    "kanban_request_review",
    (
        "Hand the task off for review: implementation, self-review, and "
        "verification are complete and you want a human (or reviewer) to "
        "look before it is marked done. Moves the task to the 'review' "
        "column and notifies the subscriber. Unlike ``kanban_block`` this is "
        "NOT a blocker — it never counts toward unblock-loop detection, so a "
        "task can cycle through review across follow-ups without ever being "
        "falsely escalated to triage. Use this instead of blocking with a "
        "free-form 'review-required:' reason."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "summary": _prop("string", (
                "What was implemented and how it was verified, in one or "
                "two sentences — shown to the reviewer. Don't paste "
                "the whole diff; the reviewer has the board and the PR."
        )),
        "reviewer": _prop("string", (
                "Optional reviewer profile. When provided, the task is "
                "reassigned to that profile before review dispatch."
        )),
        "metadata": {
            "type": "object",
            "description": (
                "Optional structured handoff facts for the reviewer, such "
                "as changed_files, tests_run, commit, or decisions."
            ),
            "additionalProperties": True,
        },
    },
    ["summary"],
)

KANBAN_REQUEST_CHANGES_SCHEMA = _schema(
    "kanban_request_changes",
    (
        "Reviewer verdict: return the current review run to the original "
        "implementer with concrete required changes. This closes the review "
        "run, reapplies parent dependency gating, and requeues the task without "
        "using block-loop accounting. Only use from a task claimed from the "
        "review column; use kanban_block only for a genuine external blocker."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "reason": _prop("string", (
                "Specific, actionable changes the implementer must make "
                "before requesting another review."
        )),
    },
    ["reason"],
)

KANBAN_HEARTBEAT_SCHEMA = _schema(
    "kanban_heartbeat",
    (
        "Signal that you're still alive during a long operation "
        "(training, encoding, large crawls). Call every few minutes so "
        "humans see liveness separately from PID checks. Pure side "
        "effect — no work changes."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "note": _prop("string", (
                "Optional short note describing current progress. "
                "Shown in the event log."
        )),
    },
    [],
)

KANBAN_COMMENT_SCHEMA = _schema(
    "kanban_comment",
    (
        "Append a comment to a task's thread. Use for durable notes "
        "that should outlive this run (questions for the next worker, "
        "partial findings, rationale). Ephemeral reasoning doesn't "
        "belong here — use your normal response instead."
    ),
    {
        "task_id": _prop("string", (
                "Task id. Required (may be your own task or "
                "another's — comment threads are per-task)."
        )),
        "body": _prop("string", "Markdown-supported comment body."),
    },
    ["task_id", "body"],
)

KANBAN_ATTACH_SCHEMA = _schema(
    "kanban_attach",
    (
        "Attach a file to a task by passing its bytes inline (base64). "
        "Use for genuine file artifacts the next worker or a human should "
        "be able to download — generated reports, images, exports. The "
        "file is stored as a real attachment (not a comment link) under "
        "the task's attachments dir, capped at 25 MB. Prefer "
        "kanban_attach_url when you only have a URL."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "filename": _prop("string", (
                "File name to store it under (e.g. 'report.pdf'). "
                "Directory components are stripped; only the leaf is kept."
        )),
        "content_base64": {
            "type": "string",
            "description": "The file contents, base64-encoded. Max 25 MB decoded.",
        },
        "content_type": _prop("string", "Optional MIME type (e.g. 'application/pdf')."),
    },
    ["filename", "content_base64"],
)

KANBAN_ATTACH_URL_SCHEMA = _schema(
    "kanban_attach_url",
    (
        "Attach a file to a task by URL — Hermes downloads it server-side "
        "and stores it as a real attachment (capped at 25 MB). Use when "
        "you have a link rather than the bytes. Only http/https URLs are "
        "accepted."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "url": _prop("string", "http(s) URL to fetch and store."),
        "filename": _prop("string", (
                "Optional name to store it under. Defaults to the URL "
                "path's leaf component."
        )),
        "content_type": _prop("string", (
                "Optional MIME type override. Defaults to the "
                "Content-Type the server returns."
        )),
    },
    ["url"],
)

KANBAN_ATTACHMENTS_SCHEMA = _schema(
    "kanban_attachments",
    (
        "List the files attached to a task: id, filename, content_type, "
        "size, who uploaded it, and the absolute on-disk path you can read."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
    },
    [],
)

KANBAN_CREATE_SCHEMA = _schema(
    "kanban_create",
    (
        "Create a new kanban task, optionally as a child of the current "
        "one (pass the current task id in ``parents``). Used by "
        "orchestrator workers to fan out — decompose work into child "
        "tasks with specific assignees, link them into a pipeline, "
        "then complete your own task. The dispatcher picks up the new "
        "tasks on its next tick and spawns the assigned profiles."
    ),
    {
        "title": _prop("string", "Short task title (required)."),
        "assignee": _prop("string", (
                "Profile name that should execute this task "
                "(e.g. 'researcher-a', 'reviewer', 'writer'). "
                "Required — tasks without an assignee are never "
                "dispatched."
        )),
        "body": _prop("string", (
                "Opening post: full spec, acceptance criteria, "
                "links. The assigned worker reads this as part of "
                "its context."
        )),
        "parents": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Parent task ids. The new task stays in 'todo' "
                "until every parent reaches 'done'; then it "
                "auto-promotes to 'ready'. Typical fan-in: list "
                "all the researcher task ids when creating a "
                "synthesizer task."
            ),
        },
        "tenant": _prop("string", (
                "Optional namespace for multi-project isolation. "
                "Defaults to HERMES_TENANT env if set."
        )),
        "priority": _prop("integer", (
                "Dispatcher tiebreaker. Higher = picked sooner "
                "when multiple ready tasks share an assignee."
        )),
        "workspace_kind": {
            "type": "string",
            "enum": ["scratch", "dir", "worktree"],
            "description": (
                "Workspace flavor: 'scratch' (fresh tmp dir, "
                "default), 'dir' (shared directory, requires "
                "absolute workspace_path), 'worktree' (git worktree)."
            ),
        },
        "workspace_path": _prop("string", (
                "Absolute path for 'dir' or 'worktree' workspace. "
                "Relative paths are rejected at dispatch."
        )),
        "project": _prop("string", (
                "Optional project id or slug to link the task to. When "
                "set, the task becomes a git worktree under the project's "
                "primary repo with a deterministic branch (project slug + "
                "task id), instead of a random branch."
        )),
        "triage": _prop("boolean", (
                "If true, task lands in 'triage' instead of 'todo' "
                "— a specifier profile is expected to flesh out "
                "the body before work starts."
        )),
        "idempotency_key": _prop("string", (
                "If a non-archived task with this key already "
                "exists, return that task's id instead of creating "
                "a duplicate. Useful for retry-safe automation."
        )),
        "max_runtime_seconds": _prop("integer", (
                "Per-task runtime cap. When exceeded, the "
                "dispatcher SIGTERMs the worker and re-queues the "
                "task with outcome='timed_out'."
        )),
        "initial_status": {
            "type": "string",
            "enum": ["running", "blocked"],
            "description": (
                "Initial card status. Use 'blocked' for tasks that "
                "require immediate human ops (R3 gate) to skip the "
                "brief running-to-blocked transition. Defaults to "
                "'running', which preserves the usual dispatch path."
            ),
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Skill names to force-load into the dispatched "
                "worker. The kanban lifecycle is already injected "
                "automatically; use this to pin a task to a specialist "
                "context — e.g. ['translation'] for a translation "
                "task, ['github-code-review'] for a reviewer task. "
                "The names must match skills installed on the "
                "assignee's profile."
            ),
        },
        "goal_mode": _prop("boolean", (
                "Run the dispatched worker in a goal loop. When true, "
                "after each turn an auxiliary judge checks the worker's "
                "response against this card's title/body; if the work "
                "isn't done and budget remains, the worker keeps going "
                "in the same session until the judge agrees it's "
                "complete (or the goal-turn budget is exhausted, which "
                "blocks the task for human review). Use this for "
                "open-ended cards where one shot rarely finishes the "
                "work. Defaults to false (classic single-shot worker)."
        )),
        "goal_max_turns": _prop("integer", (
                "Turn budget for goal_mode workers. Caps how many "
                "continuation turns the worker may take before the task "
                "is blocked for review. Ignored unless goal_mode is "
                "true. Defaults to the goal-engine default (20)."
        )),
        "model": _prop("string", (
                "Pin the dispatched worker to this model instead of "
                "the assignee profile's configured model. Use the "
                "exact model name the target provider expects. Omit "
                "to use the profile default."
        )),
        "provider": _prop("string", (
                "Provider the 'model' belongs to (e.g. 'openrouter', "
                "'anthropic', 'nous'). Set this whenever the model "
                "is not from the assignee profile's configured "
                "provider — a model name alone is resolved against "
                "the profile's provider and will fail if it belongs "
                "to a different one. Requires 'model'."
        )),
    },
    ["title", "assignee"],
)

KANBAN_UNBLOCK_SCHEMA = _schema(
    "kanban_unblock",
    (
        "Unblock a Kanban task. It moves to ready when all parents are done, "
        "or todo while any parent remains open. Orchestrator-only — only "
        "profiles with the kanban toolset can unblock routed work; "
        "dispatcher-spawned task workers never see this tool."
    ),
    {
        "task_id": _prop("string", "Blocked task id to move to ready or parent-gated todo."),
    },
    ["task_id"],
)

KANBAN_LINK_SCHEMA = _schema(
    "kanban_link",
    (
        "Add a parent→child dependency edge after both tasks already "
        "exist. The child won't promote to 'ready' until all parents "
        "are 'done'. Cycles and self-links are rejected."
    ),
    {
        "parent_id": {"type": "string", "description": "Parent task id."},
        "child_id":  {"type": "string", "description": "Child task id."},
    },
    ["parent_id", "child_id"],
)
