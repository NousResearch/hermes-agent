"""Tool schemas for tools.kanban_tools (model-facing; strings are byte-frozen)."""
from __future__ import annotations

from typing import Any

_DESC_TASK_ID_DEFAULT = (
    "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env — the "
    "task the dispatcher spawned you to work on. That default only exists for "
    "a dispatcher-spawned worker; any other caller has no default and must "
    "pass an explicit task_id (use kanban_list to discover ids)."
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
        "verbatim in your reasoning. Outside a dispatcher-spawned worker "
        "there is no default task: a bare call returns a pointer to "
        "``kanban_list`` instead of task state."
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
                "Examples: [\"~/.hermes/cache/scratch/q3-revenue.png\", "
                "\"~/.hermes/cache/scratch/report.pdf\"]. The gateway notifier "
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
                "resumes automatically when an incomplete parent finishes; "
                "if no parent is open it is recorded as needs_input instead. "
                "The others surface to a human. Omit only if none apply."
            ),
        },
    },
    ["reason"],
)

KANBAN_SCHEDULE_SCHEMA = _schema(
    "kanban_schedule",
    (
        "Park your current task in the 'scheduled' state while it waits for "
        "time or an external event. This ends the current run and makes the "
        "task non-dispatchable until an orchestrator unblocks it; it does not "
        "create a timer. Put any wake-up marker such as "
        "``SCHEDULED_UNTIL=<ISO8601>`` in ``reason``."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "reason": _prop("string", (
            "Optional reason or machine-readable wake-up marker recorded on "
            "the completed run and scheduled event."
        )),
    },
    [],
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
                "Optional reviewer profile. Ignored (and refused) on a "
                "card that was created with a required reviewer — the "
                "saved gate wins and is preserved across retries and "
                "resume. On an ungated card it must exist and carry "
                "the sdlc-review skill; when provided, the task is "
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
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of absolute paths to deliverable "
                "files this handoff names — generated charts, "
                "PDFs, spreadsheets, images, archives. Examples: "
                "['~/.hermes/cache/scratch/q3-revenue.png', '~/.hermes/cache/scratch/report.pdf']. "
                "A review handoff is the last implementer "
                "transition, so the kernel copies these into the "
                "task's durable attachments before the reviewer's "
                "completion cleans the scratch workspace up, and "
                "the gateway notifier uploads them as native "
                "attachments to the subscribed chat. A missing "
                "declared scratch artifact keeps the task in place "
                "so you can fix the path and retry."
            ),
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
                "another's — comment threads are per-task). Outside a "
                "dispatcher-spawned worker there is no default; use "
                "kanban_list to discover ids."
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
                "Every name must resolve in the assignee profile's "
                "effective skill library (its own skills plus shared/"
                "bundled/external/plugin skills): one missing name "
                "rejects the whole request atomically, naming the "
                "profile and the missing skills, before anything is "
                "written. Omit the field for default behaviour."
            ),
        },
        "reviewer": _prop("string", (
                "Optional review gate: the profile persisted as this "
                "card's required reviewer. It must exist and already "
                "carry the sdlc-review skill the dispatcher injects "
                "for review-phase startup, or the create is refused "
                "with nothing written. Gated cards cannot be completed "
                "by an implementation run: they finish with "
                "kanban_request_review, which selects this reviewer "
                "automatically and rejects any override. Omit to keep "
                "the card ungated."
        )),
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
        "completion_contract": _prop("string", (
            "Declare at creation: local-only (default), OWNER/REPO for PR publication, or an exact GitHub PR URL. "
            "PR tasks cannot complete until repository-required exact-head CI passes. On publication pass metadata.published_pr."
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
        "are 'done'. Cycles and self-links are rejected. A running child "
        "is rejected unless the active owning worker is linking its own "
        "card for a dependency handoff. The response reports the child's "
        "ACTUAL resulting status and remaining gates — a link that could "
        "not demote the child says so instead of claiming a transition "
        "that did not happen."
    ),
    {
        "parent_id": {"type": "string", "description": "Parent task id."},
        "child_id":  {"type": "string", "description": "Child task id."},
    },
    ["parent_id", "child_id"],
)

KANBAN_UNLINK_SCHEMA = _schema(
    "kanban_unlink",
    (
        "Counterpart to ``kanban_link``: drop an existing parent→child "
        "dependency edge. Same edge validation and board scoping as "
        "linking. Returns whether the edge was actually removed plus the "
        "child's REAL resulting status, whether it was promoted by the "
        "removal, and any parents still gating it — no transition is "
        "claimed unless it actually occurred. Removing the last "
        "unsatisfying parent re-evaluates the child immediately instead "
        "of leaving it parked until the next dispatcher tick."
    ),
    {
        "parent_id": {"type": "string", "description": "Parent task id (the dependency being dropped)."},
        "child_id":  {"type": "string", "description": "Child task id (the dependent being released)."},
    },
    ["parent_id", "child_id"],
)

KANBAN_GRAPH_SCHEMA = _schema(
    "kanban_graph",
    (
        "Read-only focus-task graph inspection. Returns the task's id, "
        "title and status plus its DIRECT parents and children, each "
        "with id, title and status. Strictly SELECT-only: it never "
        "writes status, events or edges and never recomputes readiness, "
        "so inspecting a graph can never move a card. Use kanban_show "
        "for the full record (comments, runs, events) and this when you "
        "only need the dependency shape around one task."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
    },
    [],
)

KANBAN_PROMOTE_SCHEMA = _schema(
    "kanban_promote",
    (
        "Promote a task out of 'triage' into the normal flow — and only "
        "that. It becomes 'ready' only when every parent is already "
        "terminal; otherwise it lands in 'todo' and the response reports "
        "the unmet parent gate(s) so you know what must finish first. "
        "The transition is a single guarded update, so a concurrent "
        "board change can never be overwritten, and the response always "
        "reports where the task ACTUALLY landed. There is no arbitrary "
        "status setter: a 'running' or any other direct transition is "
        "not expressible through this tool. Orchestrator-only."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "reason": _prop("string", (
            "Optional audit note recorded on the promotion event, e.g. "
            "'triage clarified by operator'."
        )),
    },
    [],
)

KANBAN_ARCHIVE_SCHEMA = _schema(
    "kanban_archive",
    (
        "Archive a task — but NOT one whose work is still in flight. "
        "'ready', 'running' and 'review' are protected: a direct archive is "
        "refused with the block/stop-first instruction and nothing changes. "
        "Every other status archives directly, including legacy raw statuses "
        "such as 'completed'; 'archived' is the already-archived no-op "
        "(archive is not deletion). The status check and the archive are one "
        "atomic guarded transition, so a task that changed state "
        "concurrently — or one whose worker cannot be proven stopped, or whose "
        "spawn has started a worker whose PID is not published yet — is refused "
        "without being archived. This tool never blocks a task on its own: "
        "block it first (kanban_block), then archive. `reason` is REQUIRED and "
        "is stored on the `archived` event as {source, actor, reason} so the "
        "board keeps an audit trail of why an agent archived a card; an "
        "invalid reason is rejected before anything changes, and credential-"
        "shaped text in it is masked by the same redactor that guards every "
        "other agent-authored free-text field on this board. On success it "
        "returns an impact receipt: which dependents actually changed status "
        "(before/after each), which are still waiting and why (remaining "
        "parent gates, holds, or both), which are now 'ready' and need "
        "assignment/dispatch follow-up, plus the same-card review association "
        "(the card's own review run and linked children). Orchestrator-only."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "reason": _prop("string", (
            "Required. Why this task is being archived, in one or two "
            "sentences (e.g. 'superseded by t_ab12cd34 — folded into the "
            "parent card'). Stored on the `archived` event for later audit "
            "(credential-shaped text is masked first, as everywhere else on "
            "this board); must be non-empty after trimming whitespace."
        )),
    },
    ["task_id", "reason"],
)

KANBAN_DECOMPOSE_SCHEMA = _schema(
    "kanban_decompose",
    (
        "Apply an agent-authored decomposition of a 'triage' task into a "
        "graph of child tasks. No auxiliary/LLM call is made — you supply "
        "the whole graph. The entire graph is validated first and applied "
        "atomically: if anything is invalid (missing title, bad parent "
        "index, a cycle, an unknown assignee) no child rows, edges or "
        "events are written and the root is left untouched. On success "
        "the root waits on the whole child graph and wakes when all "
        "children finish; the response returns the created child ids and "
        "the ACTUAL resulting status of the root and each child after "
        "eligibility is recomputed. Only a 'triage' task can be "
        "decomposed, and only once. Orchestrator-only."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "children": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Ordered child specs. ``parents`` entries are 0-based "
                "indexes into THIS array, expressing dependencies between "
                "the new siblings (a child with parents waits for them). "
                "The graph must be acyclic."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "title": _prop("string", "Short task title (required)."),
                    "body": _prop("string", (
                        "Opening post: full spec, acceptance criteria, "
                        "links. The assigned worker reads this as part of "
                        "its context."
                    )),
                    "assignee": _prop("string", (
                        "Profile that should execute this child. Must be "
                        "an installed profile; when omitted it falls back "
                        "to the configured default assignee so a child is "
                        "never left unassigned."
                    )),
                    "parents": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "0-based indexes into this same children "
                            "array that must complete first. Omit for "
                            "parallel work."
                        ),
                    },
                    "workspace_kind": {
                        "type": "string",
                        "enum": ["scratch", "dir", "worktree"],
                        "description": (
                            "Workspace flavor for this child; inherits "
                            "the root's when omitted."
                        ),
                    },
                    "workspace_path": _prop("string", (
                        "Absolute workspace path override for this child."
                    )),
                },
                "required": ["title"],
            },
        },
    },
    ["children"],
)

KANBAN_REASSIGN_SCHEMA = _schema(
    "kanban_reassign",
    (
        "Move an existing task to a different profile — the tool form of "
        "`hermes kanban reassign`. It runs the SAME shared kernel the CLI "
        "uses (assign/reassign in kanban_db), so the lifecycle guards are "
        "identical: a card still running under a claim is refused and "
        "nothing changes, unless `reclaim` is true, which releases the "
        "claim first (the \"this profile's model is broken\" path). The "
        "destination must be an installed profile — validated against the "
        "same enumeration the CLI and the dispatcher's spawn gate use, so a "
        "typo is refused before the board is opened and no event, status or "
        "claim is touched. On success the shared `assigned` audit event "
        "({assignee, from}) is appended and the response reads the card "
        "back from the board. Board isolation is preserved: only the board "
        "this call opens is written. Orchestrator-only."
    ),
    {
        "task_id": _prop("string", _DESC_TASK_ID_DEFAULT),
        "assignee": _prop("string", (
            "Destination profile name. Must be an installed profile "
            "(see kanban_discover for the roster)."
        )),
        "reclaim": _prop("boolean", (
            "Release the active claim before reassigning, so a card whose "
            "current profile is broken can still move. Default false: a "
            "claimed running card is refused instead."
        )),
        "reason": _prop("string", (
            "Optional audit note recorded on the reclaim when `reclaim` "
            "is true."
        )),
    },
    ["task_id", "assignee"],
)

KANBAN_DISCOVER_SCHEMA = _schema(
    "kanban_discover",
    (
        "Read-only roster of the profiles this home can spawn work to. "
        "Returns every installed profile — `default` plus each live named "
        "profile — with its optional `profile.yaml` descriptor metadata "
        "(display name, description, role), exactly as the dispatcher's "
        "roster and `hermes profile list` read them. Profiles without a "
        "descriptor are still listed, with descriptor status `missing`; an "
        "unreadable or unparseable descriptor is reported as `unreadable` / "
        "`invalid` (never silently treated as absent, never an error). "
        "Nothing is written and no profile is generated: this is the same "
        "enumeration `kanban_create` and `kanban_reassign` validate "
        "against, so it is the authoritative list of valid assignees. "
        "Only descriptor fields are returned — no config, credentials, "
        "SOUL or prompt contents are read, and every returned string is "
        "length-bounded. Profiles live under the Hermes home rather than a "
        "board, so the optional `board` argument does not apply here."
    ),
    {},
    [],
)
