---
title: Plane
sidebar_position: 37
---

# Plane

Hermes can expose a small `plane` toolset for working with Plane as the human-visible project board while keeping Hermes kanban as the execution layer.

V1 is intentionally explicit:

- no dashboard or parallel portal
- no automatic sync
- no delete operation
- writes to Plane only happen when the user asks for a create or update
- Plane remains the project source of truth

## Configuration

Set these values in `~/.hermes/.env`:

```bash
PLANE_API_KEY=...
PLANE_WORKSPACE=ai_factory
PLANE_PROJECT_ID=8695a8d1-e6fc-44e1-8bd2-9f37158b5124
# optional, defaults to https://api.plane.so
PLANE_BASE_URL=https://api.plane.so
```

The Plane client always sends:

- `X-API-Key`
- `Accept: application/json`
- a browser-like `User-Agent`

The browser `User-Agent` is required because Plane Cloud can reject valid script requests with Cloudflare `browser_signature_banned` when the request does not look browser-like.

## Tools

`plane_ping`

Runs a quick health check for the configured integration. It calls `GET /api/v1/users/me/` and the configured project endpoint using the shared client, then returns:

- `ok`
- `latency_ms`
- `user` / `user_email`
- `workspace`
- `project` / `project_name`

Use it first when debugging auth, network, Cloudflare `User-Agent`, or project access issues.

`plane_board_snapshot`

Returns a compact project snapshot by default:

```json
{
  "project": {"id": "...", "name": "...", "identifier": "AIFACTORY"},
  "states": [{"id": "...", "name": "Todo", "group": "backlog", "count": 3}],
  "counts_by_state": {"Todo": 3},
  "total_items": 12,
  "items": [{"id": "...", "sequence_id": 12, "readable_id": "AIFACTORY-12", "name": "...", "state_name": "Todo", "state_id": "...", "priority": "medium", "labels": [], "assignees_names": [], "url": "..."}]
}
```

Options:

- `include_items_per_state`
- `per_state_limit`
- `limit`
- `verbose`

When `verbose=true`, the response also includes raw `project_payload`, `states_payload`, and `items_payload`.

`plane_list_work_items`

Lists compact work items with optional filters:

- `state`
- `label`
- `assignee`
- `priority`
- `query`
- `limit`
- `verbose`

Default item shape:

```json
{"id": "...", "sequence_id": 12, "readable_id": "AIFACTORY-12", "name": "...", "state_name": "Todo", "state_id": "...", "priority": "medium", "labels": [], "assignees_names": [], "url": "..."}
```

When `verbose=true`, the response also includes raw `items_payload`.

`plane_get_work_item`

Reads one work item by `work_item_id` or `sequence_id`. The default `item` uses the same compact shape as `plane_list_work_items`. When `verbose=true`, the response also includes raw `payload` and `enriched_item`.

`plane_create_work_item`

Creates a Plane work item idempotently. Supports:

- `name`
- `description_html` or `description_markdown`
- `priority`
- `state`
- `labels`
- `assignees`
- `start_date`
- `target_date`
- `external_source`
- `external_id`

Idempotence contract:

- Hermes tries to look up an existing Plane work item before creation using `external_source` + `external_id`.
- If `external_source` is omitted, it defaults to `nova-hermes`.
- If `external_id` is omitted, Hermes generates a stable fallback from workspace, project, external source, and normalized `name`: `plane-create:<sha256-prefix>`.
- The fallback deliberately excludes mutable fields such as description, labels, state, dates, and assignees so a retry with minor payload differences still targets the same logical item.
- When the pre-create lookup misses but Plane still rejects the POST with `409 Conflict` (server-side uniqueness on `external_source` + `external_id`), Hermes now performs a second lookup and returns the existing item with `already_existed: true`, `created: null`. Callers see a stable idempotent response in both cases.

`plane_update_work_item`

Updates selected fields of an existing work item. Requires `work_item_id` and at least one update field.

Supported fields (PATCH partial):

- `name`
- `description_html` or `description_markdown`
- `priority`
- `state` (state name or id)
- `labels` (list of label names or ids)
- `assignees` (list of user ids)
- `start_date`
- `target_date`
- `external_source`
- `external_id`

PATCH partial semantics:

- A field passed as `None` (or absent from the call) is **ignored** and never sent to Plane.
- A field passed with an explicit non-null value is sent to Plane and overwrites the current value.
- For list fields (`labels`, `assignees`), an empty list `[]` is the explicit "clear" signal and is forwarded as-is. Use `None` (or omit the key) to leave the current list untouched.

This means retries with the same payload are safe, and partial updates do not accidentally clear fields the caller did not mention.

`plane_add_comment`

Adds a comment to an existing work item without changing its state. Supports:

- `work_item_id` or `sequence_id`
- `body_markdown`
- `prefix`, default `true`, prepends `[Nova]`

`body_markdown` is converted to simple HTML and sent to Plane as `comment_html` on the work item comments endpoint.

**Known limitation (V1.1):** the current Markdown pipeline is intentionally minimal. It escapes HTML and turns line breaks into `<br>` tags. Lists, bold, italic, inline code, and links are **not** rendered as Markdown by Plane. Plain text and line breaks are the only formatting that survives end-to-end. Rich Markdown support is tracked for V1.2.

`plane_sync_progress`

Reflects progress from an imported Hermes kanban task back to Plane in one call. Supports:

- `hermes_card_id`, optional when called by a kanban worker because it defaults to `HERMES_KANBAN_TASK`
- `summary`, posted as a Plane comment
- `status`, optional Plane state name or id, for example `In Progress`, `Waiting`, or `Done`
- `prefix`, default `true`, prepends `[Nova]` to the comment

The tool reads the Plane linkage fields stored by `plane_import_to_kanban` in the Hermes task body (`plane_work_item_id`, `plane_sequence_id`, `plane_url`). If the Hermes card is not linked to Plane, it fails without posting a comment. If `status` is provided, it updates the Plane state before posting the progress comment and returns the compact item, Plane URL, comment payload, and update payload.

Convention: call `plane_sync_progress` on meaningful kanban transitions, for example task start, blocked state, implementation complete, or review done. Use `plane_add_comment` only for comments that should not imply state movement.

`plane_check_kanban_links`

Checks a list of Hermes kanban tasks already linked to Plane and returns only the anomalies. Supports:

- `hermes_card_ids`, required list of Hermes task ids

Return shape: `{"items": [{"hermes_card_id", "status", "plane_work_item_id", "plane_sequence_id", "plane_state_id", "plane_state_name", "plane_url"}], "count": <int>}`.

Behavior:

- `status: "cancelled"` when the linked Plane work item still exists but is in the `Cancelled` state
- `status: "missing"` when the linked Plane work item can no longer be resolved from the stored linkage
- no automatic side effects, no kanban mutation, no Plane write
- tasks not linked to Plane fail fast with a validation error instead of being silently skipped

Use it as a read-only drift detector before starting work on imported Hermes tasks, or as a periodic audit step on in-flight kanban items.

`plane_import_to_kanban`

Creates Hermes kanban tasks from selected Plane work items. The task body stores Plane linkage fields:

- `plane_workspace_slug`
- `plane_project_id`
- `plane_work_item_id`
- `plane_sequence_id`
- `plane_url`
- `plane_state_id`

Titles use:

```text
[Plane AIFACTORY-12] <title>
```

Return shape: `{"created_tasks": [{"task_id", "plane_work_item_id", "plane_sequence_id", "workdir", "already_imported"}]}`.

The `already_imported` flag is `true` when a non-archived Hermes kanban task already exists for the same Plane work item (resolved via the `plane:<workspace>:<project>:<work_item_id>` idempotency key). In that case Hermes returns the existing task id instead of creating a duplicate. Callers can rely on this to make `plane_import_to_kanban` safe to call repeatedly without polluting the kanban.

`plane_prepare_workdir`

Creates a local work directory for deliverables:

```text
/home/emeric/AI Factory/AIFACTORY-12_<slug>/
├── README.md
├── work/
└── deliverables/
```

The folder prefix matches the Plane project identifier. If `project_key` is provided in the call, it is used directly. If omitted and Plane is configured, Hermes resolves it from the Plane project (`get_project_identifier`). Otherwise it falls back to `AIFACTORY`.

## Recommended workflow

1. Use `plane_ping` to verify auth, browser-like `User-Agent`, network, and project access.
2. Use `plane_board_snapshot` to inspect the current board.
3. Use `plane_get_work_item` for the specific card.
4. Use `plane_import_to_kanban` only when a Plane card should become executable agent work.
5. Agents work in Hermes kanban and local workdirs.
6. Use `plane_check_kanban_links` before starting or resuming work on imported Hermes tasks when you want a quick drift check against Plane.
7. Use `plane_sync_progress` from kanban workers to reflect meaningful progress and optional state transitions back to Plane.
8. Use `plane_add_comment` for extra notes that should not move the Plane state.
9. Update Plane explicitly when other fields should change on the project board.

## End-to-end example

Typical flow for picking up Plane card `AIFACTORY-13` as an Hermes execution task:

```json
// 1) Import the card and create a local workdir.
{"tool": "plane_import_to_kanban", "args": {
  "sequence_ids": [13],
  "assignee": "emeric",
  "create_workdir": true
}}
// → {"created_tasks":[{"task_id":"t_abc","plane_sequence_id":13,"workdir":"/home/emeric/AI Factory/AIFACTORY-13_…","already_imported":false}]}

// 2) From the kanban worker, post first progress update.
{"tool": "plane_sync_progress", "args": {
  "summary": "Started implementation, scaffolding in place.",
  "status": "In Progress"
}}

// 3) Add an extra note that should not move the state.
{"tool": "plane_add_comment", "args": {
  "sequence_id": 13,
  "body_markdown": "Quick design note: switching to PATCH partial semantics."
}}

// 4) Close the loop when work is done.
{"tool": "plane_sync_progress", "args": {
  "summary": "Implementation complete, validation pending.",
  "status": "Done"
}}
```

A second call to `plane_import_to_kanban` with the same `sequence_ids` returns the existing task id and `already_imported: true`, so it is safe to call from idempotent automations.

## Troubleshooting

**HTTP 403 with `browser_signature_banned` in the body.** Cloudflare in front of Plane Cloud is rejecting the request. Verify the call goes through the shared client from `tools/plane_client.py`. Do not replace it with ad hoc `curl` or `requests` calls without the browser-like `User-Agent`.

**HTTP 409 on `plane_create_work_item`.** Means another caller (or a previous Hermes attempt) already created the work item with the same `external_source` + `external_id`. The handler converts this case into a clean `already_existed: true` response, with `created: null` and `item` populated with the existing card. If you still see a `tool_error` 409 surface, capture the payload and the resolved `external_id` so the lookup path can be tightened further.

**Lookup misses but card exists.** The pre-create lookup uses the Plane filter API on `external_source` + `external_id`, then falls back to a full board scan. On the current Plane Cloud board the full scan cost is negligible, and it remains necessary because the filtered lookup can return `400`, `404`, or `422` even when the card exists. The fallback is therefore intentional best effort behavior, not dead code.

**State update succeeds only with duplicated `state_id` + `state`.** In the current Plane environment Hermes must send both keys for state transitions. The tool keeps this workaround intentionally. If you remove one of the two fields and live updates start failing again, put the duplication back before debugging anything else.

**`plane_check_kanban_links` reports `missing` or `cancelled`.** `missing` means the stored Plane linkage no longer resolves. `cancelled` means the Plane card still exists but has reached the `Cancelled` state. The tool is read-only by design: it reports drift but does not archive, cancel, or edit the Hermes task for you.

**`plane_sync_progress` says the kanban task is not linked.** The Hermes kanban task body must contain the `plane_*` linkage lines written by `plane_import_to_kanban`. Tasks created manually via `hermes kanban` are not linked unless those lines are added by hand.

**Workdir folder name does not match the Plane project key.** Before V1.1, `plane_prepare_workdir` hardcoded `AIFACTORY` regardless of the configured Plane project. From V1.1 onwards, the folder prefix follows the real Plane project identifier (or the explicit `project_key` argument), with a final fallback to `AIFACTORY` only when no project identifier can be resolved.
