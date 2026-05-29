# Kynver AgentOS Runtime Contract

Hermes is the first Kynver AgentOS adapter, not the owner of Kynver durable
state. Runtime adapters should preserve the same boundary:

- Kynver owns memory/context, task and todo control-plane records, skills,
  session lifecycle, audit traces, review artifacts, approvals, progress, and
  steer artifacts.
- The runtime owns local machine-control tools such as shell/process, file,
  browser, media, vision, and computer-use actions.
- Runtime events sent to Kynver carry provenance fields:
  `runtime`, `callSign`, `contextTag`, `sourceId`, runtime session id,
  AgentOS session id when known, platform/channel, profile identity, workspace,
  and event timestamp.

## Hermes Adapter

Hermes exposes Kynver through the memory-provider interface. When Kynver
AgentOS credentials are present and `KYNVER_AGENTOS_MODE` is not `disabled`,
the provider is selected by default even when `memory.provider` is blank.
`KYNVER_AGENTOS_MODE=observe` keeps read/search behavior available while
turning durable writes into observed/degraded metadata. In observe mode, when
`KYNVER_MEMORY_DISABLED=1`, or after a Kynver health failure, Hermes keeps
local `MEMORY.md` and `USER.md` in the prompt. Kynver suppresses local fallback
context only when configured, enabled, memory is not disabled, and recent
AgentOS calls are healthy.

Hermes agent-loop tools use a generic observer hook:

- `MemoryProvider.on_tool_observed(tool_name, args, result, metadata=None)`
- plugin hook `agent_loop_tool_observed`

This hook is intentionally generic and is fired for built-in agent-loop tools
such as `todo`, `memory`, `delegate_task`, and `session_search`. Kynver uses it
to mirror todo state and audit tool events without adding Kynver-specific code
to those tools.

When `KYNVER_PLAN_ID` is set, Forge also registers plugin hooks on `todo`:

- `pre_tool_call` — project Hermes todos to `/plans/:id/progress-rows` and
  `/progress-focus` (`in_progress` focus ≠ harness executor lease `running`).
- `transform_tool_result` — read-back reconciliation from Kynver into the todo
  list the model sees (also wired for agent-loop tools in `tool_executor.py`).

When `KYNVER_API_KEY` (and slug) are present, operating hooks default **on** unless
`KYNVER_OPERATING_TOOLS=false`. Forge logs an info line at plugin registration when
hooks are active so operators see the default-on behavior without reading env docs.

Opt out of operating hooks with `KYNVER_OPERATING_TOOLS=false` while keeping the
memory provider active.

## HTTP Runtime Contract

The current route contract comes from the installed Kynver MCP AgentOS package,
which proxies MCP tools to resource-style HTTP routes under
`/api/agent-os/{slug}`. Hermes passes suffixes such as `/memory` to
`KynverAgentOSClient`; the client adds the `/api/agent-os/{slug}` prefix.

| AgentOS area | HTTP route | Payload notes |
| --- | --- | --- |
| Memory search | `GET /agent-os/{slug}/memory?q=...&k=...` | Optional filters include `sourceId`, repeated `sourceIds`, `ticker`, `debatePersona`, `personaSlug`, and `surface`. |
| Memory write | `POST /agent-os/{slug}/memory` | Body includes `content`, optional `slug` from Hermes `key`, `sourceId`, `metadata`, `sourceRefs`, `memoryType`, `confidence`, `reviewStatus`, and optional project/goal/contact/skill/correction fields. |
| Task create | `POST /agent-os/{slug}/tasks` | Body uses `title`, `description`, `priority`, `executor`, refs/links, schedule/dependency fields, `idempotencyKey`, and `requestId`. New tasks are `ready` unless scheduled or dependency-gated. |
| Task get/list/update | `GET /agent-os/{slug}/tasks/{taskId}`, `GET /agent-os/{slug}/tasks?...`, `PATCH /agent-os/{slug}/tasks/{taskId}` | List filters include `status`, `executor`, `parentTaskId`, `personaSlug`, and `limit`. Patch accepts lifecycle and artifact fields such as `status`, `lastSummary`, `blocker`, `branch`, `worktreePath`, `prUrl`, and `headCommit`. |
| Task event/close/steer | `POST /agent-os/{slug}/tasks/{taskId}/events`, `/close`, `/steer` | Events use `type`, `payload`, `artifactVisibility`, `eventKey`; close uses terminal `status` (`done`, `failed`, `cancelled`) plus `summary`; steer uses `message`, `detail`, `eventKey`. |
| Skills | `GET /agent-os/{slug}/skills?view=manifest`, `GET /agent-os/{slug}/skills/{skillSlug}?source=...` | Hermes lists manifests and fetches full skill bodies on demand. Skill search is a local filter over the manifest list because the MCP server does not expose a separate search route. |
| Sessions | `POST /agent-os/{slug}/sessions`, `POST /agent-os/{slug}/sessions/{sessionId}/events`, `PATCH /agent-os/{slug}/sessions/{sessionId}` | Open body uses `channel` and optional `model`; event body is `{ event }`; close body includes `summary` and optional structured event/goal/project fields. |
| Daily session log | `POST /agent-os/{slug}/daily-log` | Body uses `entry` and optional `date`; Hermes does not currently call this route. |

There is no `/task:mirror_todo` route. Hermes mirrors todos by posting task
create records to `/tasks` with stable `idempotencyKey` values. When the server
returns a task id, Hermes can then patch non-ready state or close terminal todos
with `done`/`failed`/`cancelled`; without a returned id, the create call remains
the durable upsert boundary and later state mutation is intentionally not
claimed.

Unknown endpoint failures must fail open locally, use short timeouts for
observer/session side effects, and surface redacted degraded-mode metadata
rather than blocking runtime tools.
