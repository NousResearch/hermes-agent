---
title: "Relay Shared Metrics"
description: "NeMo Relay shared metrics: what is exported, consent and retention, staging validation"
---

# NeMo Relay Shared Metrics

Hermes includes NeMo Relay as a normal runtime dependency on platforms for
which Relay publishes a native wheel. The shared-metrics integration is built
into Hermes and does not require a Hermes observability plugin. Hermes remains
importable without Relay on other native targets. Those targets use an
explicit reduced-capability no-op host:
Hermes execution remains available, while Relay scopes, middleware, plugins,
and subscribers are unavailable. The `hermes-agent[nemo-relay]` extra remains
as a no-op compatibility alias for existing installation commands.

> [!WARNING]
> This removes the Hermes `observability/nemo_relay` plugin. Existing users
> must remove `observability/nemo_relay` (or its legacy `nemo_relay` alias)
> from `plugins.enabled` and move exporter configuration into a Relay
> `plugins.toml`. `HERMES_NEMO_RELAY_PLUGINS_TOML` can select an explicit
> file, and `hermes update` or `hermes migrate relay` creates one when
> migrating legacy `HERMES_NEMO_RELAY_ATOF_*` and
> `HERMES_NEMO_RELAY_ATIF_*` settings. Those legacy variables no longer
> configure Relay exporters themselves.

On supported platforms, Hermes requires NeMo Relay 0.9 for managed provider
and tool calls.

## Runtime Dependency and Data Boundary

Hermes installs the platform-specific `nemo-relay` native wheel from the
bounded `>=0.9,<0.10` dependency range. The published package is built from
the [NVIDIA NeMo Relay repository](https://github.com/NVIDIA/NeMo-Relay).
Unsupported platforms use the explicit no-op runtime described above rather
than downloading a different implementation.

When Relay managed execution is active, the provider request and response pass
through that native module in the Hermes process so configured interceptors can
operate on the real call. This is separate from the shared-metrics data
contract. Shared-metrics mode installs no rich-observability network exporter,
and its subscriber
accepts only the versioned, allowlisted projection described below. The
opt-in package sender described in Appendix A is the only outbound path, it
transmits nothing unless the user enables both `enabled` and `send`, and it
sends whole packages rather than live spans. Enabling a
separately configured rich-observability or dynamic plugin can create a
different data path and requires its own policy review.

Collection remains off unless Hermes policy enables it:

```yaml
telemetry:
  shared_metrics:
    enabled: true
```

This choice is read from the profile's own `config.yaml`. A machine-managed
configuration overlay cannot enable or disable shared metrics on the profile's
behalf.

Hermes uses Relay's normal process-wide plugin discovery. Relay reads these
files, lowest precedence first:

| Layer | Linux and macOS | Windows |
|-------|-----------------|---------|
| User | `$XDG_CONFIG_HOME/nemo-relay/plugins.toml`, or `~/.config/nemo-relay/plugins.toml` | `%USERPROFILE%\.config\nemo-relay\plugins.toml` (`XDG_CONFIG_HOME` and then `HOME` take precedence when set) |
| System | `/etc/nemo-relay/plugins.toml` | `%ProgramData%\nemo-relay\plugins.toml` |

`HERMES_NEMO_RELAY_PLUGINS_TOML` replaces the user file with an explicit file;
the system file still applies above it. Repository-local configuration is
ignored. If an explicitly selected file cannot be loaded, Hermes reports the
error and continues without Relay plugins rather than falling back to another
configuration.

Run `hermes doctor` to see which files apply. Its **NeMo Relay Plugins**
section lists each file Relay resolves, whether any plugin is enabled, and any
problem Relay reports, without loading plugin code.

## Session-Span Segmentation for Continuous Sessions

Relay exports a span when its scope closes. A continuous gateway session can
remain open for days, so its session span remains open even though each turn
span is exported normally. Optional segmentation rotates only the session
scope at a turn boundary:

```yaml
gateway:
  telemetry:
    session_segments:
      on_compaction: false  # rotate after context compaction
      max_turns: 0          # 0 = unlimited; N = turns per segment
```

| Key | Default | Behavior |
|---|---:|---|
| `on_compaction` | `false` | Rotate after compaction completes, at the next turn boundary. |
| `max_turns` | `0` | Rotate after every N completed turns; `0` disables the cap. |

Both defaults preserve one session scope for the full session. Rotated spans
retain the same `session_id` and add `hermes.session.segment` plus
`hermes.session.segment_reason` (`compaction` or `max_turns`).

## Working-Directory Scope Data

When Hermes knows a session or task's logical working directory, its
`hermes.session` and `hermes.turn` start scopes include it as `data.cwd` in
ATOF. A turn running in a task worktree can therefore differ from its owning
session. Unknown directories are omitted, and scope-end data remains reserved
for the outcome.

The working directory is Relay scope input, so it is visible to every enabled
Relay subscriber, not only ATOF. Paths can reveal usernames, repository names,
or mount layouts. Relay does not filter events by working directory; if a path
must not leave the host, use a trusted local collector or do not enable a remote
exporter for that process.

## Process-Wide Plugin Policy and Profile Isolation

Relay plugin configuration is a process-level deployment choice, not a Hermes
profile setting. The first hosted profile triggers lazy initialization, and
every additional profile hosted by that Hermes process shares the resulting
static middleware, dynamic plugins, subscribers, exporters, and guardrail
policy. After initialization succeeds, Hermes logs the files it loaded:

```text
The Relay plugin host is active process-wide and applies to all profiles hosted by this Hermes process. Configuration files: /home/user/.config/nemo-relay/plugins.toml; /etc/nemo-relay/plugins.toml
```

Profile scopes still preserve causal isolation inside that shared policy.
ATIF groups events by their top-level Agent scope, so simultaneous profile
sessions produce separate trajectories rather than one mixed trajectory.
ATOF and other global subscribers observe events from every hosted profile.
Static and dynamic middleware likewise runs for managed calls from every
profile.

A worker plugin running in a separate worker process does not create a
per-profile security boundary. One process-wide activation dispatches calls
from all hosted profiles to that worker while preserving the invoking
profile's Relay scope stack. Native dynamic plugins are loaded into the Hermes
process and share the same policy boundary.

Run profiles in separate Hermes processes when they require different trust
levels, plugin credentials, exporter destinations, or guardrail policies.
This process-wide plugin contract does not change each profile's independent
shared-metrics consent, local SQLite state, or ATIF trajectory grouping.

Hermes core owns one Relay host and one isolated Relay session scope per Hermes
session. Core lifecycle producers use
`agent.relay_runtime` to obtain the shared session handle or
run Relay scope, LLM, tool, and mark APIs in that session context. New product
marks do not require Hermes plugin registration. Shared-metrics marks must
still contain only fields approved by the versioned allowlist; the hard
dependency does not change the collection or privacy policy.

## Current Slices

The current vertical slices record pseudonymous profile activity, logical
model calls, top-level task runs, tool and approval outcomes, and skill
lifecycle and reuse:

```text
Hermes turn, API, tool, and approval hooks
  -> Relay session, task, LLM, tool, and mark lifecycle
  -> Hermes shared-metrics subscriber
  -> SQLite counters
  -> immutable JSON delta package
```

Hermes sends an empty `LLMRequest` into the metrics-owned lifecycle. This does
not describe the separate managed-execution call through the native runtime
documented above. The terminal metrics event contains the model identifier and
provider route that Hermes used for the logical call, such as
`nvidia/nemotron-3-ultra` through `openrouter`. These identifiers are
lowercased and structurally bounded, but they are not normalized through a
checked-in model catalog. Pricing and model-family classification belong to
the metrics backend. Prompts, responses, endpoints, error text, session IDs,
task IDs, and request IDs are not included in the metrics event or package.
New calls use `hermes.model_route.count`. Since package schema v3 each route row
also carries `call_role` (`primary` or `auxiliary`), `outcome` (`success`,
`failed`, `cancelled`) and `error_class`: the error classifier's own
`FailoverReason` value (`rate_limit`, `auth`, `context_overflow`, ...) for the
last failed attempt of that logical call, or `none`. A `success` row with a
non-`none` class is a call that recovered after that error. Auxiliary calls
(titles, compression, vision, ...) follow the same rules: one row per logical
call however many fallback attempts it took, classified by the same classifier
(an HTTP-200 body carrying a provider `error` object is classified from that
object), `cancelled` with `none` when Hermes aborted it (`/stop`, Ctrl+C, an
interrupt, shutdown), and `unknown` only when the classifier cannot name the
failure. An auxiliary call that runs beside the turn (title generation) and
finishes under the turn's own live scopes is still counted: its result closes
the scope when the turn drains it. Auxiliary rows report `ttft_bucket`
`unknown`: most auxiliary calls are not streamed. The previous
`hermes.model_call.count` contract remains readable only so pending local
counters created by older builds can be exported without losing data.

The first consented session start emits an empty `hermes.client.active` Relay
mark. The profile-scoped subscriber creates a random UUID install identity and
uses a transactional compare-and-set to record at most one client-active
counter in any rolling 24-hour window. The metric has no dimensions; Hermes
version, OS family, architecture, and install method remain bounded package
resources. Concurrent Hermes processes share the SQLite latch, so simultaneous
starts cannot double-count one install. A later session or task can attempt the
mark again, but the subscriber suppresses it until the rolling window expires.

Each task run is a Relay `Function` scope named `hermes.task_run`, parented to
the owning Hermes session. The start counter contains only bounded execution
surface and entrypoint values plus, for gateway tasks, the built-in messaging
`platform` (`telegram`, `discord`, `slack`, ...; platforms Hermes ships under
`plugins/platforms/` by name, a `plugin-catalog/` platform by its catalog entry
name only when the installer's own record proves a catalog install, every other
plugin platform `plugin`, every other surface `none`). The terminal counter
(`hermes.task_run.finished`) contains the start fields plus bounded outcome, end
reason, termination status, and a `failure_class` for failed tasks: the provider
`FailoverReason` when the turn died on a classified API error, otherwise a local
class (`empty_response`, `context_compression`, `repeated_errors`, `exception`,
`other`, ...). The same end event feeds `hermes.task_run.duration` with execution
surface, outcome, duration bucket and provider-retry count bucket. Package v2
carried duration, retries and per-task model/tool call counts on the terminal row
itself, which made almost every task its own row; call counts per turn live on
`hermes.task_cost.count`. Raw exit
reasons never leave the machine. Retries are additional
provider attempts for the same Hermes API request ID; they do not inflate the
logical model-call count. Tool calls are deduplicated by their Hermes tool-call
ID after a terminal tool result is observed. The outer `AIAgent` execution
boundary closes the task for normal returns, early returns, exceptions, and
cancellations. Active task ownership follows the task ID if Hermes rotates its
conversation session during context compression.

The `entrypoint` dimension (on `hermes.task_run.started`, `hermes.task_run.finished`
and `hermes.session.count`) says who dispatched the run, from a closed set:

| Value | Meaning |
|---|---|
| `interactive` | A person in a chat UI: the `hermes` REPL, a `hermes chat -q` that seeds the REPL on a TTY, `--tui`, Desktop, ACP editors. |
| `one_shot` | A finite CLI run that answers one prompt and exits: `hermes -z` / `--oneshot`, `hermes chat -q` off a TTY or with `--oneshot`, `-Q` / `--quiet`. A person's shell line and a script looping it look the same, so both read `one_shot`. Bot Chat delivery turns (`hermes -p <profile> chat -c "Bot Chat" -Q`) are one-shot runs too: their author may be a person on another connection. Surface stays `cli`. |
| `background` | An unattended run a Hermes dispatcher spawned: a kanban worker (`HERMES_SESSION_SOURCE=kanban`) or an A2A forward (`--source a2a`). |
| `delegated` | A subagent run under a parent task or session (wins over the values above). |
| `gateway_message` | A messaging-platform message. |
| `scheduled_task`, `batch`, `api`, `python` | Cron, batch runner, API server, Python embedding. |
| `other`, `unknown` | Unattributable. |

A run is `one_shot` or `background` when its process carries the
`HERMES_SINGLE_QUERY_SESSION` marker that the one-shot paths set (the same marker the
session source and `cache_ttl: auto` read). Engagement (`hermes.engagement.*`) and the
attended-only rows (task cost, tool usage per session, model friction) treat `one_shot`
like `interactive`, as they did before the value existed; `background` and `delegated`
runs are unattended and excluded there. Packages written before `one_shot` existed
carry these runs as `interactive` and still validate. `hermes -z` leaves through
`os._exit`, so it closes its metrics session before exiting rather than relying on the
atexit hook.

Each tool invocation is represented by a Relay tool lifecycle named
`hermes.tool_call`. The terminal counter contains only bounded tool category,
outcome and approval outcome; the same event feeds `hermes.tool_call.latency`
with tool category, latency bucket and explicit retry-count bucket (package v2
carried latency and retries on the terminal row, one row per few calls). Hermes
derives the category from the toolset already declared in its runtime registry;
custom and unrecognized toolsets collapse to `other` rather than exporting
tool or plugin names. The same terminal event also feeds
`hermes.tool.usage.count` with `tool_name`, `outcome` and `error_class`.
`tool_name` is exported only for tools declared in the repository's static
`toolsets.TOOLSETS` (`toolsets.BUILTIN_TOOL_NAMES`, captured before any runtime
custom toolset is created); MCP tools report `mcp` and every plugin or custom
tool reports `plugin`. `error_class` maps Hermes's own `error_type` values
(`tool_error`, `timeout`, `interrupted`, `invalid_arguments`, `blocked`,
`contract_violation`); any other value, such as an exception class name,
collapses to `exception`. Hermes does not infer retries from repeated tool names or
adjacent calls; when the
hook does not provide an explicit retry relationship, the retry bucket is
`unknown`. Approval decisions are emitted as `hermes.tool_approval` marks and
recorded as attributed to a tool call or explicitly `unattributed`. Non-built-in
tool names, call IDs, arguments, results, commands, descriptions, and error
text are not included in shared-metrics events or packages. A started tool that is still
open when its task terminates is closed as failed, timed out, or cancelled and
remains in the task's tool-count bucket.

Successful skill mutations emit `hermes.skill.lifecycle` marks with only a
bounded action and provenance. Successful loads emit `hermes.skill.load`
marks with bounded provenance, first-use or reuse state, reuse-after-patch
state, a use-count bucket and `skill_name`: the skill's name only when it is a
skill Hermes ships (`skills/` or `optional-skills/`), otherwise `custom`. Hermes
derives reuse and patch-generation continuity transactionally in its existing
`skills/.usage.json` state; local or agent-created skill names and exact counts
or generations never enter Relay metrics events, SQLite dimensions, or packages. A use after a new patch is counted once as
`reused_after_patch`; later uses remain ordinary reuse until another patch.
Task-outcome attribution after a patch remains deferred until its window and
multi-skill semantics are defined.

Once per rolling 24 hours, the first activation also emits a
`hermes.install.snapshot` mark describing how the profile is configured: the
memory provider (a bundled provider name, `builtin`, or `plugin`), bucketed
counts of MCP servers, enabled plugins, installed skills, enabled cron jobs,
profiles and connected messaging platforms, the main provider id, the terminal
backend (`local`, `docker`, `ssh`, ... or `other`), the display language (a
shipped locale or `other`) and `install_age_bucket`: how long ago the profile's
first-ever session started. Install age is what lets the backend tell a new user
from an existing one who just opted in. Server, plugin, skill, job and profile
names are never read into the event. The same compare-and-set latch as `hermes.client.active` keeps it to one
row per install per day, and the producer checks the latch before walking the
skills tree.

<!-- ---- v4 install ---- -->
The snapshot also carries six version-lag, channel and hardware fields, all read
offline (no network call, no subprocess):

- `release_channel` (`stable`, `main`, `dev`, `unknown`): a packaged build's baked
  channel (canary builds of main read `main`), a source install's channel record,
  else the checkout's branch (`main` for main/master, `dev` for any other branch).
  The git remote URL and branch names are never read into the event.
- `version_age_bucket` (`lt_7d` … `gte_90d`, `unknown`): age of the *installed*
  version, from its own commit date in the install stamp or checkout.
- `behind_bucket` (`0`, `1`, `2`, `3_to_5`, `6_to_10`, `gte_11`, `unknown`): commits
  or releases behind, only from the update check's cached result for this exact
  revision and under 7 days old; otherwise `unknown`.
- `ram_bucket` (`lt_8g` … `gte_128g`, `unknown`): installed memory rounded to its
  nominal size (the OS total scaled by 1.1 for firmware reservations).
- `gpu_class` (`nvidia`, `amd`, `intel`, `apple_silicon`, `none`, `unknown`): the
  highest-priority GPU vendor from `/proc/driver/nvidia` or DRM PCI vendor ids on
  Linux, the display-adapter registry class on Windows, native arm64 on macOS.
  Never a model name, driver version or VRAM size.
- `local_model_provider_used` (`yes`/`no`): whether the main model or any
  auxiliary task runs on a local or self-hosted server (a local provider id such
  as Ollama/LM Studio/llama.cpp, or a loopback/private-network base URL). The
  URL itself stays local.

### Decision-data metrics

These answer product questions the activity counters cannot: what makes people
stay, where new users drop off, which surfaces and models carry real usage, and
which extensions are worth investing in. Every dimension is a closed enum, a
bucket, a provider/model identifier (as on model routes) or a public name Nous
itself ships.

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.session.count` | entrypoint, surface, platform, turn/failed-turn buckets, active-duration bucket, last outcome, message / model-call / tool-call count buckets (`0` … `101_to_250`, `251_to_1000`, `gte_1001`) | How deep is real usage per surface; do sessions end right after a failure? One row per conversation, written when the surface closes it: the session ids a compression rotation hands it to merge into one row (the retired id closes as soon as its in-flight turn ends); a gateway reset, idle expiry, `/new` or `/branch` starts a new conversation even though it records the old session as its parent. Background review forks that reuse the session id add no turns, calls or messages. Messages are user turns + primary-model replies + tool results; model calls are logical primary API requests (retries excluded). |
| `hermes.install.milestone` | milestone, install age bucket | How long from install to first success, first gateway message, first cron run, first delegation, first created skill (one Hermes' background review created does not count), first long session? Recorded once per install. |
| `hermes.setup.completed` | surface (`cli`/`desktop`), provider | Which providers people choose at setup, and on which surface. |
| `hermes.model_tokens.sum` | call role, model, provider, auxiliary task, token type | Token volume per model/provider, prompt-cache share, and what auxiliary work (compression, titles, vision, ...) costs. The value is a token sum, not an event count. |
| `hermes.model_route.count` `ttft_bucket` | time to first token | Perceived latency per provider/model. |
| `hermes.compression.count` | trigger, outcome, context-fill bucket | How often compaction runs, how full contexts get, and whether it fails. `skipped` = nothing could fail: lock held elsewhere, nothing summarizable (no model call), user stop, or a newer attempt replaced it. |
| `hermes.model_switch.count` | from/to provider, surface | Which providers people leave and move to. |
| `hermes.fallback.count` | from/to provider, error class | How often fallback providers rescue a turn, and from what. |
| `hermes.slash_command.count` | command, surface | Which built-in commands are used (`/retry`, `/undo`, `/new` are friction signals). Skill and plugin commands report `skill`/`plugin`. |
| `hermes.extension.install.count` | kind, source, name, outcome | Which catalog skills, MCP servers and plugins get installed. `name` is a bundled/optional skill, `optional-mcps/` or `plugin-catalog/` entry, otherwise `custom`. |
| `hermes.memory.op.count` | op (`add`/`replace`/`remove`/`read`/`search`/`other`), provider (`builtin`, a bundled memory plugin, else `plugin`), origin (`foreground`/`background_review`), outcome (`success`/`failed`/`rejected`) | Is the learning loop writing memory, who asks for it (the user's turn or the background review), and how often writes are refused or fail. Never the memory text. |
| `hermes.curator.run.count` | trigger (`scheduled`/`manual`), outcome (`success`/`failed`/`skipped`), archived/merged/patched/created buckets | Does the skill curator run, and does it actually consolidate anything. Dry runs report `skipped`; a scheduled check that finds another process already running the pass records nothing. Never skill names. |
| `hermes.delegation.run.count` | subagent-count bucket, depth (`1`–`3`, `gte_4`), mode (`foreground`/`background`), outcome (`success`/`partial`/`failed`/`cancelled`) | How wide and deep delegate_task fan-outs go and how often every child finishes. One row per call, however many completion units it splits into. |
| `hermes.execution_backend.count` | kind (`terminal`/`browser`/`code`), backend, outcome, error class | Which sandboxes carry real work and how reliable each is. Terminal backends are the `terminal.backend` values (else `other`); browser backends are `local`, `lightpanda`, `cdp`, `camofox`, `extension` or a bundled cloud provider (else `other`); execute_code is `local` or `remote`. A command's own nonzero exit is still a backend success, but a foreground command that hits its timeout is `failed`/`timeout`; terminal and execute_code calls a guard refuses before they reach the backend, Hermes' own listings for TUI/Desktop path completion, and calls made by the background review and curator forks are not counted. |
| `hermes.platform.health` | platform, event (`connect_ok`/`connect_failed`/`reconnect`/`disconnect`), error class (`auth`/`network`/`rate_limited`/`config`/`other`) | Which messaging platforms fail to connect or drop, and why. `connect_failed` counts a failed-connect episode once per profile, platform and UTC day (the reconnect watcher's retries are not new rows; the next success ends the episode, and a platform still failing the next day counts again), with the error class of the episode's first failure. Classified from exception types, HTTP statuses and Hermes's own fatal codes, never error text. |
| `hermes.platform.delivery` | platform, outcome (`sent`/`failed`), failure class (`rate_limited`/`too_long`/`auth`/`network`/`forbidden`/`other`) | How often replies fail to reach the user per platform (one count per logical reply, retries included). |
| `hermes.gateway.reply_latency` | platform, first-response bucket (`lt_2s` … `gte_60s`) | Time from an accepted inbound message to the first visible reply text (stream first chunk or final message). |
| `hermes.cron.run` | outcome (`success`/`failed`/`missed`/`skipped`), delivery kind (`local`/`platform`/`webhook`/`none`/`other`), duration bucket | Do scheduled jobs run, fail, get skipped by a gate or overlap, or get missed while Hermes was down. Job names, prompts, schedules and targets are never included. |
| `hermes.startup.latency` | surface (`cli`, `tui`, `desktop_attach`, `gateway_boot`, `serve_boot`), latency bucket (`lt_500ms` … `gte_10s`) | How long each surface takes from launch to usable, so startup regressions show per surface and release. One row per process start: CLI = process start to first rendered prompt (or a `-q` query dispatched; Kanban workers excluded), TUI = Ink process start to gateway ready, Desktop = app start to backend attached, gateway = process start to adapters connected, `hermes serve` = process start to listening. Not counted: a process re-exec'd in place (e.g. `hermes sessions browse` resuming a session) and each dashboard Chat-tab terminal; a TUI/Desktop reconnect to the same backend never re-counts. |
| `hermes.update.run` | kind, outcome, failed_stage, duration_bucket, from_version_age_bucket, apply_mode | Whether updates succeed, how long they take, where they fail, and how stale the version being updated from was. `hermes update` rows are derived from the final update receipt, once per run (`kind` is `desktop` when Desktop's source-checkout hand-off ran it); a run the pre-update interpreter finishes is parked locally with only these fields, only while collection is on, and counted by the next start; Desktop packaged self-updates (`apply_mode=package`) are reported once by the app, after the restart that applies them. |
| `hermes.update.stage` | stage, outcome, duration_bucket | Per-stage result and wall time of `hermes update` (plan, snapshot, apply, deps, build, restart, verify), from the receipt's stage timestamps. |
| `hermes.process.exit` | process_kind, exit_kind, crash_class | How CLI / TUI / gateway / serve / cron-tick processes end (`clean`, `crash` with an exception family only, `killed`, `watchdog`), reported by the next start in the same profile from a local marker (a start with collection off deletes these markers, and pending `provider_setup` ones, unreported). Turns aborted by a turn watchdog also count as `exit_kind=watchdog`. |

Replies the relay connector carries report the platform the conversation lives
on (the inbound's platform, else the platform the connector fronts when it
fronts exactly one), never `relay`; `relay` remains only when neither is known.
A turn whose inbound the connector did not stamp is still a gateway message
(`execution_surface=gateway`, task/session `platform=relay`). `hermes.platform.health`
for the relay connector stays `relay`: its one socket fronts several platforms,
so a connect or drop belongs to none of them alone.

<!-- ---- v5 desktop ---- -->
#### Desktop app: what gets used, what gets in the way, what gets turned off

Recorded by the Desktop app into the focused profile's store, only while that
profile's collection switch is on. With it off the app keeps no local record
(switching it off deletes what was kept) and sends nothing. The app keeps one
local record per gateway connection and profile, shared by that profile's
windows; after a profile switch nothing is kept or sent until the new profile's
switch has been read, and the switch is re-read whenever a window regains focus
(so an opt-out from the CLI or another window takes effect there). There is no rating
prompt or other new UI; each fact comes from an interaction the app already has.
Every value is a closed id defined in the app's code (area, action, notice, flow,
toggle, step), a published config key, or a bucket. Message text, toast text,
session/bot/profile names, paths and setting values never leave.

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.desktop.feature_use` | area (panes, command palette, model/session pickers, voice, Bot Mode, skins, projects, each settings page, full pages, `other`) | Which Desktop areas are used at all, counted at most once per area per UTC day per profile (latched in the profile's local database, so a second window or a backend restart never re-counts). |
| `hermes.desktop.action_use` | action (the app's built-in command/keybinding ids plus a few named buttons; plugin commands and numbered slot shortcuts are `other`, collapsed in the app before anything is kept), via (`click`/`shortcut`/`palette`/`menu`), count bucket | Which buttons and commands people press, and how: aggregated in the app per profile and reported once per finished day (no per-press rows). The row lands in the period of the day it describes, not the day it was sent; each day is recorded once per profile however many windows or backend restarts report it, and a day older than 8 days is dropped. |
| `hermes.desktop.mode_use` | mode (`sessions`/`bots`), active-minutes bucket, messages-sent bucket, bot-count bucket | How Desktop time splits between Bot Mode and regular Sessions. One row per mode used that day, in that day's period (same once-per-profile-per-day latch as `action_use`); active time sums gaps between interactions of up to 5 minutes. |
| `hermes.desktop.friction` | kind (`notice_dismissed`, `error_toast`, `renderer_crash`, `backend_disconnect`, `slow_frame`), detail (notice id, error category, crash reason, drop reason, frame-duration bucket) | What gets in the way. Error toasts carry only their code-defined category; renderer crashes are recorded by the app shell only when the crashed window's own profile collects, and reported by a window of that profile after it comes back; slow frames are long frames while the window is visible, capped per day. |
| `hermes.desktop.dislike` | signal (`quick_close`, `cancelled`, `setting_off_default`, `rage_click`, `undo`, `feature_disabled`), target, setting, direction | Signals that a feature is unwanted: a pane closed within 5s of opening, a dialog/flow backed out of, a setting moved to or away from its default (the key only; the backend compares the saved value to the default itself), three clicks on one control within a second, an undo, a shipped feature switched off. Capped per signal per day. |
| `hermes.desktop.onboarding` | step (first-run steps: provider picker, sign-in, API key, local endpoint, model pick, choose later, free-tier screen, guided setup cards, consent, first message), event (`reached`/`completed`/`abandoned`) | Where first run stops. Each step event once per profile (latched in a small per-profile file); `abandoned` is a step still open when the app next starts. First run happens before the consent question, so until it is answered the app holds the step events in memory only (never on disk, never sent) and records them if the user opts in during that app session; a "no" or quitting first discards them. Switching collection off in the Desktop deletes those latches with the app's own copy. |

Sessions are summarized when they close (finalize, reset or process exit);
delegated child sessions are not counted separately. Milestones latch in the
local database, so each fires once per install however many processes reach it.

#### Per-model quality, friction and context pressure

Provider and model follow the model-route rules: a provider Hermes ships (built in,
an in-tree `plugins/model-providers/` profile or a public models.dev id) and its
model id; custom endpoints, provider plugins installed under
`$HERMES_HOME/plugins/model-providers/` or from pip (names and aliases included),
the local-server aliases of `custom` (`ollama`, `local`, `vllm`, `llamacpp`,
`llama-cpp`, `llama.cpp`) and loopback servers read `custom`. A shipped provider
whose endpoint is a loopback server (`lmstudio`, under any of its aliases) keeps its
name, but its model reads `custom`. A model whose provider is unknown, or whose id is a URL, a file path or a
network address (`host:port`, an IP address, `localhost`) or an AWS ARN (it carries the account
id), reads `custom`. On Azure providers the model id is a deployment name its owner
chose, so it passes only when it is a public model id (Hermes' model catalogs or
the local models.dev cache, e.g. `gpt-4o`); `acme-legal-prod` reads `custom`. The local
subscriber re-runs these rules on the provider/model fields of every mark and drops a
row they would rewrite.

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.model_tool_quality.count` | provider, model, call role, issue (`none`, `invalid_json`, `unknown_tool`, `schema_mismatch`, `empty_arguments`, `repaired`) | Which models emit broken tool calls, and how often Hermes had to repair them. Every emitted call counts once (clean ones as `none`), so the value is a rate denominator. `empty_arguments` only counts for tools with required parameters; `repaired` means Hermes fixed the tool name or the argument JSON and ran the call. |
| `hermes.model_friction.count` | provider, model, signal (`retry`, `undo`, `interrupt`, `quick_abandon`, `switch_away`) | Which models users fight with. Attributed to the model that produced the turn: `/retry` and `/undo` where they execute, a user interrupt of an interactive turn, a session that ends within 60 seconds of a failed turn, and `/model` switching away from the model. |
| `hermes.context_peak.count` | provider, model, peak fill bucket, window bucket (`lt_32k` … `gte_1m`), limit hit (`yes`/`no`) | How close sessions get to each model's context window, and how often they overflow it. One row per closed conversation: the session ids a compression rotation hands it to report once, with the fullest segment; `limit_hit` means a primary call was rejected as too large (context overflow or HTTP 413), the rejections Hermes answers with a forced compression. |

<!-- ---- v5 harness ---- -->
#### Agent-harness accuracy

These tune the agent loop itself. Hermes' own background review and curator
loops never count; delegated subagents do (their tool calls, loops and replies
are model behaviour too). Command text, file paths, tool arguments and reply text never
leave — only the closed values below.

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.file_edit.count` | tool (`patch`, `write_file`), mode (`replace`, `v4a`, `whole_file`), outcome (`applied`, `already_applied`, `no_match`, `ambiguous`, `failed`), match strategy (the patch tool's fuzzy-match chain: `exact`, `line_trimmed`, `whitespace_normalized`, `indentation_flexible`, `escape_normalized`, `trimmed_boundary`, `unicode_normalized`, `block_anchor`, `context_aware`; `none` when nothing was matched) | Which fuzzy-match strategies earn their keep, and how often edits miss or are ambiguous. One row per edit tool call; a multi-hunk V4A patch reports the loosest strategy any hunk needed. |
| `hermes.loop_guard.count` | provider, model, signal (`repeated_tool_call`, `loop_detected`, `iteration_cap`), detector (`exact_failure`, `idempotent_no_progress`, `same_tool_failure`, `identical_call_streak`, `identical_cycle`, `web_search_cap`, `subagent_cap`, `iteration_budget`) | How often each stuck-loop guard fires, per model. `repeated_tool_call` is a warning the call still ran with, `loop_detected` a block or halt, `iteration_cap` a turn that spent its iteration budget. At most once per turn per signal and detector. |
| `hermes.tool_recovery.count` | provider, model, tool (built-in name, else `mcp` / `plugin`), next tool (`same`, `different`, `none`), next outcome (`success`, `error`, `no_tool_call`, `gave_up`) | Whether models recover after a failed tool call. One row per failed call, resolved against the model's next round: its next call to the same tool, else its first call; `no_tool_call` when it answered in text instead, `gave_up` when the turn ended without its reply (halted, budget spent, interrupted, errored). |
| `hermes.terminal.outcome.count` | backend (the terminal backends), command kind (`git`, `package_manager`, `build`, `test_runner`, `python`, `node`, `shell_builtin`, `shell`, `file_ops`, `network`, `container`, `other`), outcome (`ok`, `nonzero`, `timeout`, `killed`) | Which kinds of commands fail or time out, per backend. The kind comes from a fixed table of the first program word (after env assignments and `sudo`-style wrappers). One row per foreground command that reached an exit status; `timeout` / `killed` come from Hermes' own deadline and interrupt flags, so a command's own `exit 124` is `nonzero`. `hermes.execution_backend.count` counts the same calls by whether the backend served them — disjoint dimensions, not a second count of outcomes. |
| `hermes.model_reply_issue.count` | provider, model, issue (`none`, `empty`, `reasoning_only`, `refusal`, `truncated_length`) | Which models return unusable replies. One row per primary model response (usable ones as `none`, the rate denominator). `refusal` and `truncated_length` come only from the structured finish reason (`content_filter`, `length`); `empty` is a valid response with no visible text, tool call or reasoning. |
<!-- ---- end v5 harness ---- -->

<!-- ---- v5 efficiency ---- -->
#### Efficiency: turn cost, waste, tool overhead and prompt-cache breaks

Provider and model follow the model-route rules above. A "user turn" is one user
message through its final reply; Hermes-owned work (background memory/skill review,
the curator, delegated subagents' own turns) is not a user turn. `cache_break` and
`tool_output_truncation` describe model and tool behaviour, so delegated
subagents count there; background review and the curator never do.

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.task_cost.count` | provider, model, tokens bucket (`lt_2k` … `gte_1m`, `unknown`), tool calls bucket, API calls bucket (`0` … `51_to_100`, `gte_101`), outcome (`completed`, `interrupted`, `failed`) | What a user turn costs per model. Tokens are prompt (cache reads/writes included) plus completion over the turn's primary calls; `unknown` when the provider reported no usage. One row per interactive turn the user saw end (a session-close abort is not a turn). |
| `hermes.wasted_tokens.count` | provider, model, reason (`interrupt`, `retry`, `undo`), tokens bucket | How many tokens users throw away. One row per turn an interrupt, `/retry` or `/undo` discarded (`/undo N` counts N turns), attributed to the model that produced that turn; a turn interrupted and then undone counts once. `unknown` when this process never saw the turn (restart, remote host). |
| `hermes.tool_output_truncation.count` | tool (shipped tool name, else `mcp` / `plugin`), truncated (`yes`/`no`), original size bucket (characters: `lt_1k` … `gte_500k`) | Which tools produce output too large to keep inline. One row per tool result; `yes` when the tool cut its own output (terminal, `execute_code` and MCP head/tail truncation; the size is then the original) or the per-result cap or per-turn budget spilled it to disk. |
| `hermes.tool_overhead.count` | enabled tool count bucket, tool schema tokens bucket (`0`, `lt_2k` … `gte_40k`), execution surface | What carrying tool definitions costs. One row per closed interactive conversation: the tools it had enabled and Hermes's own estimate of the tokens their definitions add to each request. |
| `hermes.tool_enabled_unused.count` | toolset (a toolset Hermes ships; MCP servers, plugins and user toolsets read `custom`), used (`yes`/`no`) | Which default toolsets are paid for but never used. One row per enabled toolset per closed interactive conversation (bounded by the shipped toolsets). |
| `hermes.cache_break.count` | provider, model, cause (`compression`, `model_switch`, `toolset_change`, `system_prompt_rebuild`, `provider_reported_miss`, `cache_expired`) | How often Hermes throws away a warm prompt cache, and why. `compression` is expected; `model_switch`, `toolset_change` (the tool array changed mid-conversation) and `system_prompt_rebuild` (a continuing conversation rebuilt its system prompt instead of replaying the stored bytes) are Hermes-known causes; `provider_reported_miss` is a primary call reading zero cached tokens right after a warm read on the same model with no Hermes-known cause, `cache_expired` the same after at least five idle minutes. A known cause is not counted again as a miss. |
<!-- ---- end v5 efficiency ---- -->

<!-- ---- v5 engagement ---- -->
#### Engagement and implicit model satisfaction

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.engagement.surface_day.count` | surface (`cli`, `tui`, `desktop`, `gateway`, `acp`), active-minutes bucket (`0`, `lt_5m`, `5m_to_30m`, `30m_to_2h`, `2h_to_6h`, `gte_6h`) | How long each surface is actually used per day. One row per surface used on a closed UTC day. |
| `hermes.engagement.day.count` | active-minutes bucket, surfaces-used count (`0`–`3`, `gte_4`), primary provider, primary model, active-profile count bucket | Days active per week, multi-surface use, and next-day / next-week return by model. One row per closed UTC day a person used Hermes on. The root (default) profile also writes a host row on days only other profiles were active: `surfaces_used_count` `0`, active minutes `0`, carrying the active-profile count; exclude `surfaces_used_count=0` rows when counting days active. |
| `hermes.model_switch_after.count` | provider, model (the model switched away from), turns-before-switch bucket (`1`, `2_to_3`, `4_to_10`, `11_to_30`, `gte_31`) | How long users stay on a model before `/model` leaves it. Counts the user turns sent on the old model in the conversation (compression segments included; a turn that failed over to a fallback still counts for the model it was sent on; background review forks are not turns); a switch before any turn on the current model is not counted. |

Active time is accumulated locally per UTC day: the sum of the gaps between
consecutive interactions (a user turn starting or ending on an interactive
surface or a gateway message; unattended cron runs, which `hermes.cron.run`
counts, delegated children, background review, curator, batch and API-server /
python embedding are excluded), each gap capped at 5
minutes. The day's rows are recorded once the day closes, by the first
interaction on a later day, in one database transaction, so a day is reported
exactly once per profile however many processes see the rollover; they are
dated to the day they describe. The primary model is the one that served the
most of those user turns that day (`none` when none did), named by the model-route
rules. Days active per week and return by model are derived server-side from
these daily rows and the existing `install_id`: Hermes keeps no weekly window
and no identifier beyond `install_id` for them.

`active_profile_count_bucket` counts the distinct profiles of the host with a
user-owned turn (the interactions above) that UTC day. Every profile folds its turns into one host
accumulator kept in the root (default) profile's database, as opaque local
hashes of each profile's home that never leave it, so a profile counts once
whichever process or multiplexed runtime served it. Only the root profile's
day row carries the count (it reports a day even when the root itself was
idle, with `0` active minutes and surfaces); every other profile's row reads
`0`. When the root profile has collection off, nothing is written to its
database and the count is not reported.
<!-- ---- end v5 engagement ---- -->

<!-- ---- v5 signals ---- -->
#### Onboarding and feature signals

| Metric | Dimensions | Question it answers |
|---|---|---|
| `hermes.tool_unavailable.count` | provider, model, tool name (shipped built-ins only) | Which toolsets should be on by default: the model called a tool Hermes ships that this session did not enable. Any other unknown name (plugin, MCP, hallucinated) stays a `model_tool_quality` `unknown_tool` issue only. A built-in the session enabled but deferred behind `tool_search` (reachable through `tool_call`) is not unavailable. Background reviews, delegated children and cron jobs, whose toolsets are narrowed on purpose, are excluded. |
| `hermes.provider_setup.count` | provider (catalog name; custom endpoints read `custom`), surface (`cli_setup`, `cli_model`, `tui`, `desktop`, `dashboard`), event (`started`, `completed`, `failed`, `abandoned`), failure class (`auth`, `network`, `no_models`, `other`; `none` unless failed; `cancelled` is no longer recorded, a cancel is `abandoned`) | Where connecting a provider breaks down. `started` counts once a provider is picked; the flow's end is recorded by the surface that ran it. A flow the user walked away from is `abandoned`, never `failed`: Esc or Ctrl-C in the CLI pickers, Cancel/Back on a Desktop or dashboard sign-in, consent declined on the provider's page, and a sign-in code left to expire (any provider). Back (Left arrow) in the CLI keeps the flow open, so picking the same provider again continues it (one `started`), while picking another provider or leaving the command ends it `abandoned`. A flow nobody finished leaves a local marker that the next setup start or Hermes start in the profile reports as `abandoned` (its process is gone, or it has been pending over an hour). A Desktop/dashboard sign-in that dies mid-poll keeps the class of the error that ended it (`network` for a dropped connection, `auth` for a refusal), not a bare `other`. A new or changed provider API key saved from a form (TUI/Desktop/dashboard) and a newly added custom endpoint start and complete in one action; clearing a key, re-saving the same key, editing an existing endpoint, ecosystem tokens (`GITHUB_TOKEN`, `GH_TOKEN`, `HF_TOKEN`) and keys a tool's settings panel also asks for (e.g. `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPINFRA_API_KEY`) are not counted from the generic key form (the Desktop's onboarding and model settings mark their saves as a provider connection, so those count). Never a key, token, base URL or error text. Leaving the provider picker before choosing one is not counted. |
| `hermes.feature_adoption.count` | feature (`memory`, `skills_created`, `delegation`, `cron`, `gateway_platform`, `desktop`, `tui`, `mcp`, `plugins`, `browser`, `voice`, `kanban`, `projects`, `bot_mode`, `curator`), days since install (`same_day`, `1d_to_7d`, `7d_to_30d`, `30d_to_90d`, `gte_90d`, `unknown`) | How long after install each major feature is first really used. Once per feature per install, latched in the local database, derived from the counters above (a foreground memory write, a skill created at the user's request (not by Hermes' background review), a successful MCP/plugin/browser/TTS/kanban tool call, a Desktop/TUI/gateway task, a cron run, a manual curator run; the scheduled curator pass does not count) plus direct first-use reports for Bot Mode messages and project creation. The age is the owning profile's (its first session). |
| `hermes.feature_disabled.count` | kind (`toolset`, `skill`, `plugin`, `platform`, `setting`, `memory`, `curator`, `compression`), name, surface (`cli_tools`, `cli_config`, `cli_slash`, `tui`, `desktop`, `dashboard`), event (`disabled`, `re_enabled`) | What users turn off. Diffed at the config write itself: a default-on toolset removed, a skill or plugin added to its disabled list, a default-`true` setting set false (and each moved back). Names are public only when shipped — toolset key, bundled/catalog skill, bundled/catalog plugin (messaging-platform plugins report as `platform`), `DEFAULT_CONFIG` key path (never a value) — else `custom`. Uninstalling a catalog skill counts as `disabled`. Only user entry points record (`hermes tools` / `config` / `skills` / `plugins`, chat slash commands, TUI/Desktop, dashboard); setup and migrations do not, even when a migration runs inside one of them (`hermes config migrate`, a profile created from the dashboard). A setting whose value is a `${VAR}` template is not compared. The diff and the record run on a background thread after the write, outside every config lock. At most once per (kind, name, event) per day. |
<!-- ---- end v5 signals ---- -->

Local state is written under:

```text
$HERMES_HOME/telemetry/shared_metrics/metrics.sqlite3
$HERMES_HOME/telemetry/shared_metrics/outbox/*.json
```

The database keeps transactional aggregate and package-outbox state. Package
files are immutable delta documents that conform to a closed JSON schema and
are written with atomic replacement as compact JSON (`jq .` pretty-prints one).
Once the ingest has accepted or refused a package, the database keeps only its
send state and drops its copy of the body; the file is the local history copy.
Each package records the Hermes version,
OS family, architecture, and install method as bounded client resources.
Unrecognized platform or installation values are exported as `unknown`; raw
platform strings, hostnames, and paths are never included. Fully packaged
aggregate rows and successfully exported package rows and files are retained
locally for 30 days. Pending package rows and counters with unexported deltas
are never pruned.
Package schemas v1 and v2 remain unchanged for existing outbox files. New
packages use v3, which also accepts the v2 field sets of `hermes.model_route.count`,
`hermes.tool_call.count` and the task counters so counters recorded before an upgrade
drain safely.
Vocabularies derived from in-repo registries (tool names, platforms, memory
providers, error classes) are bounded by pattern in the JSON schema; the
authoritative allowlist is `shared_metrics_contract.py`.

Each package contains an `install_id` generated as a random UUID. Despite the
schema field name, its current scope is one `HERMES_HOME`, so it is more
precisely a persistent pseudonymous profile identifier. It is not derived from
hardware, account, host, path, or credential data. It remains stable across
packages from that profile and can therefore link those local packages.
Deleting `$HERMES_HOME/telemetry/shared_metrics` resets the identifier together
with all aggregates and package files.

Remote delivery is opt-in and off by default. Reusing the persistent local
identifier remotely required a separate product and privacy decision covering
consent, identity scope, reset behavior, retention, and deletion — that
decision has been made.

> Those decisions are recorded in
> [Appendix A](#appendix-a-remote-exporter-decisions-phase-2), and the exporter
> implementing them has shipped. Collection alone still transmits nothing: the
> sender runs only when `telemetry.shared_metrics.send` is also true. Each
> transmitted package carries the stable `install_id` as-is (product decision,
> 2026-08-27 — see A.2 for the record, including the superseded
> HMAC-pseudonym design).

The install identity is scoped to one `HERMES_HOME`. To reset it, stop Hermes
processes and remove `$HERMES_HOME/telemetry/shared_metrics`. This deliberately
removes the old identity, aggregate database, and queued local packages
together; the next consented session creates a new identity. Disabling shared
metrics stops new collection but does not silently delete previously collected
local state.

## Smoke Test

Run a real Hermes CLI turn against the deterministic local model server:

```bash
./.venv/bin/python scripts/smoke_nemo_relay_shared_metrics.py
```

The script uses the installed `nemo-relay` dependency by default. Pass
`--relay-python ../nemo-relay/python` only when testing a locally built Relay
binding.

The smoke has the local model request a real `read_file` tool call before its
final response, then drives create, load, reuse, patch, edit, stale, archive,
restore, and install skill transitions through the installed Relay binding. It
verifies model, provider, task, tool, and skill counters in SQLite, validates
all exported delta packages against the closed schema, verifies the
pseudonymous client-active counter, and checks that prompt, response, tool-call
ID, tool-result, and skill-name canaries are absent from the packages.

## Appendix A: Remote Exporter Decisions (Phase 2)

Status: **implemented.** This appendix answers the product and
privacy questions that "Current Slices" defers to a future remote exporter. It
records what was decided and why, so the reasoning survives the implementation.

Sending is off by default and requires both `telemetry.shared_metrics.enabled`
and `telemetry.shared_metrics.send`.

The exporter sends the package files already written under
`$HERMES_HOME/telemetry/shared_metrics/outbox/` to the Hermes telemetry ingest
service. That service validates only the envelope (`schema_version` plus a UUID
`package_id`) and stores the body verbatim in S3.

### A.1 Consent

Transmission is a **separate opt-in** from collection, under a new config key:

```yaml
telemetry:
  shared_metrics:
    enabled: false   # collect locally
    send: false      # NEW: transmit to the Nous telemetry service
```

- `send` defaults to **false**. Collection alone never transmits.
- `send` requires `enabled`. It does **not** imply it: a transmission flag must
  not silently switch on collection. `send: true` with `enabled: false` warns
  and does nothing.
- Like `enabled`, `send` is profile-owned and is not overridden by
  managed-scope configuration.

Both keys are asked once per profile, with the same three answers everywhere
(Send to Nous / Local only / No thanks):

| Surface | Where the offer appears |
| --- | --- |
| `hermes setup` | At the end of every flow (Quick, Full, Blank Slate, Portal, `--quick`). |
| `hermes` / `hermes --tui` | Once before an interactive chat starts. Skipped for `-q`, piped or JSON output, spawned actions and Desktop-hosted panes. |
| Hermes Desktop | A strip above the composer, after first-run onboarding. It never blocks the composer or takes focus. |
| Web dashboard | A banner above every page, for the profile being managed. |

"No thanks" is the default in the terminal, so pressing Enter never opts
anyone in. Esc in the terminal and the dashboard banner's ✕ leave the question
open, so it is asked again next time. Answering on any surface writes both keys
to the profile's `config.yaml`, and a profile that already carries either key is
never asked again. A managed install is never offered. To change the answer
later, use `hermes setup telemetry`, `hermes tools`, or Desktop's Settings ›
Safety › Privacy & network.

**A package is only sent when its whole period falls inside a recorded
consent window.** Consent is stored as explicit intervals in the shared-
metrics SQLite store (`send_consent_windows`): a window opens when `send:
true` is first observed, is confirmed forward by every later observation,
and closes — at the last *confirmed* moment, never at the wall clock — when
`send: false` is observed. A single reconciler derives this table from the
config on every process start, so wizard changes, hand-edits to
`config.yaml`, and mid-pass revocations all take the same path, and no
transition can be missed by any of them.

Any package whose period predates the first window, falls between windows,
or runs past the newest confirmed moment is excluded — the gate fails
closed. A fresh package therefore waits at most one process start after its
period completes before becoming eligible.

The gate is on the **period**, not on the package's creation time. One period
is split across several packages created on different days: a day's first
package is written that day, and a tail package for the same period typically
follows the next day. Gating on creation time would send a period's tail while
dropping its head, reporting a **silently undercounted** day. Gating on the
period keeps consent forward-only and every transmitted period complete.

Local history can be up to 30 days old, and that data was collected under a
promise that nothing is uploaded. Honouring consent forward-only costs at most
30 days of backlog we never had permission to send.

### A.2 Identity scope — the stable install_id is transmitted as-is

**Decision record.** The original design of this exporter (and revisions 1–8
of this appendix) transmitted a keyed pseudonym instead of the identifier:
`HMAC-SHA256(key = locally-held rotating salt, message = install_id)`, with
the salt rotating every 30 days. On **2026-08-27**, before the feature
shipped (zero consented users, zero production transmissions), the product
owner decided the analytical need is a **stable cross-window identity** —
retention curves, longitudinal install behaviour — which rotation by design
destroys. The pseudonymization layer was removed in full rather than
weakened in place.

What is transmitted now:

- Each package carries `install_id` verbatim: the persistent, profile-scoped
  random UUID described above.
- It is generated locally (`uuid4`), contains no hardware, account, user, or
  machine-derived information, and identifies a *profile*, not a person.
- It is stable until the user deletes the shared-metrics directory, which
  regenerates it (see A.4).

Consequences stated plainly rather than papered over:

- Packages from one profile correlate **indefinitely**, not per-window.
  Long-term linkability of one install's daily envelope sequence is now the
  designed behaviour, not a residue.
- The A.3 residue analysis of the old design (stable `resource` tuple +
  contiguous periods bridging rotation windows) is moot — there is no window
  boundary left to bridge.
- The setup wizard's consent language states this identity model explicitly;
  it was updated in the same change that removed the derivation, so no
  consent was ever collected under the old wording in any shipped build.

**Byte-identical resends still hold.** The transmitted id is recorded on the
row (`sent_install_id`) when the package is first prepared, and the wire body
is always rebuilt from that recorded value, so a retry rebuilds identical
bytes. The contract requires this: resending a `package_id` with different
content is undefined behaviour. (With a stable id the recorded copy is no
longer load-bearing against rotation — it remains as the audit column and as
cheap insurance against any future change to identity semantics.)

### A.3 Rotation — removed (decision record)

Salt rotation was deleted together with the derivation (product decision,
2026-08-27). This section is retained as a record of what the earlier design
did and why the removal was accepted:

- Rotation existed to bound long-term linkability: one identity per 30-day
  window, unrelated identities across windows.
- The documented residue (see git history for the full analysis): the
  envelope's stable, low-entropy `resource` tuple plus contiguous daily
  periods could plausibly bridge windows for rare configurations anyway, so
  the boundary was a cost-raiser, not a wall.
- The product need that killed it: cross-window continuity is precisely what
  retention analysis requires. A boundary that mostly inconveniences honest
  analysis while only raising costs for a determined correlator was judged
  the wrong trade once stable identity became a requirement.

There is no salt in the store, no rotation schedule, and no derived
identifier anywhere in the pipeline.

### A.4 Reset behavior

Removing `$HERMES_HOME/telemetry/shared_metrics` still resets local identity,
aggregates, and package files, exactly as documented above. Two honest
qualifications now apply:

- Reset regenerates `install_id`, so subsequent packages transmit a **new**
  identity. Local reset does give a new remote identity.
- Reset **cannot unsend**. Packages already transmitted remain in the ingest
  service's storage under the identifier they were sent with. There is no
  read-back or delete API in the v1 contract.

Setting `send: false` stops transmission immediately: consent is re-read
before every package, so a pass already in flight stops after the package it
is currently sending rather than draining its whole batch. It does not delete
previously transmitted packages, and it does not stop local collection.

Turning sending off also **closes the consent window** — at the last moment
consent was actually observed, not at the wall clock. Packages whose periods
fall between one window and the next are never transmitted, even if sending
is later re-enabled, and this holds for any number of on/off cycles, across
hand-edits with no process running, and under a clock that jumps in either
direction (window opens are clamped above every timestamp already in the
store; observation marks advance by a bounded step per call, so one glitched
forward sample cannot drag the confirmation horizon years ahead; a close
never lands after the closing observation's own clock).
Unlike the earlier single moving opt-in date, closing and reopening does NOT
discard the still-undelivered backlog from a previous consented window —
those packages stay inside their own interval and remain eligible.

One deliberate upgrade-path consequence: packages exported under the
pre-interval consent model (before `send_consent_windows` existed) predate
the first recorded window and are therefore never transmitted after an
upgrade. This is the fail-closed direction — re-importing the old moving
day-stamp to release them would re-import the semantics five review rounds
showed to be unsound — and it costs at most the undelivered backlog, never
collected data.

### A.5 Retention

- **Local:** unchanged — 30 days for successfully exported history, and pending
  deltas are kept until exported. Send state does **not** extend local
  retention: a package that could never be sent is still pruned at 30 days.
  Unbounded local growth against a permanently unreachable endpoint is a worse
  failure than losing metrics from an install that has been broken for a month.
- **Remote:** raw packages are retained in S3 without expiry in production and
  for 30 days in staging.

### A.6 Deletion

There is no remote deletion path in the v1 contract, and this appendix does not
invent one. What a user can do:

| Action | Effect |
|---|---|
| `send: false` | No further packages leave the machine |
| `enabled: false` | Collection stops; existing local state remains |
| Remove `.../shared_metrics` | Local identity, aggregates, and files reset; future sends use a new install_id |
| Delete already-sent data | Not self-service — requires an operator acting on the S3 bucket |

If a deletion-on-request obligation is ever taken on, the lookup path is now
direct: the user's `install_id` (readable from their local store) is the key
their data is stored under. Building the service-side delete API remains a
new product decision, not an implementation detail.

### A.7 What the outbox directory is

Recorded because it was misread once during Phase 2 planning, in a way that
would have deleted user data.

The directory is **local history, not a send-queue**. `package_outbox` is the
SQLite table; its `exported_at` column means "written to disk", not "sent".
Files are immutable and pruned **by age alone**.

The ingest contract says senders should delete a package from their outbox on
`202`. **The exporter does not do this.** Deleting on acknowledgement would
repurpose the user's 30-day local history as a transmission queue and destroy
state they were promised. Send state lives in new columns on the
`package_outbox` table instead; the files are untouched by transmission.

### A.8 Scope note

The `install_id` field inside the package body is transmitted as the
generator wrote it (rewritten from the row's frozen `sent_install_id`, which
records the same value). No other payload field changes, nothing is added,
and the service treats the whole body as opaque. Payload schema evolution
therefore stays a sender-side concern, as before.
