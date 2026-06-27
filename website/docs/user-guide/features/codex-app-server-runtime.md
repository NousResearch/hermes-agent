---
title: Codex App-Server Runtime (optional)
sidebar_label: Codex App-Server Runtime
---

# Codex App-Server Runtime

Hermes can optionally hand `openai/*` and `openai-codex/*` turns to the [Codex CLI app-server](https://github.com/openai/codex) instead of running its own tool loop. When enabled, terminal commands, file edits, sandboxing, and MCP tool calls all execute inside Codex's runtime — Hermes becomes the shell around it (sessions DB, slash commands, gateway, memory and skill review).

This is **opt-in only**. Default Hermes behavior is unchanged unless you flip the flag. Hermes never auto-routes you onto this runtime.

## Why

- Run OpenAI agent turns against your **ChatGPT subscription** (no API key required) using the same auth flow Codex CLI uses.
- Use **Codex's own toolset and sandbox** — `shell` for terminal/read/write/search, `apply_patch` for structured edits, `update_plan` for planning, all running inside seatbelt/landlock sandboxing.
- **Native Codex plugins** — Linear, GitHub, Gmail, Calendar, Canva, etc. — installed via `codex plugin` are auto-migrated and active in your Hermes session.
- **Hermes' richer tools come along** — web_search, web_extract, browser automation, vision, image generation, skills, and TTS work via an MCP callback. Codex calls back into Hermes for tools it doesn't have built in.
- **Memory and skill nudges keep working** — Codex's events are projected into Hermes' message shape so the self-improvement loop sees a normal-looking transcript.

## What tools the model actually has

This is the part most users want to know up front. When this runtime is on, the model running your turn has three independent sources of tools:

### 1. Codex's built-in toolset (always on)

These ship with `codex app-server` itself — no Hermes involvement, no MCP, no plugins. All five are available the moment the runtime starts:

- **`shell`** — runs arbitrary shell commands inside the sandbox. This is how the model reads files (`cat`, `head`, `tail`), writes them (`echo > foo`, heredocs), searches them (`find`, `rg`, `grep`), navigates directories (`ls`, `cd`), runs builds, manages processes, and anything else you'd do in bash.
- **`apply_patch`** — applies a structured multi-file diff in Codex's patch format. The model uses this for non-trivial code edits (adding a function, refactoring across files); shell heredocs are still available for one-off writes.
- **`update_plan`** — codex's internal todo / plan tracker. Equivalent of Hermes' `todo` tool, but managed entirely inside codex's runtime.
- **`view_image`** — load a local image file into the conversation so the model can see it.
- **`web_search`** — codex has its own built-in web search when configured. Hermes also exposes `web_search` (Firecrawl-backed) via the callback below; the model picks whichever it prefers.

So **anything you'd do via terminal — read/write/search/find/run — codex does natively**. The sandbox profile (`:workspace` by default when you enable the runtime) controls what's writable.

### 2. Native Codex plugins (auto-migrated from your `codex plugin` install)

When you enable the runtime, Hermes queries codex's `plugin/list` RPC and writes a `[plugins."<name>@openai-curated"]` entry for every plugin you have installed. The plugins themselves are managed by codex and authorized once via codex's own UI.

Examples (the ones the OpenClaw thread highlighted as "YouTube-video-worthy"):

- **Linear** — find/update issues
- **GitHub** — search code, view PRs, comment
- **Gmail** — read/send mail
- **Google Calendar** — create/find events
- **Outlook calendar/email** — same shape via the Microsoft connector
- **Canva** — design generation
- ...whatever else you've installed via `codex plugin marketplace add openai-curated` + `codex plugin install ...`

What's NOT migrated:
- Plugins you haven't installed yet — install them in Codex first.
- ChatGPT app marketplace entries (`app/list`) — these are already enabled inside codex by virtue of your account auth.

### 3. Hermes tool callback (MCP server, registered in `~/.codex/config.toml`)

Hermes registers itself as an MCP server so codex can call back for tools codex doesn't ship with. Available via the callback:

- **`web_search`** / **`web_extract`** — Firecrawl-backed; tends to be cleaner than scraping for structured content.
- **`browser_navigate` / `browser_click` / `browser_type` / `browser_press` / `browser_snapshot` / `browser_scroll` / `browser_back` / `browser_get_images` / `browser_console` / `browser_vision`** — full browser automation via Camofox or Browserbase.
- **`vision_analyze`** — call a separate vision model to inspect an image (different from codex's `view_image` which loads it into the conversation).
- **`image_generate`** — image generation through Hermes' image_gen plugin chain.
- **`skill_view` / `skills_list`** — read from Hermes' skill library.
- **`text_to_speech`** — TTS through Hermes' configured provider.

When the model wants one of these, codex spawns the `hermes_tools_mcp_server` subprocess via stdio MCP, the call is dispatched through `model_tools.handle_function_call()` (same code path as Hermes' default runtime), and the result is returned to codex like any other MCP response.

### What's NOT available on this runtime

These four Hermes tools require the running AIAgent context (mid-loop state) to dispatch, and a stateless MCP callback can't drive them. Switch back to the default runtime (`/codex-runtime auto`) when you need any of them:

- **`delegate_task`** — spawn subagents
- **`memory`** — Hermes' persistent memory store
- **`session_search`** — cross-session search
- **`todo`** — Hermes' todo store (codex's `update_plan` is the in-runtime equivalent)

## Workflow features (`/goal`, kanban, cron)

### `/goal` (the Ralph loop)

**Works on this runtime.** Goals persist in `state_meta` keyed by session id, the continuation prompt feeds back as a normal user message through `run_conversation()`, and codex executes the next turn natively. The goal judge runs via the auxiliary client (configured via `auxiliary.goal_judge` in config.yaml), independent of which runtime is active. The judge's "blocked, needs user input" verdict is a clean escape if codex stalls on approvals.

**One thing to be aware of:** each continuation prompt is a fresh codex turn, which means codex re-evaluates command approval policy from scratch. If you're doing a long-running goal with lots of writes, expect more approval prompts than you'd see on a single in-session task. Set `default_permissions = ":workspace"` (which Hermes does automatically when you enable the runtime) so simple workspace writes don't require prompting.

### Kanban (multi-agent worktree dispatch)

**Works on this runtime, with one subtle dependency.** The kanban dispatcher spawns each worker as a separate `hermes chat -q` subprocess that reads the user's config — which means if `model.openai_runtime: codex_app_server` is set globally, workers also come up on the codex runtime.

What works inside a codex-runtime worker:
- Codex's full toolset (shell, apply_patch, update_plan, view_image, web_search) — the worker does its actual task work natively
- The migrated codex plugins — Linear, GitHub, etc.
- The Hermes tool callback for browser_*, vision, image_gen, skills, TTS

What also works because the MCP callback exposes them:
- **`kanban_complete` / `kanban_block` / `kanban_comment` / `kanban_heartbeat`** — the worker handoff tools. These read `HERMES_KANBAN_TASK` from env (set by the dispatcher), gate access correctly, and write to `~/.hermes/kanban.db`. Without these in the callback, a worker on this runtime could do its task but couldn't report back, hanging until the dispatcher's timeout.
- **`kanban_show` / `kanban_list`** — read-only board queries for the worker to check its own context.
- **`kanban_create` / `kanban_unblock` / `kanban_link`** — orchestrator-only operations. Available for orchestrator agents running on the codex runtime that need to dispatch new tasks.

The kanban tools are gated by `HERMES_KANBAN_TASK` env var the dispatcher sets — that var is propagated to the codex subprocess (codex inherits env) and from there to the spawned `hermes-tools` MCP server subprocess. So the tools see the right task id and gate correctly.

### Cron jobs

**Not specifically tested.** Cron jobs run via `cronjob` → `AIAgent.run_conversation`, the same code path as the CLI. If the cron job's config has `openai_runtime: codex_app_server` it'll run on codex. The same tool-availability rules apply — codex built-ins + plugins + MCP callback work, agent-loop tools (delegate_task, memory, session_search, todo) don't. If your cron job relies on those, scope the cron to a profile that uses the default runtime.

## Trade-offs

|  | Hermes default runtime | Codex app-server (opt-in) |
|---|---|---|
| `delegate_task` subagents | yes | not available — needs agent loop context |
| `memory`, `session_search`, `todo` | yes | not available — needs agent loop context |
| `web_search`, `web_extract` | yes | yes (via MCP callback) |
| Browser automation (Camofox/Browserbase) | yes | yes (via MCP callback) |
| `vision_analyze`, `image_generate` | yes | yes (via MCP callback) |
| `skill_view`, `skills_list` | yes | yes (via MCP callback) |
| `text_to_speech` | yes | yes (via MCP callback) |
| Codex `shell` (terminal/read/write/search/find/run) | — | yes (Codex built-in) |
| Codex `apply_patch` (structured multi-file edits) | — | yes (Codex built-in) |
| Codex `update_plan` (in-runtime todo) | — | yes (Codex built-in) |
| Codex `view_image` (load image into conversation) | — | yes (Codex built-in) |
| Codex sandbox (seatbelt/landlock, profiles) | — | yes (Codex built-in) |
| ChatGPT subscription auth | — | yes (via `openai-codex` provider) |
| Native Codex plugins (Linear, GitHub, etc.) | — | yes (auto-migrated) |
| User MCP servers | yes | yes (auto-migrated to codex) |
| Memory + skill review (background) | yes | yes (via item projection) |
| Multi-turn conversations | yes | yes |
| `/goal` (Ralph loop) | yes | yes |
| Kanban worker dispatch | yes | yes (via callback) |
| Kanban orchestrator tools | yes | yes (via callback) |
| All gateway platforms | yes | yes |
| Non-OpenAI providers | yes | n/a — OpenAI/Codex-scoped |

## Prerequisites

1. **Codex CLI installed:**
   ```bash
   npm i -g @openai/codex
   codex --version   # 0.130.0 or newer
   ```
2. **Codex OAuth login.** The codex subprocess reads `~/.codex/auth.json`. Two ways to populate it:
   ```bash
   codex login                  # writes tokens to ~/.codex/auth.json
   ```
   Hermes' own `hermes auth login codex` writes to `~/.hermes/auth.json` — that's a separate session. **Run `codex login` separately** if you haven't.

3. **(Optional) Install the Codex plugins you want.** When you enable the runtime, Hermes auto-migrates whichever curated plugins you've already installed via Codex CLI:
   ```bash
   codex plugin marketplace add openai-curated
   # then via codex's TUI, install Linear / GitHub / Gmail / etc.
   ```
   Hermes will discover them and write `[plugins."<name>@openai-curated"]` entries to `~/.codex/config.toml` automatically.

## Enabling

In a Hermes session:

```
/codex-runtime codex_app_server
```

That command:
- Verifies the `codex` CLI is installed (blocks with an install hint if not).
- Persists `model.openai_runtime: codex_app_server` to your config.yaml.
- Migrates user MCP servers from `~/.hermes/config.yaml` to `~/.codex/config.toml`.
- **Discovers and migrates installed native Codex plugins** (Linear, GitHub, Gmail, Calendar, Canva, etc.) by querying Codex's `plugin/list` RPC.
- **Registers Hermes' own tools as an MCP server** so the codex subprocess can call back for tools codex doesn't ship with.
- **Writes `default_permissions = ":workspace"`** so the sandbox allows writes within the workspace without prompting for every operation.
- Tells you what was migrated. Takes effect on the **next** session — the current cached agent keeps the prior runtime so prompt caches stay valid.

Synonyms: `/codex-runtime on`, `/codex-runtime off`, `/codex-runtime auto`.

To check current state without changing anything:
```
/codex-runtime
```

You can also set it manually in `~/.hermes/config.yaml`:
```yaml
model:
  openai_runtime: codex_app_server   # default is "auto" (= Hermes runtime)
```

## Self-improvement loop (memory + skill nudges)

Hermes' background self-improvement fires on counter thresholds:

- Every 10 user prompts → a forked review agent looks at the conversation and decides whether anything should be saved to memory.
- Every 10 tool iterations within a single turn → same idea but for skills (`skill_manage` writes).

**Both keep working on the codex runtime.** The codex path projects each completed `commandExecution` / `fileChange` / `mcpToolCall` / `dynamicToolCall` item into a synthetic `assistant tool_call` + `tool` result message, so by the time the review runs it sees the same shape it sees on the default Hermes runtime.

How the wiring stays equivalent:

| | Default runtime | Codex runtime |
|---|---|---|
| `_turns_since_memory` increments | per user prompt, in run_conversation pre-loop | same code path, before the early-return |
| `_iters_since_skill` increments | per tool iteration in the chat-completions loop | by `turn.tool_iterations` after the codex turn returns |
| Memory trigger (`_turns_since_memory >= _memory_nudge_interval`) | computed in pre-loop, fires after response | computed in pre-loop, passed through to codex helper |
| Skill trigger (`_iters_since_skill >= _skill_nudge_interval`) | computed after the loop | computed after the codex turn |
| `_spawn_background_review(messages_snapshot=..., review_memory=..., review_skills=...)` | called when either trigger fires | called identically when either trigger fires |

One detail: the review fork itself needs to call Hermes' agent-loop tools (`memory`, `skill_manage`), which require Hermes' own dispatch. So when the parent agent is on `codex_app_server`, the review fork is **downgraded to `codex_responses`** — same OAuth credentials, same `openai-codex` provider, but talks to OpenAI's Responses API directly so Hermes owns the loop and the agent-loop tools work. This is invisible to the user.

Net effect: enable the codex runtime and your memory + skill nudges keep firing exactly as they would otherwise.

## How approvals work

Codex requests approval before executing commands or applying patches. These get translated into Hermes' standard "Dangerous Command" prompt:

```
╭───────────────────────────────────────╮
│ Dangerous Command                     │
│                                       │
│ /bin/bash -lc 'echo hello > foo.txt'  │
│                                       │
│ ❯ 1. Allow once                       │
│   2. Allow for this session           │
│   3. Deny                             │
│                                       │
│ Codex requests exec in /your/cwd      │
╰───────────────────────────────────────╯
```

- **Allow once** → approve this single command.
- **Allow for this session** → Codex won't re-prompt for similar commands.
- **Deny** → command is rejected; Codex continues in read-only mode.

For `apply_patch` (file edit) approvals, Hermes shows a summary of what changed (`1 add, 1 update: /tmp/new.py, /tmp/old.py`) when codex provides the data via the corresponding `fileChange` item.

## Permission profiles

Codex has three built-in permission profiles:
- `:read-only` — no writes; every shell command requires approval
- `:workspace` — writes within the current workspace allowed without prompts (Hermes' default when you enable the runtime)
- `:danger-no-sandbox` — no sandbox at all (don't use this unless you understand it)

You can override the default in `~/.codex/config.toml` outside Hermes' managed block:

```toml
default_permissions = ":read-only"
```

(Hermes will preserve your override on re-migration as long as it lives outside the `# managed by hermes-agent` markers.)

## Auxiliary tasks and ChatGPT subscription token cost

When this runtime is on with the `openai-codex` provider, **auxiliary tasks (title generation, context compression, vision auto-detect, session search summarization, the background self-improvement review fork) also flow through your ChatGPT subscription by default**, because Hermes' auxiliary client uses the main provider/model when no per-task override is set.

This isn't specific to `codex_app_server` — it's true for the existing `codex_responses` path too — but it's more visible here because you're explicitly opting in for the subscription billing.

To route specific aux tasks to a cheaper / different model, set explicit overrides in `~/.hermes/config.yaml`:

```yaml
auxiliary:
  title_generation:
    provider: openrouter
    model: google/gemini-3-flash-preview
  context_compression:
    provider: openrouter
    model: google/gemini-3-flash-preview
  vision_detect:
    provider: openrouter
    model: google/gemini-3-flash-preview
  session_search:
    provider: openrouter
    model: google/gemini-3-flash-preview
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

The self-improvement review fork inherits the main runtime via `_current_main_runtime()` and Hermes downgrades it from `codex_app_server` to `codex_responses` automatically (so the fork can actually call `memory` and `skill_manage` — Hermes' own agent-loop tools). That fork still uses your subscription auth unless you've routed aux tasks elsewhere.

## Editing `~/.codex/config.toml` safely

Hermes wraps everything it manages between two marker comments:

```toml
# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section
default_permissions = ":workspace"
[mcp_servers.filesystem]
...
[plugins."github@openai-curated"]
...
# end hermes-agent managed section
```

Anything **outside** that block is yours. Re-running migration (via `/codex-runtime codex_app_server` or whenever you toggle the runtime on) replaces the managed block in place but preserves user content above and below it verbatim. This means you can:

- Add your own MCP servers Hermes doesn't know about
- Override `default_permissions` to `:read-only` if you prefer to be prompted
- Configure codex-only options (model, providers, otel, etc.)
- Add user-defined permission profiles in `[permissions.<name>]` tables

Anything you add **inside** the managed block will get clobbered on the next migration. If you need a tweak that requires editing the managed block, file an issue and we'll add the knob.

## Multi-profile / multi-tenant setups

By default, Hermes points the codex subprocess at `~/.codex/` regardless of which Hermes profile is active. This means `hermes -p work` and `hermes -p personal` share the same Codex auth, plugins, and config. For most users this is the right behavior — it matches what running `codex` CLI directly would do.

If you want per-profile Codex isolation (separate auth, separate installed plugins, separate config), set `CODEX_HOME` explicitly per profile. The cleanest way is to point at a directory under your `HERMES_HOME`:

```bash
# Inside the work profile, you might wrap hermes:
CODEX_HOME=~/.hermes/profiles/work/codex hermes chat
```

You'll need to re-run `codex login` once with that `CODEX_HOME` set so the OAuth tokens land in the profile-scoped location. After that, `hermes -p work` will operate on isolated Codex state.

We don't auto-scope this because moving an existing user's `~/.codex/` would silently invalidate their Codex CLI auth — anyone who already ran `codex login` would have to re-authenticate. Opt-in feels safer than surprising users.

## HOME environment variable passthrough

Hermes does NOT rewrite `HOME` when spawning the codex app-server subprocess (we use `os.environ.copy()` and only overlay `CODEX_HOME` and `RUST_LOG`). This means:

- Commands codex runs via its `shell` tool see the real user `HOME` and find `~/.gitconfig`, `~/.gh/`, `~/.aws/`, `~/.npmrc`, etc. correctly.
- Codex's internal state stays isolated through `CODEX_HOME` (which points at `~/.codex/` by default).

This matches the boundary OpenClaw arrived at after some early experimentation: isolate Codex's state, leave the user's home alone. (Cf. openclaw/openclaw#81562.)

## MCP server migration

Hermes' `mcp_servers` config is auto-translated to the TOML format Codex expects. The migration runs every time you enable the runtime and is idempotent — re-runs replace the managed section but preserve any user-edited Codex config.

What translates:

| Hermes (`config.yaml`) | Codex (`config.toml`) |
|---|---|
| `command` + `args` + `env` | stdio transport |
| `url` + `headers` | streamable_http transport |
| `timeout` | `tool_timeout_sec` |
| `connect_timeout` | `startup_timeout_sec` |
| `enabled: false` | `enabled = false` |

What's not migrated:
- Hermes-specific keys like `sampling` (Codex's MCP client has no equivalent — these are dropped with a per-server warning).

## Native Codex plugin migration

Plugins installed via `codex plugin` (Linear, GitHub, Gmail, Calendar, Canva, etc.) are discovered through Codex's `plugin/list` RPC. For each plugin where `installed: true`, Hermes writes a `[plugins."<name>@openai-curated"]` block enabling it in your Hermes session.

This means: when your friend says "I have Calendar and GitHub set up in my Codex CLI" and they enable Hermes' codex runtime, Hermes activates those automatically. No re-configuration needed.

What's NOT migrated:
- Plugins you haven't installed yet — install them in Codex first.
- Plugins where codex reports `availability != AVAILABLE` (broken install, expired OAuth, removed from marketplace, etc.). These are skipped to avoid writing config that would fail at activation time.
- ChatGPT app marketplace entries (the per-account `app/list` results — these are already enabled inside codex by virtue of your account auth).
- Plugin OAuth — you authorize each plugin once in Codex itself; Hermes doesn't touch credentials.

## Hermes tool callback (the new MCP server)

Codex's built-in toolset covers shell/file ops/patches but doesn't have web search, browser automation, vision, image generation, etc. To keep those usable in a codex turn, Hermes registers itself as an MCP server in `~/.codex/config.toml`:

```toml
[mcp_servers.hermes-tools]
command = "/path/to/python"
args = ["-m", "agent.transports.hermes_tools_mcp_server"]
env = { HERMES_HOME = "/your/.hermes", PYTHONPATH = "...", HERMES_QUIET = "1" }
startup_timeout_sec = 30.0
tool_timeout_sec = 600.0
```

When the model calls `web_search` (or another exposed Hermes tool), codex spawns the `hermes_tools_mcp_server` subprocess via stdio, the request is dispatched through `model_tools.handle_function_call()`, and the result is projected back to codex like any other MCP response.

**Tools available via the callback:** `web_search`, `web_extract`, `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`, `vision_analyze`, `image_generate`, `skill_view`, `skills_list`, `text_to_speech`.

**Tools NOT available:** `delegate_task`, `memory`, `session_search`, `todo`. These need the running AIAgent context to dispatch (mid-loop state) and a stateless MCP callback can't drive them. Use the default Hermes runtime (`/codex-runtime auto`) when you need these.

## Disabling

Switch back at any time:

```
/codex-runtime auto
```

Effective on the next session. The Codex managed block stays in `~/.codex/config.toml` so you can re-enable later without losing config — or remove it manually if you prefer.

## Limitations

This runtime is **opt-in beta**. Working as of Hermes Agent 2026.5 + Codex CLI 0.130.0:

- Multi-turn conversations
- `commandExecution` and `fileChange` (apply_patch) approvals via Hermes UI
- MCP tool calls (verified against `@modelcontextprotocol/server-filesystem` and the new `hermes-tools` callback)
- Native Codex plugin migration (verified against Linear / GitHub / Calendar inventory)
- Deny/cancel paths
- Toggle on/off cycle
- Memory and skill nudge counters (verified live via integration tests)
- Hermes web_search through codex (verified live: "OpenAI Codex CLI – Getting Started" returned end-to-end)

Known limitations:

- **Hermes auth and codex auth are separate sessions.** You need both `codex login` AND `hermes auth login codex` for the cleanest UX (the runtime uses codex's session for the LLM call). This is a deliberate design choice in Hermes' `_import_codex_cli_tokens` — Hermes won't share OAuth state with codex CLI to avoid clobbering each other on token refresh.
- **`delegate_task`, `memory`, `session_search`, `todo` are unavailable on this runtime.** They need the running AIAgent context which a stateless MCP callback can't provide. Use `/codex-runtime auto` when you need these.
- **No inline patch preview in approval prompts when codex doesn't track the changeset.** Codex's `fileChange` approval params don't always carry the changeset. Hermes caches the data from the corresponding `item/started` notification when possible, but if approval arrives before the item has streamed, the prompt falls back to whatever `reason` codex provides.
- **Sub-second cancellation isn't guaranteed.** Mid-stream interrupts (Ctrl+C while codex is responding) are sent via `turn/interrupt`, but if codex has already flushed the final message, you get the response anyway.

If you find a bug, [open an issue](https://github.com/NousResearch/hermes-agent/issues) with the output of `hermes logs --since 5m`. Mention `codex-runtime` in the title so it's easy to triage.

## Architecture

```
                ┌─── Hermes shell (CLI / TUI / gateway) ───┐
                │  sessions DB · slash commands · memory   │
                │  & skill review · cron · session pickers │
                └──┬──────────────────────────────────────┬┘
                   │ user_message               final     │
                   ▼                            text +    │
        ┌──────────────────────────────────┐   projected  │
        │  AIAgent.run_conversation()       │   messages   │
        │   if api_mode == codex_app_server │              │
        │     → CodexAppServerSession       │              │
        │   else: chat_completions / codex_responses (default)
        └────┬─────────────────────────────┘              │
             │ JSON-RPC over stdio                        │
             ▼                                            │
        ┌──────────────────────────────────┐              │
        │  codex app-server (subprocess)    │──────────────┘
        │   thread/start, turn/start        │
        │   item/* notifications            │
        │   shell + apply_patch + update_plan│
        │   view_image + sandbox            │
        │   ┌─────────────────────────┐     │
        │   │  MCP client             │     │
        │   │  ├─ user MCP servers    │     │
        │   │  ├─ native plugins      │     │
        │   │  │   (linear, github,   │     │
        │   │  │    gmail, calendar,  │     │
        │   │  │    canva, ...)       │     │
        │   │  └─ hermes-tools ───────┼─────────────────┐
        │   │       (callback to     │     │           │
        │   │        Hermes' richer  │     │           │
        │   │        tools)          │     │           │
        │   └─────────────────────────┘     │           │
        └──────────────────────────────────┘           │
                                                        │
                                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  hermes_tools_mcp_server.py (subprocess on demand)        │
        │   web_search, web_extract, browser_*, vision_analyze,    │
        │   image_generate, skill_view, skills_list, text_to_speech│
        └──────────────────────────────────────────────────────────┘
```

For implementation details, see [PR #24182](https://github.com/NousResearch/hermes-agent/pull/24182) and the [Codex app-server protocol README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md).
