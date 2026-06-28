---
sidebar_position: 2
title: "Slash Commands Reference"
description: "Complete reference for interactive CLI and messaging slash commands"
---

# Slash Commands Reference

Hermes has two slash-command surfaces, both driven by a central `COMMAND_REGISTRY` in `hermes_cli/commands.py`:

- **Interactive CLI slash commands** — dispatched by `cli.py`, with autocomplete from the registry
- **Messaging slash commands** — dispatched by `gateway/run.py`, with help text and platform menus generated from the registry

Installed skills are also exposed as dynamic slash commands on both surfaces. That includes bundled skills like `/plan`, which opens plan mode and saves markdown plans under `.hermes/plans/` relative to the active workspace/backend working directory.

## Permissions and admin/user split

Every messaging platform that supports a per-user allowlist (Telegram, Discord, Slack, Matrix, Mattermost, Signal, …) also supports a two-tier slash command split: **admins** get every registered command, **regular users** only get the names you list in `user_allowed_commands` (plus the always-allowed floor `/help` and `/whoami`). Configure `allow_admin_from` and `user_allowed_commands` (and the per-group equivalents `group_allow_admin_from` / `group_user_allowed_commands`) inside the platform's `extra:` block in `~/.hermes/gateway-config.yaml`.

See the per-platform docs for examples — the structure is identical across platforms:

- [Telegram](../user-guide/messaging/telegram.md#slash-command-access-control)
- [Discord](../user-guide/messaging/discord.md)
- [Slack](../user-guide/messaging/slack.md)
- [Matrix](../user-guide/messaging/matrix.md)
- [Mattermost](../user-guide/messaging/mattermost.md)
- [Signal](../user-guide/messaging/signal.md)

If `allow_admin_from` is unset for a scope, that scope stays in unrestricted backward-compat mode — every allowed user can run every command.

## Interactive CLI slash commands

Type `/` in the CLI to open the autocomplete menu. Built-in commands are case-insensitive.

### Session

| Command | Description |
|---------|-------------|
| `/new [name]` (alias: `/reset`) | Start a new session (fresh session ID + history). Optional `[name]` sets the initial session title — e.g. `/new my-experiment` opens a fresh session already titled `my-experiment` so it's easy to find later with `/resume` or `/sessions`. Append `now`, `--yes`, or `-y` to skip the confirmation modal — e.g. `/reset now`, `/new --yes my-experiment`. |
| `/clear` | Clear screen and start a new session |
| `/history` | Show conversation history |
| `/save` | Save the current conversation |
| `/retry` | Retry the last message (resend to agent) |
| `/undo` | Remove the last user/assistant exchange |
| `/title` | Set a title for the current session (usage: /title My Session Name) |
| `/compress [here [N] \| focus topic]` | Manually compress conversation context (flush memories + summarize). `/compress here [N]` summarizes everything except the most recent N exchanges (default 2), kept verbatim — pick your own compression boundary. A focus topic narrows what a full summary preserves. |
| `/rollback` | List or restore filesystem checkpoints (usage: /rollback [number]) |
| `/snapshot [create\|restore <id>\|prune]` (alias: `/snap`) | Create or restore state snapshots of Hermes config/state. `create [label]` saves a snapshot, `restore <id>` reverts to it, `prune [N]` removes old snapshots, or list all with no args. |
| `/stop` | Kill all running background processes |
| `/queue <prompt>` (alias: `/q`) | Queue a prompt for the next turn (doesn't interrupt the current agent response). |
| `/steer <prompt>` | Inject a mid-run note that arrives at the agent **after the next tool call** — no interrupt, no new user turn. The text is appended to the last tool result's content once the current tool completes, giving the agent new context without breaking the current tool-calling loop. Use this to nudge direction mid-task (e.g. "focus on the auth module" while the agent is running tests). |
| `/goal <text>` | Set a standing goal Hermes works toward across turns — our take on the Ralph loop. After each turn an auxiliary judge model decides whether the goal is done; if not, Hermes auto-continues. Subcommands: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. Budget defaults to 20 turns (`goals.max_turns`); any real user message preempts the continuation loop, and state survives `/resume`. See [Persistent Goals](/user-guide/features/goals) for the full walkthrough. |
| `/subgoal <text>` | Append a user-supplied criterion to the active goal mid-loop. The continuation prompt surfaces all subgoals to the agent verbatim, and the judge factors them into its DONE/CONTINUE verdict — so the goal isn't marked done until the original goal **and** every subgoal are met. Subcommands: `/subgoal` (list), `/subgoal remove <N>`, `/subgoal clear`. Requires an active `/goal`. |
| `/resume [name]` | Resume a previously-named session |
| `/sessions` (TUI alias: `/switch`) | Classic CLI: browse and resume previous sessions in an interactive picker. TUI: open the live session switcher for currently open TUI sessions. Use `/sessions new` in the TUI to start another live session immediately. |
| `/redraw` | Force a full UI repaint (recovers from terminal drift after tmux resize, mouse selection artifacts, etc.) |
| `/status` | Show session info — model, provider, profile, session ID, working directory, title, created/updated timestamps, token totals, agent-running state — followed by a local **Session recap** block (recent user/assistant turn counts, tool result count, top tools used, last few files touched, the latest user prompt, and the latest assistant reply). The recap is computed locally from the in-memory conversation; no LLM call, no prompt-cache impact. |
| `/agents` (alias: `/tasks`) | Show active agents and running tasks across the current session. |
| `/background <prompt>` (alias: `/bg`, `/btw`) | Run a prompt in a separate background session. The agent processes your prompt independently — your current session stays free for other work. Results appear as a panel when the task finishes. See [CLI Background Sessions](/user-guide/cli#background-sessions). |
| `/branch [name]` (alias: `/fork`) | Branch the current session (explore a different path) |
| `/handoff <platform>` | **CLI only.** Hand the current session off to a messaging platform (Telegram, Discord, Slack, WhatsApp, Signal, Matrix). The gateway picks it up immediately, creates a fresh thread on platforms that support threads (Telegram topics, Discord text-channel threads, Slack message-anchored threads), re-binds the destination to your CLI session_id so the full role-aware transcript replays, and forges a synthetic user turn so the agent confirms it's working in the new place. Your CLI exits cleanly on success with a `/resume` hint; resume locally any time with `/resume <title>`. Refused mid-turn. Requires the gateway to be running and a home channel configured for the target platform (`/sethome` from the destination chat). See [Cross-Platform Handoff](/user-guide/sessions#cross-platform-handoff). |

### Configuration

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/model [model-name]` | Show or change the current model. Supports: `/model claude-sonnet-4`, `/model provider:model` (switch providers), `/model custom:model` (custom endpoint), `/model custom:name:model` (named custom provider), `/model custom` (auto-detect from endpoint), and user-defined aliases (`/model fav`, `/model grok` — see [Custom model aliases](#custom-model-aliases)). Use `--global` to persist the change to config.yaml. **Note:** `/model` can only switch between already-configured providers. To add a new provider, exit the session and run `hermes model` from your terminal. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Toggle the optional [Codex app-server runtime](../user-guide/features/codex-app-server-runtime) for OpenAI/Codex models. `auto` (default) uses Hermes' standard chat completions; `codex_app_server` hands turns to a `codex app-server` subprocess for native shell, apply_patch, ChatGPT subscription auth, and migrated Codex plugins. Effective on next session. |
| `/personality` | Set a predefined personality |
| `/verbose` | Cycle tool progress display: off → new → all → verbose. Can be [enabled for messaging](#notes) via config. |
| `/fast [normal\|fast\|status]` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode. Options: `normal`, `fast`, `status`. |
| `/reasoning` | Manage reasoning effort and display (usage: /reasoning [level\|show\|hide]) |
| `/skin` | Show or change the display skin/theme |
| `/statusbar` (alias: `/sb`) | Toggle the context/model status bar on or off |
| `/voice [on\|off\|tts\|status]` | Toggle CLI voice mode and spoken playback. Recording uses `voice.record_key` (default: `Ctrl+B`). |
| `/yolo` | Toggle YOLO mode — skip all dangerous command approval prompts. |
| `/footer [on\|off\|status]` | Toggle the gateway runtime-metadata footer on final replies (shows model, context %, and cwd). |
| `/busy [queue\|steer\|interrupt\|status]` | CLI-only: control what pressing Enter does while Hermes is working — queue the new message, steer mid-turn, or interrupt immediately. |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | CLI-only: pick the TUI busy-indicator style. |

### Tools & Skills

| Command | Description |
|---------|-------------|
| `/tools [list\|disable\|enable] [name...]` | Manage tools: list available tools, or disable/enable specific tools for the current session. Disabling a tool removes it from the agent's toolset and triggers a session reset. |
| `/toolsets` | List available toolsets |
| `/browser [connect\|disconnect\|status]` | Manage a local Chromium-family CDP connection. `connect` attaches browser tools to a running Chrome, Brave, Chromium, or Edge instance (default: `http://127.0.0.1:9222`). `disconnect` detaches. `status` shows current connection. Auto-launches a supported Chromium-family browser if no debugger is detected. |
| `/skills` | Search, install, inspect, or manage skills from online registries. Also the review surface for the skill write-approval gate: `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>`, `/skills approval on\|off`. See [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval). |
| `/memory [pending\|approve\|reject\|approval]` | Review pending memory writes staged by the write-approval gate (`memory.write_approval`) and toggle the gate. See [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/bundles` | List configured skill bundles — `/<name>` slash aliases that preload several skills at once. Configure under `bundles:` in `~/.hermes/config.yaml`. See [Skill Bundles](/user-guide/features/skills#skill-bundles). |
| `/learn <what to learn from>` | Distill a reusable skill from anything you describe — a directory, a URL, the workflow you just walked the agent through, or pasted notes. Open-ended: the agent gathers the sources with its own tools and authors a `SKILL.md` following the house authoring standards. Works in the CLI, the messaging gateway, the TUI, and the dashboard Skills page. |
| `/cron` | Manage scheduled tasks (list, add/create, edit, pause, resume, run, remove) |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` (alias: `/suggest`) | Review suggested automations. Use `/suggestions` to list pending suggestions, `/suggestions accept <id>` to create the proposed automation, `/suggestions dismiss <id>` to reject one, `/suggestions catalog` to add curated starter automations, and `/suggestions clear` to clear resolved suggestion records. Accepted jobs preserve the current surface as the delivery origin. |
| `/blueprint [name] [slot=value ...]` (alias: `/bp`) | Set up an automation from a blueprint template. Bare `/blueprint` lists the catalog; `/blueprint <name>` starts a guided slot-filling flow on the next agent turn; `/blueprint <name> slot=value ...` creates the job directly. |
| `/curator` | Background skill maintenance — `status`, `run`, `pin`, `archive`. See [Curator](/user-guide/features/curator). |
| `/kanban <action>` | Drive the multi-profile, multi-project collaboration board without leaving chat. Full `hermes kanban` surface is available: `/kanban list`, `/kanban show t_abc`, `/kanban create "title" --assignee X`, `/kanban comment t_abc "text"`, `/kanban unblock t_abc`, `/kanban dispatch`, etc. Multi-board support included: `/kanban boards list`, `/kanban boards create <slug>`, `/kanban boards switch <slug>`, `/kanban --board <slug> <action>`. See [Kanban slash command](/user-guide/features/kanban#kanban-slash-command). |
| `/reload-mcp` (alias: `/reload_mcp`) | Reload MCP servers from config.yaml |
| `/reload-skills` (alias: `/reload_skills`) | Re-scan `~/.hermes/skills/` for newly installed or removed skills |
| `/reload` | Reload `.env` variables into the running session (picks up new API keys without restarting) |
| `/plugins` | List installed plugins and their status |

### Info

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/version` | Show Hermes Agent version, build, and environment info. |
| `/usage` | Show token usage, cost breakdown, session duration, and — when available from the active provider — an **Account limits** section with remaining quota / credits / plan usage pulled live from the provider's API. |
| `/credits` | Show your Nous credit balance and a top-up handoff link. |
| `/billing` | CLI terminal-billing flow for Nous — view balance, buy credits, and manage auto-reload / monthly limits. |
| `/insights` | Show usage insights and analytics (last 30 days) |
| `/platforms` (alias: `/gateway`) | Show gateway/messaging platform status (CLI-only summary view). |
| `/paste` | Attach a clipboard image |
| `/copy [number]` | Copy the last assistant response to clipboard (or the Nth-from-last with a number). CLI-only. |
| `/image <path>` | Attach a local image file for your next prompt. |
| `/debug` | Upload debug report (system info + logs) and get shareable links. Also available in messaging. |
| `/profile` | Show active profile name and home directory |

### Exit

| Command | Description |
|---------|-------------|
| `/quit` | Exit the CLI (also: `/exit`). |

### Dynamic CLI slash commands

| Command | Description |
|---------|-------------|
| `/<skill-name>` | Load any installed skill as an on-demand command. Example: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | Search, browse, inspect, install, audit, publish, and configure skills from registries and the official optional-skills catalog. |

### Quick Commands

User-defined quick commands map a short slash command to either a shell command or another slash command. Configure them in `~/.hermes/config.yaml`:

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

Then type `/status`, `/deploy`, or `/inbox` in the CLI or a messaging platform. Quick commands are resolved at dispatch time and may not appear in every built-in autocomplete/help table.

String-only prompt shortcuts are not supported as quick commands. Put longer reusable prompts in a skill, or use `type: alias` to point at an existing slash command.

### Custom model aliases

Define your own short names for models you use often, then reach them with `/model <alias>` in the CLI or any messaging platform. Aliases work identically in both, on session-only (default) and `--global` switches.

Two config formats are supported:

**Full form** — pin an exact model, provider, and optionally a base URL. Put this in `~/.hermes/config.yaml`:

```yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
  ollama-qwen:
    model: qwen3-coder:30b
    provider: custom
    base_url: http://localhost:11434/v1
```

**Short form** — `provider/model` in one string. Set from the shell without editing YAML:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

Then in chat:

```
/model fav            # session-only
/model grok --global  # also persists current-model change to config.yaml
```

User aliases take precedence over built-in short names, so naming an alias `sonnet`, `kimi`, `opus`, etc. will shadow the built-in. Alias names are case-insensitive.

### Alias Resolution

Commands support prefix matching: typing `/h` resolves to `/help`, `/mod` resolves to `/model`. When a prefix is ambiguous (matches multiple commands), the first match in registry order wins. Full command names and registered aliases always take priority over prefix matches.

## Messaging slash commands

The messaging gateway supports the following built-in commands inside Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, and Teams chats:

| Command | Description |
|---------|-------------|
| `/start` | Platform-protocol command. Many chat platforms (Telegram, Discord, …) send `/start` automatically the first time a user opens a bot conversation. Hermes acknowledges the ping silently — no agent reply, no session burn — so first-contact handshakes don't waste a turn. You can also send it explicitly to confirm the gateway is reachable. |
| `/new` | Start a new conversation. |
| `/reset` | Reset conversation history. |
| `/status` | Show session info, followed by a local **Session recap** block (recent turn counts, top tools used, files touched, latest prompt + reply). |
| `/stop` | Kill all running background processes and interrupt the running agent. |
| `/model [provider:model]` | Show or change the model. Supports provider switches (`/model zai:glm-5`), custom endpoints (`/model custom:model`), named custom providers (`/model custom:local:qwen`), auto-detect (`/model custom`), and user-defined aliases (`/model fav`, `/model grok` — see [Custom model aliases](#custom-model-aliases)). Use `--global` to persist the change to config.yaml. **Note:** `/model` can only switch between already-configured providers. To add a new provider or set up API keys, use `hermes model` from your terminal (outside the chat session). |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | Toggle the optional [Codex app-server runtime](../user-guide/features/codex-app-server-runtime). Persists to `model.openai_runtime` in config.yaml and evicts the cached agent so the next message picks up the new runtime. Effective on next session. |
| `/personality [name]` | Set a personality overlay for the session. |
| `/fast [normal\|fast\|status]` | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode. |
| `/retry` | Retry the last message. |
| `/undo` | Remove the last exchange. |
| `/sethome` (alias: `/set-home`) | Mark the current chat as the platform home channel for deliveries. |
| `/compress [here [N] \| focus topic]` | Manually compress conversation context. `/compress here [N]` keeps the most recent N exchanges (default 2) verbatim and summarizes the rest. A focus topic narrows what a full summary preserves. |
| `/topic [off\|help\|session-id]` | **Telegram DM only.** Manage user-managed multi-session topic mode. `/topic` enables it or shows status; `/topic off` disables it and clears bindings; `/topic help` shows usage; `/topic <session-id>` inside a topic restores a previous session. See [Multi-session DM mode](/user-guide/messaging/telegram#multi-session-dm-mode-topic). |
| `/title [name]` | Set or show the session title. |
| `/resume [name]` | Resume a previously named session. |
| `/usage` | Show token usage, estimated cost breakdown (input/output), context window state, session duration, and — when available from the active provider — an **Account limits** section with remaining quota / credits pulled live from the provider's API. |
| `/credits` | Show your Nous credit balance and a top-up link that opens the portal billing page in a browser. |
| `/insights [days]` | Show usage analytics. |
| `/reasoning [level\|show\|hide]` | Change reasoning effort or toggle reasoning display. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | Control spoken replies in chat. `join`/`channel`/`leave` manage Discord voice-channel mode. |
| `/rollback [number]` | List or restore filesystem checkpoints. |
| `/background <prompt>` | Run a prompt in a separate background session. Results are delivered back to the same chat when the task finishes. See [Messaging Background Sessions](/user-guide/messaging/#background-sessions). |
| `/queue <prompt>` (alias: `/q`) | Queue a prompt for the next turn without interrupting the current one. |
| `/steer <prompt>` | Inject a message after the next tool call without interrupting — the model picks it up on its next iteration rather than as a new turn. |
| `/goal <text>` | Set a standing goal Hermes works toward across turns — our take on the Ralph loop. A judge model checks after each turn; if not done, Hermes auto-continues until it is, you pause/clear it, or the turn budget (default 20) is hit. Subcommands: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. Safe to run mid-agent for status/pause/clear; setting a new goal requires `/stop` first. See [Persistent Goals](/user-guide/features/goals). |
| `/footer [on\|off\|status]` | Toggle the runtime-metadata footer on final replies (shows model, context %, and cwd). |
| `/curator [status\|run\|pin\|archive]` | Background skill maintenance controls. |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` | Review suggested automations right in chat. `/suggestions` lists pending suggestions, `catalog` adds curated starter automations, and `clear` prunes resolved suggestion records. Accepted suggestions keep this chat/thread as the job delivery origin. |
| `/blueprint [name] [slot=value ...]` | Browse cron blueprints, start a guided slot-filling conversation, or create a blueprint job directly. Directly created jobs deliver back to the current chat/thread. |
| `/memory [pending\|approve\|reject\|approval]` | Review pending memory writes staged by the write-approval gate (`memory.write_approval`) — approve or reject them right in chat — and toggle the gate with `/memory approval on\|off`. See [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval). |
| `/skills [pending\|approve\|reject\|diff\|approval]` | Review pending **skill** writes staged by the write-approval gate (`skills.write_approval`). Shows a one-line gist per staged write; `/skills diff <id>` is truncated for chat — read the full diff on the CLI or in `~/.hermes/pending/skills/<id>.json`. Only appears when the gate is on (or staged writes remain); search/install stay CLI-only. |
| `/kanban <action>` | Drive the multi-profile, multi-project collaboration board from chat — identical argument surface to the CLI. Bypasses the running-agent guard, so `/kanban unblock t_abc`, `/kanban comment t_abc "…"`, `/kanban list --mine`, `/kanban boards switch <slug>`, etc. work mid-turn. `/kanban create …` auto-subscribes the originating chat to the new task's terminal events. See [Kanban slash command](/user-guide/features/kanban#kanban-slash-command). |
| `/platform <list\|pause\|resume> [name]` | Operate a running gateway platform right from chat. `/platform list` shows every adapter and its state (running, paused-by-breaker, manually-paused); `/platform pause <name>` stops dispatching new messages to that adapter without unloading it; `/platform resume <name>` re-enables it and clears a tripped circuit breaker once the upstream is healthy. |
| `/reload-mcp` (alias: `/reload_mcp`) | Reload MCP servers from config. |
| `/yolo` | Toggle YOLO mode — skip all dangerous command approval prompts. |
| `/commands [page]` | Browse all commands and skills (paginated). |
| `/approve [session\|always]` | Approve and execute a pending dangerous command. `session` approves for this session only; `always` adds to permanent allowlist. |
| `/deny` | Reject a pending dangerous command. |
| `/update` | Update Hermes Agent to the latest version. |
| `/restart` | Gracefully restart the gateway after draining active runs. When the gateway comes back online, it sends a confirmation to the requester's chat/thread. |
| `/debug` | Upload debug report (system info + logs) and get shareable links. |
| `/help` | Show messaging help. |
| `/<skill-name>` | Invoke any installed skill by name. |

## Notes

- `/skin`, `/snapshot`, `/reload`, `/tools`, `/toolsets`, `/browser`, `/config`, `/cron`, `/platforms`, `/paste`, `/image`, `/statusbar`, `/plugins`, `/busy`, `/indicator`, `/redraw`, `/clear`, `/history`, `/save`, `/copy`, `/handoff`, `/billing`, and `/quit` are **CLI-only** commands.
- `/skills` is **CLI-only for search/browse/install**; its write-approval review subcommands (`pending`, `approve`, `reject`, `diff`, `approval`) also work on messaging platforms when `skills.write_approval` is on. `/memory` works on **both** surfaces.
- `/verbose` is **CLI-only by default**, but can be enabled for messaging platforms by setting `display.tool_progress_command: true` in `config.yaml`. When enabled, it cycles the `display.tool_progress` mode and saves to config.
- `/sethome`, `/update`, `/restart`, `/approve`, `/deny`, `/topic`, `/platform`, and `/commands` are **messaging-only** commands.
- `/status`, `/version`, `/background`, `/queue`, `/steer`, `/voice`, `/reload-mcp`, `/reload-skills`, `/rollback`, `/debug`, `/fast`, `/footer`, `/curator`, `/kanban`, `/credits`, `/suggestions`, `/blueprint`, `/learn`, `/sessions`, and `/yolo` work in **both** the CLI and the messaging gateway.
- `/voice join`, `/voice channel`, and `/voice leave` are only meaningful on Discord.
- In the TUI, `/sessions` shows live sessions in the current TUI process. Use `/resume [name]` or `hermes --tui --resume <id-or-title>` for saved or closed transcripts.

## Confirmation prompts for destructive commands

The CLI prompts before running slash commands that throw away unsaved session state. The current destructive set is:

| Command | What it destroys |
|---------|------------------|
| `/clear` | Clears the screen and starts a fresh session — current session ID and in-memory history are gone. |
| `/new` / `/reset` | Starts a fresh session (new session ID + empty history). |
| `/undo` | Removes the last user/assistant exchange from history. |
| `/exit --delete` / `/quit --delete` | Exits **and** permanently deletes the current session's SQLite history and on-disk transcripts. |

For each of these the CLI opens a three-choice modal: **Approve Once** (proceed this time), **Always Approve** (proceed and persist `approvals.destructive_slash_confirm: false` so future destructive commands run without prompting), or **Cancel**.

**Inline skip:** append `now`, `--yes`, or `-y` to bypass the modal for a single invocation — e.g. `/reset now`, `/new --yes my-session`, `/clear -y`, `/undo -y`. Useful when the modal doesn't render correctly on your terminal (see [issue #30768](https://github.com/NousResearch/hermes-agent/issues/30768) for native Windows PowerShell) or when scripting against the CLI.

Set `approvals.destructive_slash_confirm: false` in `~/.hermes/config.yaml` to disable the prompts globally; set it back to `true` to re-enable. See [Security — Destructive slash command confirmation](../user-guide/security.md#dangerous-command-approval) for context.
