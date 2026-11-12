---
title: "Antigravity Cli — Operate the Antigravity CLI (agy): plugins, auth, sandbox"
sidebar_label: "Antigravity Cli"
description: "Operate the Antigravity CLI (agy): plugins, auth, sandbox"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Antigravity Cli

Operate the Antigravity CLI (agy): plugins, auth, sandbox.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/antigravity-cli` |
| Path | `optional-skills/autonomous-ai-agents/antigravity-cli` |
| Version | `0.3.0` |
| Author | Tony Simons (asimons81), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Antigravity`, `CLI`, `Auth`, `Plugins`, `Sandbox` |
| Related skills | [`grok`](../../optional/autonomous-ai-agents/autonomous-ai-agents-grok.md), [`codex`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-codex.md), [`claude-code`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code.md), [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Antigravity CLI (`agy`)

Operator guide for the Antigravity CLI, invoked as `agy`. Run all `agy`
commands through the Hermes `terminal` tool and inspect its config and logs with
`read_file`. This skill is reference + procedure — it does not wrap a network
API, so there is nothing to authenticate from Hermes itself.

## When to Use

- Installing, updating, or smoke-testing the `agy` binary
- Driving non-interactive `agy --print` / `agy -p` one-shots
- Debugging Antigravity auth, sandbox, permissions, or plugin state
- Reading Antigravity settings, keybindings, conversations, or logs

## Mental model

Antigravity has two layers — keep them distinct or the guidance will be wrong:

1. **Shell wrapper commands** — `agy help`, `agy install`, `agy models`,
   `agy plugin`, `agy update`, `agy changelog`, `agy mcp`, `agy agent`. Run
   these through the `terminal` tool.
2. **Interactive in-session slash commands** — `/config`, `/permissions`,
   `/skills`, `/agents`, etc. These only exist inside a running `agy` TUI
   session, not on the shell wrapper.

`agy help` shows the shell wrapper surface, NOT the in-session slash commands.

## Prerequisites

- The `agy` binary on PATH. Verify through the `terminal` tool:
  `command -v agy && agy --version`.
- No env vars or API keys required by this skill — Antigravity manages its own
  auth via the OS keyring / browser sign-in (see Authentication below).

## How to Run

Invoke every `agy` command through the `terminal` tool. Examples:

```
terminal(command="agy --version")
terminal(command="agy help")
terminal(command="agy models")
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets' --print-timeout 2m", workdir="/path/to/project", timeout=150)
```

For an interactive multi-turn TUI session, launch `agy` with `pty=true` (and
tmux for capture/monitoring), the same pattern the `codex` / `claude-code`
skills use. For one-shot smoke tests and scripted prompts, prefer
`agy --print` (non-interactive).

To inspect Antigravity's own files, use `read_file` on the paths under Core
paths below — do not `cat` them through the terminal, and never print secrets.

## Delegation patterns

`agy` is a coding-agent backend in the same family as `codex` / `claude-code`,
so the same delegation shapes apply. Use these when handing real work (features,
fixes, reviews, second opinions) to Antigravity rather than just smoke-testing.

### One-shot (preferred for scripted prompts and second opinions)

```
terminal(command="agy -p 'Review this diff for bugs and security issues' --model gemini-3.8-flash-low --output-format json --print-timeout 5m", workdir="/path/to/repo", timeout=330)
```

`-p` is non-interactive: it runs the prompt and exits. Pick the engine with
`--model <id>` (run `agy models`, which prints `<id>\t<display name>`, e.g.
`gemini-4-argon-low`, `gemini-4-hydrogen-high`, `gemini-3.8-flash-low`,
`claude-opus-5-5-medium`, `claude-sonnet-5-5-low`; do not hardcode a full list).
Set reasoning effort with `--effort low|medium|high|xhigh|max` and execution
mode with `--mode accept-edits|auto|plan`. Add extra context roots with
repeatable `--add-dir`.

### Long / bounded runs (tests, builds, multi-file changes)

Background it and get notified on completion, the same as the `codex` skill:

```
terminal(command="agy -p 'Implement the change described in TASK.md and run the tests' --dangerously-skip-permissions --print-timeout 20m", workdir="/path/to/repo", background=true, notify_on_complete=true)
# then: process(action="poll"/"log"/"wait", session_id=<id>)
```

### Interactive multi-turn (PTY + tmux)

For a conversational session, launch `agy -i` (or bare `agy`) under `pty=true`
with tmux for `capture-pane` / `send-keys`, exactly the pattern documented in
the `codex` / `claude-code` skills. Resume later with `--continue` / `-c` or a
specific `--conversation <id>`.

### Parallel instances (batch sub-issue / worktree fan-out)

Create one git worktree per task and launch an independent `agy -p` in each
(background), then collect results — same worktree fan-out the `codex` skill
uses for batch issue fixing. Bound concurrency to what the machine and your
review capacity can absorb.

### Output formats and timeout bounding

- `--output-format text|json|stream-json` (default `text`). With
  `--output-format json`, `agy -p` returns a result envelope:
  `{"conversation_id":"...","status":"SUCCESS","response":"...","duration_seconds":1.86,"num_turns":1,"usage":{"input_tokens":...,"output_tokens":...,"thinking_tokens":...,"cache_read_tokens":...,"total_tokens":...}}`.
- `--input-format text|stream-json` (default `text`): `stream-json` reads one
  NDJSON message per line from stdin and runs a turn for each (requires
  `--output-format stream-json`).
- `--json-schema <string|path>` enforces structured output against a JSON schema
  string or schema file path (for `stream-json`, applies to the final result).
- `--print-timeout` defaults to `0s` (wait until the turn completes, no limit).
  **Always pass `--print-timeout` on scripted runs** (e.g. `--print-timeout 5m`)
  and pair it with the `terminal` tool's `timeout=` so neither layer hangs
  indefinitely.

### Orchestration boundary

Antigravity is a **worker execution backend or third-opinion reviewer** — an
execution detail owned by the agent/profile running a task, NOT a first-class
orchestration primitive. Do not put `agy` on a kanban board as its own card or
treat it as a coordination layer; route work through the normal task graph and
let the assigned worker choose `agy` (vs. codex/claude-code/direct tools) as its
method. Reach for it explicitly only when the user asks, when a worker is
configured to wrap it, or when you want a Gemini-family cross-check against
another agent's plan or diff.

## Core paths

- Binary / entrypoint: `agy`
- App data dir: `~/.gemini/antigravity-cli/`
- Settings file: `~/.gemini/antigravity-cli/settings.json`
- Keybindings file: `~/.gemini/antigravity-cli/keybindings.json`
- Logs: `~/.gemini/antigravity-cli/log/cli-*.log`
- Conversations: `~/.gemini/antigravity-cli/conversations/`
- Brain artifacts: `~/.gemini/antigravity-cli/brain/`
- History: `~/.gemini/antigravity-cli/history.jsonl`
- Plugin staging: `~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## Quick Reference

### Wrapper commands
- `agy acp`
- `agy agent` / `agy agents`
- `agy changelog`
- `agy help`
- `agy install`
- `agy mcp`
- `agy mic-serve`
- `agy models`
- `agy plugin` / `agy plugins`
- `agy remote-control`
- `agy update`

### Useful flags
- `--add-dir`
- `--agent`
- `--continue` / `-c`
- `--conversation`
- `--dangerously-skip-permissions`
- `--disable-slash-commands`
- `--effort low|medium|high|xhigh|max`
- `--input-format text|stream-json`
- `--json-schema <string|path>`
- `--log-file`
- `--mode accept-edits|auto|plan`
- `--model <id>`
- `--new-project`
- `--output-format text|json|stream-json`
- `--print` / `-p` / `--prompt`
- `--print-timeout` (default `0s` — always set explicitly on scripted runs)
- `--project <id|name>`
- `--prompt-interactive` / `-i`
- `--remote-agent`
- `--remote-control`
- `--sandbox`
- `--settings <json|path>`
- `--version`

### Plugin subcommands (`agy plugin --help`)
- `list [--json]`, `import [source]`, `install <target>`, `uninstall <name>`,
  `enable <name>`, `disable <name>`, `validate [path] [--strict] [--json]`,
  `link <mp> <target>`, `help`

### Install flags (`agy install --help`)
- `--dir`, `--skip-aliases`, `--skip-path`

### In-session slash commands
- **Conversation control:** `/resume` (`/switch`), `/rewind` (`/undo`),
  `/rename <name>`, `/clear`, `/fork`, `/reset`, `/new`
- **Settings & tools:** `/config`, `/settings`, `/permissions`, `/model`,
  `/keybindings`, `/statusline`, `/tasks`, `/skills`, `/mcp`, `/open <path>`,
  `/usage`, `/logout`, `/agents`
- **Prompt helpers:** `@` path autocomplete, `esc esc` clears the prompt (when
  not streaming), `!` runs a terminal command directly, `?` opens help

## Settings and permissions

### Top-level and common keys (`settings.json`)
- Top-level keys seen in practice: `model`, `permissions`, `trustedWorkspaces`
- Additional settings keys: `allowNonWorkspaceAccess`, `colorScheme`,
  `permissions.allow`, `enableTerminalSandbox`

### Permission & execution modes
- Session `--mode`: `accept-edits`, `auto`, `plan`.
- Permission modes: `request-review`, `always-proceed`, `strict`,
  `proceed-in-sandbox`.

### Sandbox behavior
- `enableTerminalSandbox` is a boolean in `settings.json`; default `false`.
- Launch-time overrides (`--sandbox`, `--dangerously-skip-permissions`,
  `--settings`) can supersede persistent settings for the current session.

## Authentication behavior

- The CLI tries the OS secure keyring first.
- With no saved session, it falls back to browser-based Google sign-in.
- Locally it opens the default browser; over SSH it prints an authorization URL
  and expects the auth code pasted back.
- `/logout` removes saved credentials.

## Plugins

- Plugins stage under `~/.gemini/antigravity-cli/plugins/<plugin_name>/`.
- They can bundle skills, agents, rules, MCP servers, and hooks.
- `agy plugin list` prints installed plugins with status, version, source, and
  components, e.g.:
  ```
  Installed plugins:
  - superpowers (enabled, 7.0.2)
      Source:     antigravity
      Components: 15 skills
  ```
- Returning no imported/installed plugins is also a valid empty state.

## Pitfalls

- `agy help` shows wrapper commands, not interactive slash commands.
- `agy --version` is the safe non-interactive version check; `agy version` is
  interactive and can fail without a real TTY.
- `--print-timeout` defaults to `0s` (no limit, waits until the turn completes).
  Always pass `--print-timeout` on scripted `agy -p` runs and pair it with the
  `terminal` tool's `timeout`. There is still no `--max-turns`.
- `--model` takes the model ID from the first column of `agy models` (e.g.
  `gemini-3.8-flash-low`), not the human display name. Run `agy models` rather
  than hardcoding model lists.
- Logs under `~/.gemini/antigravity-cli/log/` may contain `"You are not logged
  into Antigravity"` errors left over from sessions started with a fresh or
  isolated config dir (e.g. automated or sandboxed runs with their own config
  dir); confirm the current run actually failed before treating log entries as
  an active auth problem.
- Don't confuse persistent JSON settings with launch-time overrides (`--mode`,
  `--sandbox`, `--dangerously-skip-permissions`, `--settings`).
- `~/.gemini/antigravity-cli/bin/agentapi` is a thin wrapper to `agy agentapi`.
- On WSL, token storage is file-based, so auth issues are usually local-file /
  session-state problems, not browser-only problems.
- Workspace identity can depend on launch directory and the `.antigravitycli`
  project marker.

## Verification

Confirm the install is real and usable, all through the `terminal` tool (read
files with `read_file`):

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy models")`
5. `terminal(command="agy plugin list")`
6. `read_file` on `~/.gemini/antigravity-cli/settings.json`
7. `read_file` on the latest `~/.gemini/antigravity-cli/log/cli-*.log`
8. If needed, `read_file` on `~/.gemini/antigravity-cli/keybindings.json`

## Support files

- `references/cli-docs.md` — condensed notes from the getting-started, usage,
  and features docs.
