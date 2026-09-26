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
| Version | `0.2.0` |
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
commands through the Hermes `terminal` tool; inspect its config and logs with
`read_file`. This skill is reference + procedure — it does not wrap a network
API, so there is nothing to authenticate from Hermes itself.

## When to Use

- Installing, updating, or smoke-testing the `agy` binary
- Driving non-interactive `agy --print` / `agy -p` one-shots
- Debugging Antigravity auth, sandbox, permissions, or plugin state
- Reading Antigravity settings, keybindings, conversations, or logs

## Mental model

Antigravity has two layers — keep them distinct or the guidance will be wrong:

1. **Shell wrapper commands** — `agy help`, `agy install`, `agy plugin`,
   `agy update`, `agy changelog`. Run these through the `terminal` tool.
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
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets'", workdir="/path/to/project")
```

For an interactive multi-turn TUI session, launch `agy` with `pty=true` (and
tmux for capture/monitoring), the same pattern the `codex` / `claude-code`
skills use. For one-shot smoke tests and scripted prompts, prefer
`agy --print` (non-interactive).

To inspect Antigravity's own files, use `read_file` on the paths under Core
paths below — do not `cat` them through the terminal.

## Delegation patterns

`agy` is a coding-agent backend in the same family as `codex` / `claude-code`,
so the same delegation shapes apply. Use these when handing real work (features,
fixes, reviews, second opinions) to Antigravity rather than just smoke-testing.

### One-shot (preferred for scripted prompts and second opinions)

```
terminal(command="agy --model 'Gemini 3.1 Pro (High)' -p 'Review this diff for bugs and security issues'", workdir="/path/to/repo", timeout=300)
```

`-p` is non-interactive: it runs the prompt and exits. Pick the engine with
`--model` (run `agy models` for the exact display strings, e.g.
`'Gemini 3.1 Pro (High)'`, `'Claude Opus 4.6 (Thinking)'`). Add extra context
roots with repeatable `--add-dir`.

### Long / bounded runs (tests, builds, multi-file changes)

Background it and get notified on completion, the same as the `codex` skill:

```
terminal(command="agy --dangerously-skip-permissions -p 'Implement the change described in TASK.md and run the tests'", workdir="/path/to/repo", background=true, notify_on_complete=true)
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

### Output + bounding caveats

- `agy -p` supports structured output: `--output-format json` prints one JSON
  envelope on stdout (`conversation_id`, `status`, `response`, `num_turns`,
  `duration_seconds`, `usage`), and `--json-schema <file>` adds a validated
  `structured_output` object. The default (`text`) is plain stdout — parse it
  directly only when you didn't request JSON.
- The prompt is `-p`'s **value** (`-p "…"`) and flags may follow it; a bare
  positional prompt is rejected (`unexpected argument … would have been
  ignored`) and `-p` with no value fails with `flag needs an argument`. Piping
  the prompt into stdin works only when `-p` is omitted.
- There is **no `--max-turns`**. A print run is bounded by **`--print-timeout`**,
  whose default is `0` — wait until the turn completes, i.e. **no timeout** — so
  set it explicitly for unattended runs. The value needs a unit
  (`--print-timeout 20m`); a bare number is rejected and `agy` prints its help
  text instead of running. Pair with the `terminal` `timeout=` so the outer call
  doesn't cut the run short.
- A print run that writes files needs an explicit target: an absolute path is
  honoured, but a bare relative filename can land in the app's artefact scratch
  dir (`~/.gemini/antigravity-cli/scratch/`) instead of the launch directory.
  Without `--add-dir <dir>`, an allow-rule or `--dangerously-skip-permissions`,
  a headless write is **auto-denied** (nothing is written at all) — the
  launching tool's working directory is not enough on its own.

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
- Print-mode scratch dir (where writes land without `--add-dir`):
  `~/.gemini/antigravity-cli/scratch/`
- OAuth token when the keyring is unavailable (mode `0600`):
  `~/.gemini/antigravity-cli/antigravity-oauth-token`
- Onboarding/UI state: `~/.gemini/antigravity-cli/jetski_state.pbtxt`

## Quick Reference

### Wrapper commands
- `agy agent` / `agy agents`
- `agy changelog`
- `agy help`
- `agy install`
- `agy mcp`
- `agy models`
- `agy plugin` / `agy plugins`
- `agy remote-control`
- `agy update`

### Useful flags
- `--add-dir`
- `--continue` / `-c`
- `--conversation`
- `--dangerously-skip-permissions`
- `--effort`
- `--input-format`
- `--json-schema`
- `--model`
- `--output-format`
- `--print` / `-p`
- `--print-timeout`
- `--prompt`
- `--prompt-interactive` / `-i`
- `--sandbox`
- `--log-file`
- `--version`

### Plugin subcommands (`agy plugin --help`)
- `list`, `import [source]`, `install <target>`, `uninstall <name>`,
  `enable <name>`, `disable <name>`, `validate [path]`, `link <mp> <target>`,
  `help`

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

### Common settings keys (`settings.json`)
- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### Permission modes
`request-review`, `always-proceed`, `strict`, `proceed-in-sandbox`.

### Sandbox behavior
- `enableTerminalSandbox` is a boolean in `settings.json`; default `false`.
- Launch-time overrides (`--sandbox`, `--dangerously-skip-permissions`) can
  supersede persistent settings for the current session.

## Authentication behavior

- The CLI tries the OS secure keyring first. On a host with no keyring
  (`org.freedesktop.secrets`) it silently falls back to a file token at
  `~/.gemini/antigravity-cli/antigravity-oauth-token` (mode `0600`), so
  "keyring unavailable" in the log is normal, not a failure.
- With no saved session it prints an authorization URL and waits for the code
  shown on `https://antigravity.google/oauth-callback` to be pasted back;
  locally it can open the default browser instead.
- Sign-in is one-shot: **each new authorization request is a fresh Google
  sign-in**, so restarting the flow (or killing the waiting process) triggers
  another 2-Step challenge. Start it once and let it wait — an abandoned
  attempt does not resume.
- Keep the pasted code out of the transcript by running the TUI under `tmux`
  (`tmux new-session -d -s agy-auth 'agy'`) and pasting from a file the agent
  wrote: `tmux load-buffer -b agycode code.txt`, then
  `tmux paste-buffer -b agycode -t agy-auth` and
  `tmux send-keys -t agy-auth Enter` — the code reaches the TTY without
  entering the agent's context.
- After the first sign-in, the interactive run also walks theme →
  **interaction-data consent (checkbox defaults to ON)** → folder trust for the
  launch directory. `Tab` moves focus to the Previous/Done buttons — arrow keys
  alone do not.
- `/logout` removes saved credentials.

## Plugins

- Plugins stage under `~/.gemini/antigravity-cli/plugins/<plugin_name>/`.
- They can bundle skills, agents, rules, MCP servers, and hooks.
- `agy plugin list` returning no imported plugins is a valid empty state.

## Pitfalls

- `agy help` shows wrapper commands, not interactive slash commands.
- `agy --version` is the safe non-interactive version check; `agy version` is
  interactive and can fail without a real TTY.
- First place to look for failures: `~/.gemini/antigravity-cli/log/cli-*.log`
  (read with `read_file`).
- Don't confuse persistent JSON settings with launch-time overrides.
- `~/.gemini/antigravity-cli/bin/agentapi` is a thin wrapper to `agy agentapi`.
- On WSL, token storage is file-based, so auth issues are usually local-file /
  session-state problems, not browser-only problems.
- Workspace identity can depend on launch directory and the `.antigravitycli`
  project marker.
- `agy -p` takes the prompt as its **value** (`-p "…"`); a bare positional
  prompt is rejected, and `-p` with no value fails with `flag needs an
  argument`. Flags may follow `-p` — the prompt does not have to come last.
- stdin is a prompt source only when `-p` is omitted: piping a prompt into
  `agy --print-timeout 1m` runs it, while `echo … | agy -p` errors.
- A print run's file writes need an explicit target: an absolute path is
  honoured, while a bare relative filename can land under
  `~/.gemini/antigravity-cli/scratch/` instead of the launch directory. Without
  `--add-dir <repo>`, an allow-rule or `--dangerously-skip-permissions`, a
  headless write is auto-denied — `workdir=` on the `terminal` tool is not
  enough on its own.
- Bound print runs with `--print-timeout`, not `--max-turns` (which does not
  exist on `agy`). The default is `0`, i.e. unbounded, and the value needs a
  unit — a bare `--print-timeout 20` is rejected and prints help instead.

## Verification

Confirm the install is real and usable, all through the `terminal` tool (read
files with `read_file`):

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy plugin list")`
5. `terminal(command="agy models")` — needs an authenticated session; the list
   doubles as a check that sign-in is live
6. `read_file` on `~/.gemini/antigravity-cli/settings.json`
7. `read_file` on the latest `~/.gemini/antigravity-cli/log/cli-*.log`
8. If needed, `read_file` on `~/.gemini/antigravity-cli/keybindings.json`

## Support files

- `references/cli-docs.md` — condensed notes from the getting-started, usage,
  and features docs.
