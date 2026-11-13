---
title: "Codex — Delegate coding to OpenAI Codex CLI (features, PRs)"
sidebar_label: "Codex"
description: "Delegate coding to OpenAI Codex CLI (features, PRs)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Codex

Delegate coding to OpenAI Codex CLI (features, PRs).

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/autonomous-ai-agents/codex` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Codex`, `OpenAI`, `Code-Review`, `Refactoring` |
| Related skills | [`claude-code`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code.md), [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Codex CLI

Delegate coding tasks to [Codex](https://github.com/openai/codex) via the Hermes `terminal` tool. Codex is OpenAI's autonomous coding agent CLI.

## When to use

- Building features
- Refactoring
- PR reviews
- Batch issue fixing

Requires the Codex CLI. Flags checked with codex-cli 0.167.2 (2026-11-12).

## Prerequisites

- Codex installed: `npm install -g @openai/codex`
- Check authentication with `codex login status`; use `codex login` if needed.
  CLI login credentials can authenticate without `OPENAI_API_KEY`.
- Outside a trusted directory, `codex exec` requires `--skip-git-repo-check`.
  Initialise a git repository instead when you want diffs or reviews.
- Close stdin with `</dev/null` or use `pty=true` for `codex exec`.
  Non-terminal stdin is read even with a prompt argument; an open pipe can
  leave it waiting at "Reading additional input from stdin...". This matters
  especially for background runs. Use `pty=true` for interactive `codex`.

Hermes's own `openai-codex` provider uses Hermes-managed OAuth after
`hermes auth add openai-codex`, separate from the standalone Codex CLI login.

Shell examples use POSIX stdin redirection; on Windows use a compatible shell
or `pty=true`.

## One-Shot Tasks

```
terminal(command="codex exec 'Add dark mode toggle to settings' </dev/null", workdir="~/project")
```

For scratch work without a repository:
```
terminal(command="cd $(mktemp -d) && codex exec --skip-git-repo-check 'Build a snake game in Python' </dev/null")
```

## Background Mode (Long Tasks)

```
# Start in background with stdin closed
terminal(command="codex exec --sandbox workspace-write 'Refactor the auth module' </dev/null", workdir="~/project", background=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Key Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | One-shot execution, exits when done |
| `-s, --sandbox <mode>` | Command sandbox: `read-only`, `workspace-write`, or `danger-full-access`; not an approval-policy switch |
| `--dangerously-bypass-approvals-and-sandbox` | No sandbox, no approvals (fastest, most dangerous; `--yolo` still works as a hidden alias) |
| `--skip-git-repo-check` | Allow `exec` outside a Git repository |
| `--json` | Emit `exec` events as JSONL on stdout |
| `-o, --output-last-message <file>` | Write the final `exec` response to a file |
| `--ephemeral` | Run `exec` without persisting session files |
| `-C, --cd <dir>` | Set the `exec` working root |
| `-m, --model <model>` | Select the model for `exec` |

> **Removed:** `--full-auto` fails with "unexpected argument '--full-auto' found".
> Use `--sandbox workspace-write` for workspace-limited command execution.

## Hermes Gateway Caveat

When invoking the Codex CLI from a Hermes gateway/service context (for example,
Telegram-driven agent sessions), Codex `workspace-write` sandboxing may fail even
when the same command works in the user's interactive shell. A typical symptom is
bubblewrap/user-namespace errors such as `setting up uid map: Permission denied`
or `loopback: Failed RTM_NEWADDR: Operation not permitted`.

In that context, prefer:

```
codex exec --sandbox danger-full-access "<task>" </dev/null
```

Use process boundaries as the safety layer instead: explicit `workdir`, clean git
status before launch, narrow task prompts, `git diff` review, targeted tests, and
human/agent confirmation before committing broad changes.

Codex MCP servers configured in `~/.codex` (inspect with `codex mcp list`)
may print `rmcp::transport` ERROR lines on every run when authentication fails.
Those lines alone do not mean the task failed; check completion and the exit status.

## PR Reviews

`codex review` accepts `--uncommitted` (staged, unstaged and untracked changes),
`--base <BRANCH>`, or `--commit <SHA>`. It also accepts an optional review prompt;
check `codex review --help` for usage.

Clone to a temp directory for isolated review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main </dev/null")
```

## Parallel Issue Fixing with Worktrees

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 ~/.hermes/cache/scratch/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 ~/.hermes/cache/scratch/issue-99 main", workdir="~/project")

# Launch Codex in each
terminal(command="codex exec --sandbox workspace-write 'Fix issue #78: <description>. Commit when done.' </dev/null", workdir="~/.hermes/cache/scratch/issue-78", background=true)
terminal(command="codex exec --sandbox workspace-write 'Fix issue #99: <description>. Commit when done.' </dev/null", workdir="~/.hermes/cache/scratch/issue-99", background=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd ~/.hermes/cache/scratch/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove ~/.hermes/cache/scratch/issue-78", workdir="~/project")
```

## Batch PR Reviews

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86' </dev/null", workdir="~/project", background=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87' </dev/null", workdir="~/project", background=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## Rules

1. **Close stdin or use PTY** — `codex exec ... </dev/null` prevents waiting on an open pipe; `pty=true` is an alternative and supports interactive `codex`
2. **Scratch without Git** — use `codex exec --skip-git-repo-check`; use `git init` when diffs or review are wanted
3. **Use `exec` for one-shots** — supply a prompt and close stdin
4. **`--sandbox workspace-write` for building** — limits command writes; `--full-auto` is removed
5. **Background for long tasks** — use `background=true`, close stdin (or use PTY), and monitor with `process`
6. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
7. **Parallel is fine** — run multiple Codex processes at once for batch work
