---
title: "Claude Code Sdk — Orchestrate Claude Code through its official Python SDK"
sidebar_label: "Claude Code Sdk"
description: "Orchestrate Claude Code through its official Python SDK"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Claude Code Sdk

Orchestrate Claude Code through its official Python SDK.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/claude-code-sdk` |
| Path | `optional-skills/autonomous-ai-agents/claude-code-sdk` |
| Version | `1.1.0` |
| Author | Raghu Thiyagharajan (0xRaghu), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Claude`, `Anthropic`, `SDK`, `Sessions`, `Cost-Tracking` |
| Related skills | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Claude Code SDK Skill

Run persistent, programmatic Claude Code sessions through Anthropic's official
`claude-agent-sdk` Python package. The Hermes skill is named `claude-code-sdk`;
the package it installs and imports is named `claude-agent-sdk`.

This complements the interactive `claude-code` skill. Use this skill for
structured, resumable terminal calls; use `claude-code` when a visible CLI or
tmux workflow is more useful.

## When to Use

- Continue a Claude Code conversation across separate Hermes turns.
- Delegate work in several repositories using one handle per repository.
- Put a turn or dollar ceiling on an agent query.
- Track the SDK-reported cost of each query.

Queries on different handles can run in parallel. Queries on the same handle
are rejected while one is in flight so that two processes cannot resume and
overwrite the same Claude conversation concurrently.

## Prerequisites

Install Python 3.10 or newer and the official package:

```bash
python3 -m pip install --upgrade "claude-agent-sdk>=0.2.119"
```

The package bundles the Claude Code executable it uses. A separate global
`@anthropic-ai/claude-code` npm installation is not required. Set
`ANTHROPIC_API_KEY`, or use Claude Code credentials already available to the
SDK environment.

## How to Run

Invoke the dispatcher through Hermes's `terminal` tool. On Linux and macOS,
this shell setup respects the active `HERMES_HOME` profile and its default:

```bash
SCRIPT="${HERMES_HOME:-$HOME/.hermes}/skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py"
python3 "$SCRIPT" doctor
```

On native Windows, the default Hermes root is
`%LOCALAPPDATA%\hermes`; an explicit `HERMES_HOME` works on every platform.

Open a handle, query it, and close it when finished:

```bash
python3 "$SCRIPT" open /path/to/project
# {"session_id": "7c3a91fb22d4", ...}

python3 "$SCRIPT" query 7c3a91fb22d4 \
  "Inspect the authentication code and explain its main risk." \
  --max-turns 8 --max-budget-usd 1.00 --timeout 300

python3 "$SCRIPT" query 7c3a91fb22d4 \
  "Fix that risk and run the focused tests." \
  --max-turns 12 --max-budget-usd 2.00

python3 "$SCRIPT" costs 7c3a91fb22d4
python3 "$SCRIPT" close 7c3a91fb22d4
```

The first query starts a Claude session. Later queries pass its returned
`session_id` to `ClaudeAgentOptions.resume`, preserving context even though
each Hermes terminal call is a separate process.

## Quick Reference

| Command | Purpose |
|---|---|
| `doctor` | Report the installed package version and state directory. |
| `open <project_path>` | Register a project and return a 12-character handle. |
| `query <handle> <message>` | Query Claude, resuming prior context. |
| `list` | List active handles, activity, message count, and cost. |
| `costs <handle>` | Sum known per-query costs for one handle. |
| `close <handle>` | Remove a handle; repeated closes are harmless. |

`query` accepts these optional limits:

| Flag | SDK behavior |
|---|---|
| `--max-turns N` | Passes `max_turns=N` to `ClaudeAgentOptions`. |
| `--max-budget-usd N` | Passes `max_budget_usd=N` to `ClaudeAgentOptions`. |
| `--timeout N` | Cancels the local call after N seconds; default is 300. |

## Procedure

1. Run `doctor` to verify that `claude-agent-sdk` imports in the same Python
   environment that will run queries.
2. Call `open` once per project and retain the returned handle.
3. Call `query` serially for each handle. Use separate handles for parallel
   work, including parallel work in the same repository.
4. Check `total_cost_usd` in each response. A null per-query cost means the
   SDK did not report a value; it does not mean the query was free.
5. Call `costs` for an audit summary, then `close` the handle.

The dispatcher stores state under the active profile:

<!-- ascii-guard-ignore -->
```text
<HERMES_HOME>/skill-state/claude-code-sdk/
├── sessions.json
├── cost.log
├── .sessions.lock
└── .session-locks/
```
<!-- ascii-guard-ignore-end -->

When `HERMES_HOME` is unset, Hermes's normal platform fallback is used:
`~/.hermes` on Linux and macOS, or `%LOCALAPPDATA%\hermes` on Windows.
Session records idle for more than one hour are removed on the next command.
This only removes the local short handle; it does not delete Claude's own
conversation data.

## Pitfalls

- Do not install the deprecated package named `claude-code-sdk`. This skill
  imports the current package named `claude-agent-sdk`.
- Do not run two queries on one handle concurrently. The dispatcher returns a
  busy error instead of risking divergent resume state.
- A handle's project directory is fixed by `open`. Open another handle to use
  another working directory.
- `total_cost_usd` may be unavailable for some authentication or billing
  modes. Preserve `null` as unknown rather than displaying `$0.00`.
- Claude's tool permissions still apply. Configure permissions deliberately in
  the target project; this skill does not silently enable bypass mode.
- `close` removes only the Hermes record. No Claude process remains alive
  between calls, and no remote conversation is deleted.

## Verification

First verify the exact runtime without starting a billable query:

```bash
python3 "$SCRIPT" doctor
```

Expected output includes `"status": "ok"`, `"skill": "claude-code-sdk"`,
`"package": "claude-agent-sdk"`, and the installed package version.

For an authenticated end-to-end check:

```bash
python3 "$SCRIPT" open /tmp
python3 "$SCRIPT" query <handle> "Reply with exactly SDK_OK." --max-turns 1
python3 "$SCRIPT" query <handle> "Repeat your previous answer." --max-turns 1
python3 "$SCRIPT" close <handle>
```

Both query responses should contain `SDK_OK`. The second response proves that
resume works across separate processes.

## References

- [Claude Agent SDK for Python](https://platform.claude.com/docs/en/agent-sdk/python)
- [Official Python SDK repository](https://github.com/anthropics/claude-agent-sdk-python)
- [Package on PyPI](https://pypi.org/project/claude-agent-sdk/)
- [Architecture notes](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/autonomous-ai-agents/claude-code-sdk/references/architecture.md)
