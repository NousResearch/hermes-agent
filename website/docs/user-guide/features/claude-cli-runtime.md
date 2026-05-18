---
title: Claude CLI Runtime (optional)
sidebar_label: Claude CLI Runtime
---

# Claude CLI Runtime

Hermes can optionally hand a whole turn to the local Claude Code CLI instead of using Hermes' Anthropic API/OAuth transport. This is useful when you already use Claude Code locally and want Hermes gateway, Kanban, and cron flows to run through the same `claude` login and subscription.

This runtime is **opt-in only**. Default Hermes behavior is unchanged unless you configure it.

## When To Use It

Use the Claude CLI runtime when:

- You want Hermes to use Claude Code's local authentication instead of an Anthropic API key.
- You want to run Kanban workers through `claude -p` while keeping Hermes' task lifecycle.
- You prefer Claude Code's local model selection and subscription behavior for a specific Hermes profile.

Use the default Hermes runtime when:

- You need Hermes' native streaming/tool loop for fine-grained tool events.
- You depend on provider-level Anthropic features that are not exposed through `claude -p`.
- You need exact API credentials or routing through Hermes credential pools.

## Configuration

The direct form is:

```yaml
# ~/.hermes/config.yaml
model:
  provider: claude-cli
  default: sonnet
```

You can also keep `provider: anthropic` and ask Hermes to use the CLI transport:

```yaml
model:
  provider: anthropic
  default: sonnet
  claude_runtime: cli
```

Hermes resolves both forms to:

- `provider: claude-cli`
- `api_mode: claude_cli`
- no Hermes Anthropic API key required
- one `claude -p` process per turn

To use a non-default executable:

```yaml
model:
  provider: claude-cli
  default: sonnet
  claude_cli_bin: /opt/homebrew/bin/claude
```

The equivalent environment override is:

```bash
HERMES_CLAUDE_CLI_BIN=/opt/homebrew/bin/claude
```

## Runtime Options

| Setting | Description |
|---------|-------------|
| `model.default` | Model name passed to `claude -p --model`. |
| `model.claude_runtime: cli` | Enables the CLI runtime while keeping `provider: anthropic`. |
| `model.claude_cli_bin` | CLI executable path. |
| `HERMES_CLAUDE_CLI_BIN` | Environment fallback for the executable path. |
| `HERMES_CLAUDE_CLI_MODEL` | Fallback model when Hermes did not pass one. Prefer `model.default` for normal profiles. |
| `HERMES_CLAUDE_CLI_TIMEOUT_SECONDS` | Per-turn timeout, default `900`. |
| `HERMES_CLAUDE_CLI_EXTRA_ARGS` | Extra shell-style arguments appended to the `claude -p` command. |

## Status And Auth

`hermes status` shows `Claude CLI` when this runtime is active. Hermes checks `claude auth status` and reports the CLI auth method and subscription when the local CLI exposes them.

Hermes does not require `ANTHROPIC_API_KEY`, Anthropic OAuth, or Hermes credential-pool entries for this runtime. The Claude Code CLI owns authentication.

## Kanban Workers

When a Hermes Kanban worker runs through Claude CLI, Hermes attaches a small MCP bridge named `hermes-tools` to the `claude -p` process. That bridge exposes the Kanban lifecycle tools the dispatcher needs:

- `kanban_show`
- `kanban_comment`
- `kanban_block`
- `kanban_complete`
- `kanban_heartbeat`
- related list/link/unblock helpers

The worker prompt tells Claude to finish every assigned task with `kanban_complete` or `kanban_block`. If the CLI returns final prose while the task is still `running`, Hermes has a last-resort guard that calls `kanban_block` with an `external-runtime-prose:` reason so the dispatcher does not record a clean-exit protocol violation.

## Verification

After upgrading Hermes or Claude Code CLI, run:

```bash
./scripts/run_tests.sh \
  tests/agent/transports/test_claude_cli_session.py \
  tests/run_agent/test_claude_cli_integration.py \
  tests/hermes_cli/test_runtime_provider_resolution.py::test_resolve_runtime_provider_claude_cli_skips_anthropic_oauth \
  tests/hermes_cli/test_status.py::test_show_status_explains_claude_cli_runtime \
  -q
```

Then do a live smoke test from a Claude-CLI-backed profile:

```bash
HERMES_PROFILE=claude-cli-smoke hermes -z 'Reply exactly OK.'
```

Expected result: `OK`. If Hermes asks for Anthropic credentials, runtime resolution is not selecting `api_mode: claude_cli`.
