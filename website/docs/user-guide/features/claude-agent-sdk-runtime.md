---
title: Claude Agent SDK Runtime (subscription)
sidebar_label: Claude Agent SDK Runtime
---

# Claude Agent SDK Runtime

Hermes can hand entire turns to Anthropic's official [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which drives the Claude Code CLI's own agent loop under **Claude subscription OAuth by default**. Known metered lanes fail closed unless the operator opts in explicitly. It is the structural twin of the [Codex App-Server Runtime](/user-guide/features/codex-app-server-runtime): the external agent runs the loop and its tools; Hermes stays the shell around it (sessions DB, gateway platforms, memory, transcripts, slash commands).

Select it like any provider:

```bash
hermes model         # pick "Claude Agent SDK"
# or
hermes chat -q "hello" --provider claude-agent-sdk
```

Accepted spellings for `--provider` / `provider:` config / `provider:model` syntax: `claude-agent-sdk`, `claude-sdk`, `claude-code-sdk`, `claude_agent_sdk`.

## Auth: the SDK owns it

There is no Hermes login flow and no API key. The SDK-managed CLI subprocess authenticates itself with your Claude subscription:

- `claude setup-token` (or `claude login`) on the machine, or
- `CLAUDE_CODE_OAUTH_TOKEN` in the environment.

`hermes doctor` shows a structural status row (env var / `~/.claude` credential files). macOS Keychain-stored logins are not probed by doctor — they still work at session start.

The Python package is an opt-in extra that lazy-installs at first use, or explicitly:

```bash
pip install 'hermes-agent[claude-agent-sdk]'
```

The extra pins `claude-agent-sdk>=0.2.140` together with `mcp` 2.x, so it installs alongside `[mcp]`, `[dev]` and `[all]` on one `mcp` major. (Earlier SDK releases pinned `mcp<2` and could not share a venv with the `hermes-tools` stdio server.)

### Authentication-policy boundary

Anthropic's current [authentication and credential-use policy](https://code.claude.com/docs/en/legal-and-compliance#authentication-and-credential-use) and [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview#get-started) direct third-party products and services to API-key authentication unless previously approved, and prohibit offering Claude.ai login or routing subscription credentials on users' behalf.

This mode is narrowly designed for an individual operator running a local Hermes process against that operator's own local Claude Code authentication. Hermes does not implement a hosted Claude login, collect the credential, or relay it for another user. That architectural boundary is **not** an Anthropic approval or a promise that policy will remain unchanged. Do not expose this subscription mode as authentication for a hosted or managed third-party service; use API-key authentication or obtain Anthropic's approval for that use case.

## Billing posture (fail-closed)

This provider exists to bill the **subscription**. Accordingly:

- If a metered `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` is set, the runtime **refuses to start** rather than silently switch billing. Set `agent.claude_agent_sdk.allow_metered_key: true` to explicitly allow it.
- The spawned CLI's environment gets metered billing vectors neutralized (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, metered-shaped `ANTHROPIC_TOKEN`, `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, AWS static credentials, `GOOGLE_APPLICATION_CREDENTIALS`) unless `allow_metered_key` is set. Subscription-shaped OAuth/setup tokens and HOME/PATH are untouched.
- The child is the final authority: `system/init.apiKeySource` and typed `RateLimitEvent` messages are inspected. With the default guard, a reported API-key source or enabled/active subscription Extra Usage stops and retires the SDK session before Hermes can silently continue on a metered lane.
- Safe turns are recorded as `subscription_included`. When `allow_metered_key: true` admits a child-reported metered lane, Hermes labels it `sdk_reported_metered` and persists the SDK-reported cost instead of claiming it was included.

## Configuration

All keys live under `agent.claude_agent_sdk` in `config.yaml` (see `cli-config.yaml.example`):

| Key | Default | Meaning |
| --- | --- | --- |
| `streaming` | `false` | Emit the SDK's partial-message deltas into the gateway streaming pipeline. |
| `allow_metered_key` | `false` | Explicit metered-billing opt-in. Allows API-key sources and subscription Extra Usage reported by the child, and disables the env scrub/guard that would otherwise refuse them. |
| `append_file` | `""` | Operator persona/soul file appended to the system prompt. |
| `append_total_max_chars` | `null` | Whole SDK system-prompt append budget in characters. `null` uses 22,000. Blocks are packed whole; evictions warn with internal content-free labels. Positive integer overrides are accepted, while invalid values fall back. |
| `permission_mode` | `""` | An SDK permission mode literal (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto`). Empty keeps the `HERMES_TERMINAL_SECURITY_MODE` mapping (`auto` maps to the fail-closed SDK `default` mode). |
| `env` | `{}` | Extra environment for the spawned Claude CLI. Values are stringified; metered-billing vectors are rejected unless `allow_metered_key` is true. |
| `setting_sources` | `[]` | Filesystem settings sources (`user`, `project`, `local`). Empty keeps the SDK isolated from ambient Claude settings and `CLAUDE.md`. |
| `max_budget_usd` | `null` | Per-query USD cap forwarded to the SDK; the turn ends with `error_max_budget_usd` when exceeded. `null` = no budget. |
| `max_buffer_size` | `null` | Maximum size of one CLI NDJSON message. `null` uses Hermes' 10 MiB limit rather than the SDK's 1 MiB default, which can terminate a turn on a large tool result. Positive integer overrides are accepted; invalid values warn and fall back. The pinned SDK currently measures Unicode code points despite documenting bytes. |
| `turn_timeout` | `null` | Activity-aware soft turn budget in seconds. `null` uses 600; active tools, approvals, and stream output suspend the idle verdict. |
| `post_tool_quiet_timeout` | `null` | Post-tool silence watchdog. `null` uses 90 seconds with streaming enabled and disables it without streaming; `0` disables it explicitly. |
| `deliver_background_results` | `false` | Proactively deliver completed background Agent-task answers through the gateway completion lane. |
| `hybrid_mcp_bridge` | `false` | Opt in to the in-process MCP bridge that exposes the FULL Hermes tool registry (proxified third-party MCPs + agent-level tools) to the SDK loop and permits direct HTTP MCP registration. Only headerless, non-templated, credential-free HTTP(S) URLs are eligible. The SDK serializes direct MCP config into child process arguments, so authenticated servers require a credential-safe relay. `false` (default) keeps the stdio `hermes-tools` wrapper only — byte-identical to the fcava-provider default. Off by default because the wide bridge exposes agent-level tools whose enablement is a security choice. |
| `hybrid_mcp_bridge_exclude` | `[]` | Tool/server names to drop from the hybrid bridge (both `hermes-tools` and `hermes-hybrid` buckets) and direct HTTP MCP registration. Ignored when `hybrid_mcp_bridge` is `false`. Use to keep the wide bridge for proxified MCPs without inheriting high-blast tools (`delegate_task`, `cron_*`, `read_terminal`, `terminal`). Match on the raw Hermes registry/server name (no `mcp__` prefix). |

### Permission posture, honestly

The default mapping (`HERMES_TERMINAL_SECURITY_MODE=auto`) selects the SDK's `default` mode and installs Hermes' approval callback. The fixed bounded `mcp__hermes-tools__read_file` and `search_files` identities bypass the approval round-trip because their handlers retain Hermes' protected-path checks; shell and mutation tools still require normal approval. Operators can explicitly select another SDK mode, including `acceptEdits`, through `permission_mode`.

Headless runs (`hermes chat -q`, cron) have no approver to answer that round-trip: only `read_file` and `search_files` are exempt, so any other `hermes-tools` call — `session_search` included — waits for an approval that never arrives and times out. Allow-list what a headless job needs, or keep it to the exempt inspection tools.

Ambient Claude settings are isolated: the runtime pins the SDK's `setting_sources` to the empty list, so `~/.claude/settings.json` and project `.claude/settings*.json` cannot re-permission tools or add hooks underneath the configured posture. (This also means `CLAUDE.md` files are not loaded — this runtime composes its own system-prompt append from Hermes' memory, skills index, and your `append_file`.)

## What Hermes still provides

- **hermes-tools MCP server** — a curated stdio surface: memory and `session_search` shims; browser/web/media/skills/TTS tools; and bounded `read_file` / `search_files` inspection. It does not expose shell, file mutation, process control, or generic Git tools. When `hybrid_mcp_bridge: true`, the standard surface becomes an in-process MCP server under the same name (`mcp__hermes-tools__*`) — operator grants stored in `~/.claude/settings.json` keep matching without a migration step. Extra bridge-only third-party MCP and agent-level tools are exposed separately as `mcp__hermes-hybrid__*`.
- **Transcripts and continuity** — the SDK's typed message stream is projected into Hermes' messages shape and persisted; across gateway restarts the runtime resumes the same SDK session, and a failed resume retries fresh with a bounded continuity digest.
- **Interrupts** — `/stop` and new-message preemption route into the SDK's interrupt.

## Limitations

- Auxiliary text tasks (title generation, compression, extraction) auto-route through one-shot Agent SDK queries on the same subscription. Built-in and MCP tools are disabled for those calls. If that SDK route is unavailable, aux fails closed instead of selecting a metered fallback; an explicitly configured auxiliary provider remains an operator opt-in.
- The background memory/skill review pass is skipped on this runtime (the review fork cannot write through the SDK's tool surface).
- Model names are Claude model ids (e.g. `claude-opus-4-8`); leave unset to use the CLI's default model.
- With `model.provider: claude-agent-sdk` pinned in `config.yaml`, a bare `-m <claude-model-id>` stays on this provider — the pin survives model→provider inference, and short aliases (`-m sonnet`) resolve within it. Without a pinned provider, Claude model ids route to the native `anthropic` (metered API) provider as usual. Known residual: dot-form ids absent from the curated catalog (e.g. `claude-opus-4.8`) still leave the pin — use the dash-form ids.
