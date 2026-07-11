---
title: Claude Max Kanban Runtime
sidebar_label: Claude Max Kanban Runtime
---

# Claude Max Kanban Runtime

Hermes can run a Kanban worker through the official Claude Agent SDK and the
Claude Code login attached to a Claude Max subscription. This is an opt-in
whole-agent runtime: Claude owns the worker loop while Hermes keeps ownership
of the board, session record, fallback chain, and worker lifecycle.

The current boundary is intentionally narrow: **Kanban workers on macOS only**.
Keep the board orchestrator on Hermes' normal runtime.

## Cost prerequisite

Before enabling this runtime, disable **Usage Credits / extra usage** on the
Anthropic account. The runtime requires `claude auth status` to attest all of
the following before a worker starts:

- logged in through `claude.ai`;
- first-party Anthropic service;
- `max` subscription type.

It also aborts when the SDK reports an overage route and records attested usage
as subscription-included rather than estimating an API cost.

:::warning No software-only zero-charge guarantee
The Claude CLI auth response proves that the process is using a Max login, but
it does not expose the account-level Usage Credits switch. Hermes cannot turn
that setting off or verify it locally. An SDK rate-limit/overage event is only
observable after a request has begun. Treat disabled Usage Credits as a hard
operator prerequisite, not something the runtime can enforce completely.
:::

Do not put `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or another Anthropic API
credential in the worker profile. The Claude subprocess receives an exact
allowlisted environment and will not receive those values, but removing them
also prevents a confusing native API route elsewhere in the profile.

Zero-extra operation also requires **every** fallback and Mixture-of-Agents
reference/aggregator route to be subscription-backed or local. A Max-backed
acting worker does not make a native Anthropic, OpenAI, OpenRouter, or other API
advisor free; those routes keep their own billing semantics.

Automatic auxiliary tasks do not reinterpret a Claude SDK main route as native
Anthropic and do not search ambient credentials or main fallback routes. Title
generation, compression, and similar `provider: auto` tasks skip gracefully.
To enable one, configure `auxiliary.<task>.provider` and `.model` explicitly;
that route has its own billing semantics and may cost extra.

## Prerequisites

1. macOS, because the worker command boundary uses Seatbelt (`sandbox-exec`);
2. Claude Code signed into the intended Max account:

   ```bash
   claude auth login
   claude auth status
   ```

3. Usage Credits / extra usage disabled in the Anthropic account UI;
4. a dedicated Hermes worker profile and an initialized Kanban board.

The exact pinned Agent SDK dependency is installed lazily on first use. To
preinstall it for an immutable deployment, install the optional extra:

```bash
pip install 'hermes-agent[claude-agent-sdk]'
```

## Worker profile configuration

Create a dedicated profile rather than changing the orchestrator profile:

```bash
hermes profile create claude-worker --clone
```

In that profile's `config.yaml`, select the provider-neutral runtime:

```yaml
model:
  default: claude-opus-4-6
  provider: anthropic
  runtime: claude_agent_sdk

# Optional, but recommended: another subscription/local runtime to use when
# Max is capped or temporarily unavailable. Each entry owns its own runtime.
fallback_providers:
  - provider: openai-codex
    model: gpt-5.4
    runtime: codex_app_server
```

Do not use a native `provider: anthropic` fallback without
`runtime: claude_agent_sdk` if the goal is subscription-only operation. Native
Anthropic is an API billing route.

Assign ready Kanban tasks to `claude-worker`. The gateway-embedded dispatcher
spawns `hermes -p claude-worker chat -q ...` with the task and workspace
identities pinned in its environment; direct chat sessions do not satisfy this
boundary and fail closed.

## Worker capabilities and isolation

The SDK worker receives a small in-process Hermes MCP surface:

- Kanban status, heartbeat, comment, complete, block, create, link, and related
  worker-safe board operations available to the assigned task;
- `read_file` and `write_file`, confined to regular files inside the worker
  workspace with symlink and linked-file rejection;
- `terminal`, run with an exact environment, workspace-only writes, network
  disabled, and a stable default-deny Seatbelt profile that reads only the
  workspace, read-only Git common directory, and explicit system/toolchain
  roots;
- `process`, limited to processes owned by the current worker task.

When the same Claude runtime performs a background memory/skill review, it gets
an independent capability mode containing only the requested `memory` and/or
`skill_manage` tool. Curator mode gets only `skill_manage`. Kanban, terminal,
process, and file capabilities are absent from those auxiliary SDK sessions;
missing required tool definitions fail closed.

Claude's built-in Bash, web tools, plugins, skills, and user setting sources are
disabled for this runtime. The worker receives the real host `HOME` only so the
Claude login can resolve from secure OS storage; arbitrary Hermes/provider
credentials are not inherited.

## Fallback behavior

Runtime identity belongs to each route, not to the provider name. This lets one
turn move between `claude_agent_sdk`, `codex_app_server`, a local model, or the
normal Hermes loop without starting a second Hermes process.

When a Max request is rejected for auth, quota, rate limit, overload, billing,
or a server failure, Hermes advances the configured fallback chain in the same
worker process. The circuit is persisted by profile, attested account, runtime,
provider, and model, so newly created agents respect an active reset window.

Hermes replays the original user turn only while it is safe. An unresolved tool
call fails closed. If Claude completed and recorded a tool action before the
failure, the next external runtime instead receives a continuation instruction
to inspect durable workspace, board, and process state and not repeat completed
actions.

## Supported surfaces

The provider-neutral route is propagated consistently by the CLI, messaging
gateway (including Telegram), TUI/Desktop gateway, ACP, and cron constructors.
The Claude SDK execution boundary itself still requires a dispatcher-created
Kanban task on macOS.

That means Desktop or Telegram can create, monitor, and receive results from a
Claude Max worker through the normal Kanban flow. They are not alternate ways
to start a direct Claude SDK chat. ACP and cron also fail closed if configured
for this runtime outside a Kanban worker instead of silently becoming native
Anthropic.

## Verification

Safe preflight, with no generative model call:

```bash
claude auth status
```

This checks the system Claude CLI and is a useful operator preflight, but it is
not the final runtime attestation. Hermes runs the SDK's pinned bundled Claude
CLI through an exact-environment wrapper and executes `auth status` against that
same wrapped binary before the worker starts. Confirm the system check reports
a first-party `claude.ai` Max login, then separately verify in the Anthropic
account UI that Usage Credits / extra usage is disabled.

The repository's automated tests verify route propagation, exact-environment
credential isolation, Max attestation parsing, workspace/process boundaries,
fallback replay rules, circuit persistence, message projection, and token
accounting without consuming Claude tokens. A true end-to-end Kanban smoke test
necessarily consumes Max-plan usage and cannot prove the account-level Usage
Credits switch; run it only after the manual account check.

## Current limitations

- macOS only;
- Kanban workers only; the orchestrator remains on Hermes' native runtime;
- no direct Desktop, TUI, ACP, Telegram, or cron Claude SDK chat;
- no Claude built-in Bash, network/web tools, plugins, or user settings;
- Usage Credits state cannot be attested through the CLI;
- the terminal rejects pre-existing hard-linked regular files before each
  sandbox start. Default-deny path rules also stop the sandboxed worker from
  creating an external hard link or following a late symlink outside the
  workspace. macOS Seatbelt is not a separate-user or VM boundary, however: a
  hostile same-UID host process could race a new hard link into a long-running
  sandbox after preflight. Use a dedicated OS account, copied private
  workspace, container, or VM when hostile sibling host processes are in the
  threat model. Workspaces whose dependency manager created hard-linked files
  must be recreated in copy mode first (for uv, set `UV_LINK_MODE=copy`).

For board setup and worker assignment, see [Kanban](./kanban) and
[Kanban worker lanes](./kanban-worker-lanes). For the fallback-chain format,
see [Fallback Providers](./fallback-providers).
