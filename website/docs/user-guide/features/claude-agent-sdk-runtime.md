---
title: Claude Agent SDK Runtime (subscription)
sidebar_label: Claude Agent SDK Runtime
---

# Claude Agent SDK Runtime

Hermes can hand entire turns to Anthropic's official [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview), which drives the Claude Code CLI's own agent loop under **Claude subscription OAuth** — never a metered API key. It is the structural twin of the [Codex App-Server Runtime](/user-guide/features/codex-app-server-runtime): the external agent runs the loop and its tools; Hermes stays the shell around it (sessions DB, gateway platforms, memory, transcripts, slash commands).

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

## Billing posture (fail-closed)

This provider exists to bill the **subscription**. Accordingly:

- If a metered `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` is set, the runtime **refuses to start** rather than silently switch billing. Set `agent.claude_agent_sdk.allow_metered_key: true` to explicitly allow it.
- The spawned CLI's environment gets metered billing vectors neutralized (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, AWS static credentials, `GOOGLE_APPLICATION_CREDENTIALS`) unless `allow_metered_key` is set. The subscription token flow and HOME/PATH are untouched.
- Usage is recorded as `subscription_included` — token counts are tracked, cost shows as *included*.

## Configuration

All keys live under `agent.claude_agent_sdk` in `config.yaml` (see `cli-config.yaml.example`):

| Key | Default | Meaning |
| --- | --- | --- |
| `streaming` | `false` | Emit the SDK's partial-message deltas into the gateway streaming pipeline. |
| `allow_metered_key` | `false` | Allow startup with a metered Anthropic key present (disables the fail-closed guard AND the env scrub). |
| `append_file` | `""` | Operator persona/soul file appended to the system prompt. |
| `permission_mode` | `""` | An SDK permission mode literal (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto`). Empty keeps the `HERMES_TERMINAL_SECURITY_MODE` mapping. Set `default` to route SDK tool permissions through Hermes' approval flow. |
| `max_budget_usd` | `null` | Per-query USD cap forwarded to the SDK; the turn ends with `error_max_budget_usd` when exceeded. `null` = no budget. |

### Permission posture, honestly

The default mapping (`HERMES_TERMINAL_SECURITY_MODE=auto`) selects the SDK's `acceptEdits` mode: file edits under the working directory are auto-approved and **no Hermes approval callback is in the loop**. This is the closest usable-unattended mode, not codex parity. Hermes' approval callback is bridged only in `default` mode (`permission_mode: default` or `HERMES_TERMINAL_SECURITY_MODE=approval-required`).

Ambient Claude settings are isolated: the runtime pins the SDK's `setting_sources` to the empty list, so `~/.claude/settings.json` and project `.claude/settings*.json` cannot re-permission tools or add hooks underneath the configured posture. (This also means `CLAUDE.md` files are not loaded — this runtime composes its own system-prompt append from Hermes' memory, skills index, and your `append_file`.)

## What Hermes still provides

- **hermes-tools MCP server** — memory and `session_search` shims (plus the standard Hermes tool surface) are exposed into the SDK's loop over stdio.
- **Transcripts and continuity** — the SDK's typed message stream is projected into Hermes' messages shape and persisted; across gateway restarts the runtime resumes the same SDK session, and a failed resume retries fresh with a bounded continuity digest.
- **Interrupts** — `/stop` and new-message preemption route into the SDK's interrupt.

## Limitations

- Auxiliary tasks (title generation, compression) do **not** auto-detect a metered fallback while this provider is active — aux fails closed unless you explicitly configure an auxiliary provider.
- The background memory/skill review pass is skipped on this runtime (the review fork cannot write through the SDK's tool surface).
- Model names are Claude model ids (e.g. `claude-opus-4-8`); leave unset to use the CLI's default model.
- With `model.provider: claude-agent-sdk` pinned in `config.yaml`, a bare `-m <claude-model-id>` stays on this provider — the pin survives model→provider inference, and short aliases (`-m sonnet`) resolve within it. Without a pinned provider, Claude model ids route to the native `anthropic` (metered API) provider as usual. Known residual: dot-form ids absent from the curated catalog (e.g. `claude-opus-4.8`) still leave the pin — use the dash-form ids.
