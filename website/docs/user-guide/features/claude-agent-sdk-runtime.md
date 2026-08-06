---
title: Claude Agent SDK Runtime (optional)
sidebar_label: Claude Agent SDK Runtime
---

# Claude Agent SDK Runtime

Hermes can optionally hand a whole turn to Anthropic's official
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which wraps the Claude Code CLI as a subprocess. When enabled, your Claude Pro / Max / Team / Enterprise plan drives inference: Claude runs its own model loop, and Hermes becomes the shell around it (sessions DB, tools, approvals, environments, slash commands, gateway, memory and skill review).

This is **opt-in and default-off**. Nothing changes unless you flip the flag, and Hermes never auto-routes you onto this runtime.

:::caution Default-off pending a policy answer
Anthropic's Agent SDK overview says third-party developers may not *offer* claude.ai login for their products without prior approval, while Anthropic's Help Center says SDK usage still draws from your subscription's limits. Those two statements are not reconcilable by reading harder, and the risk of guessing wrong lands on your account. Hermes therefore ships the runtime turned off; enable it only for your own account, knowingly. See `docs/design/claude-subscription-via-agent-sdk.md` for the full record.
:::

## Why

- Run Claude turns against your **Claude subscription** instead of a metered API key.
- Hermes holds **no Claude credential at all** — no token reading, no credential files, no OAuth flow. You sign in with `claude auth login` and the SDK resolves credentials itself.
- The traffic **is** Claude Code's, rather than Hermes' traffic dressed up to look like it. That is the entire point of the change.
- **Claude owns its own context and compaction**, so prompt-cache continuity and message alternation are handled by the side that actually knows the exact bytes.

## Two Claude providers, and how billing differs

`claude-code` and `anthropic` are separate providers on purpose. Blurring them is how a subscription silently becomes an API bill.

|  | **Anthropic API** (`anthropic`) | **Claude subscription** (`claude-code`) |
|---|---|---|
| Auth | `ANTHROPIC_API_KEY` you pasted | `claude auth login`, handled by the SDK |
| Billed to | your Anthropic Console org, metered per token | your Claude plan's usage limits |
| Request shape | Hermes builds `/v1/messages` | the Claude Code CLI, via the SDK |
| Runs the loop | Hermes | Claude |
| Credential held by Hermes | yes | none |

Hermes still records token counts for a subscription turn so `/status`, session accounting, and the dashboard keep working — but a subscription turn is reported as `subscription_included`, not as a dollar charge. The SDK also reports what the same turn *would* have cost on the API; that number is surfaced for comparison only and is never added to your API spend.

### Why Hermes refuses to run when `ANTHROPIC_API_KEY` is set

The two things above are different accounts, and it is easy to end up on the wrong one without noticing.

- **Anthropic API** is a pay-as-you-go Console account. Every token is a charge on a card.
- **Claude subscription** is your Pro / Max / Team / Enterprise plan. Usage draws on the limits you already pay a flat fee for.

The Claude Code CLI decides which one to use by looking at your environment, and it prefers a key over your plan. Its order is:

1. Cloud provider credentials (`CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_USE_FOUNDRY`)
2. `ANTHROPIC_AUTH_TOKEN`
3. `ANTHROPIC_API_KEY`
4. an `apiKeyHelper` script in your Claude settings
5. `CLAUDE_CODE_OAUTH_TOKEN`
6. **your subscription login — last**

So if you have `ANTHROPIC_API_KEY` exported (Hermes' setup wizard encourages one, so most people do), a "Claude subscription" turn would quietly be billed to your Console account instead.

Hermes does not let that happen. Before a subscription turn starts it checks which credential would win, and if anything outranks your plan it **refuses the turn** and tells you exactly what to unset:

```
Refusing to start the Claude subscription runtime: this turn would not be billed to your Claude plan.
  • ANTHROPIC_API_KEY is set, and the Claude Code CLI prefers it over your subscription —
    requests would bill your Anthropic Console account, as metered API usage.
    Fix: unset ANTHROPIC_API_KEY
  Check what the CLI would use with: claude auth status
```

**To check it yourself:**

```bash
claude auth status
```

`"authMethod": "claude.ai"` with a `subscriptionType` means your plan pays. `"authMethod": "api-key"` means a key pays — run `claude auth login`, and unset the variable the refusal named.

**To fix it:** unset the variable in the shell you start Hermes from (`unset ANTHROPIC_API_KEY`), or remove it from `~/.hermes/.env` if that is where it comes from. If you need the key for the `anthropic` provider as well, keep them in separate shells or profiles — Hermes will not silently pick one for you.

Belt and braces: even once the check passes, Hermes launches the Claude CLI from an environment it builds explicitly, with every one of those variables removed, so nothing that appears later can redirect the bill. Your own environment is never modified — other providers keep seeing their keys exactly as before.

## Who owns what

The runtime is only tractable because each side owns its half completely and neither reconstructs the other's state.

**Hermes owns** — and these behave identically to every other provider:

- The **system prompt**. Hermes sends its own, byte-for-byte the string it already built for your conversation. Not the SDK's `claude_code` preset. Your SOUL.md identity, memory instructions, context files, and skills all still apply.
- **Every tool.** The SDK's built-in Bash / Read / Edit / Glob / Grep tools are switched off entirely. Claude calls Hermes' tools instead, through an in-process bridge, and each call runs the normal Hermes lifecycle: approvals, guardrails, checkpoints, plugin hooks, progress events.
- **Execution environments.** Tools run wherever your session runs — local, Docker, SSH, Modal, Daytona, or Singularity — because the bridge follows the turn's task id.
- **Permissions and approvals**, plugins, memory, skills, the session picker, transcripts, and the UI.
- The **toolset is pinned** for the conversation. A project or user `.mcp.json` cannot widen it mid-session, and the CLI is told not to load a second copy of `CLAUDE.md`, settings, hooks, or skills that would compete with Hermes' context assembly.

**Claude owns:**

- Claude-native context and its opaque frames.
- **Prompt-cache continuity** across turns.
- **Automatic compaction.** When Claude compacts, Hermes records the boundary and shows the usual compression status, but does not rewrite your visible transcript.
- Assistant/tool-result alternation, message UUIDs, the resume cursor, turn usage, and the terminal reason.

## What it looks like while running

Claude's stream is projected into the same callbacks every other Hermes runtime fires, so nothing renders differently:

- Live assistant text and thinking deltas in the CLI, TUI, desktop app, and messaging gateways.
- Tool start/completion cards with stable ids, using the plain Hermes tool name (`terminal`, `web_search`) rather than the bridge's internal MCP naming.
- Interim commentary, honoring `display.show_commentary`.
- Compaction status, token/usage accounting, and the memory + skill review nudges.

Interrupting works too: `/stop` asks Claude to abort, the runtime drains the interrupted response to completion, and the session stays usable for the next turn.

## Prerequisites

1. **Claude Code installed and signed in.**
   ```bash
   claude auth login
   claude auth status    # should report a plan login, not "api-key"
   ```
   If `claude auth status` says you are signed in with an API key, that is metered API billing, not your plan.

2. **The optional `claude-code` extra.** The SDK's platform wheels bundle the Claude executable and are ~80 MB, so it is never a base dependency and is not part of `[all]`.
   ```bash
   pip install 'hermes-agent[claude-code]'
   ```

3. **Turn the gate on** in `~/.hermes/config.yaml`:
   ```yaml
   claude_subscription:
     enabled: true
   ```

If any of the three is missing, the turn refuses to start and tells you which one — it does not fail with a stack trace or silently fall back to a different billing source.

## Enabling

With the three prerequisites met, select the provider the same way as any other:

```yaml
model: claude-sonnet-4-5
provider: claude-code
```

While `claude_subscription.enabled` is `false`, the legacy `claude-code` and `claude-oauth` slugs keep resolving to the `anthropic` provider, so an existing config is never silently repointed at a different billing source.

## Rolling back

Set `claude_subscription.enabled: false`. Nothing else is required: the runtime is purely additive, and it never alters or deletes Hermes transcripts, provider bindings, or Claude's own credentials. Every other provider stays available and unchanged, so a rolled-back setup is exactly the one you had before.

## Known limitations

- **Session resume is not wired up yet.** Each Hermes session starts a fresh Claude session; the SDK session id is captured but not yet used to resume or fork.
- **Auxiliary/side-LLM tasks** (title generation, the curator, vision fallbacks, the goal judge) do not route through this runtime. They keep using whatever `auxiliary.*` provider you have configured.
- **Subagents** (`delegate_task`) run on the normal provider path, not on the SDK.
- **A conflicting credential blocks the runtime rather than being worked around.** Hermes refuses instead of quietly stripping the variable, because which account pays is your decision to make, not ours. See the billing section above.
