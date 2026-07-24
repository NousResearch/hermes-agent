---
sidebar_position: 19
title: "Context Audit"
description: "Audit startup prompt and tool-schema context budget without exposing raw prompt content"
---

# Context Audit

Hermes can collect a redacted startup context-budget audit for a session. The audit answers what was loaded before the first model call, how large each source is, which sources are operationally necessary, and where the largest safe savings probably are.

It reports metadata only:

- source labels such as `system_prompt.stable`, `system_prompt.context`, `system_prompt.volatile`, and `tool_schema.<name>`
- exact character counts for prompt text
- exact serialized byte counts for API tool schemas
- rough token estimates using `chars / 4` when provider tokenization is unavailable
- SHA-256 content hashes truncated for comparison
- necessity ranks and optimization options

It does **not** print raw memory contents, raw `USER.md`, full prompt text, secrets, or full tool schemas.

## Configuration

The feature is opt-in. Default behavior is unchanged and adds no prompt text.

```yaml
agent:
  startup_context_audit: off  # off | summary | status | debug_file
```

| Mode | Behavior |
|---|---|
| `off` | Default. No startup audit is collected for normal sessions. |
| `status` | Collect once at startup and expose the redacted report through `/context-audit`. No prompt injection. |
| `summary` | Collect once and append a compact redacted summary to the startup prompt. Use sparingly; this intentionally spends a small amount of context to make the budget visible. |
| `debug_file` | Same as `status`, plus writes a redacted JSON report under `~/.hermes/sessions/context_audits/`. |

Enable with review:

```bash
hermes config set agent.startup_context_audit status
```

Then start a new session and run:

```text
/context-audit
```

:::caution Live runtime changes
Changing `agent.startup_context_audit`, enabled toolsets, model provider, API key wiring, or background-model config affects future runtime behavior. Review the config diff and restart deliberately; Hermes does not apply optimization recommendations automatically.
:::

## Necessity ranks

| Rank | Meaning |
|---|---|
| `critical` | Identity, safety, and operating posture that should not be disabled for token savings. |
| `high` | Project context, memory/profile context, or other high-value guidance. Trim only after preserving the behavior elsewhere. |
| `situational` | Useful in many sessions, but often configurable or scopeable by task. Tool schemas and broad discovery indexes usually land here. |
| `candidate_to_trim` | Duplicated, oversized, or unclassified material that should be reviewed before it keeps growing. |

## Common optimization levers

Start with the biggest safe contributors, not cosmetic edits:

1. **Tool schema payload** — use task-specific `enabled_toolsets` for cron jobs and subagents; review largest always-on toolsets before changing platform defaults.
2. **Available skills index** — keep discovery, but compact broad categories where the current mode does not need full descriptions.
3. **Project context** — keep `AGENTS.md` as a router and move deep procedure into skills or linked references.
4. **Identity and safety guidance** — do not remove critical blocks; only deduplicate repeated operational wording.
5. **Memory and user profile** — usually smaller than tools/skills, but prune when near profile limits.

The audit recommendations name config keys or commands for review. They never disable tools, rewrite prompts, or change live runtime config by themselves.
