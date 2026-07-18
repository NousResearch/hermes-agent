---
sidebar_position: 40
title: "Tool Progress Grouping"
description: "Control whether gateway tool-progress edits one message in place or sends one message per tool call, and how per-platform overrides work."
---

# Tool Progress Grouping

Tool progress shows status updates like `💻 ls -la...` as the agent runs. On platforms that support **message editing**, Hermes can either keep a single bubble and update it as each tool starts, or send a new bubble for every tool. That choice is `tool_progress_grouping`.

| Value | Behavior | When to use |
|------|----------|------|
| `accumulate` | Edit one message in place as tools run | Default. Best when editing is cheap and you want low-noise progress. |
| `separate` | Send one message per tool call | Pre-v0.9 behavior. Use it if edited bubbles feel jittery or you want an explicit run trace. |

This only applies when `display.tool_progress` is enabled (`off | new | all | verbose`). It’s irrelevant if tool progress is disabled.

## Configuring it

Global default in `~/.hermes/config.yaml`:

```yaml
display:
  tool_progress: all
  # accumulate = edit one bubble in place (default)
  # separate   = one message per tool call
  tool_progress_grouping: accumulate
```

Per-platform override:

```yaml
display:
  platforms:
    telegram:
      tool_progress: all
      tool_progress_grouping: accumulate
    discord:
      tool_progress: all
      tool_progress_grouping: separate
```

Possible values are `accumulate` and `separate`. Unknown values fall back to `accumulate`.

## Platform caveats

- **Editing-backed platforms** (Telegram, Discord, Slack, Matrix, Feishu/Lark, WeCom, etc.) can update one bubble in place. Use `accumulate` whenever your client turn-around feels slow; it keeps chat tidy.
- **No-edit / fire-and-forget platforms** such as SMS, ntfy, some webhook flows, or setups with restricted bot posting cannot edit a previous message. On these transports, `accumulate` effectively behaves like a compacted delivery of visible progress updates rather than true live editing.
- For platforms where message editing is unreliable or rate-limited, try `separate`. The main tradeoff is volume: a run with many tool calls can flood the chat.

## Interaction with other settings

- `tool_progress: off` hides tool progress entirely; grouping has no effect.
- `tool_progress: new` shows only new tool starts; `all` chains all status updates in the surface Hermes chooses for that platform.
- `display.background_process_notifications` is separate and not affected by grouping.
- `busy_input_mode` is separate and not affected by grouping.
